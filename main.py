"""
NOIMA — Greek Sign Language Recognition Server
FastAPI backend: model inference + static file serving.

Usage:
    python main.py
    # → http://localhost:8000
"""

import base64
import io as _io
import json
import logging
import webbrowser
from threading import Timer
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import mediapipe as mp
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import NOIMAv2
from features import process_keypoints

# Landmark indices — must match frontend & test_pipeline.py
_POSE_INDICES = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
_FACE_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def _extract_kps(results) -> np.ndarray:
    """Extract 225-dim keypoint vector from a MediaPipe Holistic result."""
    kps = np.zeros(225, dtype=np.float32)
    off = 0
    for idx in _POSE_INDICES:
        lm = results.pose_landmarks.landmark[idx] if results.pose_landmarks else None
        kps[off], kps[off+1], kps[off+2] = (lm.x, lm.y, lm.z) if lm else (0, 0, 0)
        off += 3
    for i in range(21):
        lm = results.left_hand_landmarks.landmark[i] if results.left_hand_landmarks else None
        kps[off], kps[off+1], kps[off+2] = (lm.x, lm.y, lm.z) if lm else (0, 0, 0)
        off += 3
    for i in range(21):
        lm = results.right_hand_landmarks.landmark[i] if results.right_hand_landmarks else None
        kps[off], kps[off+1], kps[off+2] = (lm.x, lm.y, lm.z) if lm else (0, 0, 0)
        off += 3
    for idx in _FACE_INDICES:
        lm = results.face_landmarks.landmark[idx] if results.face_landmarks else None
        kps[off], kps[off+1], kps[off+2] = (lm.x, lm.y, lm.z) if lm else (0, 0, 0)
        off += 3
    return kps

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("noima")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "noima_v2plus_best.pt"
LABEL_MAP_PATH = ROOT / "label_map.json"
STATIC_DIR = ROOT / "static"

# ---------------------------------------------------------------------------
# App state (loaded at startup)
# ---------------------------------------------------------------------------
class AppState:
    model: NOIMAv2 = None
    id_to_gloss: dict = {}
    device: torch.device = None
    holistic = None  # MediaPipe Holistic for GIF inference


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ---- startup ----
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {state.device}")

    # Load label map
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        gloss_to_id = json.load(f)
    state.id_to_gloss = {v: k for k, v in gloss_to_id.items()}
    log.info(f"Loaded {len(state.id_to_gloss)} gloss classes")

    # Load model
    state.model = NOIMAv2(
        num_classes=len(state.id_to_gloss),
        dim=256,
        nhead=8,
        num_conv_blocks=3,
        num_transformer_blocks=2,
        drop_rate=0.15,
        max_frames=64,
    )
    checkpoint = torch.load(MODEL_PATH, map_location=state.device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state.model.load_state_dict(state_dict)
    state.model.to(state.device)
    state.model.eval()
    log.info(f"Model loaded from {MODEL_PATH} ({sum(p.numel() for p in state.model.parameters()):,} params)")

    # Init MediaPipe Holistic for server-side GIF inference
    state.holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )
    log.info("MediaPipe Holistic ready")

    def _open_browser():
        webbrowser.open("http://localhost:8000")
        log.info("Opened browser at http://localhost:8000")
        
    Timer(0.5, _open_browser).start()

    yield
    # ---- shutdown ----
    state.holistic.close()
    log.info("Shutting down")


app = FastAPI(title="NOIMA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (JS, CSS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve sign GIFs
SIGN_VIDEOS_DIR = ROOT / "sign_videos"
app.mount("/sign_videos", StaticFiles(directory=str(SIGN_VIDEOS_DIR)), name="sign_videos")

# Serve logo from project root
@app.get("/noima_logo_no_bg.png")
def serve_logo():
    return FileResponse(str(ROOT / "noima_logo_no_bg.png"))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    keypoints: list  # shape (T, 225)


class Top5Item(BaseModel):
    gloss: str
    confidence: float


class PredictResponse(BaseModel):
    gloss: str
    confidence: float
    top5: list[Top5Item]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/signs")
def list_signs():
    """Return sorted list of sign gloss names that have a GIF available."""
    signs = sorted([f.stem for f in SIGN_VIDEOS_DIR.glob("*.gif")])
    return {"signs": signs}


# --- /gif_frames ---

@app.get("/gif_frames/{gloss}")
def get_gif_frames(gloss: str):
    """Return per-frame base64 images + MediaPipe landmarks + model prediction."""
    gif_path = SIGN_VIDEOS_DIR / f"{gloss}.gif"
    if not gif_path.exists():
        raise HTTPException(status_code=404, detail=f"GIF not found: {gloss}")

    try:
        gif = Image.open(gif_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open GIF: {e}")

    duration_ms = gif.info.get("duration", 100)
    fps = max(1, round(1000 / duration_ms))

    def lm_list(lms, indices=None):
        if lms is None:
            return []
        pts = lms.landmark
        if indices:
            return [[pts[i].x, pts[i].y] for i in indices]
        return [[p.x, p.y] for p in pts]

    frames_out = []
    kps_seq = []
    frame_idx = 0
    while True:
        try:
            gif.seek(frame_idx)
            frame_rgb = np.array(gif.convert("RGB"))
            h_px, w_px = frame_rgb.shape[:2]
            results = state.holistic.process(frame_rgb)

            # Resize to max 480px wide for bandwidth
            scale = min(1.0, 480 / w_px)
            if scale < 1.0:
                nw, nh = int(w_px * scale), int(h_px * scale)
                frame_pil = Image.fromarray(frame_rgb).resize((nw, nh), Image.LANCZOS)
            else:
                frame_pil = Image.fromarray(frame_rgb)
            buf = _io.BytesIO()
            frame_pil.save(buf, format="JPEG", quality=72)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            frames_out.append({
                "image": f"data:image/jpeg;base64,{img_b64}",
                "landmarks": {
                    "pose":       lm_list(results.pose_landmarks, _POSE_INDICES),  # 11 training landmarks
                    "left_hand":  lm_list(results.left_hand_landmarks),            # 21 training landmarks
                    "right_hand": lm_list(results.right_hand_landmarks),           # 21 training landmarks
                    "face":       lm_list(results.face_landmarks, _FACE_INDICES),  # 22 lip landmarks
                },
            })
            kps_seq.append(_extract_kps(results))
            frame_idx += 1
        except EOFError:
            break

    if not frames_out:
        raise HTTPException(status_code=422, detail="No frames extracted")

    prediction = None
    if len(kps_seq) >= 5:
        features, mask = process_keypoints(kps_seq)
        x = torch.from_numpy(features).to(state.device)
        m = torch.from_numpy(mask).to(state.device)
        with torch.no_grad():
            probs = torch.softmax(state.model(x, m), dim=-1)[0]
        top5_idx = probs.topk(5).indices.cpu().tolist()
        prediction = {
            "gloss": state.id_to_gloss[top5_idx[0]],
            "confidence": round(float(probs[top5_idx[0]]), 4),
            "top5": [
                {"gloss": state.id_to_gloss[i], "confidence": round(float(probs[i]), 4)}
                for i in top5_idx
            ],
        }
        log.info(f"gif_frames({gloss}) → {prediction['gloss']} ({prediction['confidence']:.3f})")

    return {"fps": fps, "frames": frames_out, "prediction": prediction}


# --- /predict_gif ---

@app.get("/predict_gif/{gloss}", response_model=PredictResponse)
def predict_gif(gloss: str):
    """Run model inference on an existing sign GIF (server-side MediaPipe)."""
    gif_path = SIGN_VIDEOS_DIR / f"{gloss}.gif"
    if not gif_path.exists():
        raise HTTPException(status_code=404, detail=f"GIF not found: {gloss}")

    # Extract frames from GIF and run MediaPipe on each
    try:
        gif = Image.open(gif_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open GIF: {e}")

    kps_seq = []
    frame_idx = 0
    while True:
        try:
            gif.seek(frame_idx)
            frame = np.array(gif.convert("RGB"))
            results = state.holistic.process(frame)
            kps_seq.append(_extract_kps(results))
            frame_idx += 1
        except EOFError:
            break

    if len(kps_seq) < 5:
        raise HTTPException(status_code=422, detail="Too few frames in GIF")

    features, mask = process_keypoints(kps_seq)
    x = torch.from_numpy(features).to(state.device)
    m = torch.from_numpy(mask).to(state.device)

    with torch.no_grad():
        logits = state.model(x, m)
        probs = torch.softmax(logits, dim=-1)[0]

    top5_indices = probs.topk(5).indices.cpu().tolist()
    top5 = [
        Top5Item(gloss=state.id_to_gloss[i], confidence=round(float(probs[i]), 4))
        for i in top5_indices
    ]
    best_id = top5_indices[0]
    log.info(f"predict_gif({gloss}) → {state.id_to_gloss[best_id]} ({float(probs[best_id]):.3f})")

    return PredictResponse(
        gloss=state.id_to_gloss[best_id],
        confidence=round(float(probs[best_id]), 4),
        top5=top5,
    )


# --- /predict ---

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    kps = req.keypoints
    if len(kps) < 5:
        raise HTTPException(status_code=422, detail="Too few frames (need ≥5)")

    features, mask = process_keypoints(kps)   # (1, 64, 450), (1, 64)

    x = torch.from_numpy(features).to(state.device)
    m = torch.from_numpy(mask).to(state.device)

    with torch.no_grad():
        logits = state.model(x, m)             # (1, 310)
        probs = torch.softmax(logits, dim=-1)[0]

    top5_indices = probs.topk(5).indices.cpu().tolist()
    top5 = [
        Top5Item(gloss=state.id_to_gloss[i], confidence=round(float(probs[i]), 4))
        for i in top5_indices
    ]

    best_id = top5_indices[0]
    log.info(f"Predicted: {state.id_to_gloss[best_id]} ({float(probs[best_id]):.3f})")

    return PredictResponse(
        gloss=state.id_to_gloss[best_id],
        confidence=round(float(probs[best_id]), 4),
        top5=top5,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
