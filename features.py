"""
Feature engineering pipeline — replicates the exact training preprocessing
from __notebook_source__.ipynb to ensure inference matches training distribution.

Pipeline:
  (T, 225) raw keypoints
    → extract xy         → (T, 150)
    → normalize          → (T, 150)  [per-sequence, non-zero frames only]
    → add vel + acc      → (T, 450)
    → pad/sample to 64   → (64, 450)
    → mask               → (64,)     [True for real frames]
"""

import numpy as np

MAX_FRAMES = 64
RAW_FEATURES = 225   # 75 landmarks × (x, y, z)
XY_FEATURES = 150    # 75 landmarks × (x, y)
FULL_FEATURES = 450  # XY_FEATURES × 3 (pos + vel + acc)


def extract_xy(kps: np.ndarray) -> np.ndarray:
    """
    Drop z-coordinate.
    Input:  (T, 225) or (N, T, 225)
    Output: (T, 150) or (N, T, 150)
    """
    if kps.ndim == 2:
        T = kps.shape[0]
        return kps.reshape(T, 75, 3)[:, :, :2].reshape(T, 150)
    else:
        N, T, _ = kps.shape
        return kps.reshape(N, T, 75, 3)[:, :, :, :2].reshape(N, T, 150)


def normalize_per_sequence(kps: np.ndarray) -> np.ndarray:
    """
    Normalize to 0-mean, 1-std using only non-zero (real) frames.
    Input: (T, 150)
    Output: (T, 150) normalized in-place clone
    """
    kps = kps.copy()
    energy = np.abs(kps).sum(axis=1)       # (T,)
    real_mask = energy > 0.01
    if real_mask.sum() > 0:
        real_frames = kps[real_mask]
        mean = real_frames.mean()
        std = real_frames.std() + 1e-8
        kps[real_mask] = (kps[real_mask] - mean) / std
    return kps


def add_velocity_acceleration(kps: np.ndarray) -> np.ndarray:
    """
    Append velocity (frame diff) and acceleration (2-frame diff).
    Input:  (T, 150)
    Output: (T, 450)
    """
    vel = np.zeros_like(kps)
    vel[1:] = kps[1:] - kps[:-1]

    acc = np.zeros_like(kps)
    acc[2:] = kps[2:] - kps[:-2]

    return np.concatenate([kps, vel, acc], axis=-1)


def pad_or_sample(kps: np.ndarray, target: int = MAX_FRAMES) -> np.ndarray:
    """
    Resize temporal dimension to exactly `target` frames.
    - If T < target: zero-pad at the end.
    - If T > target: uniform linear interpolation (same as training _resample).
    Input:  (T, 450)
    Output: (target, 450)
    """
    T = kps.shape[0]
    if T == target:
        return kps

    if T < target:
        pad = np.zeros((target - T, kps.shape[1]), dtype=kps.dtype)
        return np.concatenate([kps, pad], axis=0)

    # T > target: uniform subsampling with linear interpolation
    indices = np.linspace(0, T - 1, target).astype(np.float32)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, T - 1)
    frac = (indices - idx_floor)[:, None]
    return (kps[idx_floor] * (1 - frac) + kps[idx_ceil] * frac).astype(np.float32)


def process_keypoints(raw_kps) -> tuple:
    """
    Full inference pipeline.

    Args:
        raw_kps: array-like of shape (T, 225) — raw MediaPipe keypoints
                 [pose×11, lhand×21, rhand×21, face×22] each as (x, y, z)

    Returns:
        features: np.ndarray of shape (1, 64, 450), dtype float32
        mask:     np.ndarray of shape (1, 64),       dtype float32  (1=real, 0=padded)
    """
    kps = np.array(raw_kps, dtype=np.float32)   # (T, 225)

    # 1. Extract x, y only
    xy = extract_xy(kps)                          # (T, 150)

    # 2. Normalize per-sequence (positions first, before adding derivatives)
    xy = normalize_per_sequence(xy)               # (T, 150)

    # 3. Add velocity and acceleration
    features = add_velocity_acceleration(xy)      # (T, 450)

    # 4. Build mask from position block (first 150 features)
    T = features.shape[0]
    real_mask = (np.abs(features[:, :150]).sum(axis=1) > 0.01).astype(np.float32)

    # 5. Pad or subsample to MAX_FRAMES
    features = pad_or_sample(features, MAX_FRAMES)   # (64, 450)

    # 6. Rebuild mask for padded sequence
    mask = np.zeros(MAX_FRAMES, dtype=np.float32)
    n_real = min(T, MAX_FRAMES)
    if T <= MAX_FRAMES:
        mask[:T] = real_mask
    else:
        # After subsampling all frames are real
        mask[:] = 1.0

    return features[np.newaxis].astype(np.float32), mask[np.newaxis].astype(np.float32)
