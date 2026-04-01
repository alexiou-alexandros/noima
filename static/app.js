/**
 * NOIMA — Frontend
 * MediaPipe Holistic + sign detection state machine + REST API calls
 */

// ---------------------------------------------------------------------------
// Landmark indices (must match training data landmark extraction)
// ---------------------------------------------------------------------------
const POSE_INDICES = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16];
const FACE_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                      78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];

// ---------------------------------------------------------------------------
// Sign detection thresholds
// ---------------------------------------------------------------------------
const MOVE_THRESH     = 0.020;
const PAUSE_FRAMES    = 20;
const MIN_SIGN_FRAMES = 12;
const MAX_BUFFER      = 150;
const MIN_HAND_RATE   = 0.25;
const CONF_THRESHOLD  = 0.30;
const PRE_BUFFER_SIZE = 15;
const EMA_ALPHA       = 0.3;

// ---------------------------------------------------------------------------
// Skeleton drawing constants
// ---------------------------------------------------------------------------
const SKELETON_POSE_COLOR = '#8C9F82';
const SKELETON_HAND_L     = '#D4A373';
const SKELETON_HAND_R     = '#C6925B';
const SKELETON_FACE_COLOR = 'rgba(140,159,130,0.7)';

const HAND_CONN = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];
// POSE_INDICES remapped: 0=nose 1=l_eye 2=r_eye 3=mouth_l 4=mouth_r
// 5=l_shoulder 6=r_shoulder 7=l_elbow 8=r_elbow 9=l_wrist 10=r_wrist
const POSE_CONN = [[5,6],[5,7],[7,9],[6,8],[8,10],[0,5],[0,6]];

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const video          = document.getElementById('input-video');
const mainCanvas     = document.getElementById('main-canvas');
const mctx           = mainCanvas.getContext('2d');
const recordingBadge = document.getElementById('recording-badge');
const statusDot      = document.getElementById('status-dot');
const statusText     = document.getElementById('status-text');

// Sign library
const signGrid     = document.getElementById('sign-grid');
const signSearch   = document.getElementById('sign-search');
const libraryCount = document.getElementById('library-count');
const searchCount  = document.getElementById('search-count');

// Input source
const btnInputGif    = document.getElementById('btn-input-gif');
const btnInputCam    = document.getElementById('btn-input-cam');
const gifSourceInfo  = document.querySelector('.gif-source-info');
const gifGlossName   = document.getElementById('gif-gloss-name');
const btnClearGif    = document.getElementById('btn-clear-gif');
const gifPlaceholder = document.getElementById('gif-placeholder');
const gifLoading     = document.getElementById('gif-loading');

// Predictions
const top5Container  = document.getElementById('top5-container');
const top5Label      = document.getElementById('top5-label');

// GIF tooltip
const gifTooltip    = document.getElementById('gif-tooltip');
const gifTooltipImg = document.getElementById('gif-tooltip-img');
const gifTooltipLbl = document.getElementById('gif-tooltip-label');

// Language state
let currentLang = 'el';
const btnLangEl = document.getElementById('btn-lang-el');
const btnLangEn = document.getElementById('btn-lang-en');

function updateStaticTranslations() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    el.textContent = translateText(key, currentLang);
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const key = el.getAttribute('data-i18n-html');
    el.innerHTML = translateText(key, currentLang);
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const key = el.getAttribute('data-i18n-title');
    el.setAttribute('title', translateText(key, currentLang));
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.getAttribute('data-i18n-placeholder');
    el.setAttribute('placeholder', translateText(key, currentLang));
  });
}

btnLangEl.addEventListener('click', () => setLanguage('el'));
btnLangEn.addEventListener('click', () => setLanguage('en'));

function setLanguage(lang) {
  if (currentLang === lang) return;
  currentLang = lang;
  btnLangEl.classList.toggle('active', lang === 'el');
  btnLangEn.classList.toggle('active', lang === 'en');
  
  updateStaticTranslations();
  
  // Rebuild grid with translated glosses
  const q = signSearch.value.trim().toUpperCase();
  const filtered = q ? allSigns.filter(s => translateGloss(s, currentLang).toUpperCase().includes(q) || s.includes(q)) : allSigns;
  buildSignGrid(filtered);

  if (selectedSign) {
    gifGlossName.textContent = translateGloss(selectedSign, currentLang);
  }

  if (inputMode === 'gif') {
    top5Label.textContent = translateText('Top 5 Confidence', currentLang);
    if (!selectedSign) {
      setStatus('idle', translateText('Επέλεξε ένα νόημα από τη βιβλιοθήκη', currentLang));
    }
  } else {
    top5Label.textContent = translateText('Top 5 — Κάμερα', currentLang);
    setStatus('ready', selectedSign ? (currentLang === 'en' ? `Try: ${translateGloss(selectedSign, currentLang)}` : `Δοκίμασε: ${translateGloss(selectedSign, currentLang)}`) : translateText('Κάμερα έτοιμη', currentLang));
  }
}


// ---------------------------------------------------------------------------
// Webcam classification state
// ---------------------------------------------------------------------------
let frameBuffer      = [];
let preBuffer        = [];
let prevHandKps      = null;
let smoothedMovement = 0;
let pauseCounter     = 0;
let isSigning        = false;
let handFrames       = 0;
let isClassifying    = false;

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
let allSigns      = [];
let selectedSign  = null;
let inputMode     = 'gif';  // 'gif' | 'cam'

// GIF animation state
let frameImages    = [];
let frameLandmarks = [];
let currentFrameIdx = 0;
let animInterval   = null;

let cameraStarted  = false;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

function lm(landmark) {
  return landmark ? [landmark.x, landmark.y, landmark.z] : [0, 0, 0];
}

function extractKeypoints(results) {
  const kps = new Float32Array(225);
  let offset = 0;
  for (const idx of POSE_INDICES) {
    const lmk = results.poseLandmarks ? results.poseLandmarks[idx] : null;
    const [x, y, z] = lm(lmk);
    kps[offset++] = x; kps[offset++] = y; kps[offset++] = z;
  }
  for (let i = 0; i < 21; i++) {
    const lmk = results.leftHandLandmarks ? results.leftHandLandmarks[i] : null;
    const [x, y, z] = lm(lmk);
    kps[offset++] = x; kps[offset++] = y; kps[offset++] = z;
  }
  for (let i = 0; i < 21; i++) {
    const lmk = results.rightHandLandmarks ? results.rightHandLandmarks[i] : null;
    const [x, y, z] = lm(lmk);
    kps[offset++] = x; kps[offset++] = y; kps[offset++] = z;
  }
  for (const idx of FACE_INDICES) {
    const lmk = results.faceLandmarks ? results.faceLandmarks[idx] : null;
    const [x, y, z] = lm(lmk);
    kps[offset++] = x; kps[offset++] = y; kps[offset++] = z;
  }
  return kps;
}

function computeHandMovement(currKps) {
  if (!prevHandKps) { prevHandKps = currKps; return 0; }
  let sum = 0, count = 0;
  for (const [start, end] of [[33, 96], [96, 159]]) {
    for (let i = start; i < end; i += 3) {
      if (currKps[i] !== 0 || currKps[i + 1] !== 0) {
        sum += Math.abs(currKps[i] - prevHandKps[i]) + Math.abs(currKps[i + 1] - prevHandKps[i + 1]);
        count++;
      }
    }
  }
  prevHandKps = currKps;
  return count > 0 ? sum / count : 0;
}

// ---------------------------------------------------------------------------
// Input source toggle
// ---------------------------------------------------------------------------
btnInputGif.addEventListener('click', () => switchInput('gif'));
btnInputCam.addEventListener('click', () => switchInput('cam'));

function switchInput(mode) {
  inputMode = mode;
  btnInputGif.classList.toggle('active', mode === 'gif');
  btnInputCam.classList.toggle('active', mode === 'cam');

  isSigning = false;
  recordingBadge.classList.add('hidden');

  if (mode === 'gif') {
    signGrid.style.pointerEvents = 'auto';
    signGrid.style.opacity = '1';
    
    stopGifAnimation();
    mctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    video.style.display = 'none';
    gifSourceInfo.style.display = '';
    top5Label.textContent = translateText('Top 5 Confidence', currentLang);
    if (selectedSign) {
      gifPlaceholder.classList.add('hidden');
      loadGifFrames(selectedSign);
    } else {
      gifPlaceholder.classList.remove('hidden');
      top5Container.innerHTML = '<span class="empty-hint">—</span>';
    }
    setStatus('idle', translateText('Επέλεξε ένα νόημα από τη βιβλιοθήκη', currentLang));
  } else {
    // cam mode
    signGrid.style.pointerEvents = 'none';
    signGrid.style.opacity = '0.5';
    
    // Clear previously selected sign to ensure no expectation interference
    selectedSign = null;
    document.querySelectorAll('.sign-card.selected').forEach(c => c.classList.remove('selected'));
    
    stopGifAnimation();
    mctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    video.style.display = 'block';
    gifSourceInfo.style.display = 'none';
    gifPlaceholder.classList.add('hidden');
    gifLoading.classList.add('hidden');
    top5Label.textContent = translateText('Top 5 — Κάμερα', currentLang);
    top5Container.innerHTML = `<span class="empty-hint" data-i18n="Κάνε μια χειρονομία…">${translateText('Κάνε μια χειρονομία…', currentLang)}</span>`;
    setStatus('ready', translateText('Κάμερα έτοιμη', currentLang));
    if (!cameraStarted) startCamera();
  }
}

// ---------------------------------------------------------------------------
// Sign Library
// ---------------------------------------------------------------------------
async function loadSignLibrary() {
  try {
    const resp = await fetch('/signs');
    const data = await resp.json();
    allSigns = data.signs;
    libraryCount.textContent = allSigns.length;
    buildSignGrid(allSigns);
  } catch (err) {
    signGrid.innerHTML = `<span class="empty-hint" data-i18n="Σφάλμα φόρτωσης νοημάτων">${translateText('Σφάλμα φόρτωσης νοημάτων', currentLang)}</span>`;
  }
}

function buildSignGrid(signs) {
  signGrid.innerHTML = '';
  if (!signs.length) {
    signGrid.innerHTML = `<span class="empty-hint" data-i18n="Δεν βρέθηκαν νοήματα">${translateText('Δεν βρέθηκαν νοήματα', currentLang)}</span>`;
    return;
  }
  signs.forEach(gloss => {
    const card = document.createElement('div');
    card.className = 'sign-card' + (gloss === selectedSign ? ' selected' : '');
    card.textContent = translateGloss(gloss, currentLang);
    card.dataset.gloss = gloss;
    card.addEventListener('mouseenter', e => showGifTooltip(e, gloss));
    card.addEventListener('mousemove',  e => moveGifTooltip(e));
    card.addEventListener('mouseleave', hideGifTooltip);
    card.addEventListener('click', () => selectSign(gloss));
    signGrid.appendChild(card);
  });
}

signSearch.addEventListener('input', e => {
  const q = e.target.value.trim().toUpperCase();
  const filtered = q ? allSigns.filter(s => translateGloss(s, currentLang).toUpperCase().includes(q) || s.includes(q)) : allSigns;
  searchCount.textContent = q ? `${filtered.length} / ${allSigns.length}` : '';
  buildSignGrid(filtered);
});

// ---------------------------------------------------------------------------
// Sign Selection
// ---------------------------------------------------------------------------
function selectSign(gloss) {
  selectedSign = gloss;
  document.querySelectorAll('.sign-card').forEach(c =>
    c.classList.toggle('selected', c.dataset.gloss === gloss));

  if (inputMode === 'gif') {
    gifGlossName.textContent = translateGloss(gloss, currentLang);
    btnClearGif.classList.remove('hidden');
    gifPlaceholder.classList.add('hidden');
    loadGifFrames(gloss);
  } else {
    // cam mode: just update selectedSign for comparison, no display change
    setStatus('ready', currentLang === 'en' ? `Try: ${translateGloss(gloss, currentLang)}` : `Δοκίμασε: ${translateGloss(gloss, currentLang)}`);
  }
}

function clearGifSelection() {
  selectedSign = null;
  gifGlossName.textContent = '—';
  btnClearGif.classList.add('hidden');
  stopGifAnimation();
  mctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
  gifPlaceholder.classList.remove('hidden');
  top5Container.innerHTML = '<span class="empty-hint">—</span>';
  document.querySelectorAll('.sign-card.selected').forEach(c => c.classList.remove('selected'));
  setStatus('idle', translateText('Επέλεξε ένα νόημα από τη βιβλιοθήκη', currentLang));
}

btnClearGif.addEventListener('click', clearGifSelection);

// ---------------------------------------------------------------------------
// GIF Frames + Animation
// ---------------------------------------------------------------------------
async function loadGifFrames(gloss) {
  setStatus('classifying', currentLang === 'en' ? `Analyzing GIF: ${translateGloss(gloss, currentLang)}…` : `Αναλύω GIF: ${translateGloss(gloss, currentLang)}…`);
  top5Container.innerHTML = `<span class="empty-hint" data-i18n="Αναγνώριση…">${translateText('Αναγνώριση…', currentLang)}</span>`;
  gifLoading.classList.remove('hidden');
  stopGifAnimation();

  try {
    const resp = await fetch(`/gif_frames/${encodeURIComponent(gloss)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    frameImages = await Promise.all(data.frames.map(f =>
      new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = f.image;
      })
    ));
    frameLandmarks = data.frames.map(f => f.landmarks);

    // Set canvas dimensions from first frame
    mainCanvas.width  = frameImages[0].naturalWidth;
    mainCanvas.height = frameImages[0].naturalHeight;
    gifLoading.classList.add('hidden');

    currentFrameIdx = 0;
    renderGifFrame();

    animInterval = setInterval(() => {
      currentFrameIdx = (currentFrameIdx + 1) % frameImages.length;
      renderGifFrame();
    }, 1000 / (data.fps || 10));

    if (data.prediction) {
      renderTop5(data.prediction.top5);
      const topPredEng = translateGloss(data.prediction.gloss, 'en');
      const selectedEng = translateGloss(gloss, 'en');
      
      if (topPredEng === selectedEng) {
        setStatus('ready', currentLang === 'en' ? `✓ Correct! ${translateGloss(gloss, currentLang)} (${(data.prediction.confidence * 100).toFixed(1)}%)` : `✓ Σωστό! ${translateGloss(gloss, currentLang)} (${(data.prediction.confidence * 100).toFixed(1)}%)`);
      } else {
        setStatus('ready', currentLang === 'en' ? `Recognized: ${translateGloss(data.prediction.gloss, currentLang)} — Expected: ${translateGloss(gloss, currentLang)}` : `Αναγνωρίστηκε: ${translateGloss(data.prediction.gloss, currentLang)} — αναμενόταν: ${translateGloss(gloss, currentLang)}`);
      }
    }
  } catch (err) {
    gifLoading.classList.add('hidden');
    setStatus('error', `Error: ${err.message}`);
  }
}

function stopGifAnimation() {
  if (animInterval) { clearInterval(animInterval); animInterval = null; }
  frameImages = []; frameLandmarks = [];
}

function renderGifFrame() {
  if (!frameImages.length) return;
  const img = frameImages[currentFrameIdx];
  const lms = frameLandmarks[currentFrameIdx];
  mctx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);
  drawSkeleton(mctx, lms, mainCanvas.width, mainCanvas.height);
}

// ---------------------------------------------------------------------------
// Skeleton Drawing (shared by GIF and camera modes)
// ---------------------------------------------------------------------------
function drawSkeleton(ctx, lms, w, h) {
  const pt = (arr, i) => (arr && arr[i]) ? [arr[i][0] * w, arr[i][1] * h] : null;

  function line(arr, i, j, color, lw) {
    const a = pt(arr, i), b = pt(arr, j);
    if (!a || !b) return;
    ctx.shadowBlur = 3; ctx.shadowColor = 'rgba(0,0,0,0.7)';
    ctx.strokeStyle = color; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.shadowBlur = 0;
  }
  function dot(arr, i, color, r) {
    const p = pt(arr, i);
    if (!p) return;
    ctx.shadowBlur = 3; ctx.shadowColor = 'rgba(0,0,0,0.7)';
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  if (lms.pose?.length)       { for (const [a,b] of POSE_CONN) line(lms.pose,a,b,SKELETON_POSE_COLOR,2); lms.pose.forEach((_,i)=>dot(lms.pose,i,SKELETON_POSE_COLOR,4)); }
  if (lms.left_hand?.length)  { for (const [a,b] of HAND_CONN) line(lms.left_hand,a,b,SKELETON_HAND_L,1.5); lms.left_hand.forEach((_,i)=>dot(lms.left_hand,i,SKELETON_HAND_L,2.5)); }
  if (lms.right_hand?.length) { for (const [a,b] of HAND_CONN) line(lms.right_hand,a,b,SKELETON_HAND_R,1.5); lms.right_hand.forEach((_,i)=>dot(lms.right_hand,i,SKELETON_HAND_R,2.5)); }
  if (lms.face?.length)       { lms.face.forEach((_,i)=>dot(lms.face,i,SKELETON_FACE_COLOR,2)); }
}

// ---------------------------------------------------------------------------
// GIF Tooltip (library hover)
// ---------------------------------------------------------------------------
let tooltipTimer = null;

function showGifTooltip(e, gloss) {
  tooltipTimer = setTimeout(() => {
    gifTooltipImg.src = `/sign_videos/${encodeURIComponent(gloss)}.gif`;
    gifTooltipLbl.textContent = translateGloss(gloss, currentLang);
    gifTooltip.classList.remove('hidden');
    moveGifTooltip(e);
  }, 150);
}
function moveGifTooltip(e) {
  const tw = gifTooltip.offsetWidth || 160, th = gifTooltip.offsetHeight || 160;
  gifTooltip.style.left = Math.min(e.clientX + 16, window.innerWidth - tw - 8) + 'px';
  gifTooltip.style.top  = Math.max(e.clientY - th - 20, 8) + 'px';
}
function hideGifTooltip() {
  clearTimeout(tooltipTimer);
  gifTooltip.classList.add('hidden');
  gifTooltipImg.src = '';
}

// ---------------------------------------------------------------------------
// Top-5 rendering
// ---------------------------------------------------------------------------
function renderTop5(top5) {
  top5Container.innerHTML = '';

  if (inputMode === 'gif' && selectedSign) {
    const topPredEng = translateGloss(top5[0].gloss, 'en');
    const selectedEng = translateGloss(selectedSign, 'en');
    const isCorrect = topPredEng === selectedEng;

    const badge = document.createElement('div');
    badge.className = 'prediction-result ' + (isCorrect ? 'result-correct' : 'result-wrong');
    badge.textContent = isCorrect ? translateText('✓ Σωστή πρόβλεψη', currentLang) : translateText('✗ Λάθος πρόβλεψη', currentLang);
    top5Container.appendChild(badge);
  }

  top5.forEach((item, i) => {
    const pct = (item.confidence * 100).toFixed(1);
    const row = document.createElement('div');
    if (inputMode === 'gif' && selectedSign) {
      const itemEng = translateGloss(item.gloss, 'en');
      const selectedEng = translateGloss(selectedSign, 'en');
      const isMatch = itemEng === selectedEng && i === 0;
      const isNear  = itemEng === selectedEng && i > 0;
      row.className = 'top5-item' + (isMatch ? ' match' : isNear ? ' near-match' : '');
    } else {
      row.className = 'top5-item';
    }
    row.innerHTML = `
      <span class="top5-rank">${i + 1}</span>
      <span class="top5-gloss">${translateGloss(item.gloss, currentLang)}</span>
      <div class="top5-bar-wrap"><div class="top5-bar" style="width:${pct}%"></div></div>
      <span class="top5-pct">${pct}%</span>
    `;
    top5Container.appendChild(row);
  });
}

// ---------------------------------------------------------------------------
// Webcam Classification
// ---------------------------------------------------------------------------
async function classifySign() {
  const activeFrames = pauseCounter > 0
    ? frameBuffer.slice(0, frameBuffer.length - pauseCounter)
    : frameBuffer.slice();

  frameBuffer = []; preBuffer = []; isSigning = false;
  pauseCounter = 0; smoothedMovement = 0;
  recordingBadge.classList.add('hidden');

  const nFrames = activeFrames.length, nHands = handFrames;
  handFrames = 0;

  if (isClassifying || nFrames < MIN_SIGN_FRAMES) { isClassifying = false; return; }
  if (nHands / nFrames < MIN_HAND_RATE) {
    setStatus('ready', currentLang === 'en' ? `No hands detected (${nHands}/${nFrames} frames)` : `Χέρια δεν εντοπίστηκαν (${nHands}/${nFrames} frames)`);
    isClassifying = false; return;
  }

  isClassifying = true;
  setStatus('classifying', currentLang === 'en' ? `Recognizing… (${nFrames} frames)` : `Αναγνώριση… (${nFrames} frames)`);

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keypoints: activeFrames.map(f => Array.from(f)) }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    renderTop5(data.top5);

    const matched = selectedSign && data.top5.some(item => translateGloss(item.gloss, 'en') === translateGloss(selectedSign, 'en'));
    if (matched) {
      setStatus('ready', currentLang === 'en' ? `✓ ${translateGloss(selectedSign, currentLang)} recognized! (${(data.confidence * 100).toFixed(1)}%)` : `✓ ${translateGloss(selectedSign, currentLang)} αναγνωρίστηκε! (${(data.confidence * 100).toFixed(1)}%)`);
    } else if (data.confidence >= CONF_THRESHOLD) {
      setStatus('ready', `${translateGloss(data.gloss, currentLang)} (${(data.confidence * 100).toFixed(1)}%)${selectedSign ? (currentLang === 'en' ? ` — expected ${translateGloss(selectedSign, currentLang)}` : ` — περίμενα ${translateGloss(selectedSign, currentLang)}`) : ''}`);
    } else {
      setStatus('ready', currentLang === 'en' ? `Low confidence: ${translateGloss(data.top5[0].gloss, currentLang)} (${(data.confidence * 100).toFixed(1)}%)` : `Χαμηλή confidence: ${translateGloss(data.top5[0].gloss, currentLang)} (${(data.confidence * 100).toFixed(1)}%)`);
    }
  } catch (err) {
    setStatus('error', currentLang === 'en' ? `Error: ${err.message}` : `Σφάλμα: ${err.message}`);
  }
  isClassifying = false;
}

// ---------------------------------------------------------------------------
// MediaPipe Holistic callback (webcam)
// ---------------------------------------------------------------------------
function onResults(results) {
  if (inputMode !== 'cam') return;

  mainCanvas.width  = video.videoWidth  || mainCanvas.offsetWidth;
  mainCanvas.height = video.videoHeight || mainCanvas.offsetHeight;
  mctx.save();
  mctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);

  // Build landmark data in same format as GIF slider — calls identical drawSkeleton()
  const lmData = {
    pose: results.poseLandmarks
      ? POSE_INDICES.map(idx => {
          const p = results.poseLandmarks[idx];
          return p ? [p.x, p.y] : null;
        })
      : [],
    left_hand: results.leftHandLandmarks
      ? results.leftHandLandmarks.map(p => [p.x, p.y])
      : [],
    right_hand: results.rightHandLandmarks
      ? results.rightHandLandmarks.map(p => [p.x, p.y])
      : [],
    face: results.faceLandmarks
      ? FACE_INDICES.map(idx => [results.faceLandmarks[idx].x, results.faceLandmarks[idx].y])
      : [],
  };
  drawSkeleton(mctx, lmData, mainCanvas.width, mainCanvas.height);
  mctx.restore();

  if (isClassifying) return;

  const kps = extractKeypoints(results);
  const movement = computeHandMovement(kps);
  const hasHands = !!(results.leftHandLandmarks || results.rightHandLandmarks);

  if (!isSigning) {
    preBuffer.push(kps);
    if (preBuffer.length > PRE_BUFFER_SIZE) preBuffer.shift();
  }

  smoothedMovement = EMA_ALPHA * movement + (1 - EMA_ALPHA) * smoothedMovement;

  if (smoothedMovement > MOVE_THRESH) {
    pauseCounter = 0;
    if (!isSigning) {
      isSigning = true; handFrames = 0;
      frameBuffer = [...preBuffer]; preBuffer = [];
      recordingBadge.classList.remove('hidden');
      setStatus('signing', currentLang === 'en' ? 'Recording sign…' : 'Καταγραφή νοήματος…');
    }
  } else if (isSigning) {
    pauseCounter++;
  }

  if (isSigning) {
    frameBuffer.push(kps);
    if (hasHands) handFrames++;
    if ((pauseCounter >= PAUSE_FRAMES && frameBuffer.length >= MIN_SIGN_FRAMES)
        || frameBuffer.length >= MAX_BUFFER) {
      classifySign();
    }
  }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
async function startCamera() {
  try {
    await camera.start();
    cameraStarted = true;
    setStatus('ready', selectedSign ? (currentLang === 'en' ? `Try: ${translateGloss(selectedSign, currentLang)}` : `Δοκίμασε: ${translateGloss(selectedSign, currentLang)}`) : translateText('Κάμερα έτοιμη', currentLang));
  } catch (err) {
    setStatus('error', currentLang === 'en' ? `Camera: ${err.message}` : `Κάμερα: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Init MediaPipe (camera starts lazily on first switch to cam mode)
// ---------------------------------------------------------------------------
const holistic = new Holistic({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  smoothSegmentation: false,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
holistic.onResults(onResults);

const camera = new Camera(video, {
  onFrame: async () => { await holistic.send({ image: video }); },
  width: 640, height: 480,
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
setStatus('idle', translateText('Επέλεξε ένα νόημα από τη βιβλιοθήκη', currentLang));
loadSignLibrary();
