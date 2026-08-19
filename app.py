"""
app.py — DeepGuard: Multi-Branch Deepfake Detection System
Clean academic deployment frontend
"""

import streamlit as st
import torch
import numpy as np
import cv2
import tempfile
import os
import time
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard — Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS override (light theme base via config.toml) ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── DARK TOKEN MAP ─────────────────────────────────────────────────────── */
:root {
  --bg:        #0D0F14;
  --bg-panel:  #151820;
  --bg-raised: #1C2030;
  --border:    #252A38;
  --border-hi: #2E3448;

  --text-1:   #F0F2F6;
  --text-2:   #9BA3B2;
  --text-3:   #5C6478;

  --accent:   #3B82F6;
  --accent-d: #1D4ED8;

  --fake:     #EF4444;
  --real:     #22C55E;
}

/* Base */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
}
.stApp { background: var(--bg) !important; color: var(--text-1) !important; }
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }

/* Force all Streamlit text dark-safe */
p, span, div, label, h1, h2, h3, h4, h5, h6 { color: var(--text-1) !important; }

/* Hero */
.dg-hero {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.75rem;
  margin-bottom: 1.25rem;
}
.dg-hero-inner {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 2rem;
}
.dg-hero-title {
  font-size: 1.6rem;
  font-weight: 600;
  letter-spacing: -0.025em;
  color: var(--text-1) !important;
  line-height: 1;
}
.dg-hero-sub {
  font-size: 0.82rem;
  color: var(--text-2) !important;
  margin-top: 0.3rem;
}
.dg-status {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  font-weight: 500;
  padding: 3px 10px;
  border-radius: 20px;
  margin-top: 0.6rem;
}
.dg-status-ok  { background: rgba(34,197,94,0.15); color: #4ADE80 !important; }
.dg-status-err { background: rgba(239,68,68,0.15);  color: #F87171 !important; }
.dg-status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.dg-hero-right { text-align: right; flex-shrink: 0; }
.dg-hero-right .inst { font-size: 0.82rem; font-weight: 500; color: var(--text-1) !important; }
.dg-hero-right .dept { font-size: 0.75rem; color: var(--text-2) !important; margin-top: 2px; }
.dg-auc-row { display: flex; gap: 1.1rem; justify-content: flex-end; margin-top: 0.6rem; }
.dg-auc { text-align: center; }
.dg-auc-v { font-family: 'DM Mono', monospace; font-size: 1rem; font-weight: 500; color: var(--accent) !important; }
.dg-auc-l { font-size: 0.63rem; color: var(--text-3) !important; margin-top: 1px; }

/* Section label */
.dg-section { display: flex; align-items: center; gap: 0.6rem; margin: 1.1rem 0 0.6rem; }
.dg-section-label {
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-3) !important;
  white-space: nowrap;
}
.dg-section-rule { flex: 1; height: 1px; background: var(--border); }

/* Cards */
.dg-card {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
}
.dg-card-title {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text-3) !important;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.75rem;
}

/* Verdict */
.dg-verdict { border-radius: 10px; padding: 1.25rem 1.5rem; border: 1px solid; margin-bottom: 0.75rem; }
.dg-verdict-fake { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.3); }
.dg-verdict-real { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.3); }
.dg-verdict-inner { display: flex; justify-content: space-between; align-items: center; gap: 1.5rem; }
.dg-verdict-eyebrow {
  font-size: 0.67rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-3) !important; margin-bottom: 0.3rem;
}
.dg-verdict-label { font-size: 1.45rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; }
.dg-verdict-fake .dg-verdict-label { color: #F87171 !important; }
.dg-verdict-real .dg-verdict-label { color: #4ADE80 !important; }
.dg-verdict-desc { font-size: 0.84rem; color: var(--text-2) !important; margin-top: 0.4rem; line-height: 1.6; }
.dg-conf {
  text-align: center; flex-shrink: 0;
  padding: 0.75rem 1.1rem; border-radius: 8px;
  background: rgba(255,255,255,0.04);
}
.dg-conf-val { font-family: 'DM Mono', monospace; font-size: 1.6rem; font-weight: 500; line-height: 1; }
.dg-verdict-fake .dg-conf-val { color: #F87171 !important; }
.dg-verdict-real .dg-conf-val { color: #4ADE80 !important; }
.dg-conf-lbl { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3) !important; margin-top: 3px; }

/* Metric strip */
.dg-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 0.75rem; }
.dg-metric { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; }
.dg-metric-val { font-family: 'DM Mono', monospace; font-size: 1.25rem; font-weight: 500; line-height: 1; margin-bottom: 3px; }
.dg-metric-lbl { font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-3) !important; font-weight: 500; }

/* Bottom info */
.dg-info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1rem; }
.dg-arch-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 0.78rem;
}
.dg-arch-row:last-child { border-bottom: none; }
.dg-arch-key { color: var(--text-2) !important; }
.dg-arch-val { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-1) !important; font-weight: 500; }
.dg-about-text { font-size: 0.8rem; color: var(--text-2) !important; line-height: 1.7; }
.dg-team { margin-top: 0.65rem; padding-top: 0.55rem; border-top: 1px solid var(--border); font-size: 0.72rem; color: var(--text-3) !important; }

/* Footer */
.dg-footer {
  text-align: center; font-size: 0.7rem; color: var(--text-3) !important;
  padding: 1.25rem 0 0.5rem; border-top: 1px solid var(--border);
  margin-top: 1.5rem; line-height: 1.9;
}

/* Streamlit widgets */
div[data-testid="stFileUploaderDropzone"] {
  background: var(--bg-panel) !important;
  border: 1.5px dashed var(--border-hi) !important;
  border-radius: 10px !important;
  padding: 1.75rem 1.5rem !important;
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text-2) !important; font-size: 0.82rem !important; }
div[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--text-3) !important; font-size: 0.72rem !important; }
div[data-testid="stFileUploaderDropzone"] button {
  background: var(--accent) !important; color: #fff !important;
  border: none !important; border-radius: 6px !important;
  padding: 0.38rem 1rem !important; font-size: 0.8rem !important; font-weight: 500 !important;
}
div[data-testid="stFileUploaderDropzone"] button:hover { background: var(--accent-d) !important; }
div[data-testid="stFileUploaderFile"] { background: var(--bg-raised) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
div[data-testid="stFileUploaderFile"] * { color: var(--text-2) !important; }
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }
.stProgress > div > div > div { background: var(--accent) !important; }
div[data-testid="stToggle"] label { color: var(--text-2) !important; }
div[data-baseweb="slider"] div[data-testid="stThumbValue"] { color: var(--text-1) !important; }
/* Caption text */
.stCaptionContainer p, small { color: var(--text-3) !important; }
/* Slider label */
label[data-testid="stWidgetLabel"] p { color: var(--text-2) !important; }
/* Info / error boxes */
div[data-testid="stAlert"] { background: var(--bg-raised) !important; border-color: var(--border-hi) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224
N_FRAMES      = 16
MAX_VIDEO_MB  = 100

BRANCH_INFO = {
    "Branch A — Raw spatial":    {"color": "#3B82F6", "tag": "B3",  "desc": "Visual appearance · EfficientNet-B3 backbone"},
    "Branch B — Spectral diff":  {"color": "#14B8A6", "tag": "B0",  "desc": "Inter-channel frequency residuals"},
    "Branch C — CT handcrafted": {"color": "#8B5CF6", "tag": "MLP", "desc": "Contourlet directional features"},
    "Branch D — PRNU noise":     {"color": "#F97316", "tag": "MLP", "desc": "Camera sensor noise residuals"},
    "Branch E — rPPG temporal":  {"color": "#EC4899", "tag": "MLP", "desc": "Physiological pulse signal"},
}

tf_eval = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import sys
    sys.path.insert(0, ".")
    try:
        from models.detector_v2 import build_detector_v2
        for cand in [Path("model.pt"), Path("checkpoints/model.pt"), Path("deploy/model.pt")]:
            if cand.exists():
                ckpt_path = cand
                break
        else:
            return None, "model.pt not found. Place it in the deploy/ folder."
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(str(ckpt_path), map_location=device)
        model  = build_detector_v2(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def get_face_detector():
    try:
        import mediapipe as mp
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.5)
    except Exception:
        return None


# ── Video / face utils ─────────────────────────────────────────────────────────
def extract_frames(video_path, n_frames=N_FRAMES):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total < 1:
        cap.release(); return [], fps, 0
    duration = total / fps
    indices  = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames   = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps, duration


def detect_face_crop(frame_rgb, detector):
    if detector is None:
        h, w = frame_rgb.shape[:2]; s = min(h, w)
        y1 = (h - s) // 2; x1 = (w - s) // 2
        return cv2.resize(frame_rgb[y1:y1+s, x1:x1+s], (IMG_SIZE, IMG_SIZE))
    res = detector.process(frame_rgb)
    if res.multi_face_landmarks is None:
        h, w = frame_rgb.shape[:2]; s = min(h, w)
        y1 = (h - s) // 2; x1 = (w - s) // 2
        return cv2.resize(frame_rgb[y1:y1+s, x1:x1+s], (IMG_SIZE, IMG_SIZE))
    lm = res.multi_face_landmarks[0]
    h, w = frame_rgb.shape[:2]
    xs = [p.x for p in lm.landmark]; ys = [p.y for p in lm.landmark]
    x1 = max(0, int(min(xs)*w)-20); y1 = max(0, int(min(ys)*h)-20)
    x2 = min(w, int(max(xs)*w)+20); y2 = min(h, int(max(ys)*h)+20)
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def compute_diff_image(img_np):
    from scipy.ndimage import uniform_filter
    f = img_np.astype(np.float32) / 255.0
    def en(ch): return np.abs(ch - uniform_filter(ch, size=3))
    R, G, B = f[:,:,0], f[:,:,1], f[:,:,2]
    diff = np.stack([en(R)-en(B), en(R)-en(G), en(G)-en(B)], axis=2)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return diff.astype(np.float32)


@torch.no_grad()
def run_inference(model, frames_rgb):
    device     = next(model.parameters()).device
    frame_probs, all_gates = [], []
    hc   = torch.zeros(1, 9369).to(device)
    prnu = torch.zeros(1, 30).to(device)
    rppg = torch.zeros(1, 32).to(device)
    for img_np in frames_rgb:
        pil  = Image.fromarray(img_np)
        rr   = tf_eval(pil).unsqueeze(0).to(device)
        di   = torch.from_numpy(compute_diff_image(img_np).transpose(2,0,1)).unsqueeze(0).to(device)
        logit, gates = model(rr, di, hc, prnu, rppg, return_gates=True)
        frame_probs.append(torch.sigmoid(logit).item())
        all_gates.append(gates.squeeze(0).cpu().numpy())
    avg_prob = float(np.mean(frame_probs))
    return {
        "fake_probability": avg_prob,
        "is_fake":          avg_prob >= 0.5,
        "confidence":       max(avg_prob, 1 - avg_prob),
        "frame_probs":      frame_probs,
        "gate_weights":     np.mean(all_gates, axis=0).tolist(),
    }


# ── UI helpers ─────────────────────────────────────────────────────────────────
def section(label):
    st.markdown(f"""
    <div class="dg-section">
      <span class="dg-section-label">{label}</span>
      <span class="dg-section-rule"></span>
    </div>
    """, unsafe_allow_html=True)


def render_hero(model_ready, device):
    if model_ready:
        status_html = '<span class="dg-status dg-status-ok"><span class="dg-status-dot"></span>Model online &nbsp;·&nbsp; ' + device.upper() + '</span>'
    else:
        status_html = '<span class="dg-status dg-status-err"><span class="dg-status-dot"></span>Model offline</span>'

    st.markdown(f"""
    <div class="dg-hero">
      <div class="dg-hero-inner">
        <div>
          <div class="dg-hero-title">🛡️ DeepGuard</div>
          <div class="dg-hero-sub">Multi-branch deepfake detection &nbsp;·&nbsp; Spatial · Spectral · PRNU · Contourlet · rPPG</div>
          {status_html}
        </div>
        <div class="dg-hero-right">
          <div class="inst">Keshav Mahavidyalaya, University of Delhi</div>
          <div class="dept">4th Year B.Sc. (H) Computer Science</div>
          <div class="dg-auc-row">
            <div class="dg-auc">
              <div class="dg-auc-v">91.58%</div>
              <div class="dg-auc-l">FF++ Val AUC</div>
            </div>
            <div class="dg-auc">
              <div class="dg-auc-v">87.81%</div>
              <div class="dg-auc-l">FF++ Test AUC</div>
            </div>
            <div class="dg-auc">
              <div class="dg-auc-v">72.72%</div>
              <div class="dg-auc-l">Celeb-DF AUC</div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(prob, conf, dom_branch, dom_weight):
    is_fake = prob >= 0.5
    vcls  = "dg-verdict-fake" if is_fake else "dg-verdict-real"
    label = "Deepfake Detected" if is_fake else "Authentic Video"
    if is_fake:
        desc = (f"Flagged as synthetic with <strong>{prob*100:.1f}%</strong> fake probability. "
                f"Dominant forensic signal: <strong>{dom_branch}</strong> "
                f"({dom_weight*100:.1f}% attention-gate weight).")
    else:
        desc = (f"Forensic signals are consistent with an authentic recording "
                f"(<strong>{(1-prob)*100:.1f}%</strong> real probability). "
                f"Dominant signal: <strong>{dom_branch}</strong> "
                f"({dom_weight*100:.1f}% attention-gate weight).")
    st.markdown(f"""
    <div class="dg-verdict {vcls}">
      <div class="dg-verdict-inner">
        <div>
          <div class="dg-verdict-eyebrow">Detection result</div>
          <div class="dg-verdict-label">{label}</div>
          <div class="dg-verdict-desc">{desc}</div>
        </div>
        <div class="dg-conf">
          <div class="dg-conf-val">{conf*100:.0f}%</div>
          <div class="dg-conf-lbl">Confidence</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(prob, n_frames, t_elapsed, is_fake):
    fc = "color:#DC2626" if is_fake else "color:#16A34A"
    rc = "color:#16A34A" if not is_fake else "color:#DC2626"
    st.markdown(f"""
    <div class="dg-metrics">
      <div class="dg-metric">
        <div class="dg-metric-val" style="{fc}">{prob*100:.1f}%</div>
        <div class="dg-metric-lbl">Fake probability</div>
      </div>
      <div class="dg-metric">
        <div class="dg-metric-val" style="{rc}">{(1-prob)*100:.1f}%</div>
        <div class="dg-metric-lbl">Real probability</div>
      </div>
      <div class="dg-metric">
        <div class="dg-metric-val">{n_frames}</div>
        <div class="dg-metric-lbl">Frames analysed</div>
      </div>
      <div class="dg-metric">
        <div class="dg-metric-val">{t_elapsed:.1f}s</div>
        <div class="dg-metric-lbl">Inference time</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def build_branch_html(gates):
    dom = int(np.argmax(gates))
    html = '<div class="dg-branch">'
    for i, (name, info) in enumerate(BRANCH_INFO.items()):
        w   = gates[i] if i < len(gates) else 0.0
        pct = w * 100
        bar = max(2, pct)
        dom_cls = "dg-branch-item dg-branch-dominant" if i == dom else "dg-branch-item"
        html += f"""
        <div class="{dom_cls}">
          <div class="dg-branch-row">
            <span class="dg-branch-name">{name}</span>
            <span class="dg-branch-pct">{pct:.1f}%</span>
          </div>
          <div class="dg-branch-track">
            <div class="dg-branch-fill" style="width:{bar:.1f}%;background:{info['color']}"></div>
          </div>
          <div class="dg-branch-desc">{info['desc']}</div>
        </div>"""
    return html + "</div>"


def build_timeline_html(frame_probs, duration):
    if not frame_probs:
        return ""
    bars = ""
    for p in frame_probs:
        color = "#EF4444" if p >= 0.5 else "#22C55E"
        bars += f'<div class="dg-tbar" style="height:{max(8,p*100):.0f}%;background:{color}"></div>'
    n  = len(frame_probs)
    ts = ([f"{(i/n)*duration:.1f}s" for i in [0, n//4, n//2, 3*n//4, n-1]]
          if duration > 0 else
          [f"F1", f"F{max(1,n//4)}", f"F{max(1,n//2)}", f"F{max(1,3*n//4)}", f"F{n}"])
    aside = f"{n} frames · {duration:.1f}s" if duration > 0 else f"{n} frames"
    return f"""
    <div class="dg-timeline-bars">{bars}</div>
    <div class="dg-taxis">
      <span>{ts[0]}</span><span>{ts[1]}</span><span>{ts[2]}</span><span>{ts[3]}</span><span>{ts[4]}</span>
    </div>
    <div class="dg-tlegend">
      <span class="dg-tleg"><span class="dg-tleg-dot" style="background:#EF4444"></span>Fake (≥0.5)</span>
      <span class="dg-tleg"><span class="dg-tleg-dot" style="background:#22C55E"></span>Real (&lt;0.5)</span>
      <span style="margin-left:auto;color:#9CA3AF;font-size:0.63rem">{aside}</span>
    </div>"""


def render_bottom_info(device):
    arch = [
        ("Branch A · raw RGB",  "EfficientNet-B3"),
        ("Branch B · spectral", "EfficientNet-B0"),
        ("Branch C · CT",       "MLP 9369-d"),
        ("Branch D · PRNU",     "MLP 30-d"),
        ("Branch E · rPPG",     "MLP 32-d"),
        ("Fusion mechanism",    "Attention gate"),
        ("Total parameters",    "22.18 M"),
        ("Compute",             device.upper()),
    ]
    rows = "".join(
        f'<div class="dg-arch-row">'
        f'<span class="dg-arch-key">{k}</span>'
        f'<span class="dg-arch-val">{v}</span>'
        f'</div>' for k, v in arch
    )
    st.markdown(f"""
    <div class="dg-info-grid">
      <div class="dg-card">
        <div class="dg-card-title">Model architecture</div>
        {rows}
      </div>
      <div class="dg-card">
        <div class="dg-card-title">About this project</div>
        <div class="dg-about-text">
          <strong>DeepGuard</strong> is a multi-branch hybrid deepfake detection system combining
          five forensic signals: spatial appearance, inter-channel spectral residuals, Contourlet
          handcrafted features, PRNU camera-noise analysis, and rPPG temporal physiology.
          An attention-gated fusion mechanism dynamically weights each branch per input clip.
          <br><br>
          Trained on <strong>FaceForensics++ C23</strong> (Deepfakes, Face2Face, FaceShifter,
          FaceSwap, NeuralTextures, DeepFakeDetection). Cross-dataset generalisation verified
          on <strong>Celeb-DF v2</strong> official test split.
        </div>
        <div class="dg-team">
          Mayank &nbsp;·&nbsp; Saloni Nayyar &nbsp;·&nbsp;
          Prof. Jyoti Kumari &nbsp;·&nbsp; Prof. Ashutosh Singh<br>
          Keshav Mahavidyalaya, University of Delhi
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="dg-footer">
      <strong>DeepGuard</strong> · Multi-Branch Hybrid Deepfake Detection System<br>
      EfficientNet-B3 + EfficientNet-B0 + PRNU + Contourlet Transform + rPPG · Attention-gated fusion<br>
      For research and educational use only · Not production-grade evidence
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model, model_err = load_model()
    detector = get_face_detector()
    device   = "cuda" if (model and torch.cuda.is_available()) else "cpu"

    # 1. Hero
    render_hero(model_ready=(model is not None), device=device)

    if model_err:
        st.error(f"Model not loaded: {model_err}")
        st.info("Place `model.pt` in the `deploy/` folder and restart.")
        render_bottom_info(device)
        render_footer()
        return

    # 2. Settings
    section("Analysis Settings")
    c1, c2, c3 = st.columns([3, 1.5, 1.5], gap="medium")
    with c1:
        n_frames_sel = st.slider(
            "Frames to analyse", min_value=8, max_value=20, value=12, step=4,
            help="More frames = higher accuracy, slightly slower"
        )
    with c2:
        use_tta = st.toggle("Test-time augmentation", value=True,
                            help="Mirror-flip pass — slightly improves accuracy")
    with c3:
        show_frames = st.toggle("Show face frames", value=True,
                                help="Display per-frame grid after analysis")

    # 3. Upload
    section("Upload Video")
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        uploaded = st.file_uploader(
            "Choose video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        st.caption("MP4 · AVI · MOV · MKV &nbsp;·&nbsp; max 100 MB · processed in memory, never stored")

    # 4. Pre-upload state
    if uploaded is None:
        render_bottom_info(device)
        render_footer()
        return

    # 5. File size check
    file_mb = len(uploaded.getvalue()) / (1024 * 1024)
    if file_mb > MAX_VIDEO_MB:
        st.error(f"File too large ({file_mb:.1f} MB). Maximum: {MAX_VIDEO_MB} MB.")
        return

    # 6. Write temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    # 7. Processing
    scan_slot = st.empty()
    scan_slot.info("⌬ Running forensic analysis pipeline…")
    progress = st.progress(0, text="Extracting frames")

    t_start = time.time()
    frames_raw, _, duration = extract_frames(tmp_path, n_frames_sel)
    os.unlink(tmp_path)

    if not frames_raw:
        scan_slot.empty(); progress.empty()
        st.error("Could not extract frames. Try a different video file.")
        return

    progress.progress(25, text="Detecting and cropping faces")
    face_crops = [detect_face_crop(f, detector) for f in frames_raw]

    progress.progress(55, text="Running multi-branch detector")
    if use_tta:
        flipped = [np.fliplr(c) for c in face_crops]
        r1 = run_inference(model, face_crops)
        r2 = run_inference(model, flipped)
        result = r1.copy()
        result["fake_probability"] = (r1["fake_probability"] + r2["fake_probability"]) / 2
        result["is_fake"]          = result["fake_probability"] >= 0.5
        result["confidence"]       = max(result["fake_probability"], 1 - result["fake_probability"])
        result["frame_probs"]      = [(a+b)/2 for a,b in zip(r1["frame_probs"], r2["frame_probs"])]
        result["gate_weights"]     = [(a+b)/2 for a,b in zip(r1["gate_weights"], r2["gate_weights"])]
    else:
        result = run_inference(model, face_crops)

    progress.progress(95, text="Compiling forensic report")
    t_elapsed = time.time() - t_start
    time.sleep(0.2)
    progress.empty()
    scan_slot.empty()

    prob       = result["fake_probability"]
    conf       = result["confidence"]
    gate_names = list(BRANCH_INFO.keys())
    dom_idx    = int(np.argmax(result["gate_weights"]))
    dom_branch = gate_names[dom_idx]
    dom_weight = result["gate_weights"][dom_idx]

    # 8. Forensic report
    section("Forensic Report")

    # Verdict + metrics
    render_verdict(prob, conf, dom_branch, dom_weight)
    render_metrics(prob, len(face_crops), t_elapsed, result["is_fake"])

    # Two-column: branch weights | timeline + interpretation
    col_l, col_r = st.columns([2, 3], gap="large")

    with col_l:
        import streamlit.components.v1 as components
        branch_html_content = build_branch_html(result["gate_weights"])
        left_html = f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');
          * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }}
          body {{ background: transparent; }}
          .card {{ background: #151820; border: 1px solid #252A38; border-radius: 10px; padding: 1rem 1.25rem; }}
          .card-title {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em;
                         color: #5C6478; padding-bottom: 0.55rem; border-bottom: 1px solid #252A38; margin-bottom: 0.75rem; }}
          .card-sub {{ font-size: 0.75rem; color: #6B7280; margin-bottom: 0.65rem; }}
          .dg-branch {{ display: flex; flex-direction: column; gap: 0.6rem; }}
          .dg-branch-item {{ display: flex; flex-direction: column; gap: 3px; }}
          .dg-branch-dominant {{ border-left: 2px solid #3B82F6; padding-left: 8px; margin-left: -8px; }}
          .dg-branch-row {{ display: flex; justify-content: space-between; align-items: center; }}
          .dg-branch-name {{ font-size: 0.78rem; color: #9BA3B2; }}
          .dg-branch-pct {{ font-family: 'DM Mono', monospace; font-size: 0.75rem; font-weight: 500; color: #F0F2F6; }}
          .dg-branch-track {{ height: 5px; background: #252A38; border-radius: 3px; overflow: hidden; }}
          .dg-branch-fill {{ height: 100%; border-radius: 3px; }}
          .dg-branch-desc {{ font-size: 0.67rem; color: #5C6478; padding-left: 2px; }}
        </style>
        <div class="card">
          <div class="card-title">Branch contribution</div>
          <div class="card-sub">Attention-gate weights &middot; dominant branch highlighted</div>
          {branch_html_content}
        </div>
        """
        components.html(left_html, height=430, scrolling=False)

    with col_r:
        import streamlit.components.v1 as components

        is_fake = result["is_fake"]
        if is_fake:
            if prob > 0.85:   tier, extra = "High confidence",     "Strong synthetic artifacts detected across multiple branches."
            elif prob > 0.65: tier, extra = "Moderate confidence", "Some branches show inconsistencies. Further review recommended."
            else:             tier, extra = "Low confidence",       "Borderline case — manual expert review advised."
            interp = ("Classified as <strong>deepfake</strong> with <strong>" + f"{prob*100:.1f}%" +
                      "</strong> fake probability. Dominant signal: <strong>" + dom_branch +
                      "</strong> (gate weight <strong>" + f"{dom_weight*100:.1f}%" + "</strong>). " + extra)
        else:
            if prob < 0.15:   tier, extra = "High confidence",     "Forensic signals are fully consistent with an authentic camera recording."
            elif prob < 0.35: tier, extra = "Moderate confidence", "Signals lean authentic. No strong synthetic artifacts detected."
            else:             tier, extra = "Low confidence",       "Borderline case — manual expert review advised."
            interp = ("Classified as <strong>authentic</strong> with <strong>" + f"{(1-prob)*100:.1f}%" +
                      "</strong> real probability. Dominant signal: <strong>" + dom_branch +
                      "</strong> (gate weight <strong>" + f"{dom_weight*100:.1f}%" + "</strong>). " + extra)

        tl_html = build_timeline_html(result["frame_probs"], duration)

        right_html = f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');
          * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }}
          body {{ background: transparent; }}
          .card {{ background: #151820; border: 1px solid #252A38; border-radius: 10px; padding: 1rem 1.25rem; }}
          .card-title {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em;
                         color: #5C6478; padding-bottom: 0.55rem; border-bottom: 1px solid #252A38; margin-bottom: 0.75rem; }}
          .card-sub {{ font-size: 0.75rem; color: #5C6478; margin-bottom: 0.55rem; }}
          .dg-timeline-bars {{ display: flex; align-items: flex-end; height: 52px; gap: 2px; margin-bottom: 5px; }}
          .dg-tbar {{ flex: 1; border-radius: 2px 2px 0 0; min-height: 4px; }}
          .dg-taxis {{ display: flex; justify-content: space-between; font-family: 'DM Mono', monospace;
                       font-size: 0.62rem; color: #5C6478; border-top: 1px solid #252A38; padding-top: 4px; }}
          .dg-tlegend {{ display: flex; gap: 10px; margin-top: 6px; font-size: 0.65rem; color: #9BA3B2; flex-wrap: wrap; }}
          .dg-tleg {{ display: flex; align-items: center; gap: 4px; }}
          .dg-tleg-dot {{ width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }}
          .divider {{ margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid #252A38; }}
          .dg-interp {{ background: rgba(59,130,246,0.1); border-left: 3px solid #3B82F6;
                        border-radius: 0 6px 6px 0; padding: 0.75rem 1rem;
                        font-size: 0.82rem; color: #93C5FD; line-height: 1.65; margin-top: 0.6rem; }}
          .dg-interp strong {{ color: #BFDBFE; }}
          .dg-tier {{ display: inline-block; margin-top: 5px; font-size: 0.67rem; font-weight: 600;
                      padding: 2px 8px; border-radius: 4px;
                      background: rgba(59,130,246,0.18); color: #60A5FA; }}
          .dg-disclaimer {{ background: rgba(234,179,8,0.08); border: 1px solid rgba(234,179,8,0.25);
                            border-radius: 6px; padding: 7px 11px;
                            font-size: 0.72rem; color: #FCD34D; line-height: 1.55; margin-top: 0.6rem; }}
        </style>
        <div class="card">
          <div class="card-title">Frame-level timeline</div>
          <div class="card-sub">Per-frame fake probability across the analysed clip</div>
          {tl_html}
          <div class="divider">
            <div class="card-title">Forensic interpretation</div>
            <div class="dg-interp">
              {interp}
              <div><span class="dg-tier">&#9635; {tier}</span></div>
            </div>
          </div>
          <div class="dg-disclaimer">
            &#9888; Research prototype &nbsp;&middot;&nbsp; 87.81% AUC on FF++ test
            &nbsp;&middot;&nbsp; 72.72% on Celeb-DF
            &nbsp;&middot;&nbsp; Not for use as sole evidence in legal or official contexts.
          </div>
        </div>
        """
        components.html(right_html, height=430, scrolling=False)

    # 9. Face frame horizontal scroller
    if show_frames and face_crops:
        import base64, io

        section(f"Analysed Face Frames  ·  {len(face_crops)} frames  ·  scroll to view all")

        cards_html = ""
        for idx, (crop, p) in enumerate(zip(face_crops, result["frame_probs"])):
            pil = Image.fromarray(crop)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode()
            ts  = f"{(idx / len(face_crops)) * duration:.1f}s" if duration > 0 else f"#{idx+1}"
            lbl = "FAKE"  if p >= 0.5 else "REAL"
            top_color = "#EF4444" if p >= 0.5 else "#22C55E"
            tag_cls   = "tag-fake" if p >= 0.5 else "tag-real"
            lbl       = "FAKE"    if p >= 0.5 else "REAL"
            cards_html += f"""
            <div class="frame-card" style="border-top:2px solid {top_color};">
              <img src="data:image/jpeg;base64,{b64}"
                   style="width:100%;height:148px;object-fit:cover;display:block;"
                   alt="Frame {idx+1}"/>
              <div class="frame-meta">
                <div class="frame-row">
                  <span class="frame-num">Frame {idx+1}</span>
                  <span class="frame-ts">{ts}</span>
                </div>
                <div class="frame-row">
                  <span class="tag {tag_cls}">{lbl}</span>
                  <span class="frame-prob">{p:.2f}</span>
                </div>
              </div>
            </div>"""

        import streamlit.components.v1 as components
        n_frames_count = len(face_crops)
        scroll_height = 230
        frames_component_html = f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');
          * {{ box-sizing: border-box; margin: 0; padding: 0; }}
          body {{ background: transparent; font-family: 'DM Sans', sans-serif; }}
          .wrap {{
            background: #151820; border: 1px solid #252A38; border-radius: 10px;
            padding: 0.85rem 1rem;
          }}
          .scroller {{
            display: flex; gap: 8px; overflow-x: auto; overflow-y: hidden;
            padding-bottom: 6px; scroll-behavior: smooth;
            scrollbar-width: thin; scrollbar-color: #2E3448 #1C2030;
          }}
          .scroller::-webkit-scrollbar {{ height: 5px; }}
          .scroller::-webkit-scrollbar-track {{ background: #1C2030; border-radius: 3px; }}
          .scroller::-webkit-scrollbar-thumb {{ background: #2E3448; border-radius: 3px; }}
          .hint {{
            text-align: center; font-size: 0.65rem; color: #5C6478;
            margin-top: 6px; padding-top: 5px; border-top: 1px solid #252A38;
          }}
          .frame-card {{
            flex: 0 0 auto; width: 148px; background: #1C2030;
            border-radius: 7px; overflow: hidden;
          }}
          .frame-meta {{ padding: 5px 7px 7px; }}
          .frame-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }}
          .frame-num {{ font-size: 0.64rem; color: #5C6478; }}
          .frame-ts  {{ font-family: 'DM Mono', monospace; font-size: 0.59rem; color: #5C6478; }}
          .frame-prob {{ font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500; color: #F0F2F6; }}
          .tag {{ font-size: 0.57rem; font-weight: 700; padding: 1px 5px; border-radius: 3px;
                  text-transform: uppercase; letter-spacing: 0.05em; }}
          .tag-fake {{ background: rgba(239,68,68,0.18); color: #F87171; }}
          .tag-real {{ background: rgba(34,197,94,0.15); color: #4ADE80; }}
        </style>
        <div class="wrap">
          <div class="scroller">{cards_html}</div>
          <div class="hint">&#8592; scroll to see all {n_frames_count} frames &#8594;</div>
        </div>
        """
        components.html(frames_component_html, height=scroll_height, scrolling=False)

    # 10. Architecture + about + footer
    render_bottom_info(device)
    render_footer()


if __name__ == "__main__":
    main()
