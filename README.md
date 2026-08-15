#Title: DeepShield Deepfake Detection System


emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ DeepShield — Multi-Branch Deepfake Detection

A five-branch hybrid deepfake video detection system developed at
**Keshav Mahavidyalaya, University of Delhi**.

## Architecture

| Branch | Input | Backbone | Gate Weight |
|--------|-------|----------|-------------|
| A — Raw spatial | 224×224 RGB | EfficientNet-B3 (pretrained) | 0.14 |
| B — Spectral diff | Inter-channel energy diff | EfficientNet-B0 (scratch) | 0.50 |
| C — CT handcrafted | Contourlet Transform features | MLP | 0.02 |
| D — PRNU noise | Camera noise residuals | MLP | 0.34 |
| E — rPPG temporal | Heartbeat frequency signal | MLP | 0.00 |

## Performance

| Dataset | AUC | Split type |
|---------|-----|-----------|
| FaceForensics++ C23 (val) | **91.58%** | Video-level (proper) |
| FaceForensics++ C23 (test) | **87.81%** | Video-level (proper) |
| Celeb-DF v2 (official test) | **72.72%** | Cross-dataset |

## Usage

Upload an MP4 video clip containing a human face. The system will:
1. Extract and align face crops using MediaPipe FaceMesh
2. Compute inter-channel spectral difference maps
3. Run all five detection branches in parallel
4. Fuse results through an attention gate mechanism
5. Return a prediction with branch contribution analysis

## Disclaimer

This is a research prototype. Results should not be used as sole
evidence in legal or official contexts.
