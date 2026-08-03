"""
prepare_model.py

Run this ONCE on your local machine before deploying.
Copies your best checkpoint to model.pt in the deploy folder.

Usage:
    cd D:\\DO NOT TOUCH\\deepfake-spectral-detector
    python prepare_model.py
"""

import torch
import shutil
from pathlib import Path

SRC = Path("checkpoints/run_v2_full_b3/checkpoint_best.pt")
DST = Path("deploy/model.pt")

if not SRC.exists():
    print(f"ERROR: {SRC} not found")
    exit(1)

# Load and verify checkpoint
ckpt = torch.load(str(SRC), map_location="cpu")
print(f"Checkpoint loaded:")
print(f"  Epoch     : {ckpt['epoch']}")
print(f"  Val AUC   : {ckpt['val_auc']:.4f}")
print(f"  raw_model : {ckpt['config'].get('raw_model', 'not set')}")
print(f"  File size : {SRC.stat().st_size / 1e6:.1f} MB")

DST.parent.mkdir(exist_ok=True)
shutil.copy2(str(SRC), str(DST))
print(f"\nCopied to: {DST}")
print(f"Size: {DST.stat().st_size / 1e6:.1f} MB")
print("\nNow commit and push the deploy/ folder to Hugging Face.")