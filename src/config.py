"""
config.py — All paths, hyperparameters, and shared constants.

Edit this file once for your setup. Every other script imports from here.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS  (edit these)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_ROOT  = "/home/katerina/codes/datasets/landscapes/Landscape Classification/Landscape Classification/Training Data"
CUSTOM_ROOT = "/home/katerina/codes/datasets/ARXPHOTOS314/images"
TEST_ROOT   = "/home/katerina/codes/datasets/landscapes/Landscape Classification/Landscape Classification/Testing Data"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_DIR   = "weights"
FEATURE_DIR  = "features"   # extracted latents saved as .npz
RESULT_DIR   = "results"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# ── Epochs ──
# The original paper uses 150 epochs on CIFAR-10 (32×32, 10 classes).
# Landscape Classification is harder: 224×224 images, training from scratch.
# 150 epochs is the minimum for HAS to converge on this dataset.
# Evidence: after 50 epochs HAS conf μ=0.30 (random ≈ 0.20) — not converged.
EPOCHS       = 150

LR           = 0.1          # SGD LR for baseline — matches original paper
HAS_LR       = 0.05         # HAS needs a lower initial LR than baseline.
                             # Angular margin methods are sensitive to large
                             # gradient steps early in training — the weight
                             # vectors on the unit sphere can collapse.
                             # 0.05 with warmup (see train.py) is the sweet spot.
BATCH_SIZE   = 64           # Larger batch → more stable gradient for angular
                             # margin; 32 caused noisy penalty gradients.
ALPHA        = 1.0          # NLL / CE weight — unchanged
BETA         = 0.5          # HAS penalty weight.
                             # Original paper uses 1.0 on CIFAR-10 (10 classes).
                             # With 5 classes the penalty signal is stronger
                             # per sample; 0.5 prevents it from overwhelming
                             # the classification loss early in training.
                             # Symptom of BETA=1.0 too high: Mountain weight
                             # collapses to the centre of the sphere (all
                             # classes point toward Mountain in direction matrix).
HAS_MARGIN   = 0.1          # Paper recommendation — keep at 0.1 for 5 classes.
                             # Increasing to 0.3+ causes accuracy loss (Fig. 3
                             # of paper shows accuracy drops for large margin
                             # with fewer classes).
HAS_SCALE    = 8.0         # scale=1  → max conf ~0.65 (too flat)
                             # scale=10 → well-separated with proper training
                             # scale=30 → unstable from scratch
                             # Keep at 10 — the problem was epochs/LR, not scale.
HAS_WARMUP_EPOCHS = 5       # NEW — linear LR warmup for HAS.
                             # Angular margin methods need the classification
                             # loss to stabilise first before the penalty
                             # starts reshaping the sphere geometry.
                             # Without warmup, early large penalties push all
                             # weight vectors toward the same region (Mountain
                             # collapse observed in results).

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CONCEPT_THRESH = 0.70       # confidence below this → concept drift
DRIFT_SIGMA    = 2.0        # data drift = centroid dist > mean + DRIFT_SIGMA*std

# ADDITION 1 — HAS margin drift threshold
HAS_MARGIN_SIGMA = 2.0      # N sigma for margin-based drift threshold

# ─────────────────────────────────────────────────────────────────────────────
# LANDSCAPE CLASSES  (for training data)
# ─────────────────────────────────────────────────────────────────────────────
LANDSCAPE_CLASSES = ["Coast", "Desert", "Forest", "Glacier", "Mountain"]

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM → TRAINING class mapping
#   Keys   = folder names in CUSTOM_ROOT
#   Values = index into LANDSCAPE_CLASSES (0=Coast, 1=Desert, …)
#
#   Only these 5 folders are loaded from the custom dataset.
#   Everything else (People, Food, Art, …) is ignored.
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CLASS_MAP = {
    "Beach Landscape":      0,   # → Coast
    "Desert Landscape":     1,   # → Desert
    "Forest Landscape":     2,   # → Forest
    "Ice & Snow Landscape": 3,   # → Glacier
    "Mountain Landscape":   4,   # → Mountain
}

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    """Create all output directories."""
    for d in [WEIGHT_DIR, FEATURE_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)


def resolve_custom_root():
    """Return CUSTOM_ROOT if it exists, else TEST_ROOT."""
    if os.path.isdir(CUSTOM_ROOT):
        return CUSTOM_ROOT
    print(f"  ⚠ CUSTOM_ROOT not found ({CUSTOM_ROOT}), falling back to TEST_ROOT")
    return TEST_ROOT
