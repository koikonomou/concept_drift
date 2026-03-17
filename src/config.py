"""
config.py — All paths, hyperparameters, and shared constants.

Edit this file once for your setup. Every other script imports from here.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS  (edit these)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_ROOT  = "/home/kate/datasets/landscapes/Landscape Classification/Landscape Classification/Training Data"
CUSTOM_ROOT = "/home/kate/datasets/ARXPHOTOS314/images"
TEST_ROOT   = "/home/kate/datasets/landscapes/Landscape Classification/Landscape Classification/Testing Data"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_DIR   = "weights"
FEATURE_DIR  = "features"   # extracted latents saved as .npz
RESULT_DIR   = "results"

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS       = 10
LR           = 1e-4         # Adam LR for baseline
HAS_LR       = 1e-2         # SGD LR for HAS (higher — SGD needs it)
BATCH_SIZE   = 32
ALPHA        = 1.0          # NLL / CE weight
BETA         = 3.0          # HAS multi-class penalty weight
HAS_MARGIN   = 0.1
HAS_SCALE    = 15.0

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
CONCEPT_THRESH = 0.70       # confidence below this → concept drift
DRIFT_SIGMA    = 2.0        # data drift = centroid dist > mean + DRIFT_SIGMA*std

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
