"""
Extract 64-D latents + predictions from both models.

Loads weights from step1, runs inference on train and custom datasets,
saves everything as .npz files so later steps don't need GPU.

Usage:
    python step2_extract.py

Outputs:
    features/bl_train.npz     baseline train features
    features/bl_custom.npz    baseline custom features
    features/has_train.npz    HAS train features  (+ margins, closest_boundary)
    features/has_custom.npz   HAS custom features (+ margins, closest_boundary)
    features/meta.npz         class names + dataset sizes
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader


from config import (TRAIN_ROOT, WEIGHT_DIR, FEATURE_DIR, DEVICE,
                    LANDSCAPE_CLASSES, CUSTOM_CLASS_MAP,
                    HAS_MARGIN, HAS_SCALE,
                    ensure_dirs, resolve_custom_root)
from models import (BaselineModel, HASModel, FolderDataset,
                    STANDARD_TRANSFORM)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_baseline(model, loader):
    model.eval()
    latents, preds, confs, labels_all, subs = [], [], [], [], []
    for imgs, lbls, sub in loader:
        imgs = imgs.to(DEVICE)
        logits, latent = model(imgs)
        probs = torch.softmax(logits, dim=1)
        latents.append(latent.cpu().numpy())
        preds.append(probs.argmax(1).cpu().numpy())
        confs.append(probs.max(1).values.cpu().numpy())
        labels_all.append(np.array([int(l) for l in lbls]))
        subs.extend(sub)
    if not latents:
        return None
    return dict(latents=np.concatenate(latents),
                preds=np.concatenate(preds),
                confs=np.concatenate(confs),
                labels=np.concatenate(labels_all),
                subs=np.array(subs))


# ADDITION 1 — extract_has now also computes per-sample angular margins and
#              closest_boundary indices using the HAS weight vectors.
@torch.no_grad()
def extract_has(model, loader):
    model.eval()
    latents, preds, confs, labels_all, subs = [], [], [], [], []
    # Lists for new margin-based drift signals
    margins_all, closest_boundary_all = [], []

    # Retrieve L2-normalised weight matrix once (shape [64, n_classes])
    normed_W = model.get_normed_weights()  # stays on the model's device

    for imgs, lbls, sub in loader:
        imgs = imgs.to(DEVICE)
        logits, _, latent = model(imgs)  # no labels → penalty is 0
        probs = logits.exp()             # log_softmax → probabilities
        latents.append(latent.cpu().numpy())
        preds.append(probs.argmax(1).cpu().numpy())
        confs.append(probs.max(1).values.cpu().numpy())
        labels_all.append(np.array([int(l) for l in lbls]))
        subs.extend(sub)

        # ADDITION 1 — cosine similarities: latent is already L2-normalised
        # latent: (B, 64),  normed_W: (64, n_classes)
        # cos_t:  (B, n_classes)
        cos_t = latent @ normed_W          # (B, n_classes)
        # Sort descending along class axis
        cos_sorted, sort_idx = cos_t.sort(dim=1, descending=True)
        # Angular margin = best cosine − second-best cosine
        margin = (cos_sorted[:, 0] - cos_sorted[:, 1]).cpu().numpy()  # (B,)
        # Index of second-best class = drift direction
        cb = sort_idx[:, 1].cpu().numpy()  # (B,)

        margins_all.append(margin)
        closest_boundary_all.append(cb)

    if not latents:
        return None

    result = dict(
        latents=np.concatenate(latents),
        preds=np.concatenate(preds),
        confs=np.concatenate(confs),
        labels=np.concatenate(labels_all),
        subs=np.array(subs),
    )
    # ADDITION 1 — attach new keys to the same dict (saved into existing .npz)
    result["margins"]           = np.concatenate(margins_all).astype(np.float32)
    result["closest_boundary"]  = np.concatenate(closest_boundary_all).astype(np.int32)
    return result


def save_features(data: dict, path: str):
    np.savez_compressed(path, **data)
    n = len(data["latents"])
    print(f"  ✓ {n} samples → {path}")


def save_train_stats(latents, confs, tag):
    """Centroid + distance + confidence stats for drift thresholding."""
    centroid = latents.mean(axis=0)
    dists = np.linalg.norm(latents - centroid, axis=1)
    path = os.path.join(WEIGHT_DIR, f"{tag}_train_stats.npz")
    np.savez(path, centroid=centroid,
             dist_mean=dists.mean(), dist_std=dists.std(),
             conf_mean=confs.mean(), conf_std=confs.std())
    print(f"  ✓ Train stats ({tag}) → {path}")
    print(f"    Latent distance: μ={dists.mean():.4f}, σ={dists.std():.4f}")
    print(f"    Confidence:      μ={confs.mean():.4f}, σ={confs.std():.4f}")

def save_train_stats_has(latents, confs, margins, labels, tag):
# ── Overall centroid (kept for reference / Baseline compatibility) ──────
    centroid = latents.mean(axis=0)
    dists    = np.linalg.norm(latents - centroid, axis=1)

    # ── Per-class sphere centroids (HAS-native data drift) ─────────────────
    # For each class c: mean of all training embeddings of class c,
    # then L2-normalised back onto the unit sphere.
    # This is the geometrically correct reference point for HAS because:
    #   - HAS embeds everything on S⁶³
    #   - The overall centroid is INSIDE the sphere (not on it)
    #   - The per-class sphere centroid is ON the sphere, in the class region
    # Cosine distance from the predicted class centroid is scale-free and
    # consistent with the spherical geometry HAS operates in.
    n_classes = len(np.unique(labels))
    class_centroids    = np.zeros((n_classes, latents.shape[1]), dtype=np.float32)
    class_cos_dist_mean = np.zeros(n_classes, dtype=np.float32)
    class_cos_dist_std  = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cls_lat = latents[mask]                          # (n_c, 64)
        raw_centroid = cls_lat.mean(axis=0)              # (64,)
        norm = np.linalg.norm(raw_centroid)
        class_centroids[c] = raw_centroid / (norm + 1e-8)  # unit vector on S⁶³

        # Cosine distance = 1 - cosine_similarity
        # For unit vectors: cos_sim = dot product
        cos_sims = cls_lat @ class_centroids[c]          # (n_c,)
        cos_dists = 1.0 - cos_sims                       # (n_c,)
        class_cos_dist_mean[c] = cos_dists.mean()
        class_cos_dist_std[c]  = cos_dists.std()

    path = os.path.join(WEIGHT_DIR, f"{tag}_train_stats.npz")
    np.savez(path,
             centroid=centroid,
             dist_mean=dists.mean(),    dist_std=dists.std(),
             conf_mean=confs.mean(),    conf_std=confs.std(),
             margin_mean=margins.mean(), margin_std=margins.std(),
             # NEW — per-class sphere centroids for HAS data drift
             class_centroids=class_centroids,
             class_cos_dist_mean=class_cos_dist_mean,
             class_cos_dist_std=class_cos_dist_std)

    print(f"  ✓ Train stats ({tag}) → {path}")
    print(f"    Latent distance: μ={dists.mean():.4f}, σ={dists.std():.4f}")
    print(f"    Confidence:      μ={confs.mean():.4f}, σ={confs.std():.4f}")
    print(f"    Margin:          μ={margins.mean():.4f}, σ={margins.std():.4f}")
    for c in range(n_classes):
        print(f"    Class {c} cos-dist: μ={class_cos_dist_mean[c]:.4f}, "
              f"σ={class_cos_dist_std[c]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    custom_root = resolve_custom_root()
    print(f"Train data : {TRAIN_ROOT}")
    print(f"Custom data: {custom_root}")
    print(f"Device     : {DEVICE}\n")

    # ── Load models ──
    bl_path  = os.path.join(WEIGHT_DIR, "baseline.pth")
    has_path = os.path.join(WEIGHT_DIR, "has_model.pth")
    for p in [bl_path, has_path]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: weights not found at {p}\n"
                     f"  Run step1_train.py first.")

    baseline = BaselineModel().to(DEVICE)
    baseline.load_state_dict(torch.load(bl_path, map_location=DEVICE))

    has_model = HASModel(margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
    has_model.load_state_dict(torch.load(has_path, map_location=DEVICE))
    print("Models loaded.\n")

    # ── Train data ──
    print("Extracting train features …")
    train_ds = FolderDataset(TRAIN_ROOT, class_names=LANDSCAPE_CLASSES,
                             transform=STANDARD_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4)

    bl_train  = extract_baseline(baseline, train_loader)
    has_train = extract_has(has_model, train_loader)

    save_features(bl_train,  os.path.join(FEATURE_DIR, "bl_train.npz"))
    save_features(has_train, os.path.join(FEATURE_DIR, "has_train.npz"))
    save_train_stats(bl_train["latents"],  bl_train["confs"],  "baseline")
    # ADDITION 1 — use extended stats saver for HAS (includes margin_mean/std)
    save_train_stats_has(has_train["latents"], has_train["confs"],
                         has_train["margins"],  has_train["labels"], "has")

    # ── Custom data (only the 5 classes that map to training classes) ──
    print("\nExtracting custom features …")
    print(f"  Using class map: {CUSTOM_CLASS_MAP}")
    custom_ds = FolderDataset(custom_root, class_map=CUSTOM_CLASS_MAP,
                              transform=STANDARD_TRANSFORM)
    if len(custom_ds) == 0:
        print("\n  ERROR: Custom dataset has 0 images!")
        if os.path.isdir(custom_root):
            contents = sorted(os.listdir(custom_root))[:15]
            print(f"  Folder contents: {contents}")
        print(f"  Expected folders: {list(CUSTOM_CLASS_MAP.keys())}")
        sys.exit(1)

    custom_loader = DataLoader(custom_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"  {len(custom_ds)} images found")

    bl_custom  = extract_baseline(baseline, custom_loader)
    has_custom = extract_has(has_model, custom_loader)

    save_features(bl_custom,  os.path.join(FEATURE_DIR, "bl_custom.npz"))
    save_features(has_custom, os.path.join(FEATURE_DIR, "has_custom.npz"))

    # ── Save metadata ──
    meta_path = os.path.join(FEATURE_DIR, "meta.json")
    meta = dict(
        train_classes=LANDSCAPE_CLASSES,
        custom_classes=custom_ds.class_names,
        custom_class_map={k: v for k, v in CUSTOM_CLASS_MAP.items()},
        label_meaning={i: c for i, c in enumerate(LANDSCAPE_CLASSES)},
        train_root=TRAIN_ROOT,
        custom_root=custom_root,
        n_train=len(train_ds),
        n_custom=len(custom_ds),
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  ✓ Metadata → {meta_path}")
    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
