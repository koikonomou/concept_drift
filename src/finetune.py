"""
finetune.py — Redesigned fine-tuning ablation study.

Must be run AFTER detect.py (reads drift CSVs with file_path column).

─────────────────────────────────────────────────────────────────────────────
DESIGN
─────────────────────────────────────────────────────────────────────────────

Fine-tuning approach (both models):
  • Head-only: backbone fully frozen, only classifier head updated
      Baseline → model.classifier  (Linear 64→5,  320 params)
      HAS      → model.has_layer   (cosine head,  320 params)
  • Loss: CrossEntropy only — no HAS penalty
      The spherical geometry already exists from training.
      We only need to reposition the class boundaries.
  • LR: 1e-2 (safe with only 320 params, 10× higher than full-model FT)
  • Epochs: 20 (head-only convergence is fast)

Shared budget N:
  N is defined by the HAS Pure Data Drift pool size (e.g. 103).
  Both models use the same N for each ft_pct — fair comparison.
  ft_pct=25% → N=26 for both, ft_pct=50% → N=52, ft_pct=100% → N=103.

Four selection modes:
  has-margin      HAS only. Top N from Pure Data Drift pool ranked by
                  data_drift_score descending. Geometrically certified:
                  high angular margin → label reliable, novel territory.
                  This is the proposed method.

  baseline-conf   Baseline only. Top N from full custom set ranked by
                  confidence ascending (lowest confidence first).
                  This is the Baseline's native drift signal — its best
                  attempt at selecting informative fine-tuning data.

  random          Both models. N random images from the full custom set.
                  Fixed seed=42 → reproducible. Same N as has-margin.
                  This is the no-selection-criterion control: what a
                  practitioner without a drift detector would do.

  all             Both models. All non-Pure-Concept-Drift custom samples.
                  No selection criterion, maximum data.
                  Negative control: brute-force vs targeted.

Fixed test set:
  ALL 2150 custom images. Identical for every (model × mode × ft_pct).
  Broken down by HAS drift categories so both models are evaluated on
  the same taxonomy.

─────────────────────────────────────────────────────────────────────────────
WHAT THE COMPARISONS ANSWER
─────────────────────────────────────────────────────────────────────────────

HAS has-margin   vs  HAS random       → does geometric selection beat blind sampling?
HAS has-margin   vs  BL baseline-conf → does margin beat confidence for selection?
HAS any-mode     vs  BL same-mode     → does HAS architecture enable better adaptation?
drift 25% vs 50% vs 100%              → data efficiency curve

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python finetune.py                          # all modes, 25/50/100%
  python finetune.py --ft-pcts 50             # single percentage
  python finetune.py --ft-mode has-margin     # single mode
  python finetune.py --ft-pcts 10,25,50,100  # extended sweep
  python finetune.py --ft-epochs 20 --ft-lr 1e-2

  nohup python finetune.py > logs/finetune.log 2>&1 &
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config import (
    WEIGHT_DIR, RESULT_DIR, DEVICE,
    HAS_MARGIN, HAS_SCALE, BATCH_SIZE,
    LANDSCAPE_CLASSES, ensure_dirs,
)
from models import BaselineModel, HASModel, TRAIN_AUGMENT, STANDARD_TRANSFORM


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FINETUNE_EPOCHS = 20     # head-only converges faster than full model
FINETUNE_LR     = 1e-2   # safe with only 320 params (10× full-model LR)
RANDOM_SEED     = 42

# Drift type display order for print and CSV
_DRIFT_ORDER = [
    "overall",
    "In-Distribution",
    "Pure Data Drift",
    "Data Drift",
    "Pure Concept Drift",
    "Concept Drift",
    "Full Drift (both)",
]
_DRIFT_SHORT = {
    "overall":            "ALL",
    "In-Distribution":    "InDist",
    "Pure Data Drift":    "DataDrift",
    "Data Drift":         "DataDrift",
    "Pure Concept Drift": "Concept",
    "Concept Drift":      "Concept",
    "Full Drift (both)":  "Full",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PathLabelDataset(Dataset):
    """Dataset from explicit (path, label) lists."""
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.open(self.paths[(idx+1) % len(self.paths)]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers — all path reading goes through file_path column
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(csv_path):
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found — run detect.py first.")
    df = pd.read_csv(csv_path)
    if "file_path" not in df.columns:
        sys.exit(f"ERROR: 'file_path' column missing in {csv_path}.\n"
                 f"  Re-run detect.py with the updated version.")
    return df


def _df_to_paths_labels(df):
    """Extract valid (paths, labels) from a DataFrame."""
    cls_to_idx = {c: i for i, c in enumerate(LANDSCAPE_CLASSES)}
    paths, labels = [], []
    for _, row in df.iterrows():
        p   = str(row["file_path"])
        lbl = cls_to_idx.get(row["true_class"], -1)
        if lbl < 0 or not p or not os.path.exists(p):
            continue
        paths.append(p)
        labels.append(lbl)
    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# Pool and test set loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_has_pool(has_csv):
    """Load HAS Pure Data Drift pool sorted by data_drift_score descending.

    This pool defines the reference budget N for all modes and both models.
    Sorted descending = most spatially novel samples first.
    """
    df = _read_csv(has_csv)
    pool = (df[df["has_drift_type_margin"] == "Pure Data Drift"]
            .sort_values("data_drift_score", ascending=False)
            .copy())
    if len(pool) == 0:
        print("  ⚠ HAS Pure Data Drift pool is empty"); return None, 0
    paths, labels = _df_to_paths_labels(pool)
    print(f"  HAS pool: {len(paths)} Pure Data Drift images "
          f"(sorted by data_drift_score ↓, reference budget N={len(paths)})")
    return (paths, labels), len(paths)


def _load_full_test(csv_path, drift_col):
    """ALL custom images as (path, label, drift_type) triples.

    Fixed test set — identical for every run, both models.
    2150 images broken down by HAS drift categories.
    """
    df = _read_csv(csv_path)
    cls_to_idx = {c: i for i, c in enumerate(LANDSCAPE_CLASSES)}
    records = []
    for _, row in df.iterrows():
        p   = str(row["file_path"])
        lbl = cls_to_idx.get(row["true_class"], -1)
        if lbl < 0 or not p or not os.path.exists(p):
            continue
        records.append((p, lbl, row[drift_col]))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Selection — four modes
# ─────────────────────────────────────────────────────────────────────────────

def _select_has_margin(has_pool_paths, has_pool_labels, n_use):
    """HAS native: top N from Pure Data Drift pool by data_drift_score.

    These are the most spatially novel samples with geometrically
    certified labels (high angular margin = far from all boundaries).
    """
    sel = has_pool_paths[:n_use], has_pool_labels[:n_use]
    print(f"  [has-margin] Top {len(sel[0])}/{len(has_pool_paths)} "
          f"Pure Data Drift samples (biggest drift first)")
    return sel


def _select_baseline_conf(bl_csv, n_use):
    """Baseline native: top N from full custom set by confidence ascending.

    Lowest confidence = most uncertain = Baseline's proxy for drift.
    This is the Baseline's best attempt at selecting informative samples.
    Uses the full custom set (not just flagged-drifted) to match the
    spirit of HAS selecting from its geometrically defined pool.
    """
    df = _read_csv(bl_csv)
    # Sort ascending: lowest confidence first (most uncertain)
    selected = df.sort_values("max_confidence", ascending=True).head(n_use)
    paths, labels = _df_to_paths_labels(selected)
    print(f"  [baseline-conf] {len(paths)} images with lowest confidence "
          f"(mean conf={selected['max_confidence'].mean():.3f})")
    # Show drift type breakdown for transparency
    if "drift_type" in selected.columns:
        vc = selected["drift_type"].value_counts()
        for dt, cnt in vc.items():
            print(f"    {dt:<26}: {cnt} ({cnt/len(selected)*100:.1f}%)")
    return paths, labels


def _select_random(csv_path, n_use, seed=RANDOM_SEED):
    """Random N from the full custom set — no selection criterion.

    Same seed for both models → identical random samples → fair comparison.
    This is what a practitioner without a drift detector would do.
    """
    df = _read_csv(csv_path)
    paths_all, labels_all = _df_to_paths_labels(df)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths_all), size=min(n_use, len(paths_all)),
                     replace=False)
    paths  = [paths_all[i]  for i in idx]
    labels = [labels_all[i] for i in idx]
    print(f"  [random] {len(paths)} images from full custom set (seed={seed})")
    return paths, labels


def _select_all(csv_path, drift_col):
    """All non-Pure-Concept-Drift custom samples.

    Excludes only the samples with the most unreliable labels
    (near-boundary, ambiguous).
    """
    df = _read_csv(csv_path)
    concept_labels = {"Pure Concept Drift", "Concept Drift"}
    subset = df[~df[drift_col].isin(concept_labels)]
    paths, labels = _df_to_paths_labels(subset)
    print(f"  [all] {len(paths)} images (full custom set minus concept-drifted)")
    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model, test_records):
    """Error rate per drift type. Lower is better.

    test_records: list of (path, label, drift_type)
    Returns dict {drift_type: error_pct, "overall": error_pct}
    """
    if not test_records:
        return {}
    model.eval()
    by_type = {}
    for _, _, dt in test_records:
        by_type.setdefault(dt, {"correct": 0, "total": 0})

    BATCH = 64
    for start in range(0, len(test_records), BATCH):
        chunk = test_records[start:start+BATCH]
        imgs_list, labels_b, dtypes_b = [], [], []
        for path, label, dtype in chunk:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            imgs_list.append(STANDARD_TRANSFORM(img))
            labels_b.append(label)
            dtypes_b.append(dtype)
        if not imgs_list:
            continue
        preds = model(torch.stack(imgs_list).to(DEVICE))[0].argmax(1).cpu().tolist()
        for pred, lbl, dtype in zip(preds, labels_b, dtypes_b):
            by_type[dtype]["total"]   += 1
            by_type[dtype]["correct"] += int(pred == lbl)

    out = {}
    tc = tn = 0
    for dtype, c in by_type.items():
        n, correct = c["total"], c["correct"]
        if n > 0:
            out[dtype] = round(100.0 * (n - correct) / n, 2)
            tc += correct; tn += n
    out["overall"] = round(100.0 * (tn - tc) / tn, 2) if tn > 0 else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Head-only fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def _run_finetune(model, model_type, sel_paths, sel_labels,
                   ft_epochs, ft_lr):
    """Fine-tune ONLY the classifier head. Backbone fully frozen.

    For Baseline: updates model.classifier (Linear 64→5 = 320 params)
    For HAS:      updates model.has_layer  (cosine head = 320 params)

    Loss: CrossEntropy only — no HAS penalty.
    The spherical geometry already exists from training.
    We only reposition the class boundaries on the sphere.

    LR is higher than full-model fine-tuning because:
      - Only 320 parameters are updated (not 23M)
      - Backbone is frozen — no risk of distorting features
      - Head needs to move meaningfully on small data
    """
    # Step 1: freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: unfreeze only the classifier head
    if model_type == "baseline":
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = list(model.classifier.parameters())
    else:
        for param in model.has_layer.parameters():
            param.requires_grad = True
        trainable = list(model.has_layer.parameters())

    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_params} params (head only, backbone frozen)")

    ft_ds = PathLabelDataset(sel_paths, sel_labels, transform=TRAIN_AUGMENT)
    ft_loader = DataLoader(
        ft_ds, batch_size=min(BATCH_SIZE, len(ft_ds)),
        shuffle=True, num_workers=4, drop_last=False)

    opt = torch.optim.SGD(trainable, lr=ft_lr,
                           momentum=0.9, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()

    for ep in range(1, ft_epochs + 1):
        model.train()
        tot_loss = correct = total = 0
        for imgs, lbls in ft_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)[0]    # first return = logits for both models
            # For HAS: logits are log_softmax → exp() to get probabilities for CE
            if model_type == "has":
                loss = nn.NLLLoss()(logits,lbls)
            else:
                loss = ce(logits, lbls)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            correct  += logits.argmax(1).eq(lbls).sum().item()
            total    += len(lbls)
        print(f"    EP {ep:2d}/{ft_epochs} | "
              f"Loss {tot_loss/len(ft_loader):.4f} | "
              f"Train-Acc {100*correct/total:.1f}%")

    # Re-enable all gradients after fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_err(prefix, err_dict):
    order = ["overall", "In-Distribution", "Pure Data Drift", "Data Drift",
             "Pure Concept Drift", "Concept Drift", "Full Drift (both)"]
    parts = [f"{_DRIFT_SHORT.get(k,k)}={err_dict[k]:.1f}%err"
             for k in order if err_dict.get(k) is not None]
    print(f"  {prefix}: " + "  |  ".join(parts))


def _print_summary(rows):
    if not rows:
        return
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY — Error Rate on FULL CUSTOM SET (all 2150 images)")
    print("Head-only fine-tuning (backbone frozen)  |  Negative Δ = improvement")
    print("=" * 100)
    print(f"\n  {'Model':<10} {'Mode':<16} {'ft%':>4} {'N-FT':>6}  "
          f"{'ALL Δ':>7}  {'DataDrift Δ':>12}  {'InDist Δ':>10}  "
          f"{'Concept Δ':>11}  {'Full Δ':>9}")
    print("  " + "─" * 93)

    prev = None
    for r in rows:
        if r["model"] != prev:
            if prev is not None:
                print()
            prev = r["model"]

        def fmt(keys):
            for k in (keys if isinstance(keys, list) else [keys]):
                v = r.get(f"delta_{k}")
                if v is not None:
                    return f"{v:+.1f}%"
            return "    —  "

        print(f"  {r['model']:<10} {r['mode']:<16} "
              f"{r['ft_pct']*100:>3.0f}% {r['n_finetune']:>6}  "
              f"{fmt('overall'):>7}  "
              f"{fmt(['Pure Data Drift','Data Drift']):>12}  "
              f"{fmt('In-Distribution'):>10}  "
              f"{fmt(['Pure Concept Drift','Concept Drift']):>11}  "
              f"{fmt('Full Drift (both)'):>9}")

    print()
    print("  KEY COMPARISONS:")
    print("  HAS  has-margin  vs  HAS  random      → geometric selection vs blind sampling")
    print("  HAS  has-margin  vs  BL   baseline-conf → margin-based vs confidence-based")
    print("  HAS  any-mode    vs  BL   same-mode    → architecture advantage")
    print("  Positive InDist Δ = catastrophic forgetting (should be ~0 for head-only)")
    print("  Negative Full Δ  = geometric coupling: data-drift FT fixes boundary confusion")


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation
# ─────────────────────────────────────────────────────────────────────────────

def finetune_ablation(ft_epochs=FINETUNE_EPOCHS, ft_lr=FINETUNE_LR,
                      ft_pcts=(0.25, 0.50, 1.0)):
    """Run the full redesigned ablation study.

    For each ft_pct:
      1. Compute shared budget N from HAS pool size
      2. For each model × mode combination:
         a. Load ORIGINAL weights fresh
         b. Select N images according to mode
         c. Evaluate on FULL 2150 custom images BEFORE fine-tuning
         d. Fine-tune HEAD ONLY for ft_epochs with CrossEntropy
         e. Evaluate on SAME full test set AFTER fine-tuning
         f. Record per-drift-type error rates
    """
    print("\n" + "=" * 68)
    print("FINE-TUNING ABLATION — REDESIGNED")
    print("=" * 68)
    print(f"  ft_pcts  : {[f'{p*100:.0f}%' for p in ft_pcts]}")
    print(f"  Epochs   : {ft_epochs}  |  LR: {ft_lr}")
    print(f"  Approach : head-only (backbone frozen), CrossEntropy loss")
    print(f"  Budget N : defined by HAS Pure Data Drift pool size")
    print(f"  Test set : ALL 2150 custom images (fixed)\n")

    bl_path  = os.path.join(WEIGHT_DIR, "baseline.pth")
    has_path = os.path.join(WEIGHT_DIR, "has_model.pth")
    for p in [bl_path, has_path]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found — run train.py first.")

    bl_csv  = os.path.join(RESULT_DIR, "drift_baseline.csv")
    has_csv = os.path.join(RESULT_DIR, "drift_has.csv")

    # ── Load HAS pool once — defines budget N for all runs ───────────────────
    print("Loading HAS pool (defines shared budget N) …")
    has_pool_result, n_pool = _load_has_pool(has_csv)
    if has_pool_result is None:
        sys.exit("ERROR: HAS pool is empty — run detect.py first.")
    has_pool_paths, has_pool_labels = has_pool_result

    # ── Load fixed test sets (once per model) ─────────────────────────────────
    print("\nLoading fixed test sets …")
    # Use HAS taxonomy for both models (richer categories)
    has_test_records = _load_full_test(has_csv, "has_drift_type_margin")
    bl_test_records  = _load_full_test(has_csv, "has_drift_type_margin")
    # Both models evaluated on HAS categories — same taxonomy, fair comparison
    print(f"  Test set (HAS taxonomy): {len(has_test_records)} images")
    vc = pd.Series([r[2] for r in has_test_records]).value_counts()
    for dt, cnt in vc.items():
        print(f"    {dt:<26}: {cnt:5d}  ({cnt/len(has_test_records)*100:.1f}%)")

    all_rows = []

    for ft_pct in ft_pcts:
        n_use = max(1, int(np.ceil(ft_pct * n_pool)))
        print(f"\n{'═'*68}")
        print(f"  ft_pct = {ft_pct*100:.0f}%  →  N = {n_use} images per run")
        print(f"{'═'*68}")

        # ── Define (model, mode) combinations ────────────────────────────────
        # has-margin: only for HAS (uses HAS geometric signal)
        # baseline-conf: only for Baseline (uses Baseline native signal)
        # random: both models, same N, same seed
        # all: both models, same pool definition
        runs = [
            # (model_tag, model_type, orig_path, test_records, mode, csv_for_select, save_prefix)
            ("HAS",      "has",      has_path, has_test_records, "has-margin",     has_csv, "has_model"),
            ("HAS",      "has",      has_path, has_test_records, "random",          has_csv, "has_model"),
            ("HAS",      "has",      has_path, has_test_records, "all",             has_csv, "has_model"),
            ("Baseline", "baseline", bl_path,  bl_test_records,  "baseline-conf",   bl_csv,  "baseline"),
            ("Baseline", "baseline", bl_path,  bl_test_records,  "random",          bl_csv,  "baseline"),
            ("Baseline", "baseline", bl_path,  bl_test_records,  "all",             bl_csv,  "baseline"),
        ]

        for model_tag, model_type, orig_path, test_records, mode, csv_path, save_prefix in runs:
            print(f"\n  ── {model_tag} | {mode} | {ft_pct*100:.0f}% ({n_use} images) ──")

            # Load ORIGINAL weights — fresh every run, no contamination
            if model_tag == "Baseline":
                model = BaselineModel(n_classes=len(LANDSCAPE_CLASSES)).to(DEVICE)
            else:
                model = HASModel(n_classes=len(LANDSCAPE_CLASSES),
                                  margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
            model.load_state_dict(torch.load(orig_path, map_location=DEVICE,
                                              weights_only=True))
            print(f"  Loaded original weights: {os.path.basename(orig_path)}")

            # Evaluate BEFORE
            err_before = _evaluate(model, test_records)
            _print_err("  BEFORE", err_before)

            # Select fine-tuning samples
            if mode == "has-margin":
                sel_paths, sel_labels = _select_has_margin(
                    has_pool_paths, has_pool_labels, n_use)
            elif mode == "baseline-conf":
                sel_paths, sel_labels = _select_baseline_conf(bl_csv, n_use)
            elif mode == "random":
                sel_paths, sel_labels = _select_random(csv_path, n_use)
            else:  # "all"
                sel_paths, sel_labels = _select_all(
                    csv_path, "has_drift_type_margin" if model_type=="has"
                              else "drift_type")

            if not sel_paths:
                print("  ⚠ No samples — skipping"); continue
            print(f"  Fine-tuning on {len(sel_paths)} images …")

            # Fine-tune (head only)
            model = _run_finetune(model, model_type, sel_paths, sel_labels,
                                   ft_epochs, ft_lr)

            # Evaluate AFTER
            err_after = _evaluate(model, test_records)
            _print_err("  AFTER ", err_after)

            # Save weights
            save_name = f"{save_prefix}_{mode}_{int(ft_pct*100)}pct.pth"
            torch.save(model.state_dict(),
                       os.path.join(WEIGHT_DIR, save_name))
            print(f"  ✓ weights/{save_name}")

            # Record
            row = dict(
                model=model_tag, mode=mode,
                ft_pct=ft_pct, n_finetune=len(sel_paths),
                n_test=len(test_records),
                ft_epochs=ft_epochs, ft_lr=ft_lr,
            )
            for dtype in (err_before.keys() | err_after.keys()):
                b = err_before.get(dtype)
                a = err_after.get(dtype)
                row[f"err_before_{dtype}"] = b
                row[f"err_after_{dtype}"]  = a
                row[f"delta_{dtype}"] = (
                    round(a - b, 2)
                    if a is not None and b is not None else None)
            all_rows.append(row)

    _print_summary(all_rows)

    out = os.path.join(RESULT_DIR, "finetune_ablation.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Redesigned fine-tuning ablation. "
                    "Requires detect.py CSVs with file_path column.")
    parser.add_argument("--ft-epochs", type=int,   default=FINETUNE_EPOCHS,
                        help=f"Fine-tune epochs (default {FINETUNE_EPOCHS})")
    parser.add_argument("--ft-lr",     type=float, default=FINETUNE_LR,
                        help=f"Fine-tune LR (default {FINETUNE_LR})")
    parser.add_argument("--ft-pcts",   type=str,   default="25,50,100",
                        help="Comma-separated %% of HAS pool (default: 25,50,100)")
    args = parser.parse_args()

    ensure_dirs()

    try:
        ft_pcts = tuple(float(p.strip()) / 100.0
                        for p in args.ft_pcts.split(","))
    except ValueError:
        sys.exit(f"ERROR: --ft-pcts must be comma-separated numbers, "
                 f"got '{args.ft_pcts}'")

    finetune_ablation(
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        ft_pcts=ft_pcts,
    )
    print("\nfinetune.py complete.")


if __name__ == "__main__":
    main()
