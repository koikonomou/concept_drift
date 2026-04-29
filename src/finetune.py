"""

─────────────────────────────────────────────────────────────────────────────
DESIGN
─────────────────────────────────────────────────────────────────────────────
Fine-tune pool  = all Pure Data Drift samples, sorted by data_drift_score
                  descending (furthest from training centroid first).
                  Labels are ground-truth from folder structure.
                  Geometrically certified: high margin → label reliable.

For each (mode × ft_pct) run:

  Fine-tune set  = top ft_pct% of pool  (what the model learns from)

  Test set       = two parts combined:
    Part A — Held-out pool:
                  bottom (1-ft_pct)% of pool, NEVER seen during fine-tuning.
                  These are also Pure Data Drift but held out.
                  Answers: does fine-tuning on 25% of data-drifted samples
                  improve performance on the other 75%? (data efficiency)
    Part B — Category samples:
                  all non-Pure-Data-Drift custom samples.
                  In-Distribution   → catastrophic forgetting check
                  Pure Concept Drift → does data-drift fine-tuning also fix
                                       boundary confusion? (geometric coupling)
                  Full Drift         → hardest cases

  Category samples are built ONCE per model and reused across all runs.
  The held-out pool changes per ft_pct — test set is rebuilt each run.

Special case ft_pct = 1.0:
  All pool samples are used for fine-tuning → held-out pool is empty.
  Test set = category samples only (consistent with original design).

Special case mode = "all":
  Uses all non-concept-drift samples for fine-tuning regardless of ft_pct.
  Test set = category samples only (no held-out pool since pool is consumed).

─────────────────────────────────────────────────────────────────────────────
ABLATION TABLE COLUMNS
─────────────────────────────────────────────────────────────────────────────
  ALL Δ         overall error rate Δ across the full test set
  DataDrift Δ   error Δ on held-out Pure Data Drift (data efficiency signal)
  InDist Δ      error Δ on In-Distribution (catastrophic forgetting check)
  Concept Δ     error Δ on Pure Concept Drift (geometric coupling finding)
  Full Δ        error Δ on Full Drift (hardest samples)

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
  python finetune.py                              # all modes, 25/50/100%
  python finetune.py --ft-pcts 50                 # single percentage
  python finetune.py --ft-mode drift-ranked       # single mode
  python finetune.py --ft-pcts 10,25,50,75,100   # full sweep
  python finetune.py --ft-epochs 15 --ft-lr 5e-5

  nohup python finetune.py > logs/finetune.log 2>&1 &
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config import (
    CUSTOM_CLASS_MAP, WEIGHT_DIR, RESULT_DIR, DEVICE,
    ALPHA, BETA, HAS_MARGIN, HAS_SCALE, BATCH_SIZE,
    LANDSCAPE_CLASSES, ensure_dirs, resolve_custom_root,
)
from models import (
    BaselineModel, HASModel,
    TRAIN_AUGMENT, STANDARD_TRANSFORM,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FINETUNE_EPOCHS = 10
FINETUNE_LR     = 1e-4      # 500–1000× lower than training LR
RANDOM_SEED     = 42        # fixed → reproducible random baseline

# Drift type order for evaluation and CSV columns
_DRIFT_ORDER = [
    "overall",
    "Pure Data Drift",    # held-out pool — data efficiency signal
    "In-Distribution",    # catastrophic forgetting check
    "Pure Concept Drift", # geometric coupling finding
    "Full Drift (both)",  # hardest cases
]
_DRIFT_SHORT = {
    "overall":            "ALL",
    "Pure Data Drift":    "DataDrift",
    "In-Distribution":    "InDist",
    "Pure Concept Drift": "Concept",
    "Full Drift (both)":  "Full",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PathLabelDataset(Dataset):
    """Dataset from explicit (path, label) lists — used for fine-tune set."""
    def __init__(self, paths, labels, transform=None):
        self.paths     = list(paths)
        self.labels    = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.open(self.paths[(idx + 1) % len(self.paths)]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Sample loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _paths_from_df(df, custom_root):
    """Reconstruct (paths, labels) from CSV rows using CUSTOM_CLASS_MAP.

    Groups by unique (true_class, sub_category) before scanning folders
    to avoid adding the same subfolder multiple times — the CSV has one
    row per image but multiple rows share the same sub_category folder.
    """
    cls_to_idx = {c: i for i, c in enumerate(LANDSCAPE_CLASSES)}
    paths, labels = [], []
    seen = set()   # track (true_class, sub_category) already scanned

    for _, row in df.iterrows():
        key = (row["true_class"], row["sub_category"])
        if key in seen:
            continue
        seen.add(key)

        lbl = cls_to_idx.get(row["true_class"], -1)
        if lbl < 0:
            continue
        sub = row["sub_category"]
        for folder, idx in CUSTOM_CLASS_MAP.items():
            if idx != lbl:
                continue
            sub_path = os.path.join(custom_root, folder, sub)
            if not os.path.isdir(sub_path):
                continue
            for fname in os.listdir(sub_path):
                if fname.lower().endswith(("jpg", "jpeg", "png")):
                    paths.append(os.path.join(sub_path, fname))
                    labels.append(lbl)
            break
    return paths, labels

def _load_pool(csv_path, custom_root, drift_col):
    """Load Pure Data Drift samples sorted by data_drift_score descending.

    Sorted descending = most spatially novel samples first, so drift-ranked
    mode takes pool[:n_use] (the highest-drift samples) and the held-out
    test portion is pool[n_use:] (the lower-drift end of the same category).

    Returns (paths, labels) or None if pool is empty / CSV missing.
    """

    df = pd.read_csv(csv_path)
    pool_label = "Pure Data Drift" if "Pure Data Drift" in df[drift_col].values else "Data Drift"
    pool = (df[df[drift_col] == pool_label]
            .sort_values("data_drift_score", ascending=False)
            .copy())

    paths, labels = _paths_from_df(pool, custom_root)
    if not paths:
        print("  ⚠ Fine-tune pool is empty (no Pure Data Drift samples found)")
        return None

    print(f"  Fine-tune pool: {len(paths)} Pure Data Drift images "
          f"(sorted by data_drift_score ↓)")
    return paths, labels


def _load_category_test(csv_path, custom_root, drift_col):
    """Load all non-Pure-Data-Drift samples as (path, label, drift_type) triples.

    These are the stable part of the test set — same across all ft_pct values.
    Contains: In-Distribution, Pure Concept Drift, Full Drift (both).
    """
    df= pd.read_csv(csv_path)
    pool_label = "Pure Data Drift" if "Pure Data Drift" in df[drift_col].values \
             else "Data Drift"
    test_df = df[df[drift_col] != pool_label].copy()

    records    = []
    cls_to_idx = {c: i for i, c in enumerate(LANDSCAPE_CLASSES)}
    for _, row in test_df.iterrows():
        lbl = cls_to_idx.get(row["true_class"], -1)
        if lbl < 0:
            continue
        sub = row["sub_category"]
        for folder, idx in CUSTOM_CLASS_MAP.items():
            if idx != lbl:
                continue
            sub_path = os.path.join(custom_root, folder, sub)
            if not os.path.isdir(sub_path):
                continue
            for fname in os.listdir(sub_path):
                if fname.lower().endswith(("jpg", "jpeg", "png")):
                    records.append((
                        os.path.join(sub_path, fname),
                        lbl,
                        row[drift_col],
                    ))
            break

    print(f"  Category test samples: {len(records)} "
          f"(In-Dist + Concept Drift + Full Drift)")
    vc = pd.Series([r[2] for r in records]).value_counts()
    for dt, cnt in vc.items():
        print(f"    {dt:<26}: {cnt:5d}  ({cnt/len(records)*100:.1f}%)")
    return records


def _build_test_set(pool_paths, pool_labels, n_use, category_records):
    """Build the full test set for a specific (mode × ft_pct) run.

    Test set = held-out pool (pool[n_use:]) + category samples.

    pool[n_use:]  = Pure Data Drift samples NOT used for fine-tuning.
                    These are the data efficiency measurement:
                    did fine-tuning on n_use samples improve performance
                    on the remaining unseen data-drifted samples?

    category      = In-Distribution, Pure Concept Drift, Full Drift.
                    Same across all runs.

    When n_use == len(pool) (ft_pct = 1.0), held-out pool is empty and
    test set = category samples only.
    """
    held_out = [
        (pool_paths[i], pool_labels[i], "Pure Data Drift")
        for i in range(n_use, len(pool_paths))
    ]
    return held_out + list(category_records)


def _select_finetune(pool_paths, pool_labels, n_use, mode,
                     csv_path, custom_root, drift_col):
    """Select the fine-tune samples for this (mode × ft_pct) run.

    drift-ranked: pool[:n_use]  (highest data_drift_score first)
    random:       random n_use from pool (fixed seed → reproducible)
    all:          all non-Pure-Concept-Drift samples (broader pool,
                  ignores n_use — used as a negative control)
    """
    if mode == "drift-ranked":
        return pool_paths[:n_use], pool_labels[:n_use]

    elif mode == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(pool_paths),
                         size=min(n_use, len(pool_paths)),
                         replace=False)
        return ([pool_paths[i] for i in idx],
                [pool_labels[i] for i in idx])

    else:  # "all" — negative control: use everything (including lower-quality labels)
        df = pd.read_csv(csv_path)
        # Exclude Pure Concept Drift only — those have the most unreliable labels
        subset = df[df[drift_col] != "Pure Concept Drift"]
        paths, labels = _paths_from_df(subset, custom_root)
        return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_by_drift_type(model, test_records):
    """Error rate per drift type.

    Returns dict {drift_type: error_pct, "overall": error_pct}.
    Error rate = % wrong (100 − accuracy). Lower is better.
    Processes in mini-batches of 64 for speed.
    """
    if not test_records:
        return {}

    model.eval()
    by_type = {}
    for _, _, dt in test_records:
        by_type.setdefault(dt, {"correct": 0, "total": 0})

    BATCH = 64
    for start in range(0, len(test_records), BATCH):
        chunk = test_records[start:start + BATCH]
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
        tensor = torch.stack(imgs_list).to(DEVICE)
        preds  = model(tensor)[0].argmax(1).cpu().tolist()
        for pred, lbl, dtype in zip(preds, labels_b, dtypes_b):
            by_type[dtype]["total"]   += 1
            by_type[dtype]["correct"] += int(pred == lbl)

    out = {}
    total_c = total_n = 0
    for dtype, c in by_type.items():
        n, correct = c["total"], c["correct"]
        if n > 0:
            out[dtype] = round(100.0 * (n - correct) / n, 2)
            total_c += correct
            total_n += n
    out["overall"] = (round(100.0 * (total_n - total_c) / total_n, 2)
                      if total_n > 0 else None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_err(prefix, err_dict):
    parts = [f"{_DRIFT_SHORT.get(k, k)}={err_dict[k]:.1f}%err"
             for k in _DRIFT_ORDER if err_dict.get(k) is not None]
    print(f"  {prefix}: " + "  |  ".join(parts))


def _print_summary(rows):
    if not rows:
        return
    print("\n" + "=" * 94)
    print("ABLATION SUMMARY — Error Rate on Test Set  (negative Δ = fewer errors = better)")
    print("=" * 94)
    print(f"\n  {'Model':<10} {'Mode':<14} {'ft%':>4} {'N-FT':>6} {'N-Test':>7}  "
          f"{'ALL Δ':>7}  {'DataDrift Δ':>12}  {'InDist Δ':>10}  "
          f"{'Concept Δ':>11}  {'Full Δ':>8}")
    print("  " + "─" * 88)

    prev = None
    for r in rows:
        if r["model"] != prev:
            if prev is not None:
                print()
            prev = r["model"]

        def fmt(k):
            v = r.get(f"delta_{k}")
            return "    —  " if v is None else f"{v:+.1f}%"

        print(f"  {r['model']:<10} {r['mode']:<14} "
              f"{r['ft_pct']*100:>3.0f}% {r['n_finetune']:>6} {r['n_test']:>7}  "
              f"{fmt('overall'):>7}  "
              f"{fmt('Pure Data Drift'):>12}  "
              f"{fmt('In-Distribution'):>10}  "
              f"{fmt('Pure Concept Drift'):>11}  "
              f"{fmt('Full Drift (both)'):>8}")

    print()
    print("  DataDrift Δ: did fine-tuning on top ft% improve the held-out bottom (1-ft)?")
    print("               This measures data efficiency of the selection strategy.")
    print("  InDist Δ   : positive = catastrophic forgetting (fine-tuning hurt clean data).")
    print("  Concept Δ  : negative = geometric coupling — data-drift fine-tuning also")
    print("               reduces boundary confusion (key paper finding).")


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning — one pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_finetune(model, model_type, sel_paths, sel_labels, ft_epochs, ft_lr):
    """Fine-tune model in-place. Returns model."""
    ft_ds     = PathLabelDataset(sel_paths, sel_labels, transform=TRAIN_AUGMENT)
    ft_loader = DataLoader(ft_ds,
                           batch_size=min(BATCH_SIZE, len(ft_ds)),
                           shuffle=True, num_workers=4, drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=ft_lr,
                           momentum=0.9, weight_decay=1e-4)
    ce  = torch.nn.CrossEntropyLoss()
    nll = torch.nn.NLLLoss()

    for ep in range(1, ft_epochs + 1):
        model.train()
        tot_loss = correct = total = 0
        for imgs, lbls in ft_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            if model_type == "baseline":
                logits, _ = model(imgs)
                loss = ce(logits, lbls)
            else:
                logits, penalty, _ = model(imgs, labels=lbls)
                loss = ALPHA * nll(logits, lbls) + BETA * penalty
            loss.backward(); opt.step()
            tot_loss += loss.item()
            correct  += logits.argmax(1).eq(lbls).sum().item()
            total    += len(lbls)
        print(f"    EP {ep:2d}/{ft_epochs} | "
              f"Loss {tot_loss/len(ft_loader):.4f} | "
              f"Train-Acc {100*correct/total:.1f}%")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune_ablation(ft_epochs=FINETUNE_EPOCHS, ft_lr=FINETUNE_LR,
                      ft_pcts=(0.25, 0.50, 1.0),
                      modes=("drift-ranked", "random", "all")):
    """Run fine-tuning ablation across all (model × mode × ft_pct) combinations.

    For each combination:
      1. Load original weights fresh (modes cannot contaminate each other)
      2. Select fine-tune samples (top n_use from pool per mode)
      3. Build test set = held-out pool[n_use:] + category samples
      4. Evaluate on test set BEFORE fine-tuning
      5. Fine-tune for ft_epochs epochs
      6. Evaluate on same test set AFTER fine-tuning
      7. Record per-drift-type error rates
    """
    print("\n" + "=" * 68)
    print("FINE-TUNING ABLATION")
    print("=" * 68)
    print(f"  Modes   : {list(modes)}")
    print(f"  ft_pcts : {[f'{p*100:.0f}%' for p in ft_pcts]}")
    print(f"  Epochs  : {ft_epochs}  |  LR: {ft_lr}")
    print(f"  Pool    : Pure Data Drift, sorted by data_drift_score ↓")
    print(f"  Test    : held-out pool (bottom 1-ft%) + category samples\n")

    custom_root = resolve_custom_root()
    bl_path     = os.path.join(WEIGHT_DIR, "baseline.pth")
    has_path    = os.path.join(WEIGHT_DIR, "has_model.pth")
    for p in [bl_path, has_path]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found — run train.py first.")

    all_rows = []

    for model_tag, model_type, csv_name, drift_col, orig_path, save_prefix in [
        ("Baseline", "baseline", "drift_baseline.csv", "drift_type",
         bl_path,   "baseline"),
        ("HAS",      "has",      "drift_has.csv",       "has_drift_type_margin",
         has_path,  "has_model"),
    ]:
        csv_full = os.path.join(RESULT_DIR, csv_name)
        print(f"\n  ══ {model_tag} ══════════════════════════════════════════════")

        # ── Load pool and category test samples (once per model) ─────────────
        print(f"\n  Loading fine-tune pool …")
        pool_result = _load_pool(csv_full, custom_root, drift_col)
        if pool_result is None:
            print(f"  ⚠ No pool — skipping {model_tag}")
            continue
        pool_paths, pool_labels = pool_result
        n_pool = len(pool_paths)

        print(f"\n  Loading category test samples …")
        category_records = _load_category_test(csv_full, custom_root, drift_col)
        if not category_records and n_pool == 0:
            print(f"  ⚠ No data — skipping {model_tag}")
            continue

        # ── Reference model (no fine-tuning) ─────────────────────────────────
        # Evaluate on the ft_pct=1.0 test set (= category only, no held-out pool)
        # as a common baseline for all rows.
        if model_tag == "Baseline":
            ref = BaselineModel(n_classes=len(LANDSCAPE_CLASSES)).to(DEVICE)
        else:
            ref = HASModel(n_classes=len(LANDSCAPE_CLASSES),
                            margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
        ref.load_state_dict(torch.load(orig_path, map_location=DEVICE,
                                        weights_only=True))

        # ── Ablation loop ─────────────────────────────────────────────────────
        for mode in modes:
            for ft_pct in ft_pcts:

                # For "all" mode, ft_pct does not limit the pool — it uses
                # everything. Test set = category only (no held-out pool).
                if mode == "all":
                    n_use = n_pool   # pool fully consumed
                else:
                    n_use = max(1, int(np.ceil(ft_pct * n_pool)))

                print(f"\n  ── {model_tag} | {mode} | {ft_pct*100:.0f}% ──────────────────────")

                # ── Build test set for this run ───────────────────────────────
                # held-out pool = pool[n_use:] (unseen Pure Data Drift samples)
                # For "all" mode or ft_pct=1.0, held-out is empty.
                test_records = _build_test_set(
                    pool_paths, pool_labels, n_use, category_records)

                n_held = len(pool_paths) - n_use if mode != "all" else 0
                print(f"  Test set: {len(test_records)} images  "
                      f"(held-out pool: {n_held}  |  category: {len(category_records)})")

                # ── Evaluate BEFORE (fresh weights) ───────────────────────────
                if model_tag == "Baseline":
                    model = BaselineModel(n_classes=len(LANDSCAPE_CLASSES)).to(DEVICE)
                else:
                    model = HASModel(n_classes=len(LANDSCAPE_CLASSES),
                                     margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
                model.load_state_dict(torch.load(orig_path, map_location=DEVICE,
                                                  weights_only=True))

                err_before = _evaluate_by_drift_type(model, test_records)
                _print_err("  BEFORE", err_before)

                # ── Select fine-tune samples ───────────────────────────────────
                sel_paths, sel_labels = _select_finetune(
                    pool_paths, pool_labels, n_use,
                    mode, csv_full, custom_root, drift_col)
                if not sel_paths:
                    print(f"  ⚠ No samples selected — skipping")
                    continue
                print(f"  Fine-tuning on {len(sel_paths)} images …")

                # ── Fine-tune ─────────────────────────────────────────────────
                model = _run_finetune(model, model_type,
                                       sel_paths, sel_labels,
                                       ft_epochs, ft_lr)

                # ── Evaluate AFTER ────────────────────────────────────────────
                err_after = _evaluate_by_drift_type(model, test_records)
                _print_err("  AFTER ", err_after)

                # ── Save weights ──────────────────────────────────────────────
                save_name = f"{save_prefix}_{mode}_{int(ft_pct*100)}pct.pth"
                save_path = os.path.join(WEIGHT_DIR, save_name)
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ {save_path}")

                # ── Record row ────────────────────────────────────────────────
                row = dict(
                    model=model_tag, mode=mode,
                    ft_pct=ft_pct, n_finetune=len(sel_paths),
                    n_test=len(test_records), n_held_out=n_held,
                    ft_epochs=ft_epochs, ft_lr=ft_lr,
                )
                for dtype in _DRIFT_ORDER:
                    b = err_before.get(dtype)
                    a = err_after.get(dtype)
                    row[f"err_before_{dtype}"] = b
                    row[f"err_after_{dtype}"]  = a
                    row[f"delta_{dtype}"] = (
                        round(a - b, 2)
                        if a is not None and b is not None
                        else None)
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
        description="Fine-tuning ablation. Requires detect.py CSVs to exist.")
    parser.add_argument("--ft-epochs", type=int,   default=FINETUNE_EPOCHS,
                        help=f"Fine-tune epochs per run (default {FINETUNE_EPOCHS})")
    parser.add_argument("--ft-lr",     type=float, default=FINETUNE_LR,
                        help=f"Fine-tune learning rate (default {FINETUNE_LR})")
    parser.add_argument("--ft-pcts",   type=str,   default="25,50,100",
                        help="Comma-separated %% of pool to sweep (default: 25,50,100)")
    parser.add_argument("--ft-mode",
                        choices=["drift-ranked", "random", "all", "all-modes"],
                        default="all-modes",
                        help="Selection mode (default: all-modes runs all three)")
    args = parser.parse_args()

    ensure_dirs()

    try:
        ft_pcts = tuple(float(p.strip()) / 100.0
                        for p in args.ft_pcts.split(","))
    except ValueError:
        sys.exit(f"ERROR: --ft-pcts must be comma-separated numbers, "
                 f"got '{args.ft_pcts}'")

    modes = (["drift-ranked", "random", "all"]
             if args.ft_mode == "all-modes"
             else [args.ft_mode])

    finetune_ablation(
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        ft_pcts=ft_pcts,
        modes=modes,
    )
    print("\nfinetune.py complete.")


if __name__ == "__main__":
    main()
t
