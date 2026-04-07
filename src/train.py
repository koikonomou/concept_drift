"""
train.py — Train Baseline and HAS models, evaluate on test set,
           optionally fine-tune on drift-selected custom samples.

Training accuracy printed each epoch = TRAINING accuracy (same data
the model learns from, with augmentation).  This measures fitting.

Test accuracy is evaluated on TEST_ROOT after training with no
augmentation — this is the number to report in the paper.

Fine-tuning (--finetune):
  Reads results/drift_has.csv (or drift_baseline.csv) from detect.py.
  Selects only In-Distribution and Pure Data Drift samples — the ones
  where the model's label assignment is geometrically trustworthy.
  Fine-tunes both models briefly with a very low LR.
  Prints and saves a before/after accuracy comparison on the full custom set.
  Saved weights: weights/baseline_ft.pth  and  weights/has_model_ft.pth

Usage:
    python train.py                        # train both, evaluate on test
    python train.py --only has             # train HAS only
    python train.py --finetune             # train + fine-tune (random strategy)
    python train.py --skip-train --finetune  # fine-tune existing weights only
    python train.py --finetune --ft-strategy drift-ranked --ft-top-pct 0.5
                                           # fine-tune on top 50% subfolders by drift severity
    nohup python train.py --finetune > logs/train.log 2>&1 &
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

from config import (TRAIN_ROOT, TEST_ROOT, CUSTOM_ROOT, CUSTOM_CLASS_MAP,
                    WEIGHT_DIR, RESULT_DIR, DEVICE,
                    EPOCHS, LR, HAS_LR, BATCH_SIZE,
                    ALPHA, BETA, HAS_MARGIN, HAS_SCALE, HAS_WARMUP_EPOCHS,
                    LANDSCAPE_CLASSES, ensure_dirs, resolve_custom_root)
from models import (BaselineModel, HASModel, FolderDataset,
                    TRAIN_AUGMENT, STANDARD_TRANSFORM)


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning config
# ─────────────────────────────────────────────────────────────────────────────
FINETUNE_EPOCHS = 10
FINETUNE_LR     = 1e-4   # much lower than training LR — adaptation not re-training
FINETUNE_SELECT = ["In-Distribution", "Pure Data Drift"]
# "Pure Concept Drift" and "Full Drift (both)" excluded — label assignment
# near a boundary is geometrically unreliable for supervision.


# ─────────────────────────────────────────────────────────────────────────────
# PathLabelDataset — for fine-tuning on an explicit image list
# ─────────────────────────────────────────────────────────────────────────────

class PathLabelDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths; self.labels = labels; self.transform = transform

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
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, model_type="baseline"):
    """Compute accuracy on a DataLoader. No gradients, no augmentation."""
    model.eval()
    correct = total = 0
    for batch in loader:
        imgs   = batch[0].to(DEVICE)
        labels = batch[1]
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if model_type == "baseline":
            logits, _ = model(imgs)
        else:
            logits, _, _ = model(imgs)
        correct += logits.argmax(1).cpu().eq(labels).sum().item()
        total   += len(labels)
    return 100.0 * correct / total if total > 0 else 0.0


def make_loader(root, class_names=None, class_map=None):
    ds = FolderDataset(root, class_names=class_names, class_map=class_map, transform=STANDARD_TRANSFORM)
    if len(ds) == 0:
        return None, 0
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=4), len(ds)


def print_test_acc(model, model_type, label):
    if not os.path.isdir(TEST_ROOT):
        print(f"TEST_ROOT not found, skipping test evaluation")
        return None
    loader, n = make_loader(TEST_ROOT, class_names=LANDSCAPE_CLASSES)
    if loader is None:
        return None
    acc = evaluate(model, loader, model_type)
    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  {label} TEST SET ACCURACY: {acc:6.2f}%  │")
    print(f"  └──────────────────────────────────────────────┘\n")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(epochs, lr):
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)

    dataset = ImageFolder(root=TRAIN_ROOT, transform=TRAIN_AUGMENT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"  Train samples: {len(dataset)}  |  Classes: {dataset.classes}")

    model = BaselineModel(n_classes=len(dataset.classes)).to(DEVICE)
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    ce         = nn.CrossEntropyLoss()
    print(f"  lr={lr}  |  LR drops at epochs {milestones}")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(imgs)
            loss = ce(logits, labels)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            correct  += logits.argmax(1).eq(labels).sum().item()
            total    += len(labels)
        scheduler.step()
        # Train-Acc = accuracy on training data (measures fitting, not generalisation)
        print(f"  Epoch {ep:3d}/{epochs} | "
              f"Loss {tot_loss/len(loader):.4f} | "
              f"Train-Acc {100*correct/total:.1f}% | "
              f"lr={scheduler.get_last_lr()[0]:.1e}")

    path = os.path.join(WEIGHT_DIR, "baseline.pth")
    torch.save(model.state_dict(), path)
    print(f"Saved → {path}")
    print_test_acc(model, "baseline", "BASELINE")
    return model


def train_has(epochs, lr):
    print("\n" + "=" * 60)
    print("TRAINING HAS MODEL")
    print("=" * 60)

    dataset = ImageFolder(root=TRAIN_ROOT, transform=TRAIN_AUGMENT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"  Train samples: {len(dataset)}  |  Classes: {dataset.classes}")

    model = HASModel(n_classes=len(dataset.classes),margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
    opt   = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    nll        = nn.NLLLoss()
    print(f"  lr={lr}  |  Warmup={HAS_WARMUP_EPOCHS} epochs  |  " f"BETA={BETA}  scale={HAS_SCALE}  |  LR drops at {milestones}")

    for ep in range(1, epochs + 1):
        model.train()
        tot_nll = tot_pen = correct = total = 0
        effective_beta = BETA * min(ep / HAS_WARMUP_EPOCHS, 1.0)

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits, penalty, _ = model(imgs, labels=labels)
            loss_nll = nll(logits, labels)
            loss = ALPHA * loss_nll + effective_beta * penalty
            loss.backward(); opt.step()
            tot_nll += loss_nll.item()
            tot_pen += penalty.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total   += len(labels)

        scheduler.step()
        extra = ""
        if ep % 10 == 0 or ep <= HAS_WARMUP_EPOCHS:
            with torch.no_grad():
                nw = model.get_normed_weights()
                cos_ww = (nw.T @ nw).cpu()
                mask = ~torch.eye(len(dataset.classes), dtype=torch.bool)
                extra = f" | w_cos={cos_ww[mask].mean().item():.3f}"
        wu = f" [warmup β={effective_beta:.2f}]" if ep <= HAS_WARMUP_EPOCHS else ""
        print(f"  Epoch {ep:3d}/{epochs} | "
              f"NLL {tot_nll/len(loader):.4f} | "
              f"Pen {tot_pen/len(loader):.4f} | "
              f"Train-Acc {100*correct/total:.1f}% | "
              f"lr={scheduler.get_last_lr()[0]:.1e}{extra}{wu}")

    path = os.path.join(WEIGHT_DIR, "has_model.pth")
    torch.save(model.state_dict(), path)
    print(f"Saved → {path}")
    print_test_acc(model, "has", "HAS")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def _rank_subfolders_by_drift(df, drift_col):
    """Return subfolders ranked by concept drift severity (descending).

    Ranking score per (true_class, sub_category) group:
      - fraction of samples that are concept-drifted or fully drifted
      - mean concept_drift_score (1 - confidence) as tiebreaker
    Both columns come from detect.py and are always present.
    """
    concept_drifted_types = {"Pure Concept Drift", "Full Drift (both)",
                             "Concept Drift"}   # baseline label variant

    df = df.copy()
    df["_is_concept_drifted"] = df[drift_col].isin(concept_drifted_types)

    grp = df.groupby(["true_class", "sub_category"]).agg(
        drift_rate  = ("_is_concept_drifted", "mean"),
        drift_score = ("concept_drift_score",  "mean"),
        n_samples   = ("_is_concept_drifted",  "count"),
    ).reset_index()

    # dominant drift target — available in HAS CSV as predicted_drift_toward
    if "predicted_drift_toward" in df.columns:
        dom = (df[df["predicted_drift_toward"] != "—"]
               .groupby(["true_class", "sub_category"])["predicted_drift_toward"]
               .agg(lambda x: x.value_counts().index[0] if len(x) else "—")
               .reset_index(name="dominant_target"))
        grp = grp.merge(dom, on=["true_class", "sub_category"], how="left")
        grp["dominant_target"] = grp["dominant_target"].fillna("—")
    else:
        grp["dominant_target"] = "—"

    grp = grp.sort_values(["drift_rate", "drift_score"], ascending=False)
    return grp


def _build_finetune_dataset(csv_path, custom_root, drift_col,
                             strategy="random", top_pct=1.0):
    """Select drift-approved custom samples and return (paths, labels).

    strategy:
        "random"       — all FINETUNE_SELECT samples, random split (original)
        "drift-ranked" — restrict to the top `top_pct` fraction of subfolders
                         ranked by concept drift severity, then split
    top_pct: fraction of subfolders to keep when strategy="drift-ranked" (0-1)
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠ {csv_path} not found — run detect.py first")
        return None

    df = pd.read_csv(csv_path)
    if drift_col not in df.columns:
        print(f"  ⚠ Column '{drift_col}' missing — re-run detect.py")
        return None

    # ── Subfolder ranking (drift-ranked strategy) ──────────────────────────
    if strategy == "drift-ranked":
        ranked = _rank_subfolders_by_drift(df, drift_col)
        n_keep = max(1, int(np.ceil(top_pct * len(ranked))))
        kept   = ranked.head(n_keep)
        print(f"  Drift-ranked: keeping top {top_pct*100:.0f}% = "
              f"{n_keep}/{len(ranked)} subfolders")
        print(f"  {'Subfolder':<30} {'Class':<14} {'DriftRate':>10} "
              f"{'Score':>8} {'→ Target'}")
        print("  " + "-" * 70)
        for _, r in kept.iterrows():
            print(f"  {r['sub_category']:<30} {r['true_class']:<14} "
                  f"{r['drift_rate']:>9.1%} {r['drift_score']:>8.4f} "
                  f"  → {r['dominant_target']}")
        print()
        keep_set = set(zip(kept["true_class"], kept["sub_category"]))
        df = df[df.apply(
            lambda r: (r["true_class"], r["sub_category"]) in keep_set, axis=1)]

    # ── FINETUNE_SELECT filter (always applied) ────────────────────────────
    selected = df[df[drift_col].isin(FINETUNE_SELECT)]
    total    = len(df)
    print(f"  Selected {len(selected)}/{total} samples ({len(selected)/total*100:.1f}%)")
    for dt in FINETUNE_SELECT:
        n = (selected[drift_col] == dt).sum()
        print(f"    {dt}: {n}")

    if len(selected) == 0:
        print("  ⚠ No samples pass the filter")
        return None

    cls_to_idx = {c: i for i, c in enumerate(LANDSCAPE_CLASSES)}
    paths, labels = [], []

    for _, row in selected.iterrows():
        lbl = cls_to_idx.get(row["true_class"], -1)
        if lbl < 0:
            continue
        sub = row["sub_category"]
        for folder, idx in CUSTOM_CLASS_MAP.items():
            if idx == lbl:
                sub_path = os.path.join(custom_root, folder, sub)
                if os.path.isdir(sub_path):
                    for fname in os.listdir(sub_path):
                        if fname.lower().endswith(("jpg","jpeg","png")):
                            paths.append(os.path.join(sub_path, fname))
                            labels.append(lbl)
                    break

    if not paths:
        print("  ⚠ Could not reconstruct paths — check CUSTOM_CLASS_MAP")
        return None

    print(f"  Built fine-tune dataset: {len(paths)} images")
    return paths, labels


def finetune(ft_epochs=FINETUNE_EPOCHS, ft_lr=FINETUNE_LR,
             strategy="random", top_pct=1.0):
    print("\n" + "=" * 60)
    print("FINE-TUNING ON DRIFT-SELECTED CUSTOM SAMPLES")
    print("=" * 60)
    print(f"  Selection criteria : {FINETUNE_SELECT}")
    print(f"  Strategy           : {strategy}"
          + (f"  (top {top_pct*100:.0f}% subfolders by drift)" if strategy == "drift-ranked" else ""))
    print(f"  Fine-tune epochs   : {ft_epochs}")
    print(f"  Fine-tune LR       : {ft_lr}  (training LR was {LR})\n")

    custom_root = resolve_custom_root()

    # Load weights
    bl_path  = os.path.join(WEIGHT_DIR, "baseline.pth")
    has_path = os.path.join(WEIGHT_DIR, "has_model.pth")
    for p in [bl_path, has_path]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Train first.")

    bl  = BaselineModel(n_classes=len(LANDSCAPE_CLASSES)).to(DEVICE)
    has = HASModel(n_classes=len(LANDSCAPE_CLASSES),
                   margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
    bl.load_state_dict(torch.load(bl_path,  map_location=DEVICE))
    has.load_state_dict(torch.load(has_path, map_location=DEVICE))

    # Full custom set loader — used only as fallback when no drift samples pass the filter
    custom_loader, n_custom = make_loader(custom_root, class_map=CUSTOM_CLASS_MAP)
    if custom_loader is None:
        print("  ⚠ Custom dataset not found")
        return
    print(f"  Custom set: {n_custom} images total\n")

    results = {}
    ce  = nn.CrossEntropyLoss()
    nll = nn.NLLLoss()

    for model, model_type, csv_name, drift_col, save_name, tag in [
        (bl,  "baseline", "drift_baseline.csv", "drift_type",           "baseline_ft.pth", "Baseline"),
        (has, "has",      "drift_has.csv",       "has_drift_type_margin","has_model_ft.pth","HAS"),
    ]:
        print(f"  ─── {tag} ────────────────────────────────────────")

        # Build fine-tune dataset (returns paths + labels, not yet a Dataset)
        result = _build_finetune_dataset(
            os.path.join(RESULT_DIR, csv_name), custom_root, drift_col,
            strategy=strategy, top_pct=top_pct)

        if result is None:
            print(f"  Skipping fine-tuning for {tag}\n")
            acc_before = evaluate(model, custom_loader, model_type)
            print(f"  Custom-set accuracy (full set): {acc_before:.2f}%")
            results[tag] = dict(before=acc_before, after=acc_before)
            continue

        all_paths, all_labels = result

        # 80/20 train/test split — evaluate only on the held-out test portion
        idx      = np.random.permutation(len(all_paths))
        split    = max(1, int(0.8 * len(idx)))
        tr_idx   = idx[:split]
        te_idx   = idx[split:]

        ft_train_ds = PathLabelDataset(
            [all_paths[i] for i in tr_idx],
            [all_labels[i] for i in tr_idx],
            transform=TRAIN_AUGMENT)
        ft_test_ds  = PathLabelDataset(
            [all_paths[i] for i in te_idx],
            [all_labels[i] for i in te_idx],
            transform=STANDARD_TRANSFORM)

        ft_test_loader = DataLoader(ft_test_ds, batch_size=64, shuffle=False,
                                    num_workers=4, drop_last=False)

        print(f"  Train split: {len(ft_train_ds)} images | "
              f"Test split (held-out): {len(ft_test_ds)} images")

        # Accuracy BEFORE fine-tuning — measured on held-out test split only
        acc_before = evaluate(model, ft_test_loader, model_type)
        print(f"  Held-out accuracy BEFORE fine-tuning: {acc_before:.2f}%")

        ft_loader = DataLoader(ft_train_ds, batch_size=min(BATCH_SIZE, len(ft_train_ds)),
                               shuffle=True, num_workers=4, drop_last=False)
        opt = torch.optim.SGD(model.parameters(), lr=ft_lr,
                               momentum=0.9, weight_decay=1e-4)

        for ep in range(1, ft_epochs + 1):
            model.train()
            tot_loss = correct = total = 0
            for imgs, labels in ft_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                opt.zero_grad()
                if model_type == "baseline":
                    logits, _ = model(imgs)
                    loss = ce(logits, labels)
                else:
                    logits, penalty, _ = model(imgs, labels=labels)
                    loss = ALPHA * nll(logits, labels) + BETA * penalty
                loss.backward(); opt.step()
                tot_loss += loss.item()
                correct  += logits.argmax(1).eq(labels).sum().item()
                total    += len(labels)
            eval_acc = evaluate(model, ft_test_loader, model_type)
            model.train()
            print(f"    FT Epoch {ep:2d}/{ft_epochs} | "
                  f"Loss {tot_loss/len(ft_loader):.4f} | "
                  f"Train-Acc {100*correct/total:.1f}% | "
                  f"Eval-Acc {eval_acc:.2f}%")

        # Accuracy AFTER fine-tuning — same held-out test split
        acc_after = evaluate(model, ft_test_loader, model_type)
        print(f"  Held-out accuracy AFTER  fine-tuning: {acc_after:.2f}%")
        results[tag] = dict(before=acc_before, after=acc_after)

        # Save fine-tuned weights separately (originals preserved)
        ft_path = os.path.join(WEIGHT_DIR, save_name)
        torch.save(model.state_dict(), ft_path)
        print(f"  ✓ Saved fine-tuned weights → {ft_path}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("FINE-TUNING SUMMARY — Custom Set Accuracy")
    print("=" * 60)
    print(f"\n  {'Model':<12} {'Before':>10} {'After':>10} {'Δ':>8}")
    print("  " + "-" * 43)
    rows = []
    for tag, r in results.items():
        delta = r["after"] - r["before"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {tag:<12} {r['before']:>9.2f}% {r['after']:>9.2f}% "
              f"{sign}{delta:>6.2f}%")
        rows.append(dict(model=tag,
                         custom_acc_before=round(r["before"], 2),
                         custom_acc_after=round(r["after"], 2),
                         delta=round(delta, 2),
                         ft_epochs=ft_epochs, ft_lr=ft_lr,
                         selected_types=str(FINETUNE_SELECT)))

    csv_path = os.path.join(RESULT_DIR, "finetune_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  ✓ {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--has-lr",     type=float, default=HAS_LR)
    parser.add_argument("--only",       choices=["baseline", "has"])
    parser.add_argument("--finetune",   action="store_true", help="Fine-tune on drift-selected custom samples after training (requires detect.py CSVs)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, jump straight to fine-tuning")
    parser.add_argument("--ft-epochs",   type=int,   default=FINETUNE_EPOCHS)
    parser.add_argument("--ft-lr",       type=float, default=FINETUNE_LR)
    parser.add_argument("--ft-strategy", choices=["random", "drift-ranked"],
                        default="random",
                        help="random: all eligible samples (default); "
                             "drift-ranked: restrict to top subfolders by concept drift severity")
    parser.add_argument("--ft-top-pct",  type=float, default=0.5,
                        help="Fraction of subfolders to keep when --ft-strategy=drift-ranked (default 0.5)")
    args = parser.parse_args()

    ensure_dirs()
    print(f"Device: {DEVICE}")

    if not args.skip_train:
        if args.only != "has":
            train_baseline(args.epochs, args.lr)
        if args.only != "baseline":
            train_has(args.epochs, args.has_lr)

    if args.finetune:
        finetune(ft_epochs=args.ft_epochs, ft_lr=args.ft_lr,
                 strategy=args.ft_strategy, top_pct=args.ft_top_pct)

    print("\ntrain.py complete.")


if __name__ == "__main__":
    main()
