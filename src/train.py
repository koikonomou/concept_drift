"""
train.py — Train Baseline and HAS models, evaluate on the official test set.

Training accuracy printed each epoch = TRAINING accuracy.
It measures how well the model fits the training data — not generalisation.

Test accuracy is evaluated on TEST_ROOT after training finishes, using
STANDARD_TRANSFORM (no augmentation, no random crops).
This is the number to report in the paper.

HAS convergence health:
  w_cos ≈ 0.10–0.25  →  weight vectors well-separated on the sphere (good)
  w_cos ≈ 0.80–1.00  →  collapsing (lower BETA or increase HAS_WARMUP_EPOCHS)

Usage:
    python train.py                  # train both models
    python train.py --only has       # HAS only
    python train.py --only baseline  # Baseline only
    python train.py --epochs 150     # override epoch count

    nohup python train.py > logs/train.log 2>&1 &
"""

import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path

from config import (
    TRAIN_ROOT, TEST_ROOT,
    WEIGHT_DIR, DEVICE,
    EPOCHS, LR, HAS_LR, BATCH_SIZE,
    ALPHA, BETA, HAS_MARGIN, HAS_SCALE, HAS_WARMUP_EPOCHS,
    LANDSCAPE_CLASSES, ensure_dirs,
)
from models import (
    BaselineModel, HASModel, FolderDataset,
    TRAIN_AUGMENT, STANDARD_TRANSFORM,
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on test set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, model_type="baseline"):
    """Evaluate on TEST_ROOT with no augmentation and print result."""
    if not os.path.isdir(TEST_ROOT):
        print(f"  ⚠ TEST_ROOT not found ({TEST_ROOT}) — skipping test evaluation")
        return None

    ds = FolderDataset(TEST_ROOT, class_names=LANDSCAPE_CLASSES,
                       transform=STANDARD_TRANSFORM)
    if len(ds) == 0:
        print("  ⚠ Test set is empty — skipping")
        return None

    loader  = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    model.eval()
    correct = total = 0
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        logits = model(imgs)[0]   # first return value = logits for both models
        correct += logits.argmax(1).cpu().eq(labels).sum().item()
        total   += len(labels)

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  {model_type.upper():<32} TEST ACC: {acc:6.2f}%  │")
    print(f"  └──────────────────────────────────────────────────┘\n")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Training — Baseline
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(epochs, lr, weight_dir):
    print("\n" + "=" * 62)
    print("TRAINING BASELINE MODEL (ResNet50 + CrossEntropy)")
    print("=" * 62)

    dataset = ImageFolder(root=TRAIN_ROOT, transform=TRAIN_AUGMENT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)
    print(f"  Train: {len(dataset)} samples  |  Classes: {dataset.classes}")

    model     = BaselineModel(n_classes=len(dataset.classes)).to(DEVICE)
    opt       = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=1e-4)
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    ce         = nn.CrossEntropyLoss()
    print(f"  lr={lr}  |  LR drops at epochs {milestones}\n")

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
        # Train-Acc = accuracy on training data with augmentation.
        # Measures fitting, NOT generalisation. See TEST ACC below.
        print(f"  Epoch {ep:3d}/{epochs} | "
              f"Loss {tot_loss/len(loader):.4f} | "
              f"Train-Acc {100*correct/total:.1f}% | "
              f"lr={scheduler.get_last_lr()[0]:.1e}")

    path = weight_dir / "baseline.pth"
    print(f"\n  ✓ Saved → {path}")
    evaluate_test(model, "baseline")


# ─────────────────────────────────────────────────────────────────────────────
# Training — HAS
# ─────────────────────────────────────────────────────────────────────────────

def train_has(epochs, lr, weight_dir,  has_scale, has_margin):
    print("\n" + "=" * 62)
    print("TRAINING HAS MODEL (ResNet50 + HASeparator)")
    print("=" * 62)

    dataset = ImageFolder(root=TRAIN_ROOT, transform=TRAIN_AUGMENT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"  Train: {len(dataset)} samples  |  Classes: {dataset.classes}")

    model     = HASModel(n_classes=len(dataset.classes), margin=has_margin, scale=has_scale).to(DEVICE)
    opt       = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    nll        = nn.NLLLoss()
    print(f"  lr={lr}  |  Warmup={HAS_WARMUP_EPOCHS} epochs  |  "
          f"BETA={BETA}  scale={has_scale}  margin={has_margin}")
    print(f"  LR drops at epochs {milestones}\n")

    for ep in range(1, epochs + 1):
        model.train()
        tot_nll = tot_pen = correct = total = 0

        # Linear BETA warmup: penalty scales 0→BETA over HAS_WARMUP_EPOCHS.
        # Lets classification loss stabilise geometry before penalty reshapes sphere.
        effective_beta = BETA * min(ep / HAS_WARMUP_EPOCHS, 1.0)

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits, penalty, _ = model(imgs, labels=labels)
            loss_nll = nll(logits, labels)
            loss     = ALPHA * loss_nll + effective_beta * penalty
            loss.backward(); opt.step()
            tot_nll += loss_nll.item()
            tot_pen += penalty.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total   += len(labels)

        scheduler.step()

        # w_cos: mean pairwise cosine between class weight vectors.
        # Printed every 10 epochs and during warmup.
        extra = ""
        if ep % 10 == 0 or ep <= HAS_WARMUP_EPOCHS:
            with torch.no_grad():
                nw     = model.get_normed_weights()
                cos_ww = (nw.T @ nw).cpu()
                mask   = ~torch.eye(len(dataset.classes), dtype=torch.bool)
                extra  = f" | w_cos={cos_ww[mask].mean().item():.3f}"

        wu = f" [warmup β={effective_beta:.2f}]" if ep <= HAS_WARMUP_EPOCHS else ""
        print(f"  Epoch {ep:3d}/{epochs} | "
              f"NLL {tot_nll/len(loader):.4f} | "
              f"Pen {tot_pen/len(loader):.4f} | "
              f"Train-Acc {100*correct/total:.1f}% | "
              f"lr={scheduler.get_last_lr()[0]:.1e}{extra}{wu}")

    path = weight_dir / "has_model.pth"
    print(f"\n  ✓ Saved → {path}")
    evaluate_test(model, "has")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Baseline and HAS models, evaluate on test set.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default {EPOCHS})")
    parser.add_argument("--lr", type=float, default=LR, help=f"Baseline SGD learning rate (default {LR})")
    parser.add_argument("--has-lr", type=float, default=HAS_LR, help=f"HAS SGD learning rate (default {HAS_LR})")
    parser.add_argument("--only", choices=["baseline", "has"], help="Train only one model")
    parser.add_argument("--has-scale", type=float, default=HAS_SCALE)
    parser.add_argument("--has-margin", type=float, default=HAS_MARGIN)
    parser.add_argument("--weight-dir", default=WEIGHT_DIR)
    args = parser.parse_args()
    weight_dir = Path(args.weight_dir)
    weight_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs()
    print(f"Device: {DEVICE}\n")

    if args.only != "has":
        train_baseline(args.epochs, args.lr, weight_dir)
    if args.only != "baseline":
        train_has(args.epochs, args.has_lr, weight_dir, args.has_scale, args.has_margin)

    print("train.py complete.")


if __name__ == "__main__":
    main()
