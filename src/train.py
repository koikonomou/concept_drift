"""
step1_train.py — Train baseline and HAS models, save weights.

Usage:
    python step1_train.py                  # train both, 10 epochs
    python step1_train.py --epochs 15      # custom epoch count
    python step1_train.py --only baseline  # train just one model

Outputs:
    weights/baseline.pth
    weights/has_model.pth
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import (TRAIN_ROOT, WEIGHT_DIR, DEVICE,
                    EPOCHS, LR, HAS_LR, BATCH_SIZE, ALPHA, BETA,
                    HAS_MARGIN, HAS_SCALE, ensure_dirs)
from models import BaselineModel, HASModel, TRAIN_AUGMENT

import os


def train_baseline(epochs, lr):
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL (no HAS)")
    print("=" * 60)

    dataset = ImageFolder(root=TRAIN_ROOT, transform=TRAIN_AUGMENT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Classes: {dataset.classes}  |  Samples: {len(dataset)}")

    model = BaselineModel(n_classes=len(dataset.classes)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(imgs)
            loss = ce(logits, labels)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            correct  += logits.argmax(1).eq(labels).sum().item()
            total    += len(labels)
        print(f"  Epoch {ep:2d}/{epochs} | "
              f"Loss {tot_loss/len(loader):.4f} | Acc {100*correct/total:.1f}%")

    path = os.path.join(WEIGHT_DIR, "baseline.pth")
    torch.save(model.state_dict(), path)
    print(f"  ✓ Saved → {path}\n")


def train_has(epochs, lr):
    print("\n" + "=" * 60)
    print("TRAINING HAS MODEL (paper-faithful HASeparator)")
    print("=" * 60)

    dataset = ImageFolder(root=TRAIN_ROOT, transform=TRAIN_AUGMENT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Classes: {dataset.classes}  |  Samples: {len(dataset)}")

    model = HASModel(n_classes=len(dataset.classes),
                     margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)

    # SGD with momentum — matches the original paper.
    # Angular margin methods need uniform gradient scaling across
    # the L2-normalized weights; Adam's per-parameter rates break this.
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, nesterov=True, weight_decay=1e-4)

    # Step-decay scheduler: drop LR by 10× at 60% and 80% of training
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)

    nll = nn.NLLLoss()
    print(f"  Optimizer: SGD(lr={lr}, momentum=0.9, nesterov=True)")
    print(f"  LR drops at epochs {milestones}")

    for ep in range(1, epochs + 1):
        model.train()
        tot_nll, tot_pen, correct, total = 0.0, 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            opt.zero_grad()
            logits, penalty, _ = model(imgs, labels=labels)
            loss_nll = nll(logits.clamp(min=1e-7).log(), labels)
            loss = ALPHA * loss_nll + BETA * penalty
            loss.backward(); opt.step()

            tot_nll += loss_nll.item()
            tot_pen += penalty.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total   += len(labels)

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {ep:2d}/{epochs} | "
              f"NLL {tot_nll/len(loader):.4f} | "
              f"HAS-Pen {tot_pen/len(loader):.4f} | "
              f"Acc {100*correct/total:.1f}% | "
              f"lr={cur_lr:.1e}")

    path = os.path.join(WEIGHT_DIR, "has_model.pth")
    torch.save(model.state_dict(), path)
    print(f"  ✓ Saved → {path}\n")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Train models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR,
                        help="Baseline (Adam) learning rate")
    parser.add_argument("--has-lr", type=float, default=HAS_LR,
                        help="HAS (SGD) learning rate")
    parser.add_argument("--only", choices=["baseline", "has"],
                        help="Train only one model")
    args = parser.parse_args()

    ensure_dirs()
    print(f"Device: {DEVICE}")

    if args.only != "has":
        train_baseline(args.epochs, args.lr)
    if args.only != "baseline":
        train_has(args.epochs, args.has_lr)

    print("Step 1 complete.")


if __name__ == "__main__":
    main()
