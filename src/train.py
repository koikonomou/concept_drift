import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import (TRAIN_ROOT, WEIGHT_DIR, DEVICE,
                    EPOCHS, LR, HAS_LR, BATCH_SIZE, ALPHA, BETA,
                    HAS_MARGIN, HAS_SCALE, HAS_WARMUP_EPOCHS, ensure_dirs)
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

    # SGD with momentum — same optimizer as HAS and original paper
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, nesterov=True, weight_decay=1e-4)
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    ce         = nn.CrossEntropyLoss()
    print(f"  Optimizer: SGD(lr={lr}, momentum=0.9, nesterov=True)")
    print(f"  LR drops at epochs {milestones}")

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
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {ep:2d}/{epochs} | "
              f"Loss {tot_loss/len(loader):.4f} | Acc {100*correct/total:.1f}% | "
              f"lr={cur_lr:.1e}")

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
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, nesterov=True, weight_decay=1e-4)

    # Step-decay scheduler: drop LR by 10× at 60% and 80% of training.
    # Milestones computed on the post-warmup epoch count so the decay
    # still fires at the right fraction of productive training.
    milestones = [int(epochs * 0.6), int(epochs * 0.8)]
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)

    nll = nn.NLLLoss()
    print(f"  Optimizer  : SGD(lr={lr}, momentum=0.9, nesterov=True)")
    print(f"  Warmup     : {HAS_WARMUP_EPOCHS} epochs (BETA linearly 0→{BETA})")
    print(f"  LR drops at: epochs {milestones}")
    print(f"  BETA={BETA}, ALPHA={ALPHA}, margin={HAS_MARGIN}, scale={HAS_SCALE}")

    for ep in range(1, epochs + 1):
        model.train()
        tot_nll, tot_pen, correct, total = 0.0, 0.0, 0, 0

        # ── LINEAR WARMUP for BETA ──────────────────────────────────────────
        # During the first HAS_WARMUP_EPOCHS epochs, scale the penalty weight
        # from 0 up to BETA linearly.  This lets the classification loss
        # stabilise the embedding space first, preventing all weight vectors
        # from collapsing toward a single class (the Mountain collapse seen
        # in results with BETA=1.0 and no warmup from epoch 1).
        if ep <= HAS_WARMUP_EPOCHS:
            effective_beta = BETA * (ep / HAS_WARMUP_EPOCHS)
        else:
            effective_beta = BETA

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
        cur_lr = scheduler.get_last_lr()[0]

        # ── Diagnostic: print weight vector spread every 10 epochs ─────────
        # A healthy HAS model has weight vectors spread around the sphere.
        # If all cosines between weight pairs are high (> 0.8), the vectors
        # are collapsing — a sign that BETA is too high or warmup is missing.
        extra = ""
        if ep % 10 == 0 or ep <= HAS_WARMUP_EPOCHS:
            with torch.no_grad():
                nw = model.get_normed_weights()          # (64, n_classes)
                # Pairwise cosines between class weight vectors
                cos_ww = (nw.T @ nw).cpu()              # (n_classes, n_classes)
                mask = ~torch.eye(len(dataset.classes), dtype=torch.bool)
                mean_off_diag = cos_ww[mask].mean().item()
            extra = f" | w_cos={mean_off_diag:.3f}"
            # w_cos close to 0 = well-separated weights (good)
            # w_cos close to 1 = weights collapsing (bad — increase warmup or lower BETA)

        warmup_str = f" [warmup β={effective_beta:.2f}]" if ep <= HAS_WARMUP_EPOCHS else ""
        print(f"  Epoch {ep:3d}/{epochs} | "
              f"NLL {tot_nll/len(loader):.4f} | "
              f"HAS-Pen {tot_pen/len(loader):.4f} | "
              f"Acc {100*correct/total:.1f}% | "
              f"lr={cur_lr:.1e}{extra}{warmup_str}")

    path = os.path.join(WEIGHT_DIR, "has_model.pth")
    torch.save(model.state_dict(), path)
    print(f"  ✓ Saved → {path}\n")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Train models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR,
                        help="Baseline (SGD) learning rate")
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
