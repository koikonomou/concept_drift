import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader

TRAIN_DIR = "/home/katerina/codes/datasets/landscapes/Landscape Classification/Landscape Classification/Training Data"
MODEL_SAVE_PATH = "baseline_landscape_has.pth"
DIST_STATS_PATH = "landscape_has_train_stats.npz"  # centroid + std saved for drift thresholding

ALPHA = 1.0   # CrossEntropy weight
BETA  = 3.0   # HAS penalty weight


# ---------------------------------------------------------------------------
# 1. LATENT CONV EMBEDDER
#    Takes the spatial feature map from ResNet layer4 (B, 2048, 7, 7)
#    instead of the globally-pooled flat vector.
#    Two conv layers preserve spatial structure before projecting to 64D.
# ---------------------------------------------------------------------------
class LatentConvEmbedder(nn.Module):
    def __init__(self, in_channels=2048, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(128, out_dim)
        self.bn_out = nn.BatchNorm1d(out_dim)

    def forward(self, feature_map):
        x = F.relu(self.bn1(self.conv1(feature_map)))   # (B, 512, 7, 7)
        x = F.relu(self.bn2(self.conv2(x)))             # (B, 128, 7, 7)
        x = self.pool(x)                                # (B, 128, 1, 1)
        x = torch.flatten(x, 1)                         # (B, 128)
        x = self.bn_out(self.fc(x))                     # (B, 64)
        return x


# ---------------------------------------------------------------------------
# 2. HASeparator LAYER  (unchanged from hasseparator.py — reused as-is)
#    Binary anchors: [0] = out-of-distribution, [1] = in-distribution
#    Penalty forces training embeddings toward the in-distribution anchor.
# ---------------------------------------------------------------------------
class HASeparatorRegressor(nn.Module):
    def __init__(self, feat_dim=64, margin=0.35, scale=30.0):
        super().__init__()
        self.margin = margin
        self.scale  = scale
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, 2))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        normed_x = F.normalize(x, p=2, dim=1)
        normed_w = F.normalize(self.weight, p=2, dim=0)

        logits = torch.mm(normed_x, normed_w)
        probs  = F.softmax(self.scale * logits, dim=1)
        in_dist_score = probs[:, 1]   # proximity to in-distribution anchor

        penalty = torch.tensor(0.0, device=x.device)
        if labels is not None:
            target_indices = torch.round(labels).long().clamp(0, 1)
            gr_w    = normed_w[:, target_indices].t()
            other_w = normed_w[:, 1 - target_indices].t()
            dw      = gr_w - other_w
            normed_dw = F.normalize(dw, p=2, dim=1)
            win     = torch.sum(normed_x * normed_dw, dim=1)
            penalty = torch.mean(self.margin - torch.clamp(win, max=self.margin))

        return in_dist_score, penalty


# ---------------------------------------------------------------------------
# 3. FULL MODEL
#    ResNet50 backbone split BEFORE global avg pool to expose (B, 2048, 7, 7).
#    Two heads share that spatial feature map:
#      • classifier  → CrossEntropy loss (5 landscape classes)
#      • embedder+HAS → structural HAS penalty (in-distribution anchoring)
# ---------------------------------------------------------------------------
class LandscapeHASModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')

        # Backbone: everything up to (and including) layer4, no global pool
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # → output shape: (B, 2048, 7, 7) for 224×224 input

        # Classification head (data-drift uses confidence from here)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5)
        )

        # Spatial conv embedder → 64D latent for HAS + t-SNE
        self.embedder  = LatentConvEmbedder(in_channels=2048, out_dim=64)
        self.has_layer = HASeparatorRegressor(feat_dim=64)

    def forward(self, x, has_labels=None):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)            # (B, 2048, 7, 7)  ← spatial info intact

        logits   = self.classifier(feat_map)           # (B, 5)
        latent   = self.embedder(feat_map)              # (B, 64)
        in_dist_score, penalty = self.has_layer(latent, has_labels)

        return logits, in_dist_score, penalty, latent


# ---------------------------------------------------------------------------
# 4. TRAINING
#    HAS label = 1.0 for all training samples (all are in-distribution).
#    The penalty continuously pushes training embeddings toward the
#    in-distribution anchor so that OOD images stand out at evaluation time.
# ---------------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"Classes: {dataset.classes} | Samples: {len(dataset)}")

    model     = LandscapeHASModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ce_loss   = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_ce, total_pen, correct, total = 0, 0, 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # All training images are in-distribution → HAS label = 1.0
            has_labels = torch.ones(imgs.size(0), device=device)

            optimizer.zero_grad()
            logits, _, penalty, _ = model(imgs, has_labels)

            loss_ce = ce_loss(logits, labels)
            loss    = (ALPHA * loss_ce) + (BETA * penalty)
            loss.backward()
            optimizer.step()

            total_ce  += loss_ce.item()
            total_pen += penalty.item()
            _, preds   = logits.max(1)
            correct   += preds.eq(labels).sum().item()
            total     += labels.size(0)

        acc = 100. * correct / total
        print(f"Epoch {epoch+1} | CE: {total_ce/len(loader):.4f} | "
              f"HAS Penalty: {total_pen/len(loader):.4f} | Acc: {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}")

    # Save training distribution statistics for drift thresholding at eval time
    _save_train_distribution(model, loader, device)


def _save_train_distribution(model, loader, device):
    """Collect 64D embeddings and Euclidean distances from centroid on training
    data. These statistics define the in-distribution baseline thresholds."""
    model.eval()
    all_latents = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            _, _, _, latent = model(imgs)
            all_latents.append(latent.cpu().numpy())

    latents  = np.concatenate(all_latents, axis=0)   # (N, 64)
    centroid = latents.mean(axis=0)                   # (64,)
    dists    = np.linalg.norm(latents - centroid, axis=1)

    np.savez(DIST_STATS_PATH,
             centroid=centroid,
             dist_mean=dists.mean(),
             dist_std=dists.std())
    print(f"Training distribution stats saved → {DIST_STATS_PATH}")
    print(f"  Centroid distance: mean={dists.mean():.4f}, std={dists.std():.4f}")


if __name__ == "__main__":
    train()
