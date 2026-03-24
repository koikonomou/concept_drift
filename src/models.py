"""
Baseline ResNet50 and HAS-augmented model.

HASeparator follows the original paper:
    Kansizoglou et al., "HASeparator: Hyperplane-Assisted Softmax,"
    IEEE ICMLA 2020, pp. 519-526.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset
from PIL import Image, PngImagePlugin, ImageFile
 
from config import LANDSCAPE_CLASSES
 
# Handle PNGs with oversized ICC profiles or truncated files
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 10
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
 
TRAIN_AUGMENT = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class FolderDataset(Dataset):
    """Loads images from class sub-folders.
 
    Two modes:
      1. class_names=["Coast", "Desert", ...]
         Loads these folders, assigns labels 0, 1, 2, … in order.
 
      2. class_map={"Beach Landscape": 0, "Desert Landscape": 1, ...}
         Loads only the specified folders, assigns the given label index.
         Use this when custom folder names differ from training names.
 
    If neither matches anything on disk, auto-discovers all folders.
    """
 
    def __init__(self, root_dir, class_names=None, class_map=None,
                 transform=None):
        self.transform   = transform
        self.samples     = []
        self.class_names = []
 
        if not os.path.isdir(root_dir):
            print(f"  ⚠ Dataset root does not exist: {root_dir}")
            return
 
        all_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
 
        # ── Build folder→label mapping ──
        if class_map is not None:
            # Explicit mapping: folder name → label index
            folder_to_label = {}
            for folder, label_idx in class_map.items():
                if folder in all_dirs:
                    folder_to_label[folder] = label_idx
                else:
                    # Try case-insensitive match
                    lower_map = {d.lower(): d for d in all_dirs}
                    if folder.lower() in lower_map:
                        folder_to_label[lower_map[folder.lower()]] = label_idx
                    else:
                        print(f"  ⚠ '{folder}' not found in {root_dir}")
            self.class_names = list(folder_to_label.keys())
 
        elif class_names is not None:
            lower_map = {d.lower(): d for d in all_dirs}
            folder_to_label = {}
            for idx, cn in enumerate(class_names):
                if cn in all_dirs:
                    folder_to_label[cn] = idx
                elif cn.lower() in lower_map:
                    folder_to_label[lower_map[cn.lower()]] = idx
            if not folder_to_label:
                print(f"  ⚠ None of {class_names} found — auto-discovering")
                folder_to_label = {d: i for i, d in enumerate(all_dirs)}
            self.class_names = list(folder_to_label.keys())
 
        else:
            folder_to_label = {d: i for i, d in enumerate(all_dirs)}
            self.class_names = all_dirs
 
        # ── Load images ──
        for folder, label_idx in folder_to_label.items():
            cls_path = os.path.join(root_dir, folder)
            count = 0
            for entry in os.listdir(cls_path):
                entry_path = os.path.join(cls_path, entry)
                if os.path.isdir(entry_path):
                    for img in os.listdir(entry_path):
                        if img.lower().endswith(("png", "jpg", "jpeg")):
                            self.samples.append(
                                dict(path=os.path.join(entry_path, img),
                                     label=label_idx, sub=entry))
                            count += 1
                elif entry.lower().endswith(("png", "jpg", "jpeg")):
                    self.samples.append(
                        dict(path=entry_path, label=label_idx, sub=folder))
                    count += 1
            if count == 0:
                print(f"  ⚠ '{folder}' folder exists but contains 0 images")
 
        print(f"  FolderDataset: {len(self.samples)} images, "
              f"{len(self.class_names)} classes {self.class_names}")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["path"]).convert("RGB")
        except (ValueError, OSError, SyntaxError):
            alt = (idx + 1) % len(self.samples)
            s = self.samples[alt]
            img = Image.open(s["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, s["label"], s["sub"]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Embedder → 64-D latent  (matches the paper's architecture)
#   GlobalAveragePooling → BN → Dropout → Dense(64) → BN
# ─────────────────────────────────────────────────────────────────────────────
class Embedder64(nn.Module):
    def __init__(self, in_dim=2048, out_dim=64, dropout=0.3):
        super().__init__()
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.bn_in  = nn.BatchNorm1d(in_dim)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(in_dim, out_dim)
        self.bn_out = nn.BatchNorm1d(out_dim)
 
    def forward(self, fm):
        x = self.pool(fm).flatten(1)   # (B, 2048)
        x = self.bn_in(x)
        x = self.drop(x)
        x = self.fc(x)                 # (B, 64)
        return self.bn_out(x)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HASeparator — Multi-class (faithful to the paper)
# ─────────────────────────────────────────────────────────────────────────────
class HASeparatorMultiClass(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 64,
                 margin: float = 0.1, scale: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin      = margin
        self.scale       = scale
        self.weight = nn.Parameter(torch.empty(feat_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)
 
    def forward(self, embds, labels=None):
        normed_embds = F.normalize(embds, p=2, dim=1)
        normed_w     = F.normalize(self.weight, p=2, dim=0)
 
        cos_t  = normed_embds @ normed_w
        logits = F.log_softmax(self.scale * cos_t, dim=1)
 
        penalties = torch.zeros(embds.size(0), self.num_classes, device=embds.device)
        if labels is not None:
            labels = labels.long()
            B = embds.size(0)
            gr_w = normed_w[:, labels].t().unsqueeze(-1)
            temp = normed_w.unsqueeze(0).expand(B, -1, -1)
            dw = gr_w - temp
            normed_dw = F.normalize(dw, p=2, dim=1)
            win = torch.einsum('bd,bdc->bc', normed_embds, normed_dw)
            penalties = self.margin - torch.clamp(win, max=self.margin)  # (B, C) raw -- matches original

        return logits, penalties  # matches original signature: (logits, penalties)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ResNet50 backbone
# ─────────────────────────────────────────────────────────────────────────────
def _resnet50_backbone():
    resnet = models.resnet50(weights=None)   # train from scratch — matches original paper
    stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    return stem, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline (no HAS)
#    Classification goes THROUGH the 64-D latent:
#      feature_map → Embedder64 → 64-D → Linear(64, n_classes) → softmax
#    This ensures the 64-D space is shaped by the classification objective,
#    making it directly comparable to HAS which also classifies from 64-D.
# ─────────────────────────────────────────────────────────────────────────────
class BaselineModel(nn.Module):
    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()
        self.stem, self.l1, self.l2, self.l3, self.l4 = _resnet50_backbone()
        self.embedder   = Embedder64(2048, 64, dropout)
        self.classifier = nn.Linear(64, n_classes)
 
    def forward(self, x):
        x  = self.stem(x)
        x  = self.l1(x); x = self.l2(x); x = self.l3(x)
        fm = self.l4(x)
        latent = self.embedder(fm)           # (B, 64)
        logits = self.classifier(latent)     # (B, n_classes)
        return logits, latent
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. HAS model (multi-class HAS — faithful to the paper)
# ─────────────────────────────────────────────────────────────────────────────
class HASModel(nn.Module):
    def __init__(self, n_classes=5, margin=0.1, scale=1.0, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.stem, self.l1, self.l2, self.l3, self.l4 = _resnet50_backbone()
        self.embedder = Embedder64(2048, 64, dropout)
        self.has_layer = HASeparatorMultiClass(
            num_classes=n_classes, feat_dim=64, margin=margin, scale=scale)
 
    def forward(self, x, labels=None):
        x  = self.stem(x)
        x  = self.l1(x); x = self.l2(x); x = self.l3(x)
        fm = self.l4(x)
        latent_raw = self.embedder(fm)
        logits, penalties = self.has_layer(latent_raw, labels)
        penalty = penalties.mean()  # reduce_mean over (B, C) -- matches original loss: CE + reduce_mean(penalties)
        latent = F.normalize(latent_raw, p=2, dim=1)  # HAS operates on unit sphere — drift measured in the same space
        return logits, penalty, latent