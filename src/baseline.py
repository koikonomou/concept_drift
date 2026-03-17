import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from hasseparator import HASeparator
import torch.nn.functional as F


TRAIN_DIR = "/home/kate/datasets/landscapes/Landscape Classification/Landscape Classification/Training Data"
TEST_DIR = "/home/kate/datasets/landscapes/Landscape Classification/Landscape Classification/Testing Data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LandscapeBaseline(nn.Module):
    def __init__(self, num_classes=5, proj_dim=256):
        super(LandscapeBaseline, self).__init__()
        base_resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1]) # Output: 2048
        
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.SiLU()
        )
        

        self.has_head = HASeparator(input_dim=proj_dim, num_classes=num_classes, margin=0.1, scale=5.0)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embed = self.embedder(features)
        
        logits, normalized_embed, penalties = self.has_head(embed, labels)
        
        return logits, penalties, normalized_embed

#LANDSCAPE_CLASSES = ['coast', 'desert', 'forest', 'glacier', 'mountain']

if __name__ == "__main__":
    train_dataset = ImageFolder(root=TRAIN_DIR , transform=transform)
    test_dataset = ImageFolder(root=TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size= 32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Detected classes: {train_dataset.classes}")
    print(f"Training on: {len(train_dataset)} samples")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandscapeBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Training...")
    for epoch in range(10):
        model.train()
        total_loss_accum = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            logits, penalties, _ = model(imgs, labels)
            ce_loss = F.cross_entropy(logits, labels)
            penalty_loss = penalties.mean() if penalties is not None else 0
            loss = ce_loss + 1.0 * penalty_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss_accum += loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss_accum/len(train_loader):.4f} | Acc: {acc:.2f}%")


    torch.save(model.state_dict(), "landscape_has_model.pth")
    print("HAS Baseline model saved!")