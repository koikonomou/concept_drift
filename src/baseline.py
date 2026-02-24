import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image


TRAIN_DIR = "/home/kate/datasets/landscapes/Landscape Classification/Landscape Classification/Training Data"
TEST_DIR = "/home/kate/datasets/landscapes/Landscape Classification/Landscape Classification/Testing Data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LandscapeBaseline(nn.Module):
    def __init__(self):
        super(LandscapeBaseline, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.backbone(x)

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
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
    # Save Baseline
    torch.save(model.state_dict(), "baseline_landscape.pth")
    print("Baseline model saved!")
