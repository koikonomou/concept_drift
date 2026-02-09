import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from dotenv import load_dotenv
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

ANNOTATION_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/annotation"
IMAGE_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/images"
class LapisArtDataset(Dataset):
    def __init__(self, dataframe, img_dir,transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_filename'])
        
        image = Image.open(img_path).convert('RGB')
        
        label = torch.tensor(row['mean_response'] / 100.0, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        return image, label
# Define standard ResNet transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ArtAestheticBaseline(nn.Module):
    def __init__(self):
        super(ArtAestheticBaseline, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.backbone(x)

RENAISSANCE_STYLES = ['Early_Renaissance', 'High_Renaissance','Northern_Renaissance', 'Mannerism_Late_Renaissance']

if __name__ == "__main__":
    train_df_full = pd.read_csv(os.path.join(ANNOTATION_DIR, "LAPIS_GIAA_Trainsplit.csv"))
    val_df_full = pd.read_csv(os.path.join(ANNOTATION_DIR, "LAPIS_GIAA_Valsplit.csv"))
    
    train_df = train_df_full[train_df_full['style'].isin(RENAISSANCE_STYLES)]
    val_df = val_df_full[val_df_full['style'].isin(RENAISSANCE_STYLES)]
    
    print(f"Training on: {len(train_df)} Renaissance samples")
    print(f"Validating on: {len(val_df)} Renaissance samples")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(LapisArtDataset(train_df, IMAGE_DIR, transform), batch_size=32, shuffle=True)
    test_loader = DataLoader(LapisArtDataset(val_df, IMAGE_DIR, transform), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtAestheticBaseline().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Training...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # Save Baseline
    torch.save(model.state_dict(), "baseline_renaissance.pth")
    print("Baseline model saved!")
