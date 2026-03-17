import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# --- PATHS ---
ANNOTATION_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/annotation"
IMAGE_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/images"
RENAISSANCE_STYLES = ['Early_Renaissance', 'High_Renaissance', 'Northern_Renaissance', 'Mannerism_Late_Renaissance']

# --- 1. THE HASeparator LAYER (PyTorch Version) ---
class HASeparatorRegressor(nn.Module):
    def __init__(self, feat_dim=64, margin=0.35, scale=30.0):
        super(HASeparatorRegressor, self).__init__()
        self.margin = margin
        self.scale = scale
        # Two anchors: [0] for Low Quality, [1] for High Quality
        self.weight = nn.Parameter(torch.FloatTensor(feat_dim, 2))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        normed_x = F.normalize(x, p=2, dim=1)
        normed_w = F.normalize(self.weight, p=2, dim=0)

        # Cosine Similarity Logits
        logits = torch.mm(normed_x, normed_w)
        probs = F.softmax(self.scale * logits, dim=1)
        predicted_score = probs[:, 1]  # Proximity to "High Quality" anchor

        penalty = torch.tensor(0.0).to(x.device)
        if labels is not None:
            # Penalty logic: Discretize labels for hyperplane indexing
            target_indices = torch.round(labels).long()
            gr_w = normed_w[:, target_indices].t()
            other_w = normed_w[:, 1 - target_indices].t()
            
            dw = gr_w - other_w
            normed_dw = F.normalize(dw, p=2, dim=1)
            win = torch.sum(normed_x * normed_dw, dim=1)
            penalty = torch.mean(self.margin - torch.clamp(win, max=self.margin))

        return predicted_score, penalty

# --- 2. THE MODEL ---
class ArtAestheticHASModel(nn.Module):
    def __init__(self):
        super(ArtAestheticHASModel, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 64D Latent Space for t-SNE visualization
        self.embeddings = nn.Sequential(
            nn.Linear(2048, 64),
            nn.BatchNorm1d(64)
        )
        self.has_layer = HASeparatorRegressor(feat_dim=64)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        latent = self.embeddings(x)
        score, penalty = self.has_layer(latent, labels)
        return score, penalty, latent

# --- 3. DATA & TRAINING ---
class LapisGiaaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_filename'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['mean_response'] / 100.0, dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, label


ALPHA = 1.0  # Weight for MSE (Concept)
BETA = 5.0   # Weight for HAS Penalty (Pixel-level Structure - WE INCREASE THIS)

def train_refined_has():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ESSENTIAL: Define the Transforms (ResNet50 requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Load Data
    train_df = pd.read_csv(os.path.join(ANNOTATION_DIR, "LAPIS_GIAA_Trainsplit.csv"))
    train_df = train_df[train_df['style'].isin(RENAISSANCE_STYLES)]
    
    # FIX: Pass the transform to the dataset here
    train_dataset = LapisGiaaDataset(train_df, IMAGE_DIR, transform=transform)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = ArtAestheticHASModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse_criterion = nn.MSELoss()

    print(f"Refining model with high penalty weight (BETA={BETA}) for structural detection...")
    
    for epoch in range(10): 
        model.train()
        epoch_mse = 0
        epoch_penalty = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass: model returns score, penalty, and latent vector
            preds, penalty, _ = model(imgs, labels)
            
            # Combined Loss: High BETA forces the model to prioritize geometric fit
            loss_mse = mse_criterion(preds, labels)
            loss = (ALPHA * loss_mse) + (BETA * penalty)
            
            loss.backward()
            optimizer.step()
            
            epoch_mse += loss_mse.item()
            epoch_penalty += penalty.item()
            
        avg_mse = epoch_mse / len(loader)
        avg_pen = epoch_penalty / len(loader)
        print(f"Epoch {epoch+1} | MSE: {avg_mse:.4f} | Structural Penalty: {avg_pen:.4f}")

    # Save the refined weights
    torch.save(model.state_dict(), "baseline_has_refined.pth")
    print("\nRefined Model Saved! It is now highly sensitive to pixel-level drift.")

if __name__ == "__main__":
    train_refined_has()