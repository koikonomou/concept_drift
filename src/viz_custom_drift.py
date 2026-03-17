import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
from hasseparator import ArtAestheticHASModel # Your PyTorch HAS model

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_has_renaissance.pth"
CUSTOM_IMG_DIR = "/home/katerina/codes/concept_drift/custom_images/"
LAPIS_IMG_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/images"

def extract_features(model, img_paths, img_dir):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features = []
    with torch.no_grad():
        for path in img_paths:
            full_path = os.path.join(img_dir, path)
            img = Image.open(full_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(DEVICE)
            
            # The HAS model returns: score, penalty, latent_embeddings
            _, _, latent = model(img) 
            features.append(latent.cpu().numpy())
            
    return np.concatenate(features)

def run_custom_overlay():
    # 1. Load Model
    model = ArtAestheticHASModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # 2. Get LAPIS Reference (Renaissance)
    lapis_df = pd.read_csv("/home/katerina/codes/datasets/LAPIS/LAPIS github/annotation/LAPIS_GIAA_Testsplit.csv")
    lapis_ren = lapis_df[lapis_df['style'].str.contains('Renaissance', na=False)].sample(100)['image_filename']
    f_lapis = extract_features(model, lapis_ren, LAPIS_IMG_DIR)

    # 3. Get Custom Data
    # Assuming you have a folder of custom images
    custom_files = os.listdir(CUSTOM_IMG_DIR)
    f_custom = extract_features(model, custom_files, CUSTOM_IMG_DIR)

    # 4. Combine and Project
    all_features = np.vstack([f_lapis, f_custom])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(all_features)

    # 5. Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:100, 0], reduced[:100, 1], c='blue', label='LAPIS Renaissance (Reference)', alpha=0.5)
    plt.scatter(reduced[100:, 0], reduced[100:, 1], c='red', label='Custom Dataset (Target)', marker='x')
    
    plt.title("Latent Space Overlay: Where does Custom Art land?")
    plt.legend()
    plt.savefig("custom_drift_overlay.png")
    print("Overlay plot saved as custom_drift_overlay.png")

if __name__ == "__main__":
    run_custom_overlay()