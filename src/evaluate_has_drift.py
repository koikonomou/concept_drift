import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader
from torchvision import transforms

# Import the new architecture from your baseline_has file
from hasseparator import ArtAestheticHASModel, LapisGiaaDataset 

ANNOTATION_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/annotation"
IMAGE_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/images"
MODEL_PATH = "baseline_has_renaissance.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RENAISSANCE_STYLES = ['Early_Renaissance', 'High_Renaissance', 'Northern_Renaissance', 'Mannerism_Late_Renaissance']
DRIFT_STYLES = ['Pop_Art', 'Abstract_Expressionism']

def evaluate_has_drift():
    # 1. Load HAS Model
    model = ArtAestheticHASModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. Load Data
    test_csv_path = os.path.join(ANNOTATION_DIR, "LAPIS_GIAA_Testsplit.csv")
    test_df = pd.read_csv(test_csv_path)

    df_id = test_df[test_df['style'].isin(RENAISSANCE_STYLES)]
    df_ood = test_df[test_df['style'].isin(DRIFT_STYLES)]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader_id = DataLoader(LapisGiaaDataset(df_id, IMAGE_DIR, transform), batch_size=20)
    loader_ood = DataLoader(LapisGiaaDataset(df_ood, IMAGE_DIR, transform), batch_size=20)

    def get_results(loader):
        preds, labels, latents, penalties = [], [], [], []
        with torch.no_grad():
            for imgs, target in loader:
                imgs, target = imgs.to(DEVICE), target.to(DEVICE)
                
                # The HAS model returns: score, penalty, latent_embeddings
                output_score, penalty, latent = model(imgs, target)
                
                preds.extend(output_score.cpu().numpy())
                labels.extend(target.cpu().numpy())
                latents.append(latent.cpu().numpy())
                penalties.append(penalty.item())
                
        return np.array(preds), np.array(labels), np.concatenate(latents), np.mean(penalties)

    print("Processing Renaissance (ID)...")
    p_id, l_id, f_id, pen_id = get_results(loader_id)
    
    print("Processing Modern Art (OOD)...")
    p_ood, l_ood, f_ood, pen_ood = get_results(loader_ood)

    # 3. Performance Analysis (Concept Drift)
    mse_id = np.mean((p_id - l_id)**2)
    mse_ood = np.mean((p_ood - l_ood)**2)

    print(f"\n--- Results (HASeparator) ---")
    print(f"Renaissance MSE: {mse_id:.4f} | Avg Penalty: {pen_id:.4f}")
    print(f"Drift MSE:       {mse_ood:.4f} | Avg Penalty: {pen_ood:.4f}")
    
    # 4. Data Drift Analysis
    # We use the 64D features directly for the KS test
    stat, p_val = ks_2samp(f_id.mean(axis=1), f_ood.mean(axis=1))
    print(f"Data Drift P-value: {p_val:.4e}")

    # 5. Visualizations
    plot_tsne(f_id, f_ood)
    plot_error_distribution(l_id, p_id, l_ood, p_ood)

def plot_tsne(f_id, f_ood):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(np.vstack([f_id, f_ood]))
    plt.figure(figsize=(10,6))
    plt.scatter(reduced[:len(f_id), 0], reduced[:len(f_id), 1], label='Renaissance (ID)', color='blue', alpha=0.5)
    plt.scatter(reduced[len(f_id):, 0], reduced[len(f_id):, 1], label='Pop/Abstract (Drift)', color='red', alpha=0.5)
    plt.legend()
    plt.title("HAS Latent Space: Renaissance vs. Drift Styles")
    plt.savefig("has_tsne_drift.png")
    print("Plot saved as has_tsne_drift.png")

def plot_error_distribution(l_id, p_id, l_ood, p_ood):
    error_id = np.abs(l_id - p_id)
    error_ood = np.abs(l_ood - p_ood)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(error_id, fill=True, label="Renaissance Error", color="blue")
    sns.kdeplot(error_ood, fill=True, label="Drift Error", color="red")
    plt.title("Error Density Comparison")
    plt.legend()
    plt.savefig("has_error_density.png")
    print("Error distribution plot saved!")

if __name__ == "__main__":
    evaluate_has_drift()