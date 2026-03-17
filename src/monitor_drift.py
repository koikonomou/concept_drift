import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from hasseparator import ArtAestheticHASModel, LapisGiaaDataset
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_has_renaissance.pth"
ANNOTATION_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/annotation"
IMAGE_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/images"

def run_monitor():
    # 1. Load Model
    model = ArtAestheticHASModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. Load the official Test set
    test_df = pd.read_csv(os.path.join(ANNOTATION_DIR, "LAPIS_GIAA_Testsplit.csv"))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # We use a batch size of 1 to get per-image penalties
    dataset = LapisGiaaDataset(test_df, IMAGE_DIR, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []

    print("Analyzing pixel-level drift across the test set...")
    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            # The model returns: score, penalty, latent
            _, penalty, _ = model(img, label)
            
            # Record the details
            results.append({
                'image_filename': test_df.iloc[i]['image_filename'],
                'style': test_df.iloc[i]['style'],
                'actual_score': label.item() * 100,
                'has_penalty': penalty.item()
            })

    # 3. Create Monitoring Report
    report_df = pd.DataFrame(results)
    
    # Sort by penalty to find the most "Alien" images
    alien_images = report_df.sort_values(by='has_penalty', ascending=False)

    print("\n--- TOP 5 MOST 'ALIEN' IMAGES (Highest Data Drift) ---")
    print(alien_images[['image_filename', 'style', 'has_penalty']].head(5))

    print("\n--- TOP 5 MOST 'NATURAL' IMAGES (Lowest Data Drift) ---")
    print(alien_images[['image_filename', 'style', 'has_penalty']].tail(5))

    # Save to CSV for your own review
    report_df.to_csv("drift_monitoring_report.csv", index=False)
    print("\nFull report saved to drift_monitoring_report.csv")
def visualize_results(csv_path="drift_monitoring_report.csv"):
    df = pd.read_csv(csv_path)
    
    # Check if we have the right column names
    penalty_col = 'has_penalty' if 'has_penalty' in df.columns else 'penalty'
    
    plt.figure(figsize=(10, 6))
    
    # Styles to compare
    styles = ['High_Renaissance', 'Pop_Art', 'Abstract_Expressionism']
    colors = ['#2ecc71', '#e74c3c', '#f1c40f']
    
    for style, color in zip(styles, colors):
        subset = df[df['style'] == style]
        if not subset.empty:
            # Use the corrected column name here
            sns.kdeplot(subset[penalty_col], fill=True, label=style, color=color, bw_adjust=0.5)

    # Calculate Drift Threshold (Mean + 2*Std of Renaissance)
    ren_data = df[df['style'].str.contains('Renaissance', na=False)][penalty_col]
    if not ren_data.empty:
        threshold = ren_data.mean() + (2 * ren_data.std())
        plt.axvline(threshold, color='black', linestyle='--', label=f'Drift Threshold ({threshold:.2f})')
    
    plt.title("Pixel-Level Data Drift: Structural Penalty Distribution", fontsize=14)
    plt.xlabel("HAS Penalty (Higher = More Alien)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("refined_penalty_distribution.png")
    print(f"Distribution plot saved as refined_penalty_distribution.png")
if __name__ == "__main__":
    run_monitor()
    visualize_results(csv_path="drift_monitoring_report.csv")