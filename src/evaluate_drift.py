import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from baseline import LandscapeBaseline 
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances

CUSTOM_DATA_ROOT = "/home/kate/datasets/ARXPHOTOS314/images"
MODEL_PATH = "baseline_landscape.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAP = {
    'Beach Landscape': 0,      # coast
    'Desert Landscape': 1,     # desert
    'Forest Landscape': 2,     # forest
    'Ice & Snow Landscape': 3, # glacier
    'Mountain Landscape': 4    # mountain
}

class CustomLandscapeEvalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        for folder_name, label_idx in CLASS_MAP.items():
            main_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(main_path): continue
            
            for sub_folder in os.listdir(main_path):
                sub_path = os.path.join(main_path, sub_folder)
                if os.path.isdir(sub_path):
                    for img_name in os.listdir(sub_path):
                        if img_name.lower().endswith(('png')):
                            self.samples.append({
                                'path': os.path.join(sub_path, img_name),
                                'label': label_idx,
                                'sub_category': sub_folder
                            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, s['label'], s['sub_category']

def evaluate_landscape_drift():
    model = LandscapeBaseline().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    custom_ds = CustomLandscapeEvalDataset(CUSTOM_DATA_ROOT, transform)
    loader = DataLoader(custom_ds, batch_size=32, shuffle=False)

    all_features, all_preds, all_labels, all_subs = [], [], [], []

    print(f"Evaluating drift on {len(custom_ds)} custom samples...")
    with torch.no_grad():
        for imgs, labels, subs in loader:
            imgs = imgs.to(DEVICE)

            feat = model.backbone(imgs) 
            feat = torch.flatten(feat, 1)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            all_features.append(feat.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_subs.extend(subs)

    features = np.concatenate(all_features)
    
    # --- DATA DRIFT CALCULATION ---
    baseline_centroid = np.mean(features, axis=0)
    
    results_list = []
    for i in range(len(features)):
        dist = np.linalg.norm(features[i] - baseline_centroid)
        
        results_list.append({
            'label': all_labels[i],
            'pred': all_preds[i],
            'sub_category': all_subs[i],
            'pixel_drift_score': dist # Higher = More Data Drift
        })

    results_df = pd.DataFrame(results_list)
    results_df['correct'] = results_df['label'] == results_df['pred']

    # Grouping to see the relationship between Data Drift and Concept Drift
    drift_report = results_df.groupby('sub_category').agg({
        'pixel_drift_score': 'mean', # DATA DRIFT
        'correct': 'mean'            # CONCEPT DRIFT (Accuracy)
    }).sort_values(by='pixel_drift_score', ascending=False)

    print("\n--- DRIFT ANALYSIS BY SUB-CATEGORY ---")
    print(drift_report)
    # Identify Pure Concept Drift (Low Visual Change, High Error)
    # We look for categories with Drift below the median but Accuracy below a threshold
    median_drift = drift_report['pixel_drift_score'].median()

    pure_concept_drift = drift_report[
        (drift_report['pixel_drift_score'] <= median_drift) & 
        (drift_report['correct'] < 0.70)
    ]

    print("\n--- PURE CONCEPT DRIFT DETECTED ---")
    if not pure_concept_drift.empty:
        print(pure_concept_drift)
    else:
        print("No pure concept drift found. All errors are linked to visual changes.")
    #Statistical Data Drift (KS-Test)
    # Comparing two sub-categories directly (e.g., Jungle vs. Olive Tree)
    # A low p-value confirms the two sets of pixels are from different distributions
    sample_a = results_df[results_df['sub_category'] == results_df['sub_category'].iloc[0]]['pixel_drift_score']
    sample_b = results_df[results_df['sub_category'] == results_df['sub_category'].iloc[-1]]['pixel_drift_score']
    stat, p_val = ks_2samp(sample_a, sample_b)
    print(f"\nStatistical Data Drift (KS-Test) P-Value: {p_val:.4e}")

    plot_landscape_tsne(features, all_labels, all_subs)

def plot_landscape_tsne(features, labels, subs):
    print("Computing t-SNE... this may take a minute.")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(14, 10))
    # Using a professional palette for the 5 main categories
    scatter = sns.scatterplot(
        x=reduced[:,0], 
        y=reduced[:,1], 
        hue=[list(CLASS_MAP.keys())[l] for l in labels], 
        style=[list(CLASS_MAP.keys())[l] for l in labels],
        palette='deep', 
        alpha=0.7,
        s=60
    )
    
    plt.title("Latent Space Data Drift: Custom Landscapes vs. Baseline Concepts", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Main Categories")
    plt.tight_layout()
    plt.savefig("landscape_drift_tsne.png")
    print("TSNE plot saved as landscape_drift_tsne.png")

if __name__ == "__main__":
    evaluate_landscape_drift()