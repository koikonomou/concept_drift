"""
Dual Drift Evaluation: Data Drift vs Concept Drift
====================================================
Per-image drift is measured along two independent axes:

  DATA DRIFT    — how visually/structurally different is the image from the
                  training distribution?
                  Signal: Euclidean distance in 64D latent space from training
                  centroid (saved by landscape_has.py during training).
                  High distance → the pixel/feature distribution has shifted.

  CONCEPT DRIFT — has the model's understanding of the semantic concept broken?
                  Signal: max softmax confidence of the classifier head.
                  Low confidence → the model cannot map the image to any known
                  concept, even if it looks similar to training data.

Quadrant classification (after thresholding):
  ┌─────────────────────┬──────────────────────────┐
  │ LOW data drift       │ HIGH data drift           │
  │ HIGH concept drift   │ HIGH concept drift        │
  │ → PURE CONCEPT DRIFT │ → FULL DRIFT (both)       │
  ├─────────────────────┼──────────────────────────┤
  │ LOW data drift       │ HIGH data drift           │
  │ LOW concept drift    │ LOW concept drift         │
  │ → IN DISTRIBUTION    │ → PURE DATA DRIFT         │
  └─────────────────────┴──────────────────────────┘

The "Ice Skating Rink" example from the README sits in PURE CONCEPT DRIFT:
visually similar to Glacier (low data drift) but semantically wrong (high
concept drift) — the classic hard failure mode for deployed classifiers.
"""
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from landscape_has import LandscapeHASModel

# ---------------------------------------------------------------------------
# CONFIG — CUSTOM_DATA_ROOT is your target/deployment dataset.
#          Falls back to the Landscape test split if the custom path is missing.
# ---------------------------------------------------------------------------
CUSTOM_DATA_ROOT  = "/home/kate/datasets/ARXPHOTOS314/images"
TEST_DATA_ROOT    = "/home/katerina/codes/datasets/landscapes/Landscape Classification/Landscape Classification/Testing Data"
MODEL_PATH        = "baseline_landscape_has.pth"
DIST_STATS_PATH   = "landscape_has_train_stats.npz"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANDSCAPE_CLASSES = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']


# ---------------------------------------------------------------------------
# DATASET
# Handles two layouts:
#   hierarchical  root/ClassName/SubFolder/img.jpg  (custom deployment data)
#   flat          root/ClassName/img.jpg            (standard ImageFolder layout)
# ---------------------------------------------------------------------------
class CustomLandscapeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples   = []
        self.transform = transform

        for label_idx, class_name in enumerate(LANDSCAPE_CLASSES):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue
            for entry in os.listdir(class_path):
                entry_path = os.path.join(class_path, entry)
                if os.path.isdir(entry_path):
                    # hierarchical layout — entry is a sub-category folder
                    for img_name in os.listdir(entry_path):
                        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                            self.samples.append({
                                'path':         os.path.join(entry_path, img_name),
                                'label':        label_idx,
                                'sub_category': entry
                            })
                elif entry.lower().endswith(('png', 'jpg', 'jpeg')):
                    # flat layout — entry is an image file directly in the class folder
                    self.samples.append({
                        'path':         entry_path,
                        'label':        label_idx,
                        'sub_category': class_name
                    })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, s['label'], s['sub_category']


# ---------------------------------------------------------------------------
# CORE EVALUATION
# ---------------------------------------------------------------------------
def evaluate():
    # --- Load model ---
    model = LandscapeHASModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Load training distribution statistics ---
    stats    = np.load(DIST_STATS_PATH)
    centroid = stats['centroid']        # (64,)
    dist_mean = float(stats['dist_mean'])
    dist_std  = float(stats['dist_std'])

    # Thresholds (μ + 2σ of training distances)
    DATA_DRIFT_THRESH    = dist_mean + 2 * dist_std
    # Concept drift: confidence below this signals concept failure
    CONCEPT_DRIFT_THRESH = 0.70

    print(f"Data drift threshold    : {DATA_DRIFT_THRESH:.4f}  (train μ={dist_mean:.4f}, σ={dist_std:.4f})")
    print(f"Concept drift threshold : max_softmax < {CONCEPT_DRIFT_THRESH}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Fall back to local test split if the custom deployment path is missing
    data_root = CUSTOM_DATA_ROOT if os.path.exists(CUSTOM_DATA_ROOT) else TEST_DATA_ROOT
    print(f"\nUsing dataset: {data_root}")
    dataset = CustomLandscapeDataset(data_root, transform)
    if len(dataset) == 0:
        print("ERROR: No images found. Check CUSTOM_DATA_ROOT / TEST_DATA_ROOT paths.")
        return
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Evaluating {len(dataset)} images...")

    records = []
    all_latents, all_labels_list = [], []

    with torch.no_grad():
        for imgs, labels, sub_cats in loader:
            imgs = imgs.to(DEVICE)

            logits, in_dist_score, penalty, latent = model(imgs)
            probs      = torch.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values   # max softmax — concept drift signal
            preds      = probs.argmax(dim=1)

            for i in range(len(imgs)):
                lat   = latent[i].cpu().numpy()
                # DATA DRIFT: distance from training centroid in 64D space
                data_drift_score = float(np.linalg.norm(lat - centroid))

                # CONCEPT DRIFT: inverse of classifier confidence
                concept_drift_score = float(1.0 - confidence[i].cpu())

                records.append({
                    'sub_category':       sub_cats[i],
                    'true_label':         int(labels[i]),
                    'pred_label':         int(preds[i].cpu()),
                    'correct':            int(labels[i]) == int(preds[i].cpu()),
                    'max_confidence':     float(confidence[i].cpu()),
                    'has_in_dist_score':  float(in_dist_score[i].cpu()),
                    'data_drift_score':   data_drift_score,
                    'concept_drift_score': concept_drift_score,
                })
                all_latents.append(lat)
                all_labels_list.append(int(labels[i]))

    df = pd.DataFrame(records)

    # --- Assign drift quadrant ---
    df['data_drifted']    = df['data_drift_score']    > DATA_DRIFT_THRESH
    df['concept_drifted'] = df['max_confidence']      < CONCEPT_DRIFT_THRESH

    def quadrant(row):
        if not row['data_drifted'] and not row['concept_drifted']:
            return 'In-Distribution'
        elif row['data_drifted'] and not row['concept_drifted']:
            return 'Pure Data Drift'
        elif not row['data_drifted'] and row['concept_drifted']:
            return 'Pure Concept Drift'
        else:
            return 'Full Drift (both)'

    df['drift_type'] = df.apply(quadrant, axis=1)

    # --- Summary report ---
    print("\n" + "="*60)
    print("DRIFT SUMMARY BY SUB-CATEGORY")
    print("="*60)
    report = df.groupby('sub_category').agg(
        data_drift_mean   =('data_drift_score',    'mean'),
        concept_drift_mean=('concept_drift_score', 'mean'),
        accuracy          =('correct',             'mean'),
        drift_type        =('drift_type', lambda x: x.value_counts().index[0])
    ).sort_values('data_drift_mean', ascending=False)
    print(report.to_string())

    print("\n" + "="*60)
    print("DRIFT TYPE DISTRIBUTION")
    print("="*60)
    print(df['drift_type'].value_counts().to_string())

    # KS-test between in-distribution and drifted latent means
    id_latents  = np.array([all_latents[i] for i in range(len(df)) if not df.iloc[i]['data_drifted']])
    ood_latents = np.array([all_latents[i] for i in range(len(df)) if df.iloc[i]['data_drifted']])
    if len(id_latents) > 0 and len(ood_latents) > 0:
        stat, p_val = ks_2samp(id_latents.mean(axis=1), ood_latents.mean(axis=1))
        print(f"\nKS-Test (ID vs data-drifted latents): stat={stat:.4f}, p={p_val:.4e}")

    df.to_csv("landscape_has_drift_report.csv", index=False)
    print("\nFull report → landscape_has_drift_report.csv")

    # --- Visualisations ---
    latents_np = np.array(all_latents)
    plot_drift_quadrant(df)
    plot_tsne(latents_np, df)
    plot_drift_by_category(df)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def plot_drift_quadrant(df):
    """Scatter: Data Drift score vs Concept Drift score, coloured by drift type."""
    palette = {
        'In-Distribution':   '#2ecc71',
        'Pure Data Drift':   '#3498db',
        'Pure Concept Drift':'#e74c3c',
        'Full Drift (both)': '#8e44ad',
    }
    plt.figure(figsize=(10, 7))
    for drift_type, colour in palette.items():
        subset = df[df['drift_type'] == drift_type]
        plt.scatter(subset['data_drift_score'], subset['concept_drift_score'],
                    c=colour, label=drift_type, alpha=0.6, s=40, edgecolors='none')

    plt.xlabel("Data Drift Score  (latent distance from training centroid)", fontsize=12)
    plt.ylabel("Concept Drift Score  (1 − max softmax confidence)", fontsize=12)
    plt.title("Dual Drift Map: Data Drift vs Concept Drift", fontsize=14)
    plt.legend(title="Drift Type", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("landscape_has_dual_drift.png", dpi=150)
    print("Drift quadrant plot → landscape_has_dual_drift.png")


def plot_tsne(latents, df):
    """t-SNE of 64D latent space, coloured by drift type."""
    print("Computing t-SNE...")
    from sklearn.manifold import TSNE
    reduced = TSNE(n_components=2, random_state=42, init='pca',
                   learning_rate='auto').fit_transform(latents)

    palette = {
        'In-Distribution':   '#2ecc71',
        'Pure Data Drift':   '#3498db',
        'Pure Concept Drift':'#e74c3c',
        'Full Drift (both)': '#8e44ad',
    }
    plt.figure(figsize=(11, 7))
    for drift_type, colour in palette.items():
        mask = df['drift_type'] == drift_type
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    c=colour, label=drift_type, alpha=0.6, s=30, edgecolors='none')

    plt.title("t-SNE of 64D HAS Latent Space (coloured by drift type)", fontsize=13)
    plt.legend(title="Drift Type", fontsize=10)
    plt.tight_layout()
    plt.savefig("landscape_has_tsne.png", dpi=150)
    print("t-SNE plot → landscape_has_tsne.png")


def plot_drift_by_category(df):
    """Bar chart: mean data-drift and concept-drift scores per sub-category."""
    cat_stats = df.groupby('sub_category')[['data_drift_score', 'concept_drift_score']].mean()
    cat_stats = cat_stats.sort_values('data_drift_score', ascending=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(cat_stats.index, cat_stats['data_drift_score'], color='#3498db')
    axes[0].set_ylabel("Data Drift Score")
    axes[0].set_title("Data Drift per Sub-Category (latent distance from centroid)")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(cat_stats.index, cat_stats['concept_drift_score'], color='#e74c3c')
    axes[1].set_ylabel("Concept Drift Score")
    axes[1].set_title("Concept Drift per Sub-Category (1 − classifier confidence)")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("landscape_has_drift_by_category.png", dpi=150)
    print("Category drift bars → landscape_has_drift_by_category.png")


if __name__ == "__main__":
    evaluate()
