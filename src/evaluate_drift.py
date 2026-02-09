import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader
from torchvision import transforms
from baseline import ArtAestheticBaseline, LapisArtDataset

ANNOTATION_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/annotation"
IMAGE_DIR = "/home/katerina/codes/datasets/LAPIS/LAPIS github/images"
MODEL_PATH = "baseline_renaissance.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RENAISSANCE_STYLES = [ 'Early_Renaissance', 'High_Renaissance', 'Northern_Renaissance', 'Mannerism_Late_Renaissance']

DRIFT_STYLES = ['Pop_Art', 'Abstract_Expressionism']

def evaluate_official_drift():
    model = ArtAestheticBaseline().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_csv_path = os.path.join(ANNOTATION_DIR, "LAPIS_GIAA_Testsplit.csv")
    test_df = pd.read_csv(test_csv_path)

    df_id = test_df[test_df['style'].isin(RENAISSANCE_STYLES)]
    df_ood = test_df[test_df['style'].isin(DRIFT_STYLES)]

    print(f"ID Samples (Renaissance): {len(df_id)}")
    print(f"OOD Samples (Drift): {len(df_ood)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class GIAADataset(LapisArtDataset):
        def __getitem__(self, idx):
            img_filename = self.data.iloc[idx]['image_filename']
            img_path = os.path.join(self.img_dir, img_filename)
            image = Image.open(img_path).convert('RGB')
            label = torch.tensor(self.data.iloc[idx]['mean_response'] / 100.0, dtype=torch.float32)
            if self.transform: image = self.transform(image)
            return image, label

    loader_id = DataLoader(GIAADataset(df_id, IMAGE_DIR, transform), batch_size=20)
    loader_ood = DataLoader(GIAADataset(df_ood, IMAGE_DIR, transform), batch_size=20)

    def get_results(loader):
        preds, labels, features = [], [], []
        with torch.no_grad():
            for imgs, target in loader:
                imgs = imgs.to(DEVICE)
                # Feature extraction
                feat = model.backbone.avgpool(model.backbone.layer4(model.backbone.layer3(
                    model.backbone.layer2(model.backbone.layer1(model.backbone.maxpool(
                    model.backbone.relu(model.backbone.bn1(model.backbone.conv1(imgs))))))))).flatten(1)
                
                output = model(imgs).squeeze()
                preds.extend(output.cpu().numpy())
                labels.extend(target.numpy())
                features.append(feat.cpu().numpy())
        return np.array(preds), np.array(labels), np.concatenate(features)

    p_id, l_id, f_id = get_results(loader_id)
    p_ood, l_ood, f_ood = get_results(loader_ood)

    # 4. Analysis
    mse_id = np.mean((p_id - l_id)**2)
    mse_ood = np.mean((p_ood - l_ood)**2)

    print(f"\n--- Results ---")
    print(f"Renaissance (ID) MSE: {mse_id:.4f}")
    print(f"Drift (OOD) MSE: {mse_ood:.4f}")
    
    # Statistical Drift Test (Data Drift)
    stat, p_val = ks_2samp(f_id.mean(axis=1), f_ood.mean(axis=1))
    print(f"Data Drift P-value: {p_val:.4e}")

    # Plotting
    plot_tsne(f_id, f_ood)
    
    plot_error_distribution(l_id,p_id,l_ood, p_ood)
def plot_tsne(f_id, f_ood):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(np.vstack([f_id, f_ood]))
    plt.figure(figsize=(10,6))
    plt.scatter(reduced[:len(f_id),0], reduced[:len(f_id),1], label='Renaissance (ID)', alpha=0.5)
    plt.scatter(reduced[len(f_id):,0], reduced[len(f_id):,1], label='Pop/Abstract (Drift)', alpha=0.5)
    plt.legend()
    plt.title("Drift Visualization: Renaissance vs. Modern Styles")
    plt.savefig("drift_visualization.png")
    print("Plot saved as drift_visualization.png")
import seaborn as sns
import matplotlib.pyplot as plt

def plot_error_distribution(l_id, p_id, l_ood, p_ood):
    error_id = np.abs(l_id - p_id)
    error_ood = np.abs(l_ood - p_ood)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(error_id, fill=True, label="Renaissance (ID) Error", color="blue")
    sns.kdeplot(error_ood, fill=True, label="Pop Art (OOD) Error", color="red")
    plt.title("Concept Drift: Distribution of Absolute Prediction Errors")
    plt.xlabel("Absolute Error (Normalized)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("error_distribution.png")
    print("Error distribution plot saved!")


if __name__ == "__main__":
    evaluate_official_drift()
