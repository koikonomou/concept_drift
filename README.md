# Landscape Monitoring: Data & Concept Drift Analysis

This project implements a pipeline to detect and distinguish between **Data Drift** (visual/pixel-level changes) and **Concept Drift** (semantic/label-level changes) in landscape classification models.

## 📌 Research Objective
The goal is to monitor how a model trained on a "Generic" landscape dataset performs when deployed on a "Custom" dataset containing specific sub-categories (e.g., "Olive Tree Forests" or "Ice Skating Rinks").


## 📊 Datasets

### 1. Baseline Dataset (Source Domain)
We use the [Landscape Recognition Image Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images) from Kaggle.
- **Classes:** `Coast`, `Desert`, `Forest`, `Glacier`, `Mountain`.
- **Purpose:** Used to train the ResNet50 baseline model and establish the "Standard" feature distribution for natural landscapes.

### 2. Custom Dataset (Target Domain)
A hierarchical dataset containing specific sub-categories:
- **Forest Landscape:** Jungle, Olive Tree Cluster, Thick Treeline.
- **Ice & Snow:** Icecap, Snowy Villages, **Ice Skating Rink** (Concept Drift trigger).
- **Desert:** Oasis, Sand Dune, Rocky Hills.

---

## 🚀 Methodology

### 1. Baseline Training (`src/baseline.py`)
Trains a ResNet50 classifier on the 5 baseline classes using Cross-Entropy Loss.
- **Input:** 224x224 RGB images.
- **Output:** 5-class probability vector.

### 2. Drift Evaluation (`src/evaluate_drift.py`)
Calculates dual-drift metrics:
- **Pixel Drift Score:** Measures the Euclidean distance in the latent space (ResNet feature layer) between custom images and the baseline average.
- **Accuracy:** Measures the stability of the "Concept"


## 🛠 Installation

### Requirements
- Python 3.10+
- PyTorch / Torchvision
- Scikit-Learn (for t-SNE)
- Seaborn / Matplotlib
- Poetry (recommended)

### Running the pipeline
1. **Train Baseline:**
   ```bash
   poetry run python src/baseline.py