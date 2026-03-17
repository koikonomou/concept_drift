# Drift Detection: Baseline vs HAS

Compares a standard ResNet50 classifier against a
[HAS-augmented](https://doi.org/10.1109/ICMLA51294.2020.00087) ResNet50
for detecting **data drift** and **concept drift** on a custom evaluation dataset.

Both models share an identical architecture up to a 64-D latent bottleneck.
The only difference is the final classification layer — plain softmax vs
HAS angular margin softmax — so drift scores are directly comparable.

## Project Structure

```
├── pyproject.toml
├── .gitignore
├── README.md
└── src/
    ├── config.py          ← paths, hyperparams (edit this once)
    ├── models.py          ← BaselineModel, HASModel, FolderDataset
    ├── drift_stats.py     ← statistical tests (KS, MMD, chi², etc.)
    ├── train.py           ← train both models         → weights/
    ├── extract.py         ← extract 64-D features      → features/
    ├── detect.py          ← run drift tests            → results/*.csv, .txt
    ├── visualize.py       ← UMAP, dashboards, plots   → results/*.png
    ├── analysis.py        ← subcategory drift ranking  → results/*.png, .csv
    └── run_all.py         ← orchestrate all steps
```

## Setup

```bash
# Install with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Quick Start

```bash
# 1. Edit src/config.py with your dataset paths

# 2. Full pipeline
python src/run_all.py --epochs 10

# 3. Or run steps individually
python src/train.py --epochs 10
python src/extract.py
python src/detect.py
python src/visualize.py
python src/analysis.py
```

## Re-entering After a Failure

Each step saves its outputs to disk. If step 3 crashes, you don't
need to retrain or re-extract:

```bash
python src/run_all.py --from-step 3

# or directly:
python src/detect.py
```

## Step Outputs

| Step | Script | Writes | Reads |
|------|--------|--------|-------|
| 1. Train | `train.py` | `weights/*.pth` | training images |
| 2. Extract | `extract.py` | `features/*.npz`, `features/meta.json` | weights + images |
| 3. Detect | `detect.py` | `results/*.csv`, `results/drift_report.txt` | features |
| 4. Visualize | `visualize.py` | `results/*.png` | features + CSVs |
| 5. Analysis | `analysis.py` | `results/*.png`, `results/*.csv` | features + CSVs |

## CLI Options

```bash
# Train
python src/train.py --epochs 15 --only has --has-lr 0.01

# Detect
python src/detect.py --drift-sigma 3.0

# Visualize
python src/visualize.py --skip-umap

# Orchestrator
python src/run_all.py --skip-train
python src/run_all.py --from-step 2
python src/run_all.py --skip-umap
```

## Architecture

Both models share the same path to the 64-D latent space:

```
ResNet50 → GAP → BN → Dropout(0.3) → Linear(2048, 64) → BN → 64-D latent
```

They differ only in the classifier head:

| | Baseline | HAS |
|---|---|---|
| **Classifier** | `Linear(64, 5)` | `HASeparatorMultiClass(64, 5)` |
| **Loss** | CrossEntropy | NLL + angular margin penalty |
| **Optimizer** | Adam (lr=1e-4) | SGD (lr=1e-2, momentum=0.9, nesterov) |
| **Latent geometry** | Unconstrained | L2-normalized, angular margins |

## Drift Detection

**Data drift** — measured in the 64-D latent space:
- KS test on latent norms
- Centroid shift (Euclidean + z-score)
- MMD with RBF kernel (permutation p-value)
- Per-feature KS across all 64 dimensions

**Concept drift** — measured from classifier outputs:
- Chi² test on prediction distributions
- KS test on confidence distributions

Per-image thresholds are computed per-model from training statistics
(`mean ± 2σ`), not a fixed cutoff.


