# Drift Detection: Baseline vs HAS

## Project Structure

```
config.py              ← paths, hyperparams (edit this once)
models.py              ← BaselineModel, HASModel, FolderDataset
drift_stats.py         ← statistical tests (KS, MMD, chi², etc.)

step1_train.py         ← train both models        → weights/
step2_extract.py       ← extract 64-D features     → features/
step3_detect.py        ← run drift tests           → results/*.csv + .txt
step4_visualize.py     ← UMAP, dashboards, plots   → results/*.png

run_all.py             ← orchestrate all steps
```

## Quick Start

```bash
# 1. Edit config.py with your dataset paths

# 2. Full pipeline
python run_all.py --epochs 10

# 3. Or run steps individually
python step1_train.py --epochs 10
python step2_extract.py
python step3_detect.py
python step4_visualize.py
```

## Re-entering After a Failure

Each step saves its outputs to disk. If step 3 crashes you don't
need to retrain or re-extract:

```bash
python run_all.py --from-step 3
# or directly:
python step3_detect.py
```

## Step Outputs

| Step | Writes | Reads |
|------|--------|-------|
| 1. Train | `weights/*.pth` | training images |
| 2. Extract | `features/*.npz`, `features/meta.json` | weights + images |
| 3. Detect | `results/*.csv`, `results/drift_report.txt` | features |
| 4. Visualize | `results/*.png` | features + CSVs |

## CLI Options

```bash
# Step 1
python step1_train.py --epochs 15 --lr 5e-5 --only has

# Step 3
python step3_detect.py --concept-thresh 0.60 --drift-sigma 3.0

# Step 4
python step4_visualize.py --skip-umap   # skip slow UMAP computation

# Orchestrator
python run_all.py --skip-train          # reuse existing weights
python run_all.py --from-step 2         # start from extraction
python run_all.py --skip-umap           # pass through to step4
```

## Dependencies

```
torch torchvision numpy pandas scipy scikit-learn
matplotlib seaborn umap-learn pillow
```
