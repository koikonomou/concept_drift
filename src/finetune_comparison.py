"""
finetune_comparison.py — Pro Developer Version
Executes a controlled experiment comparing Global vs Targeted fine-tuning.
Saves training history for plotting and comparative performance deltas.
"""

import argparse, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import configurations and models
from config import (
    TEST_ROOT, WEIGHT_DIR, RESULT_DIR, DEVICE,
    HAS_MARGIN, HAS_SCALE, BATCH_SIZE, ALPHA,
    LANDSCAPE_CLASSES, ensure_dirs
)
from models import HASModel, STANDARD_TRANSFORM, TRAIN_AUGMENT

# Import the core utility functions from the original finetune.py
from finetune import (
    load_taxonomy, filter_records, sort_pool_a, sort_pool_b,
    PathLabelDataset, set_trainable, evaluate_records, evaluate_source,
    add_acc
)

# Visual styling
plt.rcParams.update({"font.family": "sans-serif", "axes.grid": True, "grid.alpha": 0.3})
POOL_COLORS = {
    "Global_All": "#7f8c8d",       # Neutral Gray
    "Targeted_Concept": "#e74c3c", # Alert Red
    "Targeted_Data": "#3498db"     # Action Blue
}

def finetune_with_logging(model, train_records, args, pool_name):
    """Fine-tuning loop that records loss and accuracy history for plotting."""
    if not train_records: return None
    
    transform = TRAIN_AUGMENT if args.augment else STANDARD_TRANSFORM
    ds = PathLabelDataset(train_records, transform)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True)

    modules, params = set_trainable(model, args.ft_scope)
    opt = torch.optim.AdamW(params, lr=args.ft_lr, weight_decay=args.weight_decay)
    nll = nn.NLLLoss()
    
    history = []
    model.eval()
    for _, m in modules: m.train()

    print(f"  Fine-tuning {pool_name}...")
    for ep in range(1, args.ft_epochs + 1):
        loss_sum = acc_sum = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.long().to(DEVICE)
            opt.zero_grad()
            logits, penalty, _ = model(imgs, labels=labels)
            loss = ALPHA * nll(logits, labels) + args.ft_beta * penalty
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
            acc_sum += logits.argmax(1).eq(labels).sum().item()
            total += len(labels)
            
        history.append({
            "epoch": ep, 
            "loss": loss_sum/len(loader), 
            "train_acc": 100.*acc_sum/total
        })
    
    return pd.DataFrame(history)

def plot_results(summary_df, history_dict, output_dir):
    """Generates Training Curves and Performance Delta charts."""
    # Plot 1: Training Convergence
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name, df in history_dict.items():
        axes[0].plot(df["epoch"], df["loss"], label=name, color=POOL_COLORS[name], lw=2)
        axes[1].plot(df["epoch"], df["train_acc"], label=name, color=POOL_COLORS[name], lw=2)
    
    axes[0].set_title("Training Loss"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Training Accuracy"); axes[1].set_ylabel("Accuracy (%)")
    for ax in axes: ax.set_xlabel("Epoch"); ax.legend()
    plt.savefig(output_dir / "ft_training_curves.png", dpi=200)

    # Plot 2: Final Improvement (Delta) across populations
    # Plot 2: Final Improvement (Delta) across populations
    metrics = {
        "concept_drift_acc_delta": "Concept Set", 
        "stable_acc_delta": "Stable Set", 
        "source_acc_delta": "Source Test"
    }
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (name, color) in enumerate(POOL_COLORS.items()):
        # Ensure the condition exists in the dataframe
        if name in summary_df["condition"].values:
            row = summary_df[summary_df["condition"] == name].iloc[0]
            # Use .get() or fillna to ensure we don't pass None to ax.bar
            deltas = [row.get(m, 0) if pd.notnull(row.get(m, 0)) else 0 for m in metrics.keys()]
            ax.bar(x + (i - 1) * width, deltas, width, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(metrics.values())
    ax.set_ylabel("Accuracy Improvement (%)")
    ax.set_title("Pool Strategy Comparison (Fixed Budget)")
    ax.axhline(0, color='black', lw=1); ax.legend()
    plt.savefig(output_dir / "ft_test_comparison.png", dpi=200)

def run_comparative_study(args):
    """The core experimental logic."""
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_recs = load_taxonomy(args.input_csv)
    
    # Define the 3 experimental pools
    concept_pool = sort_pool_b(filter_records(all_recs, ["Pure Concept Drift", "Full Drift (both)"]))
    data_pool = sort_pool_a(filter_records(all_recs, ["Pure Data Drift"]))
    global_pool = list(all_recs); random.shuffle(global_pool)

    # Ensure a Fair Comparison: Fix the budget N
    N = min(len(concept_pool), len(data_pool), args.max_samples if args.max_samples > 0 else 9999)
    pools = {
        "Global_All": global_pool[:N], 
        "Targeted_Concept": concept_pool[:N], 
        "Targeted_Data": data_pool[:N]
    }
    
    print(f"\n--- Starting Comparative Study (N={N}) ---")
    
    # Evaluation sets to measure impact
    eval_concept = concept_pool
    eval_stable = filter_records(all_recs, ["In-Distribution"])
    
    results, histories = [], {}
    for name, train_subset in pools.items():
        # Load fresh weights for every experiment run
        model = HASModel(n_classes=len(LANDSCAPE_CLASSES), margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
        model.load_state_dict(torch.load(args.has_weights, map_location=DEVICE))
        
        # Pre-evaluation
        pre_c = evaluate_records(model, eval_concept, args.batch_size)
        pre_s = evaluate_records(model, eval_stable, args.batch_size)
        pre_t = evaluate_source(model, TEST_ROOT, args.batch_size)
        
        # Fine-tune with logging
        histories[name] = finetune_with_logging(model, train_subset, args, name)
        
        # Post-evaluation
        post_c = evaluate_records(model, eval_concept, args.batch_size)
        post_s = evaluate_records(model, eval_stable, args.batch_size)
        post_t = evaluate_source(model, TEST_ROOT, args.batch_size)
        
        res = {"condition": name, "n": N}
        add_acc(res, "concept_drift", pre_c, post_c)
        add_acc(res, "stable", pre_s, post_s)
        add_acc(res, "source", pre_t, post_t)
        results.append(res)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "comparison_results.csv", index=False)
    plot_results(summary_df, histories, out_dir)
    print(f"\nResults and plots saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--input-csv", default="results/drift_has.csv")
    parser.add_argument("--has-weights", default="weights/has_model.pth")
    parser.add_argument("--output-dir", default="results/finetune_comparison")
    
    # Study Hyperparams
    parser.add_argument("--max-samples", type=int, default=150, help="Fixed budget N")
    parser.add_argument("--ft-epochs", type=int, default=50)
    parser.add_argument("--ft-lr", type=float, default=5e-5)
    parser.add_argument("--ft-beta", type=float, default=2.0)
    parser.add_argument("--ft-scope", default="embedder-fc-head")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    args = parser.parse_args()
    run_comparative_study(args)
