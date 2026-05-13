"""
baseline_comparison.py
Fine-tunes the Baseline model on random (Global) samples to provide 
a comparison point for HAS.
"""
import argparse, os, random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (TEST_ROOT, DEVICE, LANDSCAPE_CLASSES, BATCH_SIZE)
from models import BaselineModel, STANDARD_TRANSFORM
from finetune_comparison import (
    load_taxonomy, filter_records, PathLabelDataset, set_trainable, 
    evaluate_records, evaluate_source, add_acc
)

@torch.no_grad()
def evaluate_baseline_records(model, records, batch_size):
    """Accuracy evaluation specifically for the Baseline model."""
    if not records:
        return {"n": 0, "acc": None, "err": None}
    model.eval()
    ds = PathLabelDataset(records, STANDARD_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    total = correct = 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.long().to(DEVICE)
        # FIX: Baseline only returns 2 values (logits, latent)
        logits, _ = model(imgs)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += len(labels)
    acc = 100.0 * correct / max(total, 1)
    return {"n": total, "acc": acc, "err": 100.0 - acc}
def finetune_baseline(model, train_records, args):
    """Standard Cross-Entropy fine-tuning for Baseline."""
    ds = PathLabelDataset(train_records, STANDARD_TRANSFORM)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    # USE THE NEW BASELINE-SPECIFIC FUNCTION HERE
    modules, params = set_trainable_baseline(model, args.ft_scope)
    opt = torch.optim.AdamW(params, lr=args.ft_lr)
    
    model.eval()
    for _, m in modules: m.train()

    for ep in range(1, args.ft_epochs + 1):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.long().to(DEVICE)
            opt.zero_grad()
            logits, _ = model(imgs) 
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
    return model

def set_trainable_baseline(model, scope):
    """Freeze weights and unfreeze only the baseline-specific modules."""
    for p in model.parameters():
        p.requires_grad = False

    if scope == "head":
        # Baseline uses 'classifier', HAS uses 'has_layer'
        modules = [("classifier", model.classifier)]
    elif scope == "embedder-fc-head":
        modules = [
            ("embedder.fc", model.embedder.fc),
            ("classifier",  model.classifier),
        ]
    else:
        raise ValueError(f"Unknown ft_scope: {scope}")

    params = []
    for _, m in modules:
        for p in m.parameters():
            p.requires_grad = True
            params.append(p)
    return modules, params

def run_baseline_study(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_recs = load_taxonomy(args.input_csv)
    eval_concept = filter_records(all_recs, ["Pure Concept Drift", "Full Drift (both)"])
    eval_stable  = filter_records(all_recs, ["In-Distribution"])
    
    results = []
    # Test across multiple budgets N
    for N in [50, 150, 300]:
        print(f"--- Fine-tuning Baseline (N={N}) ---")
        model = BaselineModel(n_classes=len(LANDSCAPE_CLASSES)).to(DEVICE)
        model.load_state_dict(torch.load(args.baseline_weights, map_location=DEVICE))
        
        # --- CRITICAL FIX: Use the baseline-specific evaluation ---
        pre_c = evaluate_baseline_records(model, eval_concept, args.batch_size)
        pre_s = evaluate_baseline_records(model, eval_stable, args.batch_size)
        
        # Select N random samples for naive retraining
        train_subset = random.sample(all_recs, min(N, len(all_recs)))
        model = finetune_baseline(model, train_subset, args)
        
        # --- CRITICAL FIX: Use the baseline-specific evaluation here too ---
        post_c = evaluate_baseline_records(model, eval_concept, args.batch_size)
        post_s = evaluate_baseline_records(model, eval_stable, args.batch_size)
        
        res = {"condition": "Baseline_Global", "n": N}
        add_acc(res, "concept_drift", pre_c, post_c)
        add_acc(res, "stable", pre_s, post_s)
        results.append(res)

    pd.DataFrame(results).to_csv(out_dir / "baseline_comparison_results.csv", index=False)
    print(f"Results saved to {out_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--output-dir", default="finetune_results/baseline")
    parser.add_argument("--ft-epochs", type=int, default=50)
    parser.add_argument("--ft-lr", type=float, default=5e-5)
    parser.add_argument("--ft-scope", default="embedder-fc-head")
    parser.add_argument("--batch-size", type=int, default=32)
    run_baseline_study(parser.parse_args())
