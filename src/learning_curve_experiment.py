import argparse, torch, random
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (DEVICE, LANDSCAPE_CLASSES, HAS_MARGIN, HAS_SCALE)
from models import HASModel, BaselineModel
from finetune_comparison import (
    load_taxonomy, filter_records, sort_pool_b, 
    evaluate_records, evaluate_source, finetune_with_logging
)
from baseline_comparison import evaluate_baseline_records, finetune_baseline

def run_learning_curve(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_recs = load_taxonomy(args.input_csv)
    # The "Gold Standard" severe set (Targeted)
    concept_pool = sort_pool_b(filter_records(all_recs, ["Pure Concept Drift", "Full Drift (both)"]))
    
    # Define our budget steps
    budgets = list(range(100, args.max_n + 1, 20))
    results = []

    for N in budgets:
        print(f"\n>>> TESTING BUDGET N={N} <<<")
        
        # --- 1. HAS STRATEGY (Targeted Selection) ---
        model_has = HASModel(n_classes=len(LANDSCAPE_CLASSES), margin=HAS_MARGIN, scale=HAS_SCALE).to(DEVICE)
        model_has.load_state_dict(torch.load(args.has_weights, map_location=DEVICE))
        
        has_train_subset = concept_pool[:N]
        # Evaluate before/after
        pre_h = evaluate_records(model_has, concept_pool, args.batch_size)
        _ = finetune_with_logging(model_has, has_train_subset, args, f"HAS_N{N}")
        post_h = evaluate_records(model_has, concept_pool, args.batch_size)
        
        # --- 2. BASELINE STRATEGY (Random Selection) ---
        model_bl = BaselineModel(n_classes=len(LANDSCAPE_CLASSES)).to(DEVICE)
        model_bl.load_state_dict(torch.load(args.baseline_weights, map_location=DEVICE))
        
        bl_train_subset = random.sample(all_recs, min(N, len(all_recs)))
        # Count how many "Critical" images the baseline accidentally picked
        critical_paths = set(r['file_path'] for r in has_train_subset)
        matching_instances = sum(1 for r in bl_train_subset if r['file_path'] in critical_paths)
        
        pre_b = evaluate_baseline_records(model_bl, concept_pool, args.batch_size)
        _ = finetune_baseline(model_bl, bl_train_subset, args)
        post_b = evaluate_baseline_records(model_bl, concept_pool, args.batch_size)

        results.append({
            "N": N,
            "HAS_Acc": post_h['acc'],
            "Baseline_Acc": post_b['acc'],
            "Critical_Overlap": matching_instances
        })

    # Save and Plot
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "learning_curve_data.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["N"], df["HAS_Acc"], 'r-o', label="HAS (Targeted Severity)", linewidth=2)
    ax1.plot(df["N"], df["Baseline_Acc"], 'b-s', label="Baseline (Random Selection)", linewidth=2)
    ax1.set_ylabel("Accuracy on Concept-Drifted Data (%)")
    ax1.set_xlabel("Labeling Budget (Number of Instances)")
    
    # Secondary axis for the overlap
    ax2 = ax1.twinx()
    ax2.bar(df["N"], df["Critical_Overlap"], alpha=0.2, color='gray', label="Shared Critical Instances")
    ax2.set_ylabel("Count of Serious Samples in Baseline Pool")
    
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.title("Learning Efficiency: Accuracy vs. Information Density")
    plt.savefig(out_dir / "instance_learning_plot.png", dpi=200)
    print(f"Experiment complete. Plot saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--has-weights", required=True)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--max-n", type=int, default=300)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-lr", type=float, default=5e-5)
    parser.add_argument("--ft-scope", default="embedder-fc-head")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results/learning_curve")
    # --- ADD THESE TO FIX THE ATTRIBUTE ERRORS ---
    parser.add_argument("--augment", action="store_true", help="Fix for imported functions")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Fix for imported functions")
    parser.add_argument("--ft-beta", type=float, default=2.0, help="Fix for imported functions")
    # ---------------------------------------------
    run_learning_curve(parser.parse_args())
