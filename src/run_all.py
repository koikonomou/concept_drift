"""
run_all.py — Orchestrates the full paper pipeline.

Steps:
  1  train.py      Train Baseline + HAS, evaluate on test set
  2  extract.py    Extract 64-D latents, margins, closest_boundary
  3  detect.py     Drift tests, save all CSVs and terminal tables
  4  visualize.py  3 paper figures
  5  finetune      Fine-tune on drift-selected custom samples,
                   re-extract, re-detect to measure improvement
                   (only runs with --finetune flag)

Usage:
    python run_all.py                         # steps 1-4
    python run_all.py --finetune              # steps 1-4 + fine-tuning loop
    python run_all.py --skip-train            # skip step 1
    python run_all.py --from-step 3           # restart from detect
    python run_all.py --from-step 5           # fine-tune only (weights exist)
    python run_all.py --epochs 150 --finetune # full run with finetune

nohup:
    nohup python run_all.py --finetune > logs/run.log 2>&1 &
"""

import argparse, os, subprocess, sys

_SRC = os.path.dirname(os.path.abspath(__file__))

# Steps 1-4 are always available
STEPS = [
    (os.path.join(_SRC, "train.py"),     "Train models + test evaluation"),
    (os.path.join(_SRC, "extract.py"),   "Extract features"),
    (os.path.join(_SRC, "detect.py"),    "Drift detection"),
    (os.path.join(_SRC, "visualize.py"), "Visualization"),
]


def run_step(script, label, extra_args=None):
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'─' * 62}")
    print(f"  STEP: {label}")
    print(f"{'─' * 62}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        # Find step number for helpful restart message
        step_n = next(
            (i for i, (s, _) in enumerate(STEPS, 1) if s == script),
            None)
        print(f"\n  ✗ FAILED — fix the issue then re-run:")
        print(f"    python {os.path.basename(script)}")
        if step_n:
            print(f"    python run_all.py --from-step {step_n}"
                  + (" --finetune" if step_n >= 5 else ""))
        sys.exit(result.returncode)
    print(f"  ✓ {label} done.")


def main():
    parser = argparse.ArgumentParser(
        description="Full drift detection + fine-tuning pipeline")
    parser.add_argument("--finetune",    action="store_true",
                        help="Run fine-tuning loop after step 4 "
                             "(train → extract → detect → finetune → "
                             "re-extract → re-detect)")
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip step 1 (reuse existing weights)")
    parser.add_argument("--from-step",   type=int, default=1,
                        choices=[1, 2, 3, 4, 5],
                        help="Start from this step (5 = fine-tuning loop only)")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--drift-sigma", type=float, default=None)
    parser.add_argument("--ft-epochs",   type=int,   default=None,
                        help="Fine-tune epochs (default from train.py)")
    parser.add_argument("--ft-lr",       type=float, default=None,
                        help="Fine-tune learning rate (default from train.py)")
    parser.add_argument("--ft-strategy", choices=["random", "drift-ranked"],
                        default=None,
                        help="Fine-tune subfolder selection strategy (default from train.py)")
    parser.add_argument("--ft-top-pct",  type=float, default=None,
                        help="Top fraction of subfolders when --ft-strategy=drift-ranked")
    args = parser.parse_args()

    # --from-step 5 implies --finetune
    if args.from_step == 5:
        args.finetune = True

    start = max(args.from_step, 2 if args.skip_train else 1)

    # ── Steps 1-4 ─────────────────────────────────────────────────────────────
    for i, (script, label) in enumerate(STEPS, 1):
        if i < start:
            print(f"  Skipping step {i}: {label}")
            continue
        extra = []
        if i == 1 and args.epochs:
            extra += ["--epochs", str(args.epochs)]
        if i == 3 and args.drift_sigma:
            extra += ["--drift-sigma", str(args.drift_sigma)]
        run_step(script, label, extra)

    # ── Step 5 — Fine-tuning loop (optional) ──────────────────────────────────
    if args.finetune and args.from_step <= 5:
        print(f"\n{'─' * 62}")
        print(f"  STEP 5: Fine-tuning on drift-selected custom samples")
        print(f"{'─' * 62}")

        # 5a — fine-tune both models
        ft_extra = ["--skip-train", "--finetune"]
        if args.ft_epochs:
            ft_extra += ["--ft-epochs", str(args.ft_epochs)]
        if args.ft_lr:
            ft_extra += ["--ft-lr", str(args.ft_lr)]
        if args.ft_strategy:
            ft_extra += ["--ft-strategy", args.ft_strategy]
        if args.ft_top_pct is not None:
            ft_extra += ["--ft-top-pct", str(args.ft_top_pct)]
        run_step(os.path.join(_SRC, "train.py"),
                 "Fine-tune on selected custom samples", ft_extra)

        # 5b — re-extract features using the fine-tuned weights
        # We need to point extract.py at the ft weights temporarily.
        # Simplest approach: rename ft weights → main weights, extract, restore.
        print(f"\n{'─' * 62}")
        print(f"  STEP 5b: Re-extract features with fine-tuned weights")
        print(f"{'─' * 62}")

        import shutil
        weight_dir = "weights"
        pairs = [
            ("baseline_ft.pth",  "baseline.pth",  "baseline_pretrain.pth"),
            ("has_model_ft.pth", "has_model.pth", "has_pretrain.pth"),
        ]
        # Check fine-tuned weights exist before swapping
        ft_missing = [p[0] for p in pairs
                      if not os.path.exists(os.path.join(weight_dir, p[0]))]
        if ft_missing:
            print(f"  ⚠ Fine-tuned weights not found: {ft_missing}")
            print(f"    Skipping re-extract and re-detect.")
        else:
            # Swap: pretrain → backup, ft → main
            for ft_name, main_name, backup_name in pairs:
                ft_path     = os.path.join(weight_dir, ft_name)
                main_path   = os.path.join(weight_dir, main_name)
                backup_path = os.path.join(weight_dir, backup_name)
                shutil.copy(main_path, backup_path)   # backup original
                shutil.copy(ft_path,   main_path)     # promote ft to main
                print(f"  Swapped {ft_name} → {main_name}  "
                      f"(original backed up as {backup_name})")

            # Re-extract with fine-tuned weights
            run_step(os.path.join(_SRC, "extract.py"),
                     "Re-extract features (fine-tuned weights)")

            # Re-detect — saves results to separate CSVs with _ft suffix
            print(f"\n{'─' * 62}")
            print(f"  STEP 5c: Re-detect drift with fine-tuned models")
            print(f"{'─' * 62}")
            detect_extra = []
            if args.drift_sigma:
                detect_extra += ["--drift-sigma", str(args.drift_sigma)]

            # Run detect.py — outputs overwrite the existing CSVs.
            # We rename them first so both before/after are preserved.
            import glob
            result_dir = "results"
            csv_files  = glob.glob(os.path.join(result_dir, "*.csv"))
            for csv in csv_files:
                base, ext = os.path.splitext(csv)
                os.rename(csv, base + "_pretrain" + ext)
                print(f"  Backed up {os.path.basename(csv)} "
                      f"→ {os.path.basename(base)}_pretrain{ext}")

            run_step(os.path.join(_SRC, "detect.py"),
                     "Re-detect drift (fine-tuned)", detect_extra)

            # Restore original weights so subsequent runs start clean
            for ft_name, main_name, backup_name in pairs:
                backup_path = os.path.join(weight_dir, backup_name)
                main_path   = os.path.join(weight_dir, main_name)
                if os.path.exists(backup_path):
                    shutil.copy(backup_path, main_path)
                    print(f"  Restored {backup_name} → {main_name}")

        print(f"  ✓ Fine-tuning loop done.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print("  ALL STEPS COMPLETE")
    print(f"{'═' * 62}")
    print("  Weights  → weights/")
    print("  Features → features/")
    print("  Results  → results/")
    print()
    print("  Core CSVs:")
    for f in ["drift_baseline.csv", "drift_has.csv",
              "population_tests.csv", "taxonomy_comparison.csv",
              "concept_drift_direction.csv",
              "direction_matrix_train.csv", "direction_matrix_custom.csv",
              "direction_matrix_custom_drifted.csv",
              "hierarchical_baseline.csv", "hierarchical_has.csv"]:
        print(f"    results/{f}")
    if args.finetune:
        print()
        print("  Fine-tuning CSVs:")
        print("    results/finetune_results.csv    (before/after accuracy)")
        print("    results/*_pretrain.csv          (pre-finetune drift results)")
        print("    results/drift_baseline.csv      (post-finetune drift results)")
        print("    results/drift_has.csv           (post-finetune drift results)")
        print()
        print("  Fine-tuned weights:")
        print("    weights/baseline_ft.pth")
        print("    weights/has_model_ft.pth")
    print()
    print("  Figures:")
    for f in ["fig1_drift_taxonomy.png",
              "fig2_has_geometry.png",
              "fig3_concept_drift_direction.png"]:
        print(f"    results/{f}")


if __name__ == "__main__":
    main()
