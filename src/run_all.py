"""
run_all.py — Orchestrates the full paper pipeline.

Steps:
  1  train.py      Train Baseline + HAS, evaluate on test set
  2  extract.py    Extract 64-D latents, margins, closest_boundary
  3  detect.py     Drift tests, print tables, save all CSVs
  4  visualize.py  3 paper figures
  5  finetune.py   Fine-tuning ablation study (only with --finetune):
                     5a  finetune.py        (produces fine-tuned weights)
                     5b  extract.py         (re-extract with best ft weights)
                     5c  detect.py          (re-detect, backup old CSVs)
                   Pre-finetune CSVs backed up as *_pretrain.csv.
                   Original weights restored after step 5c.

────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────
  python run_all.py                          # steps 1–4
  python run_all.py --finetune               # steps 1–4 + fine-tuning
  python run_all.py --skip-train             # skip step 1
  python run_all.py --from-step 3            # restart from detect
  python run_all.py --from-step 5            # fine-tuning loop only
  python run_all.py --epochs 150 --finetune  # custom epochs + finetune

Fine-tuning options (passed to finetune.py):
  --ft-pcts   25,50,100      %% of pool to sweep (default)
  --ft-mode   all-modes      all three modes (default), or one of:
                             drift-ranked | random | all
  --ft-epochs 10             fine-tune epochs per run
  --ft-lr     1e-4           fine-tune learning rate

nohup examples:
  nohup python run_all.py --epochs 150 > logs/run.log 2>&1 &
  nohup python run_all.py --skip-train --finetune > logs/run.log 2>&1 &
  nohup python run_all.py --from-step 5 --ft-pcts 25,50,100 > logs/ft.log 2>&1 &
"""

import argparse, glob, os, shutil, subprocess, sys

_SRC = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (os.path.join(_SRC, "train.py"),     "Train models + test evaluation"),
    (os.path.join(_SRC, "extract.py"),   "Extract features"),
    (os.path.join(_SRC, "detect.py"),    "Drift detection"),
    (os.path.join(_SRC, "visualize.py"), "Visualization"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Step runner
# ─────────────────────────────────────────────────────────────────────────────

def run_step(script, label, extra_args=None):
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'─' * 66}")
    print(f"  STEP : {label}")
    args_str = " ".join(extra_args or [])
    print(f"  CMD  : python {os.path.basename(script)}"
          + (f" {args_str}" if args_str else ""))
    print(f"{'─' * 66}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        step_n = next(
            (i for i, (s, _) in enumerate(STEPS, 1) if s == script), None)
        print(f"\n  ✗ FAILED")
        print(f"    Re-run: python {os.path.basename(script)} {args_str}")
        if step_n:
            print(f"    Or:     python run_all.py --from-step {step_n}")
        sys.exit(result.returncode)
    print(f"  ✓ {label} done.")


# ─────────────────────────────────────────────────────────────────────────────
# Weight file management for step 5b/5c
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_ft_weights(weight_dir, pcts_str, mode="drift-ranked"):
    """Find the fine-tuned weight files for the best (highest) ft_pct.

    Looks for baseline_{mode}_{pct}pct.pth and has_model_{mode}_{pct}pct.pth.
    Falls back through lower pcts if the highest is not found.
    Returns (bl_ft, has_ft) paths or (None, None).
    """
    pcts = sorted(
        [float(p.strip()) for p in pcts_str.split(",")],
        reverse=True)

    for pct in pcts:
        ipct = int(pct)
        bl  = os.path.join(weight_dir, f"baseline_{mode}_{ipct}pct.pth")
        has = os.path.join(weight_dir, f"has_model_{mode}_{ipct}pct.pth")
        if os.path.exists(bl) and os.path.exists(has):
            return bl, has

    return None, None


def _swap_weights(bl_ft, has_ft, weight_dir="weights"):
    """Backup originals and promote fine-tuned weights to active slots."""
    pairs = [
        (bl_ft,  os.path.join(weight_dir, "baseline.pth"),
                 os.path.join(weight_dir, "baseline_pretrain.pth")),
        (has_ft, os.path.join(weight_dir, "has_model.pth"),
                 os.path.join(weight_dir, "has_pretrain.pth")),
    ]
    for ft, main, backup in pairs:
        shutil.copy(main, backup)
        shutil.copy(ft,   main)
        print(f"  Swapped  {os.path.basename(ft)} → {os.path.basename(main)}"
              f"  (original → {os.path.basename(backup)})")


def _restore_weights(weight_dir="weights"):
    """Restore original weights from backups."""
    for main_name, backup_name in [
        ("baseline.pth",  "baseline_pretrain.pth"),
        ("has_model.pth", "has_pretrain.pth"),
    ]:
        backup = os.path.join(weight_dir, backup_name)
        main   = os.path.join(weight_dir, main_name)
        if os.path.exists(backup):
            shutil.copy(backup, main)
            print(f"  Restored {backup_name} → {main_name}")


def _backup_csvs(result_dir="results"):
    """Rename existing CSVs with _pretrain suffix before re-detect."""
    csv_files = [f for f in glob.glob(os.path.join(result_dir, "*.csv"))
                 if "_pretrain" not in f]
    for csv in csv_files:
        base, ext = os.path.splitext(csv)
        dest = base + "_pretrain" + ext
        os.rename(csv, dest)
        print(f"  Backed up → {os.path.basename(dest)}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(finetune=False):
    print(f"\n{'═' * 66}")
    print("  ALL STEPS COMPLETE")
    print(f"{'═' * 66}")
    print("  weights/    trained model files (.pth)")
    print("  features/   latent vectors (.npz)")
    print("  results/    CSVs and figures\n")

    print("  Core CSVs:")
    for f in ["drift_baseline.csv", "drift_has.csv",
              "population_tests.csv", "taxonomy_comparison.csv",
              "concept_drift_direction.csv",
              "direction_matrix_train.csv", "direction_matrix_custom.csv",
              "hierarchical_baseline.csv", "hierarchical_has.csv"]:
        print(f"    results/{f}")

    if finetune:
        print()
        print("  Fine-tuning outputs:")
        print("    results/finetune_ablation.csv  "
              "(error rates per model × mode × ft_pct × drift type)")
        print("    results/*_pretrain.csv          (pre-finetune results)")
        print("    results/drift_*.csv             (post-finetune results)")
        print("    weights/baseline_<mode>_<pct>pct.pth")
        print("    weights/has_model_<mode>_<pct>pct.pth")

    print()
    print("  Figures:")
    for f in ["fig1_drift_taxonomy.png",
              "fig2_has_geometry.png",
              "fig3_concept_drift_direction.png"]:
        print(f"    results/{f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full drift detection + fine-tuning pipeline.",
        formatter_class=argparse.RawTextHelpFormatter)

    # Step control
    parser.add_argument("--from-step",   type=int, default=1,
                        choices=[1, 2, 3, 4, 5],
                        help=("Start from this step (default: 1).\n"
                              "  5 = fine-tuning loop only (implies --finetune)"))
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip step 1 — reuse existing weights.")

    # Training
    parser.add_argument("--epochs",      type=int,   default=None,
                        help="Override EPOCHS in config.py.")
    parser.add_argument("--drift-sigma", type=float, default=None,
                        help="Override DRIFT_SIGMA for detect.py.")

    # Fine-tuning
    parser.add_argument("--finetune",    action="store_true",
                        help="Run finetune.py after step 4.")
    parser.add_argument("--ft-epochs",   type=int,   default=None,
                        help="Fine-tune epochs per run.")
    parser.add_argument("--ft-lr",       type=float, default=None,
                        help="Fine-tune learning rate.")
    parser.add_argument("--ft-pcts",     type=str,   default="25,50,100",
                        help="Comma-separated %% of pool to sweep (default: 25,50,100).")
    parser.add_argument("--ft-mode",
                        choices=["drift-ranked", "random", "all", "all-modes"],
                        default="all-modes",
                        help="Fine-tuning mode(s) (default: all-modes).")

    args = parser.parse_args()

    # --from-step 5 implies --finetune
    if args.from_step == 5:
        args.finetune = True

    start = max(args.from_step, 2 if args.skip_train else 1)

    # ── Steps 1–4 ─────────────────────────────────────────────────────────────
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

    # ── Step 5 — Fine-tuning ablation ─────────────────────────────────────────
    if not args.finetune:
        _print_summary(finetune=False)
        return

    print(f"\n{'═' * 66}")
    print(f"  STEP 5: Fine-tuning ablation  (finetune.py)")
    print(f"{'═' * 66}")

    # 5a — Run finetune.py
    ft_script = os.path.join(_SRC, "finetune.py")
    ft_extra  = ["--ft-pcts", args.ft_pcts,
                 "--ft-mode",  args.ft_mode]
    if args.ft_epochs:
        ft_extra += ["--ft-epochs", str(args.ft_epochs)]
    if args.ft_lr:
        ft_extra += ["--ft-lr", str(args.ft_lr)]

    run_step(ft_script, "Fine-tuning ablation (all modes × ft_pcts)", ft_extra)

    # 5b — Re-extract with best fine-tuned weights
    print(f"\n{'─' * 66}")
    print(f"  STEP 5b: Re-extract with fine-tuned weights")
    print(f"{'─' * 66}")

    bl_ft, has_ft = _find_best_ft_weights("weights", args.ft_pcts,
                                           mode="drift-ranked")
    if bl_ft is None:
        print("  ⚠ No fine-tuned weights found — skipping re-extract/re-detect.")
        _print_summary(finetune=True)
        return

    print(f"  Using: {os.path.basename(bl_ft)}, {os.path.basename(has_ft)}")
    _swap_weights(bl_ft, has_ft, "weights")

    run_step(os.path.join(_SRC, "extract.py"),
             "Re-extract features (fine-tuned weights)")

    # 5c — Re-detect
    print(f"\n{'─' * 66}")
    print(f"  STEP 5c: Re-detect drift (fine-tuned models)")
    print(f"{'─' * 66}")

    print("  Backing up pre-finetune CSVs …")
    _backup_csvs("results")

    detect_extra = []
    if args.drift_sigma:
        detect_extra += ["--drift-sigma", str(args.drift_sigma)]
    run_step(os.path.join(_SRC, "detect.py"),
             "Re-detect drift (fine-tuned)", detect_extra)

    # Restore original weights so future runs use pre-finetune models by default
    print("  Restoring original weights …")
    _restore_weights("weights")

    print(f"  ✓ Step 5 complete.")
    _print_summary(finetune=True)


if __name__ == "__main__":
    main()
