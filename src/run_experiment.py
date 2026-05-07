"""
Creates:
  runs/run1/   (or run2, run3, ... auto-incremented)
    weights/        has_model.pth, baseline.pth (copied from base weights)
    features/       bl_train.npz, has_train.npz, bl_custom.npz, has_custom.npz
    results/        drift_has.csv, drift_baseline.csv, all detect tables
    results/finetune/  final_finetune_summary.csv, checkpoints/
    logs/           pipeline.log  (full stdout+stderr of every step)
    config.json     all hyperparameters used in this run

    # Aggregate all runs
    python run_experiment.py --aggregate
"""

import argparse
import glob
import json
import os
import shutil
import shlex
import subprocess
import sys
from datetime import datetime
from config import HAS_SCALE, HAS_MARGIN
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

BASE_RUNS_DIR   = "runs"
BASE_WEIGHT_DIR = "weights"   # where the original trained weights live
PYTHON = shlex.quote(sys.executable)  # preserves the active Poetry/venv interpreter


def next_run_dir(tag=""):
    """Auto-increment: runs/run1, runs/run2, ... or runs/run1_sigma_1_5 etc."""
    os.makedirs(BASE_RUNS_DIR, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    i = 1
    while os.path.exists(os.path.join(BASE_RUNS_DIR, f"run{i}{suffix}")):
        i += 1
    return os.path.join(BASE_RUNS_DIR, f"run{i}{suffix}")


def make_run_dirs(run_dir):
    """Create the full directory tree for one run."""
    for sub in ["weights", "features", "results/finetune", "logs"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)


def copy_weights(run_dir):
    """Copy original trained weights into the run dir.

    This makes every run self-contained — if you retrain later,
    old runs still have their own weights and results are reproducible.
    """
    for fname in ["baseline.pth", "has_model.pth", "has_best.pth"]:
        src = os.path.join(BASE_WEIGHT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(run_dir, "weights", fname))


def _read_config_py():
    """Import config.py and extract all uppercase constants.

    Captures every hyperparameter defined in config.py so the JSON
    is a complete record of the exact settings used in this run.
    Skips non-serialisable values (torch.device, functions, modules).
    """
    try:
        import importlib.util
        candidates = ["config.py", os.path.join("src", "config.py")]
        config_path = next((p for p in candidates if os.path.exists(p)), None)
        if config_path is None:
            return {"config_py_load_error": "config.py not found in project root or src/"}
        spec = importlib.util.spec_from_file_location("config", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return {"config_py_load_error": str(e)}

    out = {}
    for name in dir(mod):
        if not name.isupper():
            continue          # skip functions, private attrs, device, etc.
        val = getattr(mod, name)
        # Only keep JSON-serialisable types
        if isinstance(val, (int, float, str, bool, list, dict)):
            out[name] = val
        elif isinstance(val, tuple):
            out[name] = list(val)
        # Skip torch.device, functions, modules — not serialisable
    return out


def save_config(run_dir, args, run_dir_path):
    config = {
        "run_meta": {
            "run_dir":    run_dir_path,
            "timestamp":  datetime.now().isoformat(),
            "git_commit": _git_commit(),
            "tag":        args.tag,
        },
        "cli_args": {
            "drift_sigma":  args.drift_sigma,
            "mmd_perms":    args.mmd_perms,
            "pool":         args.pool,
            "ft_scope":     args.ft_scope,
            "ft_lr":        args.ft_lr,
            "ft_beta":      args.ft_beta,
            "ft_epochs":    args.ft_epochs,
            "ft_pcts":      args.ft_pcts,
            "batch_size":   args.batch_size,
            "grad_clip":    args.grad_clip,
            "augment":      args.augment,
            "only":         args.only,
            "skip_extract": args.skip_extract,
            "has_scale": args.has_scale,
            "has_margin": args.has_margin,
            },
        "config_py": _read_config_py(),
    }

    # Flat version at top level for easy pandas reading in aggregate_runs()
    flat = {}
    flat.update(config["run_meta"])
    flat.update({f"cli_{k}": v for k, v in config["cli_args"].items()})
    flat.update({f"cfg_{k}": v for k, v in config["config_py"].items()})
    config["flat"] = flat

    path = os.path.join(run_dir_path, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ {path}")

    # Also save a human-readable summary
    summary_path = os.path.join(run_dir_path, "config_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Run: {run_dir_path}\n")
        f.write(f"Time: {config['run_meta']['timestamp']}\n")
        f.write(f"Git:  {config['run_meta']['git_commit']}\n\n")
        f.write("── CLI args ──────────────────────────────\n")
        for k, v in config["cli_args"].items():
            f.write(f"  {k:<20} = {v}\n")
        f.write("\n── config.py ─────────────────────────────\n")
        for k, v in sorted(config["config_py"].items()):
            f.write(f"  {k:<22} = {v}\n")
    print(f"  ✓ {summary_path}")

    return config


def _git_commit():
    """Record the git commit hash so results are traceable to code."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Step runner — logs everything automatically
# ─────────────────────────────────────────────────────────────────────────────

def run_step(name, cmd, log_path, dry_run=False):
    """Run one pipeline step, tee output to log file and print to console."""
    print(f"\n{'═'*68}")
    print(f"  STEP: {name}")
    print(f"  CMD : {cmd}")
    print(f"{'═'*68}")

    if dry_run:
        print("  [dry-run — skipping]")
        return True

    with open(log_path, "a") as log:
        log.write(f"\n{'='*68}\n")
        log.write(f"STEP: {name}\n")
        log.write(f"CMD:  {cmd}\n")
        log.write(f"TIME: {datetime.now().isoformat()}\n")
        log.write(f"{'='*68}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        proc.wait()

    if proc.returncode != 0:
        print(f"\n  ✗ FAILED (exit {proc.returncode}) — see {log_path}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation — combine all runs into one CSV
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_runs(output_path="runs/all_runs_summary.csv"):
    """Collect final_finetune_summary.csv from every run and merge with configs.

    Each row in the output CSV contains:
      - All finetune result columns (concept_custom_acc_delta etc.)
      - All CLI args from that run (cli_drift_sigma, cli_ft_beta etc.)
      - All config.py constants from that run (cfg_EPOCHS, cfg_HAS_SCALE etc.)
      - run metadata (run_dir, timestamp, git_commit, tag)

    This lets you directly compare runs with different hyperparameters
    using pandas groupby or any plotting tool.
    """
    pattern = os.path.join(BASE_RUNS_DIR,
                           "*/results/finetune/final_finetune_summary.csv")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        print(f"No run results found in {BASE_RUNS_DIR}/")
        return

    dfs = []
    for path in paths:
        run_dir  = path.split(os.sep)[1]
        cfg_path = os.path.join(BASE_RUNS_DIR, run_dir, "config.json")

        df = pd.read_csv(path)
        df["run_dir"] = run_dir

        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)

            # Use the flat dict if present (new format)
            # else fall back to old top-level format
            flat = cfg.get("flat", cfg)
            for k, v in flat.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    df[k] = v
                elif isinstance(v, list):
                    df[k] = str(v)   # lists become strings for CSV compat

        dfs.append(df)
        print(f"  ✓ {run_dir} ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\n  ✓ {output_path}  ({len(combined)} rows across {len(dfs)} runs)")

    # Print comparison table with key result + config columns
    key_cols = [
        # Identity
        "run_dir", "tag", "pool", "ft_pct", "n_finetune",
        # CLI hyperparams
        "cli_drift_sigma", "cli_ft_beta", "cli_ft_lr", "cli_ft_epochs",
        # config.py training params
        "cfg_HAS_SCALE", "cfg_HAS_MARGIN", "cfg_ALPHA", "cfg_BETA",
        # Key results
        "concept_custom_acc_delta",   "concept_custom_margin_delta",
        "stable_custom_acc_delta",    "source_test_acc_delta",
        "full_custom_margin_delta",
    ]
    key_cols = [c for c in key_cols if c in combined.columns]
    if key_cols:
        print("\nComparison across runs:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(combined[key_cols].to_string(index=False))

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline orchestrator with auto-versioned run directories.")

    # Run control
    parser.add_argument("--tag", default="", help="Name suffix for this run (e.g. sigma_1_5, beta_3)")
    parser.add_argument("--only", default=None, choices=["extract", "detect", "finetune"], help="Run only one step (assumes previous steps are done)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction (reuse features from same run dir)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--aggregate", action="store_true", help="Only aggregate existing runs into all_runs_summary.csv")

    # detect.py hyperparameters
    parser.add_argument("--drift-sigma", type=float, default=2.0, help="Sigma for drift detection thresholds (default 2.0)")
    parser.add_argument("--mmd-perms",   type=int,   default=300, help="MMD permutation test samples (default 300)")

    # finetune.py hyperparameters
    parser.add_argument("--pool", choices=["data", "concept", "both"], default="both")
    parser.add_argument("--ft-scope", choices=["head", "embedder-fc-head"], default="embedder-fc-head")
    parser.add_argument("--ft-lr", type=float, default=5e-5)
    parser.add_argument("--ft-beta", type=float, default=2.0)
    parser.add_argument("--ft-epochs", type=int,   default=100)
    parser.add_argument("--ft-pcts",    default="25,50,100")
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--grad-clip",  type=float, default=5.0)
    parser.add_argument("--augment",    action="store_true")
    parser.add_argument("--has-scale", type=float, default=HAS_SCALE)
    parser.add_argument("--has-margin", type=float, default=HAS_MARGIN)

    args = parser.parse_args()

    # ── Aggregate only ────────────────────────────────────────────────────────
    if args.aggregate:
        aggregate_runs()
        return

    # ── Create run directory ──────────────────────────────────────────────────
    run_dir = next_run_dir(args.tag)
    make_run_dirs(run_dir)
    copy_weights(run_dir)

    log_path = os.path.join(run_dir, "logs", "pipeline.log")
    config   = save_config(run_dir, args, run_dir)

    wd  = os.path.join(run_dir, "weights")
    fd  = os.path.join(run_dir, "features")
    rd  = os.path.join(run_dir, "results")
    ftd = os.path.join(run_dir, "results", "finetune")

    print(f"\n{'='*68}")
    print(f"  RUN DIR : {run_dir}")
    print(f"  LOG     : {log_path}")
    print(f"  TAG     : {args.tag or '(none)'}")
    print(f"{'='*68}")

    ok = True
    # ── Step 0: Train ───────────────────────────────────────────────────────
    if args.only is None:
        ok = run_step(
            "Train models",
            f"{PYTHON} src/train.py "
            f"--has-scale {args.has_scale} "
            f"--has-margin {args.has_margin} "
            f"--weight-dir {wd}",
            log_path, args.dry_run)
        if not ok:
            sys.exit(1)
    # ── Step 1: Extract ───────────────────────────────────────────────────────
    if args.only in (None, "extract") and not args.skip_extract:
        ok = run_step(
            "Extract features",
            (f"{PYTHON} src/extract.py "
            f"--weight-dir {wd} "
            f"--feature-dir {fd}"),
            log_path, args.dry_run)
        if not ok:
            sys.exit(1)
    elif args.skip_extract:
        print("\n  Skipping extract — reusing existing features")

    # ── Step 2: Detect ────────────────────────────────────────────────────────
    if args.only in (None, "detect"):
        ok = run_step(
            "Detect drift",
            f"{PYTHON} src/detect.py "
            f"--feature-dir {fd} "
            f"--weight-dir  {wd} "
            f"--result-dir  {rd} "
            f"--drift-sigma {args.drift_sigma} "
            f"--mmd-perms   {args.mmd_perms}",
            log_path, args.dry_run)
        if not ok:
            sys.exit(1)

    # ── Step 3: Fine-tune ─────────────────────────────────────────────────────
    if args.only in (None, "finetune"):
        aug_flag = "--augment" if args.augment else ""
        ok = run_step(
            "Fine-tune (two-pool)",
            f"{PYTHON} src/finetune.py "
            f"--weight-dir  {wd} "
            f"--result-dir  {rd} "
            f"--output-dir  {ftd} "
            f"--pool        {args.pool} "
            f"--ft-scope    {args.ft_scope} "
            f"--ft-lr       {args.ft_lr} "
            f"--ft-beta     {args.ft_beta} "
            f"--ft-epochs   {args.ft_epochs} "
            f"--ft-pcts     {args.ft_pcts} "
            f"--batch-size  {args.batch_size} "
            f"--grad-clip   {args.grad_clip} "
            f"{aug_flag}",
            log_path, args.dry_run)
        if not ok:
            sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  Run complete: {run_dir}")
    print(f"  Results    : {rd}/")
    print(f"  Fine-tune  : {ftd}/final_finetune_summary.csv")
    print(f"  Log        : {log_path}")
    print(f"{'='*68}")

    # Auto-aggregate if more than one run exists
    existing = glob.glob(os.path.join(BASE_RUNS_DIR,
                                       "*/results/finetune/final_finetune_summary.csv"))
    if len(existing) > 1:
        print("\nAggregating all runs …")
        aggregate_runs()


if __name__ == "__main__":
    main()
