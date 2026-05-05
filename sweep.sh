#!/bin/bash
# =============================================================================
# sweep.sh — Multi-experiment sweep for the HAS drift detection pipeline.
#
# Each experiment:
#   1. Trains Baseline + HAS from scratch (or skips if --skip-train)
#   2. Extracts features
#   3. Detects drift
#   4. Fine-tunes (two-pool: Pool A + Pool B)
#   5. Saves everything in a versioned run directory
#
# Results:
#   runs/run1_*/    runs/run2_*/    ...
#   runs/all_runs_summary.csv       (auto-aggregated after all experiments)
# =============================================================================

set -e   # exit on any error

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
SKIP_TRAIN=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --dry-run)    DRY_RUN=true    ;;
    esac
done

SKIP_FLAG=""
if $SKIP_TRAIN; then
    SKIP_FLAG="--skip-extract"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one experiment
# ─────────────────────────────────────────────────────────────────────────────
run_experiment() {
    local TAG="$1"
    shift
    local EXTRA_ARGS="$@"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  EXPERIMENT: $TAG"
    echo "  ARGS:       $EXTRA_ARGS"
    echo "════════════════════════════════════════════════════════"

    if $DRY_RUN; then
        echo "  [dry-run] python src/run_experiment.py --tag $TAG $SKIP_FLAG $EXTRA_ARGS"
        return
    fi

    python src/run_experiment.py \
        --tag "$TAG" \
        $SKIP_FLAG \
        $EXTRA_ARGS
}

# ─────────────────────────────────────────────────────────────────────────────
# Make sure logs dir exists
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p logs

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Sweep started: $START_TIME"
echo "Skip train:    $SKIP_TRAIN"
echo "Dry run:       $DRY_RUN"

# =============================================================================
# QUICK VALIDATION GROUP — runs first to confirm pipeline works on VM
# Uses same settings as quick_test.sh but saves into runs/ properly
# Check this group before committing to the full sweep
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 0: Quick validation (pipeline smoke test)   ║"
echo "╚══════════════════════════════════════════════════════╝"

run_experiment "validation" \
    --drift-sigma 2.0 \
    --ft-pcts 25 \
    --ft-epochs 3 \
    --ft-beta 2.0 \
    --pool both

echo ""
echo "  Validation complete. Check runs/run*_validation/ before continuing."
echo "  Press Ctrl+C to stop sweep here, or wait 10s to continue..."
sleep 10

# =============================================================================
# EXPERIMENT GROUP 1 — Drift sigma sensitivity
# Does the threshold sigma (1.5 / 2.0 / 2.5) change the findings?
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 1: Drift sigma sensitivity                   ║"
echo "╚══════════════════════════════════════════════════════╝"

run_experiment "sigma_1_5" \
    --drift-sigma 1.5 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool both

run_experiment "sigma_2_0" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool both

run_experiment "sigma_2_5" \
    --drift-sigma 2.5 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool both

# =============================================================================
# EXPERIMENT GROUP 2 — Fine-tuning beta sensitivity
# Does the HAS penalty weight during fine-tuning matter?
# ft_beta=2.0 is the default; 1.0 = weaker, 3.0 = stronger
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 2: Fine-tuning beta sensitivity              ║"
echo "╚══════════════════════════════════════════════════════╝"

run_experiment "beta_1_0" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 1.0 \
    --pool both

run_experiment "beta_2_0" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool both

run_experiment "beta_3_0" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 3.0 \
    --pool both

# =============================================================================
# EXPERIMENT GROUP 3 — Scope sensitivity
# head-only (320 params) vs embedder-fc-head (~131K params)
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 3: Fine-tuning scope                         ║"
echo "╚══════════════════════════════════════════════════════╝"

run_experiment "scope_head" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --ft-scope head \
    --pool both

run_experiment "scope_embedder_fc_head" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --ft-scope embedder-fc-head \
    --pool both

# =============================================================================
# EXPERIMENT GROUP 4 — Pool ablation
# Pool A only (does data FT change concept drift?)
# Pool B only (does concept FT recover margins?)
# Both pools together
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 4: Pool ablation                             ║"
echo "╚══════════════════════════════════════════════════════╝"

run_experiment "pool_A_only" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool data

run_experiment "pool_B_only" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool concept

run_experiment "pool_both" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool both

# =============================================================================
# EXPERIMENT GROUP 5 — Epoch sensitivity
# Does more fine-tuning help? 50 / 100 / 150 epochs
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 5: Fine-tuning epoch sensitivity             ║"
echo "╚══════════════════════════════════════════════════════╝"

run_experiment "epochs_50" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 50 \
    --ft-beta 2.0 \
    --pool both

run_experiment "epochs_100" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 100 \
    --ft-beta 2.0 \
    --pool both

run_experiment "epochs_150" \
    --drift-sigma 2.0 \
    --ft-pcts 25,50,100 \
    --ft-epochs 150 \
    --ft-beta 2.0 \
    --pool both

# =============================================================================
# AGGREGATE — Combine all run results into one CSV
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════"
echo "  AGGREGATING all runs → runs/all_runs_summary.csv"
echo "════════════════════════════════════════════════════════"

if ! $DRY_RUN; then
    python src/run_experiment.py --aggregate
fi

END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Sweep complete"
echo "  Started : $START_TIME"
echo "  Finished: $END_TIME"
echo "  Results : runs/all_runs_summary.csv"
echo "════════════════════════════════════════════════════════"
