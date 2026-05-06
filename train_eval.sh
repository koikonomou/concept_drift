#!/bin/bash
set -e

SKIP_TRAIN=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --dry-run)    DRY_RUN=true ;;
    esac
done

SKIP_FLAG=""
if $SKIP_TRAIN; then
    SKIP_FLAG="--skip-train"
fi

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

mkdir -p logs

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Sweep started: $START_TIME"
echo "Skip train:    $SKIP_TRAIN"
echo "Dry run:       $DRY_RUN"

# =============================================================================
# GROUP 0 — Quick validation
# =============================================================================

run_experiment "validation" \
    --has-scale 8 \
    --has-margin 0.2 \
    --drift-sigma 2.0 \
    --ft-pcts 25 \
    --ft-epochs 3 \
    --ft-beta 2.0 \
    --pool both

echo ""
echo "Validation complete. Press Ctrl+C to stop, or wait 10s to continue..."
sleep 10

# =============================================================================
# GROUP 1 — HAS scale × margin training sweep
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GROUP 1: HAS scale × margin training search        ║"
echo "╚══════════════════════════════════════════════════════╝"

for SCALE in 6 8 10; do
  for MARGIN in 0.1 0.2 0.3; do
    TAG="has_s${SCALE}_m${MARGIN//./_}"

    run_experiment "$TAG" \
      --has-scale "$SCALE" \
      --has-margin "$MARGIN" \
      --drift-sigma 2.0 \
      --ft-pcts 25 \
      --ft-epochs 3 \
      --ft-beta 2.0 \
      --pool both
  done
done

# =============================================================================
# AGGREGATE
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
