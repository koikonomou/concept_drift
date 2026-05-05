#!/bin/bash
# =============================================================================
# quick_test.sh — Smoke test: verify the full pipeline runs end-to-end.
#
# Uses minimal settings to complete fast.
# Does NOT produce meaningful scientific results — only checks that every
# script runs without crashing before you commit to a long VM sweep.
#
# Expected runtime:
#   with    training : ~10 min
#   --skip-train     : ~2 min
#
# Usage:
#   chmod +x quick_test.sh
#   ./quick_test.sh
#   ./quick_test.sh --skip-train
# =============================================================================

set -e

SKIP_TRAIN=false
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
done

# Use the Poetry environment when this repo has pyproject.toml.
# Falls back to system python3 so the script still works outside Poetry.
if command -v poetry >/dev/null 2>&1 && [ -f pyproject.toml ]; then
    PY_CMD=(poetry run python)
else
    PY_CMD=(python3)
fi

mkdir -p logs
TMPDIR_CHECKS=$(mktemp -d)   # temp dir for Python check scripts
LOG="logs/quick_test.log"
> "$LOG"

PASS=0
FAIL=0
FAILED_STEPS=()
START=$(date +%s)

# ─────────────────────────────────────────────────────────────────────────────
# Helper — runs a command, tees output to log, records pass/fail
# ─────────────────────────────────────────────────────────────────────────────
check() {
    local NAME="$1"
    shift

    echo "" | tee -a "$LOG"
    echo "  ┌─ $NAME" | tee -a "$LOG"
    echo "  │  $*"    | tee -a "$LOG"

    if "$@" >> "$LOG" 2>&1; then
        echo "  └─ ✓ PASS" | tee -a "$LOG"
        PASS=$((PASS + 1))
    else
        echo "  └─ ✗ FAIL  (see $LOG)" | tee -a "$LOG"
        FAIL=$((FAIL + 1))
        FAILED_STEPS+=("$NAME")
    fi
}

# Helper — write a Python script to a temp file and run it
# Avoids all bash quoting issues with inline Python strings
pycheck() {
    local NAME="$1"
    local SCRIPT="$2"
    local TMPPY="$TMPDIR_CHECKS/$(echo "$NAME" | tr ' ' '_').py"
    # Prepend src/ so config.py and models.py are importable as config/models.
    {
        printf '%s
' "import sys, os"
        printf '%s
' "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))"
        printf '%s
' "$SCRIPT"
    } > "$TMPPY"
    check "$NAME" "${PY_CMD[@]}" "$TMPPY"
}

# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════"
echo "  QUICK TEST — smoke test for full pipeline"
echo "  Log: $LOG"
echo "  Skip train: $SKIP_TRAIN"
echo "  Python: ${PY_CMD[*]}"
echo "═══════════════════════════════════════════════"

# ── Step 0: Python imports ────────────────────────────────────────────────────
pycheck "Python imports" "
import sys
errors = []
for mod in ['torch','numpy','pandas','PIL','sklearn','scipy']:
    try:
        __import__(mod)
    except ImportError as e:
        errors.append(str(e))
try:
    from config import LANDSCAPE_CLASSES, DEVICE, ensure_dirs
except Exception as e:
    errors.append(str(e))
try:
    from models import BaselineModel, HASModel, FolderDataset
except Exception as e:
    errors.append(str(e))
if errors:
    for e in errors:
        print('MISSING:', e)
    sys.exit(1)
import torch
device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print('imports OK | device:', device_str)
"

# ── Step 1: Train (minimal epochs) ───────────────────────────────────────────
if ! $SKIP_TRAIN; then
    check "Train (3 epochs)" "${PY_CMD[@]}" src/train.py --epochs 3
else
    echo ""
    echo "  ─ Skipping train (--skip-train)" | tee -a "$LOG"

    pycheck "Weights exist" "
import os, sys
missing = []
for f in ['weights/baseline.pth', 'weights/has_model.pth']:
    if not os.path.exists(f):
        missing.append(f)
    else:
        print('OK:', f)
if missing:
    for f in missing:
        print('MISSING:', f)
    sys.exit(1)
"
fi

# ── Step 2: Extract ───────────────────────────────────────────────────────────
check "Extract features" "${PY_CMD[@]}" src/extract.py

# ── Step 3: Detect (fast — only 10 MMD permutations) ─────────────────────────
check "Detect drift" "${PY_CMD[@]}" src/detect.py --drift-sigma 2.0 --mmd-perms 10

# ── Step 4: Check CSVs produced ──────────────────────────────────────────────
pycheck "CSVs exist and have file_path column" "
import os, sys, pandas as pd
required = [
    'results/drift_has.csv',
    'results/drift_baseline.csv',
    'results/population_tests.csv',
]
ok = True
for f in required:
    if not os.path.exists(f):
        print('MISSING:', f)
        ok = False
    else:
        df = pd.read_csv(f)
        has_fp = 'file_path' in df.columns
        print(f'OK: {f}  ({len(df)} rows, file_path={has_fp})')
        if f.endswith('drift_has.csv') and not has_fp:
            print('  ERROR: file_path column missing — finetune.py will fail')
            ok = False
if not ok:
    sys.exit(1)
"

# ── Step 5: Fine-tune (1 pool, 25%, 3 epochs) ────────────────────────────────
check "Fine-tune (Pool A, 25%, 3 epochs)" "${PY_CMD[@]}" src/finetune.py \
    --pool data \
    --ft-pcts 25 \
    --ft-epochs 3 \
    --ft-beta 2.0 \
    --ft-scope embedder-fc-head \
    --log-every 1

# ── Step 6: Check finetune outputs ───────────────────────────────────────────
pycheck "Finetune outputs exist" "
import os, sys, pandas as pd
required = [
    'results/finetune/final_finetune_summary.csv',
    'results/finetune/final_finetune_summary_long.csv',
    'results/finetune/finetune_config.json',
]
ok = True
for f in required:
    if not os.path.exists(f):
        print('MISSING:', f)
        ok = False
    else:
        print('OK:', f)
if not ok:
    sys.exit(1)
df = pd.read_csv('results/finetune/final_finetune_summary.csv')
print('Summary:', len(df), 'rows')
print('Columns:', list(df.columns[:8]))
"

# ── Step 7: run_experiment.py dry-run ────────────────────────────────────────
check "run_experiment --dry-run" "${PY_CMD[@]}" src/run_experiment.py \
    --tag quicktest \
    --dry-run \
    --drift-sigma 2.0 \
    --ft-epochs 3 \
    --ft-pcts 25 \
    --pool data

# ── Step 8: GPU info ─────────────────────────────────────────────────────────
pycheck "GPU check" "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name}')
    print(f'VRAM: {vram:.1f} GB')
else:
    print('No GPU — running on CPU (sweep will be slow but functional)')
"

# ── Step 9: sweep.sh dry-run ─────────────────────────────────────────────────
if [ -f sweep.sh ]; then
    check "sweep.sh --dry-run" bash sweep.sh --skip-train --dry-run
else
    echo ""
    echo "  ─ sweep.sh not found — skipping" | tee -a "$LOG"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup temp files
# ─────────────────────────────────────────────────────────────────────────────
rm -rf "$TMPDIR_CHECKS"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
END=$(date +%s)
ELAPSED=$(( END - START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo "" | tee -a "$LOG"
echo "═══════════════════════════════════════════════" | tee -a "$LOG"
echo "  QUICK TEST COMPLETE"                           | tee -a "$LOG"
echo "  Time   : ${MINS}m ${SECS}s"                   | tee -a "$LOG"
echo "  PASSED : $PASS"                                | tee -a "$LOG"
echo "  FAILED : $FAIL"                                | tee -a "$LOG"

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    echo ""                                            | tee -a "$LOG"
    echo "  Failed steps:"                             | tee -a "$LOG"
    for s in "${FAILED_STEPS[@]}"; do
        echo "    ✗ $s"                                | tee -a "$LOG"
    done
    echo ""                                            | tee -a "$LOG"
    echo "  Full log: $LOG"                            | tee -a "$LOG"
    echo "═══════════════════════════════════════════════" | tee -a "$LOG"
    exit 1
else
    echo ""                                            | tee -a "$LOG"
    echo "  ✓ All checks passed — pipeline is ready."  | tee -a "$LOG"
    echo "  Run the full sweep with:"                  | tee -a "$LOG"
    echo "    nohup ./sweep.sh --skip-train > logs/sweep.log 2>&1 &" | tee -a "$LOG"
    echo "═══════════════════════════════════════════════" | tee -a "$LOG"
fi
