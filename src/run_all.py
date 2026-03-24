import argparse
import os
import subprocess
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (os.path.join(_SRC, "train.py"),     "Train models"),
    (os.path.join(_SRC, "extract.py"),   "Extract features"),
    (os.path.join(_SRC, "detect.py"),    "Drift detection"),
    (os.path.join(_SRC, "visualize.py"), "Visualization"),
    (os.path.join(_SRC, "analysis.py"),  "Analysis"),
]


def run_step(script, label, extra_args=None):
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'─' * 62}")
    print(f"  STEP: {label}  ({script})")
    print(f"{'─' * 62}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ✗ FAILED at: {label}")
        step_n = next(i for i, (s, _) in enumerate(STEPS, 1) if s == script)
        print(f"    Fix the issue, then re-run:  python {os.path.basename(script)}")
        print(f"    Or restart from this step:   python run_all.py --from-step {step_n}")
        sys.exit(result.returncode)
    print(f"  ✓ {label} done.")


def main():
    parser = argparse.ArgumentParser(description="Full drift experiment pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip step1 (use existing weights)")
    parser.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Start from this step number")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--drift-sigma", type=float, default=None)
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()

    start = args.from_step
    if args.skip_train:
        start = max(start, 2)

    for i, (script, label) in enumerate(STEPS, 1):
        if i < start:
            print(f"  Skipping step {i}: {label}")
            continue

        extra = []
        if i == 1 and args.epochs:
            extra += ["--epochs", str(args.epochs)]
        if i == 3 and args.drift_sigma:
            extra += ["--drift-sigma", str(args.drift_sigma)]
        if i == 4 and args.skip_umap:
            extra += ["--skip-umap"]

        run_step(script, label, extra)

    print(f"\n{'═' * 62}")
    print("  ALL STEPS COMPLETE")
    print(f"{'═' * 62}")
    print("  Weights  → weights/")
    print("  Features → features/")
    print("  Results  → results/")


if __name__ == "__main__":
    main()
