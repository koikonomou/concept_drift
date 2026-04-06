import argparse, os, subprocess, sys

_SRC = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (os.path.join(_SRC, "train.py"),     "Train models"),
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
        step_n = next(i for i, (s, _) in enumerate(STEPS, 1) if s == script)
        print(f"\n  ✗ FAILED — fix the issue then re-run:")
        print(f"    python {os.path.basename(script)}")
        print(f"    python run_all.py --from-step {step_n}")
        sys.exit(result.returncode)
    print(f"  ✓ {label} done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip step 1 (use existing weights)")
    parser.add_argument("--from-step",  type=int, default=1,
                        choices=[1, 2, 3, 4])
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--drift-sigma",type=float, default=None)
    args = parser.parse_args()

    start = max(args.from_step, 2 if args.skip_train else 1)

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

    print(f"\n{'═' * 62}")
    print("  ALL STEPS COMPLETE")
    print(f"{'═' * 62}")
    print("  Weights  → weights/")
    print("  Features → features/")
    print("  Results  → results/")
    print()
    print("  CSVs saved:")
    for f in ["drift_baseline.csv", "drift_has.csv",
              "population_tests.csv", "taxonomy_comparison.csv",
              "concept_drift_direction.csv",
              "direction_matrix_train.csv", "direction_matrix_custom.csv",
              "hierarchical_baseline.csv", "hierarchical_has.csv"]:
        print(f"    results/{f}")
    print()
    print("  Figures saved:")
    for f in ["fig1_drift_taxonomy.png",
              "fig2_has_geometry.png",
              "fig3_concept_drift_direction.png"]:
        print(f"    results/{f}")


if __name__ == "__main__":
    main()
