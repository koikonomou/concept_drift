"""
Run drift detection on saved features.

No GPU needed — loads .npz files from step2, runs all statistical tests,
saves textual report and per-image CSV.

Usage:
    python step3_detect.py
    python step3_detect.py --concept-thresh 0.60

Outputs:
    results/drift_report.txt
    results/drift_baseline.csv
    results/drift_has.csv
    results/drift_comparison.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from config import (WEIGHT_DIR, FEATURE_DIR, RESULT_DIR,
                    DRIFT_SIGMA, LANDSCAPE_CLASSES,
                    ensure_dirs)
from drift_stats import full_drift_report, print_report


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_features(name):
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run step2_extract.py first.")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_train_stats(tag):
    path = os.path.join(WEIGHT_DIR, f"{tag}_train_stats.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run step2_extract.py first.")
    s = np.load(path)
    return dict(
        centroid=s["centroid"],
        dist_mean=float(s["dist_mean"]),
        dist_std=float(s["dist_std"]),
        conf_mean=float(s["conf_mean"]),
        conf_std=float(s["conf_std"]),
    )


def build_records(custom, train_stats, class_names, drift_sigma):
    """Per-image drift scores → DataFrame.

    Thresholds are computed per-model from training statistics:
      Data drift:    latent distance  >  train_dist_mean + drift_sigma * train_dist_std
      Concept drift: confidence       <  train_conf_mean - drift_sigma * train_conf_std
    """
    centroid = train_stats["centroid"]
    data_thresh    = train_stats["dist_mean"] + drift_sigma * train_stats["dist_std"]
    concept_thresh = train_stats["conf_mean"] - drift_sigma * train_stats["conf_std"]
    # Floor at 0 — confidence can't be negative
    concept_thresh = max(concept_thresh, 0.0)

    rows = []
    for i in range(len(custom["latents"])):
        dd = float(np.linalg.norm(custom["latents"][i] - centroid))
        conf = float(custom["confs"][i])
        d_flag = dd > data_thresh
        c_flag = conf < concept_thresh

        if not d_flag and not c_flag:
            dt = "In-Distribution"
        elif d_flag and not c_flag:
            dt = "Data Drift"
        elif not d_flag and c_flag:
            dt = "Concept Drift"
        else:
            dt = "Full Drift (both)"

        lbl = int(custom["labels"][i])
        rows.append(dict(
            sub_category=str(custom["subs"][i]),
            true_label=lbl,
            true_class=class_names[lbl] if lbl < len(class_names) else str(lbl),
            pred_label=int(custom["preds"][i]),
            correct=(lbl == int(custom["preds"][i])),
            max_confidence=conf,
            data_drift_score=dd,
            concept_drift_score=1.0 - conf,
            data_drifted=d_flag,
            concept_drifted=c_flag,
            drift_type=dt,
        ))
    return pd.DataFrame(rows), data_thresh, concept_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 3: Drift detection")
    parser.add_argument("--drift-sigma", type=float, default=DRIFT_SIGMA,
                        help="N sigma for both data and concept thresholds")
    parser.add_argument("--mmd-perms", type=int, default=300)
    args = parser.parse_args()

    ensure_dirs()

    # ── Load features ──
    print("Loading features …")
    bl_train   = load_features("bl_train")
    bl_custom  = load_features("bl_custom")
    has_train  = load_features("has_train")
    has_custom = load_features("has_custom")

    bl_stats  = load_train_stats("baseline")
    has_stats = load_train_stats("has")

    meta_path = os.path.join(FEATURE_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    custom_classes = meta["custom_classes"]

    print(f"  Baseline train: {len(bl_train['latents'])} samples")
    print(f"  HAS train:      {len(has_train['latents'])} samples")
    print(f"  Custom:         {len(bl_custom['latents'])} samples")
    print(f"  Custom classes: {custom_classes}")

    # Show per-model training confidence ranges
    print(f"\n  Baseline train confidence: μ={bl_stats['conf_mean']:.4f}, σ={bl_stats['conf_std']:.4f}")
    print(f"  HAS train confidence:      μ={has_stats['conf_mean']:.4f}, σ={has_stats['conf_std']:.4f}")

    # ── Statistical tests ──
    print("\n" + "=" * 62)
    print("BASELINE MODEL — DRIFT TESTS")
    print("=" * 62)
    bl_results = full_drift_report(
        bl_train["latents"], bl_custom["latents"],
        bl_train["preds"],   bl_custom["preds"],
        bl_train["confs"],   bl_custom["confs"],
        mmd_perms=args.mmd_perms)
    print_report(bl_results)

    print("=" * 62)
    print("HAS MODEL — DRIFT TESTS")
    print("=" * 62)
    has_results = full_drift_report(
        has_train["latents"], has_custom["latents"],
        has_train["preds"],   has_custom["preds"],
        has_train["confs"],   has_custom["confs"],
        mmd_perms=args.mmd_perms)
    print_report(has_results)

    # ── Save text report ──
    rpt_path = os.path.join(RESULT_DIR, "drift_report.txt")
    with open(rpt_path, "w") as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print("BASELINE MODEL")
            print_report(bl_results)
            print("\nHAS MODEL")
            print_report(has_results)
        f.write(buf.getvalue())
    print(f"  ✓ Text report → {rpt_path}")

    # ── Per-image drift DataFrames ──
    # Thresholds are per-model: mean ± drift_sigma * std from training
    df_bl, bl_data_th, bl_conf_th = build_records(
        bl_custom, bl_stats, LANDSCAPE_CLASSES, args.drift_sigma)
    df_has, has_data_th, has_conf_th = build_records(
        has_custom, has_stats, LANDSCAPE_CLASSES, args.drift_sigma)

    print(f"\n  Thresholds (σ={args.drift_sigma}):")
    print(f"    Baseline — data drift > {bl_data_th:.4f}, "
          f"concept drift < {bl_conf_th:.4f} confidence")
    print(f"    HAS      — data drift > {has_data_th:.4f}, "
          f"concept drift < {has_conf_th:.4f} confidence")

    bl_csv  = os.path.join(RESULT_DIR, "drift_baseline.csv")
    has_csv = os.path.join(RESULT_DIR, "drift_has.csv")
    df_bl.to_csv(bl_csv, index=False)
    df_has.to_csv(has_csv, index=False)
    print(f"  ✓ {bl_csv}")
    print(f"  ✓ {has_csv}")

    # Combined comparison
    df_cmp = pd.concat([df_bl.add_prefix("bl_"), df_has.add_prefix("has_")], axis=1)
    cmp_path = os.path.join(RESULT_DIR, "drift_comparison.csv")
    df_cmp.to_csv(cmp_path, index=False)
    print(f"  ✓ {cmp_path}")

    # ── Summary ──
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    for tag, df in [("Baseline", df_bl), ("HAS", df_has)]:
        acc = df["correct"].mean() * 100
        dd  = df["data_drifted"].mean() * 100
        cd  = df["concept_drifted"].mean() * 100
        print(f"  {tag:10s} | Acc {acc:5.1f}% | "
              f"Data-drifted {dd:5.1f}% | Concept-drifted {cd:5.1f}%")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
