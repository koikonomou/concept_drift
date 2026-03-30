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
    results/hierarchical_drift.json   (ADDITION 2)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from config import (WEIGHT_DIR, FEATURE_DIR, RESULT_DIR,
                    DRIFT_SIGMA, LANDSCAPE_CLASSES,
                    HAS_MARGIN_SIGMA,
                    ensure_dirs)
from drift_stats import (full_drift_report, print_report,
                         has_margin_drift_test, has_boundary_direction_test,
                         has_drift_direction_matrix, print_direction_matrix,
                         hierarchical_drift_report)


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
    stats = dict(
        centroid=s["centroid"],
        dist_mean=float(s["dist_mean"]),
        dist_std=float(s["dist_std"]),
        conf_mean=float(s["conf_mean"]),
        conf_std=float(s["conf_std"]),
    )
    # ADDITION 1 — margin stats present only for HAS (saved by save_train_stats_has)
    if "margin_mean" in s:
        stats["margin_mean"] = float(s["margin_mean"])
        stats["margin_std"]  = float(s["margin_std"])
    return stats


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


# ADDITION 1 — build_records_has extends the HAS DataFrame with margin columns
def build_records_has(custom, train_stats, class_names, drift_sigma, margin_sigma):
    """Per-image drift scores → DataFrame for the HAS model.

    Adds these columns on top of the standard build_records() columns:

      has_margin              (float) — per-sample angular margin
                                        = cos_best − cos_second_best
      has_margin_drifted      (bool)  — margin below training threshold
      closest_boundary        (int)   — index of second-best class
      closest_boundary_class  (str)   — human-readable name
      has_drift_type_margin   (str)   — ADDITION: geometrically grounded 4-way
                                        label using margin for concept drift
                                        instead of softmax confidence.

    has_drift_type_margin logic (the 2×2 table):
    ─────────────────────────────────────────────────────────────────
    data_drifted  margin_drifted  →  has_drift_type_margin
    ─────────────────────────────────────────────────────────────────
    False         False           →  "In-Distribution"
    True          False           →  "Pure Data Drift"
                                     Far from training cloud but well
                                     inside a class region — distribution
                                     shift without functional confusion.
    False         True            →  "Pure Concept Drift"
                                     Familiar-looking image but near a
                                     decision boundary — genuine ambiguity.
    True          True            →  "Full Drift (both)"
                                     Alien image AND near a boundary —
                                     the worst case.
    ─────────────────────────────────────────────────────────────────

    This is strictly more informative than the confidence-based label
    (drift_type) because:
      • margin is scale-independent (σ=10 does not inflate it)
      • margin directly measures HAS's training objective erosion
      • it distinguishes Pure Data Drift from Full Drift even when
        confidence stays high due to the scale factor
    """
    # Base records via the shared function
    df, data_thresh, concept_thresh = build_records(
        custom, train_stats, class_names, drift_sigma)

    # ── Margin threshold from saved training stats ──────────────────────────
    # Prefer stats loaded from .npz (saved by save_train_stats_has).
    # Fall back to computing from the raw training margins array if the .npz
    # was produced by an older extract.py that didn't save margin stats.
    if "margin_mean" in train_stats and "margin_std" in train_stats:
        tmean = train_stats["margin_mean"]
        tstd  = train_stats["margin_std"]
    else:
        # Legacy fallback — should not be needed after re-running extract.py
        import warnings
        warnings.warn("margin_mean not in train_stats — re-run extract.py")
        tmean = float(np.mean(custom["margins"]))   # approximate only
        tstd  = float(np.std(custom["margins"]))

    margin_thresh = tmean - margin_sigma * tstd

    margins = custom["margins"]           # (N,)
    cb      = custom["closest_boundary"]  # (N,) int

    df["has_margin"]             = margins.astype(float)
    df["has_margin_drifted"]     = margins < margin_thresh
    df["closest_boundary"]       = cb.astype(int)
    df["closest_boundary_class"] = [
        class_names[int(c)] if int(c) < len(class_names) else str(c)
        for c in cb
    ]

    # ── Geometrically grounded 4-way drift label ────────────────────────────
    def _margin_drift_type(row):
        d = row["data_drifted"]
        m = row["has_margin_drifted"]
        if   not d and not m: return "In-Distribution"
        elif     d and not m: return "Pure Data Drift"
        elif not d and     m: return "Pure Concept Drift"
        else:                 return "Full Drift (both)"

    df["has_drift_type_margin"] = df.apply(_margin_drift_type, axis=1)

    return df, data_thresh, concept_thresh, tmean, tstd, margin_thresh


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

    # ADDITION 1 — HAS margin drift tests
    print("=" * 62)
    print("HAS MODEL — MARGIN DRIFT TESTS (ADDITION 1)")
    print("=" * 62)
    train_margins  = has_train["margins"]
    custom_margins = has_custom["margins"]

    margin_result = has_margin_drift_test(train_margins, custom_margins)
    boundary_result = has_boundary_direction_test(
        has_train["closest_boundary"],
        has_custom["closest_boundary"],
        LANDSCAPE_CLASSES,
        n_classes=len(LANDSCAPE_CLASSES),
    )
    print(f"\n  {margin_result['test']:<30} "
          f"stat={margin_result['statistic']:.4f}  "
          f"p={margin_result['p_value']:.2e}  "
          f"drifted={'YES ⚠' if margin_result['drifted'] else 'no'}")
    print(f"    Train margin mean: {margin_result['train_margin_mean']:.4f}  "
          f"Custom margin mean: {margin_result['custom_margin_mean']:.4f}  "
          f"Drop: {margin_result['margin_drop']:.4f}")
    print(f"\n  {boundary_result['test']:<30} "
          f"stat={boundary_result['statistic']:.4f}  "
          f"p={boundary_result['p_value']:.2e}  "
          f"drifted={'YES ⚠' if boundary_result['drifted'] else 'no'}")
    print(f"    Dominant drift direction: {boundary_result['dominant_drift_direction']}")

    # ADDITION 1 — Drift direction matrix: true_class × closest_boundary_class
    # This is the key table that answers "Mountain → Glacier?" specifically.
    # Each row shows, for images of that true class, which class boundary they
    # approach most often.  Diagonal = self-boundary (well-classified).
    # Off-diagonal values > 15% with ◄ annotation = meaningful drift direction.
    print()
    dir_matrix_train, _ = has_drift_direction_matrix(
        has_train["labels"], has_train["closest_boundary"],
        LANDSCAPE_CLASSES, normalise=True)
    dir_matrix_custom, _ = has_drift_direction_matrix(
        has_custom["labels"], has_custom["closest_boundary"],
        LANDSCAPE_CLASSES, normalise=True)
    print_direction_matrix(dir_matrix_train,  LANDSCAPE_CLASSES,
                           "Drift Direction Matrix — HAS TRAINING SET")
    print_direction_matrix(dir_matrix_custom, LANDSCAPE_CLASSES,
                           "Drift Direction Matrix — HAS CUSTOM SET")

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
            # ADDITION 1 — include margin test in text report
            print("\nHAS MODEL — MARGIN DRIFT")
            print(f"  {margin_result['test']}: "
                  f"stat={margin_result['statistic']:.4f}, "
                  f"p={margin_result['p_value']:.2e}, "
                  f"drifted={margin_result['drifted']}")
            print(f"  Train margin mean: {margin_result['train_margin_mean']:.4f}, "
                  f"Custom margin mean: {margin_result['custom_margin_mean']:.4f}, "
                  f"Drop: {margin_result['margin_drop']:.4f}")
            print(f"  {boundary_result['test']}: "
                  f"stat={boundary_result['statistic']:.4f}, "
                  f"p={boundary_result['p_value']:.2e}, "
                  f"drifted={boundary_result['drifted']}")
            print(f"  Dominant drift direction: {boundary_result['dominant_drift_direction']}")
        f.write(buf.getvalue())
    print(f"  ✓ Text report → {rpt_path}")

    # ── Per-image drift DataFrames ──
    # Thresholds are per-model: mean ± drift_sigma * std from training
    df_bl, bl_data_th, bl_conf_th = build_records(
        bl_custom, bl_stats, LANDSCAPE_CLASSES, args.drift_sigma)

    # ADDITION 1 — HAS DataFrame: margin stats now loaded from train_stats .npz
    df_has, has_data_th, has_conf_th, tmean, tstd, margin_thresh = build_records_has(
        has_custom, has_stats, LANDSCAPE_CLASSES,
        args.drift_sigma, HAS_MARGIN_SIGMA)

    print(f"\n  Thresholds (σ={args.drift_sigma}):")
    print(f"    Baseline — data drift > {bl_data_th:.4f}, "
          f"concept drift < {bl_conf_th:.4f} confidence")
    print(f"    HAS      — data drift > {has_data_th:.4f}, "
          f"concept drift < {has_conf_th:.4f} confidence")
    print(f"    HAS margin < {margin_thresh:.4f} "
          f"(train μ={tmean:.4f}, σ={tstd:.4f}) → "
          f"{df_has['has_margin_drifted'].mean() * 100:.1f}% margin-drifted")

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
    # ADDITION 1
    md = df_has["has_margin_drifted"].mean() * 100
    print(f"  {'HAS':10s} | Margin-drifted {md:5.1f}%")

    # ── ADDITION 1 — Drift taxonomy comparison ──────────────────────────────
    # Side-by-side table of the three drift classification systems:
    #   (A) Baseline:   confidence-based 4-way label
    #   (B) HAS conf:   same formula applied to HAS (scale-distorted)
    #   (C) HAS margin: geometrically grounded 4-way label  ← the new one
    #
    # Key question: what does margin catch that confidence misses?
    # Specifically: Pure Data Drift cases where softmax confidence is high
    # (because σ=10 inflates it) but the angular margin has eroded — the
    # embedding is near a boundary despite high confidence.
    print("\n" + "=" * 62)
    print("DRIFT TAXONOMY COMPARISON")
    print("  (A) Baseline confidence  (B) HAS confidence  (C) HAS margin")
    print("=" * 62)

    def _pct(df_in, col, label):
        return (df_in[col] == label).mean() * 100

    print(f"\n  {'Category':<28} {'Baseline(A)':>12} {'HAS-conf(B)':>12} {'HAS-margin(C)':>14}")
    print("  " + "-" * 70)

    rows_cmp = [
        ("In-Distribution",       "In-Distribution",    "In-Distribution",    "In-Distribution"),
        ("Data Drift",             "Data Drift",          "Data Drift",         "Pure Data Drift"),
        ("Concept Drift",          "Concept Drift",       "Concept Drift",      "Pure Concept Drift"),
        ("Full Drift (both)",      "Full Drift (both)",   "Full Drift (both)",  "Full Drift (both)"),
    ]
    for label_display, col_bl, col_has_conf, col_has_margin in rows_cmp:
        a = _pct(df_bl,  "drift_type",           col_bl)
        b = _pct(df_has, "drift_type",            col_has_conf)
        c = _pct(df_has, "has_drift_type_margin", col_has_margin)
        print(f"  {label_display:<28} {a:>11.1f}% {b:>11.1f}% {c:>13.1f}%")

    # Highlight images that margin flags but confidence misses
    print()
    pure_data   = df_has["has_drift_type_margin"] == "Pure Data Drift"
    hidden_dd   = (df_has.loc[pure_data, "drift_type"] == "In-Distribution").sum()
    pure_conc   = df_has["has_drift_type_margin"] == "Pure Concept Drift"
    hidden_cd   = (df_has.loc[pure_conc, "drift_type"] == "In-Distribution").sum()

    if hidden_dd > 0:
        print(f"  ► {hidden_dd} images ({hidden_dd/len(df_has)*100:.1f}%) are Pure Data Drift by margin")
        print(f"    but In-Distribution by confidence — σ={10} scale was hiding them.")
    if hidden_cd > 0:
        print(f"  ► {hidden_cd} images ({hidden_cd/len(df_has)*100:.1f}%) are Pure Concept Drift by margin")
        print(f"    but In-Distribution by confidence — boundary erosion invisible in conf.")
    if hidden_dd == 0 and hidden_cd == 0:
        print(f"  ► Margin and confidence agree on all samples (model well-trained or")
        print(f"    margin threshold negative — check HAS training convergence).")

    # ═══════════════════════════════════════════════════════════════════════
    # ADDITION 2 — Hierarchical drift analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 62)
    print("HIERARCHICAL DRIFT ANALYSIS (ADDITION 2)")
    print("=" * 62)

    def _build_key_dicts(custom_data):
        """Group latents and confs by 'ClassName/subname' key."""
        lat_by_key  = {}
        conf_by_key = {}
        labels = custom_data["labels"]
        subs   = custom_data["subs"]
        lats   = custom_data["latents"]
        confs  = custom_data["confs"]
        for i in range(len(lats)):
            lbl = int(labels[i])
            cls = LANDSCAPE_CLASSES[lbl] if lbl < len(LANDSCAPE_CLASSES) else str(lbl)
            sub = str(subs[i])
            key = f"{cls}/{sub}"
            if key not in lat_by_key:
                lat_by_key[key]  = []
                conf_by_key[key] = []
            lat_by_key[key].append(lats[i])
            conf_by_key[key].append(confs[i])
        lat_by_key  = {k: np.array(v) for k, v in lat_by_key.items()}
        conf_by_key = {k: np.array(v) for k, v in conf_by_key.items()}
        return lat_by_key, conf_by_key

    bl_lat_by_key,  bl_conf_by_key  = _build_key_dicts(bl_custom)
    has_lat_by_key, has_conf_by_key = _build_key_dicts(has_custom)

    hier_bl = hierarchical_drift_report(
        bl_lat_by_key, bl_conf_by_key,
        bl_train["latents"], bl_train["confs"],
        sigma=args.drift_sigma,
        class_names=LANDSCAPE_CLASSES,
    )
    hier_has = hierarchical_drift_report(
        has_lat_by_key, has_conf_by_key,
        has_train["latents"], has_train["confs"],
        sigma=args.drift_sigma,
        class_names=LANDSCAPE_CLASSES,
    )

    # Print formatted table
    def _print_hier(hier, model_tag):
        print(f"\n  {model_tag}")
        header = f"  {'Class':<12} {'Drifted':>8} {'SubFrac':>9}  Drifted subclasses"
        print(header)
        print("  " + "-" * 60)
        for cls, info in hier.items():
            drifted_subs = [s for s, d in info["subclasses"].items()
                            if d["data_drifted"] or d["concept_drifted"]]
            flag = "YES ⚠" if info["class_drifted"] else "no"
            frac = f"{info['subclass_fraction_drifted']:.0%}"
            subs_str = ", ".join(drifted_subs[:3])
            if len(drifted_subs) > 3:
                subs_str += f" (+{len(drifted_subs)-3} more)"
            print(f"  {cls:<12} {flag:>8} {frac:>9}  {subs_str}")

    _print_hier(hier_bl,  "Baseline")
    _print_hier(hier_has, "HAS Model")

    # Save JSON
    hier_out = {"baseline": hier_bl, "has": hier_has}
    json_path = os.path.join(RESULT_DIR, "hierarchical_drift.json")

    # ADDITION 1 — also save direction matrices for visualize.py
    dir_matrix_custom_raw, _ = has_drift_direction_matrix(
        has_custom["labels"], has_custom["closest_boundary"],
        LANDSCAPE_CLASSES, normalise=True)
    hier_out["has_direction_matrix"] = dir_matrix_custom_raw.tolist()

    # Convert numpy bools/floats to Python natives for JSON serialisation
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(_jsonify(hier_out), f, indent=2)
    print(f"\n  ✓ Hierarchical results → {json_path}")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
