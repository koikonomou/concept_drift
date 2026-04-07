import os, sys, json, argparse, io, contextlib
import numpy as np
import pandas as pd
import warnings
from config import (WEIGHT_DIR, FEATURE_DIR, RESULT_DIR,
                    DRIFT_SIGMA, HAS_MARGIN_SIGMA, LANDSCAPE_CLASSES,
                    ensure_dirs)
from drift_stats import (
    mmd_rbf, confidence_drift_test,
    has_margin_drift_test, has_boundary_direction_test,
    has_drift_direction_matrix, hierarchical_drift_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_features(name):
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run extract.py first.")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_train_stats(tag):
    path = os.path.join(WEIGHT_DIR, f"{tag}_train_stats.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run extract.py first.")
    s = np.load(path)
    stats = dict(centroid  = s["centroid"],
                 dist_mean = float(s["dist_mean"]),
                 dist_std  = float(s["dist_std"]),
                 conf_mean = float(s["conf_mean"]),
                 conf_std  = float(s["conf_std"]))
    if "margin_mean" in s:
        stats["margin_mean"] = float(s["margin_mean"])
        stats["margin_std"]  = float(s["margin_std"])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Per-image DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_df(custom, train_stats, class_names, drift_sigma):
    """Per-image drift DataFrame for the Baseline model.

    Data drift  : latent distance from training centroid > threshold
    Concept drift: softmax confidence < threshold
    Both measured by the same 2-sigma rule from training statistics.
    """
    centroid      = train_stats["centroid"]
    data_thresh   = train_stats["dist_mean"] + drift_sigma * train_stats["dist_std"]
    concept_thresh = max(
        train_stats["conf_mean"] - drift_sigma * train_stats["conf_std"], 0.0)

    rows = []
    for i in range(len(custom["latents"])):
        dd   = float(np.linalg.norm(custom["latents"][i] - centroid))
        conf = float(custom["confs"][i])
        d = dd   > data_thresh
        c = conf < concept_thresh
        lbl = int(custom["labels"][i])
        rows.append(dict(
            sub_category       = str(custom["subs"][i]),
            true_class         = class_names[lbl] if lbl < len(class_names) else str(lbl),
            pred_label         = int(custom["preds"][i]),
            correct            = lbl == int(custom["preds"][i]),
            max_confidence     = round(conf, 4),
            data_drift_score   = round(dd, 4),
            concept_drift_score= round(1.0 - conf, 4),
            data_drifted       = d,
            concept_drifted    = c,
            drift_type         = ("In-Distribution" if not d and not c else
                                  "Data Drift"       if d and not c else
                                  "Concept Drift"    if not d and c else
                                  "Full Drift (both)"),
        ))
    return pd.DataFrame(rows), data_thresh, concept_thresh


def build_has_df(custom, train_stats, class_names, drift_sigma, margin_sigma):
    """Per-image drift DataFrame for the HAS model.

    Data drift   : same centroid-distance rule as Baseline.
    Concept drift: angular margin < training threshold (NOT softmax confidence).
                   Margin = cos_best − cos_second_best.
                   This is scale-independent and directly measures HAS objective erosion.

    Pure Data Drift  (data=True,  margin=False):
        The embedding is far from the training cloud spatially, but still
        well inside a class region — no boundary confusion.  The model is
        in unfamiliar territory but not confused.

    Pure Concept Drift (data=False, margin=True):
        The embedding is geometrically familiar but near a decision boundary.
        predicted_drift_toward = the class it is drifting toward.

    Full Drift (both): spatial AND boundary confusion — worst case.

    Extra columns vs Baseline:
        has_margin             angular margin score
        has_margin_drifted     bool — margin below training threshold
        closest_boundary_class the runner-up class (drift destination)
        has_drift_type_margin  the geometrically grounded 4-way label
        predicted_drift_toward closest_boundary_class if concept-drifted,
                               else "—" — directly answers "toward which class?"
    """
    df, data_thresh, concept_thresh = build_baseline_df(
        custom, train_stats, class_names, drift_sigma)

    if "margin_mean" in train_stats and "margin_std" in train_stats:
        tm, ts = train_stats["margin_mean"], train_stats["margin_std"]
    else:
        warnings.warn("margin_mean missing — re-run extract.py")
        tm = float(np.mean(custom["margins"]))
        ts = float(np.std(custom["margins"]))

    margin_thresh = tm - margin_sigma * ts
    margins = custom["margins"]
    cb      = custom["closest_boundary"]

    df["has_margin"]             = np.round(margins.astype(float), 4)
    df["has_margin_drifted"]     = margins < margin_thresh
    df["closest_boundary_class"] = [
        class_names[int(c)] if int(c) < len(class_names) else str(c)
        for c in cb
    ]

    def _type(row):
        d, m = row["data_drifted"], row["has_margin_drifted"]
        if   not d and not m: return "In-Distribution"
        elif     d and not m: return "Pure Data Drift"
        elif not d and     m: return "Pure Concept Drift"
        else:                 return "Full Drift (both)"

    df["has_drift_type_margin"] = df.apply(_type, axis=1)

    # predicted_drift_toward: only meaningful when concept-drifted (near a boundary)
    concept_mask = df["has_margin_drifted"]
    df["predicted_drift_toward"] = "—"
    df.loc[concept_mask, "predicted_drift_toward"] = \
        df.loc[concept_mask, "closest_boundary_class"]

    return df, data_thresh, concept_thresh, tm, ts, margin_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(n=62): print("=" * n)

def _build_key_dicts(custom_data, class_names):
    lat, conf = {}, {}
    for i in range(len(custom_data["latents"])):
        lbl = int(custom_data["labels"][i])
        cls = class_names[lbl] if lbl < len(class_names) else str(lbl)
        key = f"{cls}/{custom_data['subs'][i]}"
        lat.setdefault(key, []).append(custom_data["latents"][i])
        conf.setdefault(key, []).append(custom_data["confs"][i])
    return ({k: np.array(v) for k, v in lat.items()},
            {k: np.array(v) for k, v in conf.items()})


def _jsonify(obj):
    if isinstance(obj, dict):       return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):       return [_jsonify(v) for v in obj]
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj


def save_csv(df, name, description=""):
    path = os.path.join(RESULT_DIR, name)
    df.to_csv(path, index=False)
    n = len(df)
    print(f"  ✓ {name}  ({n} rows{', ' + description if description else ''})")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-sigma", type=float, default=DRIFT_SIGMA)
    parser.add_argument("--mmd-perms",   type=int,   default=300)
    args = parser.parse_args()
    ensure_dirs()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading features …")
    bl_train  = load_features("bl_train")
    bl_custom = load_features("bl_custom")
    ht        = load_features("has_train")
    hc        = load_features("has_custom")
    bl_stats  = load_train_stats("baseline")
    has_stats = load_train_stats("has")
    with open(os.path.join(FEATURE_DIR, "meta.json")) as f:
        meta = json.load(f)
    n_custom = len(bl_custom["latents"])
    print(f"  Train: {len(bl_train['latents'])} | Custom: {n_custom}")
    print(f"  Baseline conf:  μ={bl_stats['conf_mean']:.4f}  σ={bl_stats['conf_std']:.4f}")
    print(f"  HAS      conf:  μ={has_stats['conf_mean']:.4f}  σ={has_stats['conf_std']:.4f}")
    if "margin_mean" in has_stats:
        print(f"  HAS    margin:  μ={has_stats['margin_mean']:.4f}  σ={has_stats['margin_std']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 1 — Population-level statistical tests
    # ══════════════════════════════════════════════════════════════════════════
    _sep(); print("TABLE 1 — POPULATION-LEVEL DRIFT TESTS"); _sep()

    bl_mmd   = mmd_rbf(bl_train["latents"], bl_custom["latents"], n_perm=args.mmd_perms)
    bl_conf  = confidence_drift_test(bl_train["confs"], bl_custom["confs"])
    has_mmd  = mmd_rbf(ht["latents"], hc["latents"], n_perm=args.mmd_perms)
    has_conf = confidence_drift_test(ht["confs"], hc["confs"])
    has_mar  = has_margin_drift_test(ht["margins"], hc["margins"])
    has_dir  = has_boundary_direction_test(
        ht["closest_boundary"], hc["closest_boundary"],
        LANDSCAPE_CLASSES, n_classes=len(LANDSCAPE_CLASSES))

    rows_t1 = []
    print(f"\n  {'Test':<28} {'Model':<10} {'Statistic':>10} {'p-value':>10} {'Drifted?':>9}")
    print("  " + "-" * 72)
    for r, model in [(bl_mmd,  "Baseline"), (has_mmd,  "HAS"),
                     (bl_conf, "Baseline"), (has_conf, "HAS"),
                     (has_mar, "HAS"),      (has_dir,  "HAS")]:
        flag = "YES" if r["drifted"] else "no"
        print(f"  {r['test']:<28} {model:<10} "
              f"{r['statistic']:>10.4f} {r['p_value']:>10.2e} {flag:>9}")
        rows_t1.append(dict(test=r["test"], model=model,
                            statistic=r["statistic"], p_value=r["p_value"],
                            drifted=r["drifted"]))

    print(f"\n  HAS margin: train μ={has_mar['train_margin_mean']:.4f}  "
          f"custom μ={has_mar['custom_margin_mean']:.4f}  "
          f"drop={has_mar['margin_drop']:.4f}")
    print(f"  Dominant drift direction: {has_dir['dominant_drift_direction']}")
    save_csv(pd.DataFrame(rows_t1), "population_tests.csv")

    # ── Per-image DataFrames ──────────────────────────────────────────────────
    df_bl, bl_dth, bl_cth = build_baseline_df(
        bl_custom, bl_stats, LANDSCAPE_CLASSES, args.drift_sigma)

    df_has, has_dth, has_cth, tmean, tstd, mth = build_has_df(
        hc, has_stats, LANDSCAPE_CLASSES, args.drift_sigma, HAS_MARGIN_SIGMA)

    print(f"\n  Thresholds (σ={args.drift_sigma}):")
    print(f"    Baseline — data > {bl_dth:.4f}  |  conf < {bl_cth:.4f}")
    print(f"    HAS      — data > {has_dth:.4f}  |  margin < {mth:.4f} "
          f"(μ={tmean:.4f}, σ={tstd:.4f})")
    print(f"    HAS margin-drifted: {df_has['has_margin_drifted'].mean()*100:.1f}%")

    save_csv(df_bl,  "drift_baseline.csv", "per-image Baseline")
    save_csv(df_has, "drift_has.csv",      "per-image HAS")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 2 — Drift taxonomy comparison
    # Three columns: Baseline-conf | HAS-conf | HAS-margin (proposed)
    # ══════════════════════════════════════════════════════════════════════════
    _sep(); print("TABLE 2 — DRIFT TAXONOMY COMPARISON"); _sep()
    print("  (A) Baseline-confidence  (B) HAS-confidence  (C) HAS-margin (proposed)")

    def pct(df, col, lbl):
        return (df[col] == lbl).mean() * 100

    rows_t2, rows_t2_print = [], [
        ("In-Distribution",   "In-Distribution",   "In-Distribution",   "In-Distribution"),
        ("Data Drift",        "Data Drift",         "Data Drift",        "Pure Data Drift"),
        ("Concept Drift",     "Concept Drift",      "Concept Drift",     "Pure Concept Drift"),
        ("Full Drift (both)", "Full Drift (both)",  "Full Drift (both)", "Full Drift (both)"),
    ]
    print(f"\n  {'Category':<26} {'Baseline(A)':>12} {'HAS-conf(B)':>12} {'HAS-margin(C)':>14}")
    print("  " + "-" * 67)
    for display, ca, cb_, cc in rows_t2_print:
        a = pct(df_bl,  "drift_type",           ca)
        b = pct(df_has, "drift_type",            cb_)
        c = pct(df_has, "has_drift_type_margin", cc)
        print(f"  {display:<26} {a:>11.1f}% {b:>11.1f}% {c:>13.1f}%")
        rows_t2.append(dict(category=display,
                            baseline_conf_pct=round(a,1),
                            has_conf_pct=round(b,1),
                            has_margin_pct=round(c,1)))

    # Hidden drift: what margin catches that confidence misses
    print()
    for mtype, label in [
        ("Pure Concept Drift", "Pure Concept Drift by margin — hidden from confidence"),
        ("Pure Data Drift",    "Pure Data Drift by margin   — hidden from confidence"),
    ]:
        mask   = df_has["has_drift_type_margin"] == mtype
        hidden = (df_has.loc[mask, "drift_type"] == "In-Distribution").sum()
        if hidden > 0:
            print(f"  ► {hidden} images ({hidden/n_custom*100:.1f}%): {label}")

    save_csv(pd.DataFrame(rows_t2), "taxonomy_comparison.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 3 — Concept drift direction (HAS only)
    # For each true class: how many concept-drifted samples go toward each boundary
    # This directly answers "concept drift toward which class?"
    # ══════════════════════════════════════════════════════════════════════════
    _sep(); print("TABLE 3 — CONCEPT DRIFT DIRECTION (HAS)"); _sep()
    print("  Rows = true class  |  Columns = predicted drift destination")
    print("  Only samples flagged as concept-drifted (has_margin_drifted=True)\n")

    concept_df = df_has[df_has["has_margin_drifted"]].copy()
    direction_rows = []

    if len(concept_df) == 0:
        print("  No concept-drifted samples (margin threshold may be negative).")
        print("  Re-train with more epochs to get meaningful margins.\n")
    else:
        # Cross-tabulation: true_class × predicted_drift_toward
        xtab = pd.crosstab(
            concept_df["true_class"],
            concept_df["predicted_drift_toward"],
            normalize="index"
        ) * 100

        # Print
        dest_classes = xtab.columns.tolist()
        header = f"  {'True class':<14}" + "".join(f"{c:>12}" for c in dest_classes)
        print(header)
        print("  " + "-" * len(header.rstrip()))
        for true_cls in LANDSCAPE_CLASSES:
            if true_cls not in xtab.index:
                continue
            row = xtab.loc[true_cls]
            dominant = row.idxmax()
            line = f"  {true_cls:<14}" + "".join(f"{row.get(c,0):>11.1f}%" for c in dest_classes)
            line += f"  → {dominant} ({row[dominant]:.1f}%)"
            print(line)
            for dest in dest_classes:
                direction_rows.append(dict(
                    true_class=true_cls,
                    predicted_drift_toward=dest,
                    pct_of_concept_drifted=round(row.get(dest, 0), 1),
                ))
        print()
        # Summary: most common direction per true class
        print("  Most common concept drift direction per class:")
        for true_cls in LANDSCAPE_CLASSES:
            if true_cls not in xtab.index:
                continue
            row = xtab.loc[true_cls]
            dom = row.idxmax()
            n = int((concept_df["true_class"] == true_cls).sum())
            print(f"    {true_cls:<12} → {dom:<12} ({row[dom]:.1f}% of {n} drifted samples)")
        print()

    save_csv(pd.DataFrame(direction_rows), "concept_drift_direction.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 4 — Direction matrices (training vs custom)
    # ══════════════════════════════════════════════════════════════════════════
    _sep(); print("TABLE 4 — HAS DRIFT DIRECTION MATRICES"); _sep()

    dir_train,  _ = has_drift_direction_matrix(
        ht["labels"], ht["closest_boundary"], LANDSCAPE_CLASSES)
    dir_custom, _ = has_drift_direction_matrix(
        hc["labels"], hc["closest_boundary"], LANDSCAPE_CLASSES)

    for mat, label in [(dir_train, "Training Set"), (dir_custom, "Custom Set")]:
        n = len(LANDSCAPE_CLASSES)
        print(f"\n  {label}  (row=origin, col=destination, diagonal=self)")
        header = f"  {'':14}" + "".join(f"{c:>12}" for c in LANDSCAPE_CLASSES)
        print(header)
        print("  " + "-" * len(header.rstrip()))
        for i, rn in enumerate(LANDSCAPE_CLASSES):
            row = f"  {rn:<14}" + "".join(
                f"({'diag':>9})" if i == j else f"{mat[i,j]:>11.1%} "
                for j in range(n)
            )
            off = mat[i].copy(); off[i] = 0
            dj = int(off.argmax())
            if off[dj] > 0.15:
                row += f"  → {LANDSCAPE_CLASSES[dj]} ({off[dj]:.1%})"
            print(row)
    print()

    # Save both matrices as CSVs
    for mat, name in [(dir_train, "direction_matrix_train.csv"),
                      (dir_custom, "direction_matrix_custom.csv")]:
        df_mat = pd.DataFrame(mat, index=LANDSCAPE_CLASSES, columns=LANDSCAPE_CLASSES)
        df_mat.index.name = "true_class"
        save_csv(df_mat.reset_index(), name)

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE 5 — Hierarchical drift (class / subclass)
    # ══════════════════════════════════════════════════════════════════════════
    _sep(); print("TABLE 5 — HIERARCHICAL DRIFT ANALYSIS"); _sep()

    bl_lk, bl_ck   = _build_key_dicts(bl_custom, LANDSCAPE_CLASSES)
    has_lk, has_ck = _build_key_dicts(hc,        LANDSCAPE_CLASSES)

    hier_bl  = hierarchical_drift_report(bl_lk,  bl_ck,
                                          bl_train["latents"], bl_train["confs"],
                                          LANDSCAPE_CLASSES)
    hier_has = hierarchical_drift_report(has_lk, has_ck,
                                          ht["latents"], ht["confs"],
                                          LANDSCAPE_CLASSES)

    hier_csv_rows = []
    for model_tag, hier in [("Baseline", hier_bl), ("HAS", hier_has)]:
        print(f"\n  {model_tag}")
        print(f"  {'Class':<14} {'Drifted':>8} {'SubFrac':>9}  Drifted subclasses")
        print("  " + "-" * 60)
        for cls, info in hier.items():
            drifted_subs = [s for s, d in info["subclasses"].items()
                            if d["data_drifted"] or d["concept_drifted"]]
            flag = "YES" if info["class_drifted"] else "no"
            frac = f"{info['subclass_fraction_drifted']:.0%}"
            sstr = ", ".join(drifted_subs[:3])
            if len(drifted_subs) > 3:
                sstr += f" (+{len(drifted_subs)-3})"
            print(f"  {cls:<14} {flag:>8} {frac:>9}  {sstr}")
            # Subclass rows
            for sub, si in info["subclasses"].items():
                hier_csv_rows.append(dict(
                    model=model_tag, class_name=cls, subclass=sub,
                    data_drifted=si["data_drifted"],
                    concept_drifted=si["concept_drifted"],
                    n_samples=si["n_samples"],
                    ks_stat=round(si["ks_stat"], 4),
                    ks_p=round(si["ks_p"], 4),
                    conf_mean=round(si["conf_mean"], 4),
                    class_drifted=info["class_drifted"],
                    subclass_fraction_drifted=round(info["subclass_fraction_drifted"], 2),
                ))

    # Agreement between models
    bl_drifted  = {c for c, v in hier_bl.items()  if v["class_drifted"]}
    has_drifted = {c for c, v in hier_has.items() if v["class_drifted"]}
    only_bl  = bl_drifted  - has_drifted
    only_has = has_drifted - bl_drifted
    print()
    if only_bl:  print(f"  Baseline-only flags : {sorted(only_bl)}")
    if only_has: print(f"  HAS-only flags      : {sorted(only_has)}")
    if not only_bl and not only_has:
        print(f"  Both models agree: {sorted(bl_drifted)}")

    # Split into two CSVs (one per model) for clarity
    df_hier = pd.DataFrame(hier_csv_rows)
    save_csv(df_hier[df_hier["model"] == "Baseline"].drop("model", axis=1),
             "hierarchical_baseline.csv")
    save_csv(df_hier[df_hier["model"] == "HAS"].drop("model", axis=1),
             "hierarchical_has.csv")

    print("\ndetect.py complete.")


if __name__ == "__main__":
    main()
