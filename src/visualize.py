import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FEATURE_DIR, RESULT_DIR, LANDSCAPE_CLASSES, ensure_dirs

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.2,
})

CLASS_COLORS = {
    "Coast":    "#0077b6", "Desert":  "#e76f51",
    "Forest":   "#2d6a4f", "Glacier": "#48cae4",
    "Mountain": "#7b2cbf",
}

DRIFT_COLORS = {
    "In-Distribution":    "#2ecc71",
    "Pure Data Drift":    "#3498db",
    "Data Drift":         "#3498db",
    "Pure Concept Drift": "#e74c3c",
    "Concept Drift":      "#e74c3c",
    "Full Drift (both)":  "#8e44ad",
}


def _load_csv(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run detect.py first.")
    return pd.read_csv(path)


def _save(fig, name):
    path = os.path.join(RESULT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Drift taxonomy comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_taxonomy(df_bl, df_has):
    """Grouped bar chart comparing three detection systems.

    Each group of 3 bars = one drift category.
    Bar 1 (light)  = Baseline confidence
    Bar 2 (mid)    = HAS confidence (same formula, different model)
    Bar 3 (hatched)= HAS margin — the proposed method

    The gap between bar 2 and bar 3 is the key result: what the geometric
    margin catches that softmax confidence misses.
    """
    categories = [
        ("In-Distribution",   "In-Distribution",   "In-Distribution",   "In-Distribution"),
        ("Data Drift",        "Data Drift",         "Data Drift",        "Pure Data Drift"),
        ("Concept Drift",     "Concept Drift",      "Concept Drift",     "Pure Concept Drift"),
        ("Full Drift (both)", "Full Drift (both)",  "Full Drift (both)", "Full Drift (both)"),
    ]

    def pct(df, col, lbl):
        return (df[col] == lbl).mean() * 100

    x = np.arange(len(categories))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 5))

    bar_colors = ["#2ecc71", "#3498db", "#e74c3c", "#8e44ad"]

    for i, (lbl, ca, cb, cc) in enumerate(categories):
        a = pct(df_bl,  "drift_type",           ca)
        b = pct(df_has, "drift_type",            cb)
        c = pct(df_has, "has_drift_type_margin", cc)
        col = bar_colors[i]
        kw  = dict(edgecolor="white", linewidth=0.8)
        b1 = ax.bar(x[i] - w, a, w, color=col, alpha=0.45, **kw)
        b2 = ax.bar(x[i],     b, w, color=col, alpha=0.75, **kw)
        b3 = ax.bar(x[i] + w, c, w, color=col, alpha=1.00,
                    hatch="///", **kw)
        for bar, val in [(b1,a),(b2,b),(b3,c)]:
            if val > 0.8:
                ax.text(bar[0].get_x() + bar[0].get_width()/2,
                        val + 0.5, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in categories], fontsize=10)
    ax.set_ylabel("% of custom images")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("Figure 1 — Drift Taxonomy Comparison\n"
                 "Light = Baseline (confidence)  |  Mid = HAS (confidence)  "
                 "|  Dark hatched = HAS (margin) — proposed",
                 fontweight="bold")

    from matplotlib.patches import Patch
    legend = [
        Patch(fc="#888", alpha=0.45, label="Baseline (confidence)"),
        Patch(fc="#888", alpha=0.75, label="HAS (confidence)"),
        Patch(fc="#888", alpha=1.00, hatch="///",
              label="HAS (angular margin) — proposed"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, "fig1_drift_taxonomy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — HAS geometric scatter (the 4-quadrant view)
# ─────────────────────────────────────────────────────────────────────────────

def plot_has_geometry(df_has, data_thresh, margin_thresh):
    """Scatter: data_drift_score × has_margin, coloured by has_drift_type_margin.

    Threshold lines create 4 quadrants:
      Top-left    = In-Distribution     (low distance, high margin)
      Top-right   = Pure Data Drift     (high distance, high margin)
      Bottom-left = Pure Concept Drift  (low distance, low margin)
      Bottom-right= Full Drift (both)   (high distance, low margin)

    This is the core geometric argument of the paper: data drift and concept
    drift are orthogonal dimensions in the HAS feature space.  Softmax
    confidence conflates them; margin and centroid distance separate them.
    """
    if "has_margin" not in df_has.columns or "data_drift_score" not in df_has.columns:
        print("  ⚠ Skipping Fig 2 — has_margin or data_drift_score missing from df_has.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for dtype, color in [
        ("In-Distribution",    "#2ecc71"),
        ("Pure Data Drift",    "#3498db"),
        ("Pure Concept Drift", "#e74c3c"),
        ("Full Drift (both)",  "#8e44ad"),
    ]:
        mask = df_has["has_drift_type_margin"] == dtype
        if mask.any():
            ax.scatter(
                df_has.loc[mask, "data_drift_score"],
                df_has.loc[mask, "has_margin"],
                c=color, label=dtype, alpha=0.55, s=18,
                edgecolors="none", rasterized=True
            )

    # Threshold lines
    ax.axvline(data_thresh,   color="#333", linestyle="--", lw=1.2,
               label=f"Data drift threshold ({data_thresh:.3f})")
    ax.axhline(margin_thresh, color="#c0392b", linestyle="--", lw=1.2,
               label=f"Margin threshold ({margin_thresh:.3f})")

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xmid = (xlim[0] + data_thresh) / 2
    xmid2 = (data_thresh + xlim[1]) / 2
    ymid_hi = (margin_thresh + ylim[1]) / 2
    ymid_lo = (ylim[0] + margin_thresh) / 2

    for xp, yp, label, col in [
        (xmid,  ymid_hi, "In-Distribution",    "#27ae60"),
        (xmid2, ymid_hi, "Pure\nData Drift",   "#2980b9"),
        (xmid,  ymid_lo, "Pure\nConcept Drift","#c0392b"),
        (xmid2, ymid_lo, "Full Drift\n(both)", "#7d3c98"),
    ]:
        ax.text(xp, yp, label, ha="center", va="center",
                fontsize=8, color=col, alpha=0.6,
                fontweight="bold")

    ax.set_xlabel("Data Drift Score  (latent distance from training centroid)")
    ax.set_ylabel("HAS Angular Margin  (cos_best − cos_second_best)\n"
                  "← More concept-drifted       Less concept-drifted →",
                  labelpad=8)
    ax.set_title("Figure 2 — HAS Geometric View: Data Drift vs Concept Drift\n"
                 "Each point = one custom image. "
                 "Threshold lines define the 4-quadrant taxonomy.",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    _save(fig, "fig2_has_geometry.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Concept drift direction per class
# ─────────────────────────────────────────────────────────────────────────────

def plot_concept_drift_direction(df_has):
    """For each true class, show where concept-drifted samples are heading.

    Only includes samples flagged as concept-drifted by the margin signal
    (has_margin_drifted=True).  X-axis = true class, stacked/grouped bars
    = destination class (predicted_drift_toward).

    Directly answers: "For Mountain images that are concept-drifted,
    what fraction are heading toward Glacier vs Forest vs Coast?"
    """
    concept_df = df_has[df_has["has_margin_drifted"]].copy()

    if len(concept_df) == 0:
        print("  ⚠ Skipping Fig 3 — no concept-drifted samples.")
        print("    Re-train with more epochs for meaningful margin signal.")
        return

    # Cross-tab: true_class × predicted_drift_toward (fraction)
    xtab = pd.crosstab(
        concept_df["true_class"],
        concept_df["predicted_drift_toward"],
        normalize="index"
    ) * 100

    # Keep only classes that appear
    present_true  = [c for c in LANDSCAPE_CLASSES if c in xtab.index]
    present_dest  = [c for c in LANDSCAPE_CLASSES if c in xtab.columns]

    if not present_true or not present_dest:
        print("  ⚠ Skipping Fig 3 — insufficient data for direction plot.")
        return

    xtab = xtab.reindex(index=present_true, columns=present_dest, fill_value=0)

    x   = np.arange(len(present_true))
    w   = 0.8 / len(present_dest)
    fig, ax = plt.subplots(figsize=(10, 6))

    for j, dest in enumerate(present_dest):
        vals = [xtab.loc[tc, dest] if tc in xtab.index else 0
                for tc in present_true]
        bars = ax.bar(x + (j - len(present_dest)/2 + 0.5) * w,
                      vals, w,
                      color=CLASS_COLORS.get(dest, "#999"),
                      alpha=0.85, label=dest,
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + 0.5, f"{val:.0f}%",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(present_true, fontsize=10)
    ax.set_xlabel("True Class (origin of concept-drifted samples)")
    ax.set_ylabel("% of concept-drifted samples per true class\n"
                  "heading toward each boundary")
    ax.set_title("Figure 3 — Concept Drift Direction per Class (HAS)\n"
                 "Bar colour = destination boundary class  "
                 "|  Only margin-drifted samples included",
                 fontweight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.legend(title="Predicted drift toward", fontsize=9,
              loc="upper right", title_fontsize=9)

    # Annotate counts
    counts = concept_df["true_class"].value_counts()
    for i, tc in enumerate(present_true):
        n = counts.get(tc, 0)
        ax.text(x[i], -4.5, f"n={n}", ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    _save(fig, "fig3_concept_drift_direction.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    ensure_dirs()

    print("Loading CSVs …")
    df_bl  = _load_csv("drift_baseline.csv")
    df_has = _load_csv("drift_has.csv")

    # Check that detect.py was run with the updated version
    if "has_drift_type_margin" not in df_has.columns:
        sys.exit("ERROR: has_drift_type_margin column missing. "
                 "Re-run detect.py with the updated version.")

    # Load thresholds from population tests CSV to draw lines on Fig 2
    pop_df = _load_csv("population_tests.csv")
    # We need data_thresh and margin_thresh — these aren't in population_tests,
    # so we recompute from the df itself as approximate reference lines.
    data_thresh   = df_has.loc[df_has["data_drifted"],   "data_drift_score"].min() \
                    if df_has["data_drifted"].any() else df_has["data_drift_score"].quantile(0.90)
    margin_thresh = df_has.loc[df_has["has_margin_drifted"], "has_margin"].max() \
                    if df_has["has_margin_drifted"].any() else df_has["has_margin"].quantile(0.10)

    print("\nFigure 1: drift taxonomy comparison …")
    plot_taxonomy(df_bl, df_has)

    print("Figure 2: HAS geometric scatter (4-quadrant view) …")
    plot_has_geometry(df_has, data_thresh, margin_thresh)

    print("Figure 3: concept drift direction per class …")
    plot_concept_drift_direction(df_has)

    print("\nOutputs:")
    for f in ["fig1_drift_taxonomy.png",
              "fig2_has_geometry.png",
              "fig3_concept_drift_direction.png"]:
        print(f"  results/{f}")


if __name__ == "__main__":
    main()
