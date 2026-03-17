"""
Generate all plots from saved features + CSV.

No GPU needed. Reads .npz from step2 and CSVs from step3.

Usage:
    python step4_visualize.py
    python step4_visualize.py --skip-umap    # faster, skip UMAP computation

Outputs:
    results/umap_baseline.png
    results/umap_has.png
    results/umap_side_by_side.png
    results/drift_dashboard_baseline.png
    results/drift_dashboard_has.png
    results/per_feature_ks_baseline.png
    results/per_feature_ks_has.png
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from config import FEATURE_DIR, RESULT_DIR, LANDSCAPE_CLASSES, ensure_dirs
from drift_stats import per_feature_ks


# ─────────────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#17becf", "#9467bd",
    "#d62728", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

DRIFT_PALETTE = {
    "In-Distribution":   "#2ecc71",
    "Data Drift":        "#3498db",
    "Concept Drift":     "#e74c3c",
    "Full Drift (both)": "#8e44ad",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(name):
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run step2_extract.py first.")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_csv(name):
    path = os.path.join(RESULT_DIR, f"{name}.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run step3_detect.py first.")
    return pd.read_csv(path)


def load_meta():
    path = os.path.join(FEATURE_DIR, "meta.json")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_umap(ref_latents, cur_latents, n_neighbors=15, min_dist=0.1):
    try:
        import umap
    except ImportError:
        sys.exit("Install umap-learn:  pip install umap-learn")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42, metric="euclidean")
    combined = np.vstack([ref_latents, cur_latents])
    emb = reducer.fit_transform(combined)
    n = len(ref_latents)
    return emb[:n], emb[n:]


def plot_umap_panel(ax, emb_ref, emb_cur, ref_labels, drift_labels, class_names):
    for i, cls in enumerate(class_names):
        mask = ref_labels == i
        if mask.any():
            c = CLASS_COLORS[i % len(CLASS_COLORS)]
            ax.scatter(emb_ref[mask, 0], emb_ref[mask, 1],
                       c=c, alpha=0.2, s=12, label=f"Train: {cls}")

    for dtype, colour in DRIFT_PALETTE.items():
        mask = drift_labels == dtype
        if mask.any():
            ax.scatter(emb_cur[mask, 0], emb_cur[mask, 1],
                       c=colour, marker="x", s=35, linewidths=1.0,
                       alpha=0.8, label=dtype)
    ax.set_xticks([]); ax.set_yticks([])


def plot_single_umap(emb_ref, emb_cur, ref_labels, drift_labels,
                     class_names, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_umap_panel(ax, emb_ref, emb_cur, ref_labels, drift_labels, class_names)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ {path}")


def plot_side_by_side(emb_ref_bl, emb_cur_bl, ref_lbl_bl, drift_bl,
                      emb_ref_has, emb_cur_has, ref_lbl_has, drift_has,
                      train_classes):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    plot_umap_panel(axes[0], emb_ref_bl, emb_cur_bl,
                    ref_lbl_bl, drift_bl, train_classes)
    axes[0].set_title("UMAP — Baseline (no HAS)", fontsize=13, fontweight="bold")

    plot_umap_panel(axes[1], emb_ref_has, emb_cur_has,
                    ref_lbl_has, drift_has, train_classes)
    axes[1].set_title("UMAP — HAS Model", fontsize=13, fontweight="bold")

    # Shared legend
    handles = []
    for i, cls in enumerate(train_classes):
        c = CLASS_COLORS[i % len(CLASS_COLORS)]
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=7, alpha=0.5,
                              label=f"Train: {cls}"))
    for dt, c in DRIFT_PALETTE.items():
        handles.append(Line2D([0], [0], marker="x", color=c, linestyle="",
                              markersize=8, label=dt))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               frameon=True, fancybox=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    path = os.path.join(RESULT_DIR, "umap_side_by_side.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(df, tag):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

    # A — dual drift scatter
    ax1 = fig.add_subplot(gs[0, 0])
    for dt, c in DRIFT_PALETTE.items():
        m = df["drift_type"] == dt
        ax1.scatter(df.loc[m, "data_drift_score"],
                    df.loc[m, "concept_drift_score"],
                    c=c, label=dt, alpha=0.5, s=25, edgecolors="none")
    ax1.set_xlabel("Data Drift Score"); ax1.set_ylabel("Concept Drift Score")
    ax1.set_title("Dual Drift Map"); ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    # B — confidence by drift type
    ax2 = fig.add_subplot(gs[0, 1])
    for dt, c in DRIFT_PALETTE.items():
        vals = df.loc[df["drift_type"] == dt, "max_confidence"]
        if len(vals):
            ax2.hist(vals, bins=25, alpha=0.45, color=c, label=dt, density=True)
    ax2.set_xlabel("Max Softmax Confidence"); ax2.set_ylabel("Density")
    ax2.set_title("Confidence by Drift Type"); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    # C — per-class accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    acc = df.groupby("true_class")["correct"].mean().sort_values()
    ax3.barh(acc.index, acc.values, color="#3498db")
    ax3.set_xlabel("Accuracy"); ax3.set_title("Per-Class Accuracy")
    ax3.set_xlim(0, 1); ax3.grid(axis="x", alpha=0.2)

    # D — pie chart
    ax4 = fig.add_subplot(gs[1, 1])
    counts = df["drift_type"].value_counts()
    ax4.pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=[DRIFT_PALETTE.get(k, "#999") for k in counts.index],
            startangle=140)
    ax4.set_title("Drift Type Distribution")

    fig.suptitle(f"Drift Dashboard — {tag}", fontsize=15, fontweight="bold")
    path = os.path.join(RESULT_DIR,
                        f"drift_dashboard_{tag.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-feature KS bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_feature_ks_chart(ref_lat, cur_lat, tag):
    result = per_feature_ks(ref_lat, cur_lat)
    fig, ax = plt.subplots(figsize=(14, 4))
    colours = ["#e74c3c" if p < 0.05 else "#2ecc71" for p in result["p_values"]]
    ax.bar(range(result["n_features"]), result["stats"], color=colours, width=0.8)
    ax.set_xlabel("Latent Dimension"); ax.set_ylabel("KS Statistic")
    ax.set_title(f"Per-Feature KS — {tag}  "
                 f"({result['n_drifted']}/{result['n_features']} drifted)")
    ax.axhline(0.1, color="#999", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = os.path.join(RESULT_DIR,
                        f"per_feature_ks_{tag.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 4: Visualize")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP (saves time on large datasets)")
    args = parser.parse_args()

    ensure_dirs()
    meta = load_meta()
    train_classes  = meta["train_classes"]
    custom_classes = meta["custom_classes"]

    # Load features
    print("Loading features …")
    bl_train   = load_npz("bl_train")
    bl_custom  = load_npz("bl_custom")
    has_train  = load_npz("has_train")
    has_custom = load_npz("has_custom")

    # Load drift CSVs
    df_bl  = load_csv("drift_baseline")
    df_has = load_csv("drift_has")

    # ── UMAP ──
    if not args.skip_umap:
        print("\nComputing UMAP (baseline) …")
        emb_ref_bl, emb_cur_bl = compute_umap(bl_train["latents"],
                                               bl_custom["latents"])
        print("Computing UMAP (HAS) …")
        emb_ref_has, emb_cur_has = compute_umap(has_train["latents"],
                                                 has_custom["latents"])

        drift_bl  = df_bl["drift_type"].values
        drift_has = df_has["drift_type"].values

        plot_single_umap(emb_ref_bl, emb_cur_bl,
                         bl_train["labels"], drift_bl,
                         train_classes, "UMAP — Baseline (no HAS)",
                         "umap_baseline.png")
        plot_single_umap(emb_ref_has, emb_cur_has,
                         has_train["labels"], drift_has,
                         train_classes, "UMAP — HAS Model",
                         "umap_has.png")
        plot_side_by_side(emb_ref_bl, emb_cur_bl,
                          bl_train["labels"], drift_bl,
                          emb_ref_has, emb_cur_has,
                          has_train["labels"], drift_has,
                          train_classes)
    else:
        print("\nSkipping UMAP (--skip-umap).")

    # ── Dashboards ──
    print("\nGenerating dashboards …")
    plot_dashboard(df_bl, "Baseline")
    plot_dashboard(df_has, "HAS Model")

    # ── Per-feature KS ──
    print("\nPer-feature KS plots …")
    plot_per_feature_ks_chart(bl_train["latents"], bl_custom["latents"], "Baseline")
    plot_per_feature_ks_chart(has_train["latents"], has_custom["latents"], "HAS Model")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
