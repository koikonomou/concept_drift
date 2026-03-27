"""
Subclass drift analysis with publication-quality plots.

Reads CSVs from step3, produces per-subcategory drift ranking,
class-level comparison, and polished UMAP overlays.

Usage:
    python step5_analysis.py

Outputs:
    results/subclass_drift_ranking.png    — which subcategories drifted most
    results/class_drift_comparison.png    — baseline vs HAS per class
    results/drift_summary_table.csv       — full per-subcategory stats
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

from config import FEATURE_DIR, RESULT_DIR, LANDSCAPE_CLASSES, ensure_dirs

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
})

CLASS_COLORS = {
    "Coast":    "#0077b6",
    "Desert":   "#e76f51",
    "Forest":   "#2d6a4f",
    "Glacier":  "#48cae4",
    "Mountain": "#7b2cbf",
}

DRIFT_COLORS = {
    "In-Distribution":   "#2ecc71",
    "Data Drift":        "#3498db",
    "Concept Drift":     "#e74c3c",
    "Full Drift (both)": "#8e44ad",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(name):
    path = os.path.join(RESULT_DIR, f"{name}.csv")
    return pd.read_csv(path)

def load_npz(name):
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def load_meta():
    with open(os.path.join(FEATURE_DIR, "meta.json")) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Subclass drift ranking  (the key analysis)
# ─────────────────────────────────────────────────────────────────────────────

def compute_subclass_stats(df, tag):
    """Per-subcategory drift statistics."""
    stats = df.groupby(["true_class", "sub_category"]).agg(
        n=("correct", "count"),
        accuracy=("correct", "mean"),
        mean_data_drift=("data_drift_score", "mean"),
        mean_confidence=("max_confidence", "mean"),
        pct_data_drifted=("data_drifted", "mean"),
        pct_concept_drifted=("concept_drifted", "mean"),
    ).reset_index()
    stats["model"] = tag
    return stats


def plot_subclass_drift_ranking(bl_stats, has_stats):
    """Horizontal bar chart: top-N most data-drifted subcategories, both models."""

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=False)

    for ax, stats, title in [
        (axes[0], bl_stats.nlargest(20, "mean_data_drift"), "Baseline (no HAS)"),
        (axes[1], has_stats.nlargest(20, "mean_data_drift"), "HAS Model"),
    ]:
        labels = [f"{row.sub_category}\n({row.true_class})"
                  for _, row in stats.iterrows()]
        y = range(len(labels))
        colors = [CLASS_COLORS.get(row.true_class, "#999")
                  for _, row in stats.iterrows()]

        bars = ax.barh(y, stats["mean_data_drift"].values,
                       color=colors, alpha=0.85, height=0.7,
                       edgecolor="white", linewidth=0.5)

        # Accuracy annotation on each bar
        for i, (_, row) in enumerate(stats.iterrows()):
            ax.text(row.mean_data_drift + 0.15, i,
                    f"acc={row.accuracy:.0%}",
                    va="center", fontsize=8, color="#555")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Data Drift Score (latent distance from centroid)")
        ax.set_title(title, fontweight="bold", fontsize=14)

    # Class legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="white")
               for c in CLASS_COLORS.values()]
    fig.legend(handles, CLASS_COLORS.keys(), loc="lower center",
               ncol=5, fontsize=10, frameon=True, fancybox=True,
               title="Training Class", title_fontsize=11)

    fig.suptitle("Subcategory Data Drift Ranking",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    path = os.path.join(RESULT_DIR, "subclass_drift_ranking.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class-level comparison  (baseline vs HAS)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_comparison(df_bl, df_has):
    """Side-by-side comparison per class: accuracy, data drift, concept drift."""

    classes = LANDSCAPE_CLASSES
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    metrics = [
        ("accuracy",         "correct",         "mean", "Accuracy", "%"),
        ("data_drift_rate",  "data_drifted",    "mean", "Data Drift Rate", "%"),
        ("concept_drift_rate","concept_drifted", "mean", "Concept Drift Rate", "%"),
    ]

    x = np.arange(len(classes))
    w = 0.35

    for ax, (_, col, agg, title, unit) in zip(axes, metrics):
        bl_vals = [df_bl[df_bl["true_class"] == c][col].mean() for c in classes]
        has_vals = [df_has[df_has["true_class"] == c][col].mean() for c in classes]

        bars1 = ax.bar(x - w/2, [v * 100 for v in bl_vals], w,
                       label="Baseline", color="#3498db", alpha=0.8,
                       edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + w/2, [v * 100 for v in has_vals], w,
                       label="HAS", color="#e74c3c", alpha=0.8,
                       edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

        # Value labels
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Baseline vs HAS — Per-Class Comparison",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(RESULT_DIR, "class_drift_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. UMAP — clean, publication-quality
# ─────────────────────────────────────────────────────────────────────────────

def compute_umap(ref_latents, cur_latents):
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2, random_state=42, metric="euclidean")
    combined = np.vstack([ref_latents, cur_latents])
    emb = reducer.fit_transform(combined)
    n = len(ref_latents)
    return emb[:n], emb[n:]


def plot_umap_publication(emb_ref, emb_cur, ref_labels, df_custom,
                          train_classes, title, filename):
    """Single clean UMAP with separate train cloud and custom overlay."""

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f8f9fa")

    # Train points — very light, grouped by class
    for i, cls in enumerate(train_classes):
        mask = ref_labels == i
        if mask.any():
            ax.scatter(emb_ref[mask, 0], emb_ref[mask, 1],
                       c=CLASS_COLORS.get(cls, "#ccc"),
                       alpha=0.10, s=8, rasterized=True)

    # Custom points coloured by drift type
    for dtype, colour in DRIFT_COLORS.items():
        mask = df_custom["drift_type"].values == dtype
        if mask.any():
            marker = "o" if dtype == "In-Distribution" else "X"
            size = 20 if dtype == "In-Distribution" else 35
            ax.scatter(emb_cur[mask, 0], emb_cur[mask, 1],
                       c=colour, marker=marker, s=size,
                       alpha=0.7, linewidths=0.3, edgecolors="white",
                       zorder=3)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Custom legend
    from matplotlib.lines import Line2D
    handles = []
    for cls, c in CLASS_COLORS.items():
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=6, alpha=0.4,
                              label=f"Train: {cls}"))
    handles.append(Line2D([], [], color="none", label=""))  # spacer
    for dt, c in DRIFT_COLORS.items():
        m = "o" if dt == "In-Distribution" else "X"
        handles.append(Line2D([0], [0], marker=m, color="w",
                              markerfacecolor=c, markersize=7,
                              markeredgecolor="white", markeredgewidth=0.3,
                              label=dt))

    ax.legend(handles=handles, loc="upper left", fontsize=8,
              frameon=True, fancybox=True, framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_umap_side_by_side(emb_ref_bl, emb_cur_bl, ref_lbl_bl, df_bl,
                           emb_ref_has, emb_cur_has, ref_lbl_has, df_has,
                           train_classes):
    """Side-by-side publication UMAP."""

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, emb_ref, emb_cur, ref_labels, df, title in [
        (axes[0], emb_ref_bl, emb_cur_bl, ref_lbl_bl, df_bl,
         "Baseline (no HAS)"),
        (axes[1], emb_ref_has, emb_cur_has, ref_lbl_has, df_has,
         "HAS Model"),
    ]:
        ax.set_facecolor("#f8f9fa")

        for i, cls in enumerate(train_classes):
            mask = ref_labels == i
            if mask.any():
                ax.scatter(emb_ref[mask, 0], emb_ref[mask, 1],
                           c=CLASS_COLORS.get(cls, "#ccc"),
                           alpha=0.08, s=6, rasterized=True)

        for dtype, colour in DRIFT_COLORS.items():
            mask = df["drift_type"].values == dtype
            if mask.any():
                marker = "o" if dtype == "In-Distribution" else "X"
                size = 15 if dtype == "In-Distribution" else 30
                ax.scatter(emb_cur[mask, 0], emb_cur[mask, 1],
                           c=colour, marker=marker, s=size,
                           alpha=0.7, linewidths=0.3, edgecolors="white",
                           zorder=3)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    handles = []
    for cls, c in CLASS_COLORS.items():
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=7, alpha=0.5,
                              label=f"Train: {cls}"))
    for dt, c in DRIFT_COLORS.items():
        m = "o" if dt == "In-Distribution" else "X"
        handles.append(Line2D([0], [0], marker=m, color="w",
                              markerfacecolor=c, markersize=8,
                              markeredgecolor="white", markeredgewidth=0.3,
                              label=dt))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               frameon=True, fancybox=True, framealpha=0.95,
               title="", bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("UMAP — Latent Space Comparison", fontsize=16,
                 fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    path = os.path.join(RESULT_DIR, "umap_side_by_side.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Subclass heatmap  (data drift × concept drift)
# ─────────────────────────────────────────────────────────────────────────────

def plot_subclass_heatmap(stats, tag):
    """Scatter plot of subcategories: x=data drift, y=concept drift, size=n."""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#f8f9fa")

    for _, row in stats.iterrows():
        c = CLASS_COLORS.get(row.true_class, "#999")
        ax.scatter(row.mean_data_drift, row.pct_concept_drifted * 100,
                   s=row.n * 2, c=c, alpha=0.7,
                   edgecolors="white", linewidth=0.5, zorder=3)
        # Label subcategories with significant drift
        if row.mean_data_drift > stats.mean_data_drift.quantile(0.75) or \
           row.pct_concept_drifted > 0.25:
            ax.annotate(row.sub_category, (row.mean_data_drift, row.pct_concept_drifted * 100),
                        fontsize=7, alpha=0.8, ha="left",
                        xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Mean Data Drift Score (latent distance)")
    ax.set_ylabel("Concept Drift Rate (%)")
    ax.set_title(f"Subcategory Drift Map — {tag}",
                 fontsize=14, fontweight="bold")

    # Legend
    handles = [plt.scatter([], [], c=c, s=60, label=cls, edgecolors="white")
               for cls, c in CLASS_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"subclass_heatmap_{tag.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    meta = load_meta()
    train_classes = meta["train_classes"]

    print("Loading data …")
    df_bl  = load_csv("drift_baseline")
    df_has = load_csv("drift_has")

    # ── Subclass statistics ──
    bl_stats  = compute_subclass_stats(df_bl, "Baseline")
    has_stats = compute_subclass_stats(df_has, "HAS")

    # Save combined table
    all_stats = pd.concat([bl_stats, has_stats])
    table_path = os.path.join(RESULT_DIR, "drift_summary_table.csv")
    all_stats.to_csv(table_path, index=False)
    print(f"  ✓ {table_path}")

    # ── Plots ──
    print("\nGenerating subclass drift ranking …")
    plot_subclass_drift_ranking(bl_stats, has_stats)

    print("Generating class comparison …")
    plot_class_comparison(df_bl, df_has)

    print("Generating subclass drift maps …")
    plot_subclass_heatmap(bl_stats, "Baseline")
    plot_subclass_heatmap(has_stats, "HAS Model")

    # ── UMAP (only if features exist) ──
    try:
        print("\nComputing UMAPs …")
        bl_train  = load_npz("bl_train")
        bl_custom = load_npz("bl_custom")
        has_train = load_npz("has_train")
        has_custom = load_npz("has_custom")

        emb_ref_bl, emb_cur_bl = compute_umap(
            bl_train["latents"], bl_custom["latents"])
        emb_ref_has, emb_cur_has = compute_umap(
            has_train["latents"], has_custom["latents"])

        plot_umap_publication(
            emb_ref_bl, emb_cur_bl, bl_train["labels"], df_bl,
            train_classes, "UMAP — Baseline (no HAS)", "umap_baseline.png")
        plot_umap_publication(
            emb_ref_has, emb_cur_has, has_train["labels"], df_has,
            train_classes, "UMAP — HAS Model", "umap_has.png")
        plot_umap_side_by_side(
            emb_ref_bl, emb_cur_bl, bl_train["labels"], df_bl,
            emb_ref_has, emb_cur_has, has_train["labels"], df_has,
            train_classes)
    except Exception as e:
        print(f"  ⚠ UMAP skipped: {e}")
        print("    Run step2_extract.py first for UMAP plots.")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("SUBCATEGORY DRIFT ANALYSIS")
    print("=" * 70)
    for tag, stats in [("Baseline", bl_stats), ("HAS", has_stats)]:
        top5 = stats.nlargest(5, "mean_data_drift")
        print(f"\n  {tag} — Top 5 most data-drifted subcategories:")
        for _, r in top5.iterrows():
            print(f"    {r.sub_category:30s} ({r.true_class:8s}) | "
                  f"drift={r.mean_data_drift:.2f} | "
                  f"acc={r.accuracy:.0%} | "
                  f"n={r.n}")

    # Worst concept drift
    for tag, stats in [("Baseline", bl_stats), ("HAS", has_stats)]:
        top5 = stats.nlargest(5, "pct_concept_drifted")
        print(f"\n  {tag} — Top 5 most concept-drifted subcategories:")
        for _, r in top5.iterrows():
            print(f"    {r.sub_category:30s} ({r.true_class:8s}) | "
                  f"concept_drift={r.pct_concept_drifted:.0%} | "
                  f"acc={r.accuracy:.0%}")

    # ═══════════════════════════════════════════════════════════════════════
    # ADDITION 2 — Hierarchical drift summary
    # ═══════════════════════════════════════════════════════════════════════
    hier_path = os.path.join(RESULT_DIR, "hierarchical_drift.json")
    if os.path.exists(hier_path):
        print("\n" + "=" * 70)
        print("HIERARCHICAL DRIFT SUMMARY (ADDITION 2)")
        print("=" * 70)
        with open(hier_path) as f:
            hier = json.load(f)

        hier_bl  = hier.get("baseline", {})
        hier_has = hier.get("has", {})

        # Which classes are drifted under each model
        bl_drifted  = [c for c, v in hier_bl.items()  if v.get("class_drifted")]
        has_drifted = [c for c, v in hier_has.items() if v.get("class_drifted")]

        print(f"\n  Baseline drifted classes : {bl_drifted  or 'none'}")
        print(f"  HAS      drifted classes : {has_drifted or 'none'}")

        # Agreement between models
        agree = set(bl_drifted) == set(has_drifted)
        if agree:
            print(f"  Agreement               : YES — both models flag the same classes")
        else:
            only_bl  = set(bl_drifted)  - set(has_drifted)
            only_has = set(has_drifted) - set(bl_drifted)
            if only_bl:
                print(f"  Only Baseline flags     : {sorted(only_bl)}")
            if only_has:
                print(f"  Only HAS flags          : {sorted(only_has)}")

        # Most drifted subclass across both models (highest KS stat)
        best_ks   = -1.0
        best_info = ("—", "—", "—", -1.0)
        for model_tag, hier_data in [("Baseline", hier_bl), ("HAS", hier_has)]:
            for cls, cls_info in hier_data.items():
                for sub, sub_info in cls_info.get("subclasses", {}).items():
                    if sub_info["ks_stat"] > best_ks:
                        best_ks   = sub_info["ks_stat"]
                        best_info = (model_tag, cls, sub, best_ks)

        print(f"\n  Most drifted subclass    : {best_info[1]}/{best_info[2]} "
              f"({best_info[0]}, KS={best_info[3]:.4f})")
    else:
        print(f"\n  ⚠ {hier_path} not found — skipping hierarchical summary.")
        print("    Run step3_detect.py first.")

    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()
