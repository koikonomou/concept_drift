"""
fig_tsne.py — Reproduce Figure 1 from the HASeparator paper for your dataset.

Uses t-SNE to project the 64-D training embeddings into 2-D, then plots
Baseline vs HAS side-by-side showing class discrimination quality.

Reads:  features/bl_train.npz   features/has_train.npz
Writes: results/fig0_tsne_comparison.png

Usage:
    python fig_tsne.py
    python fig_tsne.py --n-samples 3000   # faster, fewer points
    python fig_tsne.py --perplexity 50    # adjust t-SNE perplexity
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE

from config import FEATURE_DIR, RESULT_DIR, LANDSCAPE_CLASSES, ensure_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — one distinct colour per class, publication quality
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Coast":    "#0077b6",   # deep blue
    "Desert":   "#e76f51",   # terracotta
    "Forest":   "#2d6a4f",   # dark green
    "Glacier":  "#48cae4",   # ice blue
    "Mountain": "#7b2cbf",   # purple
}
CLASS_MARKERS = {
    "Coast":    "o",
    "Desert":   "s",
    "Forest":   "^",
    "Glacier":  "D",
    "Mountain": "P",
}


def load_npz(name):
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run extract.py first.")
    d = np.load(path, allow_pickle=True)
    return d["latents"], d["labels"]


def subsample(latents, labels, n, seed=42):
    """Subsample to n points, stratified by class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = n // len(classes)
    idx = []
    for c in classes:
        ci = np.where(labels == c)[0]
        chosen = rng.choice(ci, size=min(per_class, len(ci)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return latents[idx], labels[idx]


def run_tsne(latents, perplexity=40, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity,
                max_iter=1000, random_state=seed,
                init="pca", learning_rate="auto")
    return tsne.fit_transform(latents)


def plot_panel(ax, emb, labels, class_names, title, show_legend=False):
    """Plot one t-SNE panel."""
    ax.set_facecolor("#f8f9fa")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot each class
    for i, cls in enumerate(class_names):
        mask = labels == i
        if not mask.any():
            continue
        color  = CLASS_COLORS.get(cls, "#999")
        marker = CLASS_MARKERS.get(cls, "o")
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, marker=marker,
            s=14, alpha=0.65,
            linewidths=0.0,
            label=cls,
            rasterized=True,
        )

    # Compute and plot per-class centroid labels
    for i, cls in enumerate(class_names):
        mask = labels == i
        if not mask.any():
            continue
        cx, cy = emb[mask, 0].mean(), emb[mask, 1].mean()
        color = CLASS_COLORS.get(cls, "#999")
        ax.text(cx, cy, cls,
                fontsize=9, fontweight="bold", color="white",
                ha="center", va="center",
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground=color),
                ])

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(left=False, bottom=False)

    if show_legend:
        handles = [
            plt.scatter([], [], c=CLASS_COLORS[cls],
                        marker=CLASS_MARKERS[cls], s=40, label=cls)
            for cls in class_names if cls in CLASS_COLORS
        ]
        ax.legend(handles=handles, loc="lower right",
                  fontsize=9, framealpha=0.9,
                  edgecolor="#ccc", fancybox=False)


def compute_separation(emb, labels, class_names):
    """Inter-class / intra-class distance ratio — higher = better separation."""
    centroids = np.array([
        emb[labels == i].mean(axis=0)
        for i in range(len(class_names))
        if (labels == i).any()
    ])
    # Mean pairwise inter-class centroid distance
    n = len(centroids)
    inter = np.mean([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(n) for j in range(i+1, n)
    ])
    # Mean intra-class spread
    intra = np.mean([
        np.linalg.norm(emb[labels == i] -
                        emb[labels == i].mean(axis=0), axis=1).mean()
        for i in range(len(class_names)) if (labels == i).any()
    ])
    return inter, intra, inter / (intra + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",  type=int,   default=2000,
                        help="Points to plot per model (default 2000, max 10000)")
    parser.add_argument("--perplexity", type=float, default=40,
                        help="t-SNE perplexity (default 40)")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    ensure_dirs()
    n = min(args.n_samples, 10000)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading features …")
    bl_lat,  bl_lbl  = load_npz("bl_train")
    has_lat, has_lbl = load_npz("has_train")
    print(f"  Baseline : {len(bl_lat)} samples")
    print(f"  HAS      : {len(has_lat)} samples")

    # Subsample
    bl_lat,  bl_lbl  = subsample(bl_lat,  bl_lbl,  n, args.seed)
    has_lat, has_lbl = subsample(has_lat, has_lbl, n, args.seed)
    print(f"  Using {len(bl_lbl)} samples per model (stratified)\n")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={args.perplexity}) …")
    print("  Baseline … ", end="", flush=True)
    bl_emb  = run_tsne(bl_lat,  args.perplexity, args.seed)
    print("done")
    print("  HAS     … ", end="", flush=True)
    has_emb = run_tsne(has_lat, args.perplexity, args.seed)
    print("done\n")

    # ── Separation metrics ─────────────────────────────────────────────────────
    bl_inter,  bl_intra,  bl_ratio  = compute_separation(bl_emb,  bl_lbl,  LANDSCAPE_CLASSES)
    has_inter, has_intra, has_ratio = compute_separation(has_emb, has_lbl, LANDSCAPE_CLASSES)
    print(f"  Baseline: inter={bl_inter:.1f}  intra={bl_intra:.1f}  "
          f"ratio={bl_ratio:.2f}")
    print(f"  HAS     : inter={has_inter:.1f}  intra={has_intra:.1f}  "
          f"ratio={has_ratio:.2f}")
    improvement = (has_ratio - bl_ratio) / bl_ratio * 100
    print(f"  Separation improvement: {improvement:+.1f}%\n")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#f8f9fa",
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("white")

    plot_panel(axes[0], bl_emb,  bl_lbl,
               LANDSCAPE_CLASSES,
               "Baseline (Softmax)",
               show_legend=False)

    plot_panel(axes[1], has_emb, has_lbl,
               LANDSCAPE_CLASSES,
               "HASeparator",
               show_legend=True)

    # Shared subtitle with separation metrics
    fig.suptitle(
        "Figure 1 — t-SNE Visualisation of Training Embeddings\n"
        f"Baseline: inter/intra ratio = {bl_ratio:.2f}   "
        f"HAS: inter/intra ratio = {has_ratio:.2f}   "
        f"(Δ = {improvement:+.1f}%)",
        fontsize=11, y=0.02, va="bottom", color="#444",
    )

    # Annotation box explaining what the figure shows
    fig.text(
        0.5, 0.98,
        "Each point = one training image projected to 2-D via t-SNE. "
        "Tighter clusters and wider gaps between them indicate better feature discrimination.",
        ha="center", va="top", fontsize=9, color="#666",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    out = os.path.join(RESULT_DIR, "fig0_tsne_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {out}")


if __name__ == "__main__":
    main()
