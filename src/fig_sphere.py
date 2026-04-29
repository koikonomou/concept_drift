"""
fig_sphere.py — Three figures showing HAS spherical embedding geometry vs Baseline.

Correct visualisation for HAS: all embeddings live on the unit hypersphere S⁶³.
These figures show the angular/spherical structure that HAS explicitly learns,
unlike t-SNE which ignores the sphere geometry.

Figure 1  fig1_sphere_3d.png
    PCA projection to 3-D, points plotted on the sphere surface.
    Baseline embeddings are L2-normalised for fair comparison.
    Shows: how tightly each class clusters on the sphere.

Figure 2  fig2_angle_distributions.png
    Histogram of angles between same-class pairs (positive) and
    different-class pairs (negative) for both models.
    Replicates Figure 4 from Kansizoglou et al. (ICMLA 2020).
    Shows: HAS pushes positive pairs toward 0° and negative pairs
    toward 90°, increasing the angular discrimination margin.

Figure 3  fig3_cosine_heatmap.png
    5×5 matrix of mean cosine similarities between class pairs.
    Diagonal = intra-class compactness (higher = better).
    Off-diagonal = inter-class similarity (lower = better).
    Shows: the specific class boundaries that are weak/strong.

Usage:
    python fig_sphere.py
    python fig_sphere.py --n-samples 1500   # more points, slower
    python fig_sphere.py --n-angle 300      # samples per class for angles
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from sklearn.decomposition import PCA

from config import FEATURE_DIR, RESULT_DIR, LANDSCAPE_CLASSES, ensure_dirs


# ─────────────────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Coast":    "#0077b6",
    "Desert":   "#e76f51",
    "Forest":   "#2d6a4f",
    "Glacier":  "#48cae4",
    "Mountain": "#7b2cbf",
}
CLASS_MARKERS = {
    "Coast":    "o",
    "Desert":   "s",
    "Forest":   "^",
    "Glacier":  "D",
    "Mountain": "P",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size":        11,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(name):
    path = os.path.join(FEATURE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run extract.py first.")
    d = np.load(path, allow_pickle=True)
    return d["latents"], d["labels"]


def l2_norm(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-8)


def subsample_stratified(latents, labels, n_per_class, seed=42):
    """Take n_per_class samples from each class. Returns arrays."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        chosen = rng.choice(ci, size=min(n_per_class, len(ci)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    return latents[idx], labels[idx]


def save(fig, name):
    path = os.path.join(RESULT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 3-D sphere projection
# ─────────────────────────────────────────────────────────────────────────────

def _wireframe_sphere(ax, alpha=0.06, color="#aaa"):
    """Draw a faint unit sphere wireframe for spatial reference."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(x, y, z, color=color, alpha=0.08,
                      linewidth=0.4, rstride=3, cstride=3)


def _sphere_panel(ax, emb3, labels, class_names, title):
    """Plot one 3-D sphere panel."""
    _wireframe_sphere(ax)
    for i, cls in enumerate(class_names):
        mask = labels == i
        if not mask.any():
            continue
        ax.scatter(emb3[mask, 0], emb3[mask, 1], emb3[mask, 2],
                   c=CLASS_COLORS.get(cls, "#999"),
                   marker=CLASS_MARKERS.get(cls, "o"),
                   s=16, alpha=0.65, linewidths=0,
                   depthshade=True, label=cls,
                   rasterized=True)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("PC 1", fontsize=8, labelpad=2)
    ax.set_ylabel("PC 2", fontsize=8, labelpad=2)
    ax.set_zlabel("PC 3", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)

    # Keep aspect equal and view angle consistent
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 1])

    # Remove grid lines for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)


def figure1_sphere(bl_lat, has_lat, bl_lbl, has_lbl,
                   class_names, n_per_class=400):
    """PCA to 3-D, then plot on unit sphere surface."""
    print("  Figure 1: 3-D sphere projection …")

    # Subsample
    bl_s,  bl_l  = subsample_stratified(bl_lat,  bl_lbl,  n_per_class)
    has_s, has_l = subsample_stratified(has_lat, has_lbl, n_per_class)

    # Baseline: L2-normalise to put on sphere for fair comparison
    bl_unit = l2_norm(bl_s)
    # HAS: already unit vectors
    has_unit = has_s   # already normalised

    # PCA to 3-D — fit jointly so both are in the same PCA space per model
    bl_3d  = PCA(n_components=3, random_state=42).fit_transform(bl_unit)
    has_3d = PCA(n_components=3, random_state=42).fit_transform(has_unit)

    # Re-normalise after PCA to keep points near sphere surface
    bl_3d  = l2_norm(bl_3d)
    has_3d = l2_norm(has_3d)

    fig = plt.figure(figsize=(14, 6.5), facecolor="white")
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    _sphere_panel(ax1, bl_3d,  bl_l,  class_names, "Baseline (Softmax) — L2-normalised")
    _sphere_panel(ax2, has_3d, has_l, class_names, "HASeparator")

    # Shared legend
    handles = [
        plt.scatter([], [], c=CLASS_COLORS[cls],
                    marker=CLASS_MARKERS[cls], s=40, label=cls)
        for cls in class_names if cls in CLASS_COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9, edgecolor="#ccc",
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Figure 1 — Training Embeddings Projected onto the Unit Sphere (S⁶³ → S²)\n"
        "Each point = one training image. HAS explicitly pushes embeddings away from "
        "class boundaries, producing tighter clusters.",
        fontsize=10, y=0.98, va="top", color="#333"
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    save(fig, "fig1_sphere_3d.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Angular distribution histograms (replicates paper Figure 4)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_angles(latents, labels, n_per_class=200, seed=42):
    """Compute angles between positive pairs (same class) and
    negative pairs (different classes).

    Returns (pos_angles_deg, neg_angles_deg).
    Subsamples n_per_class per class for efficiency.
    """
    rng = np.random.default_rng(seed)
    lat, lbl = subsample_stratified(latents, labels, n_per_class, seed)
    unit = l2_norm(lat)   # ensure unit vectors

    pos_cos, neg_cos = [], []
    n = len(unit)
    classes = np.unique(lbl)

    for c in classes:
        ci = np.where(lbl == c)[0]
        others = np.where(lbl != c)[0]

        # Positive pairs: within class c
        if len(ci) > 1:
            a_idx = rng.choice(ci, size=min(len(ci), 300), replace=False)
            for i in range(len(a_idx)):
                for j in range(i + 1, len(a_idx)):
                    cos_v = np.clip(unit[a_idx[i]] @ unit[a_idx[j]], -1, 1)
                    pos_cos.append(cos_v)

        # Negative pairs: class c vs all other classes (sampled)
        if len(ci) > 0 and len(others) > 0:
            a_idx = rng.choice(ci, size=min(50, len(ci)), replace=False)
            b_idx = rng.choice(others, size=min(200, len(others)), replace=False)
            cos_m = unit[a_idx] @ unit[b_idx].T   # (len_a, len_b)
            neg_cos.extend(np.clip(cos_m.ravel(), -1, 1).tolist())

    pos_angles = np.degrees(np.arccos(pos_cos))
    neg_angles = np.degrees(np.arccos(neg_cos))
    return pos_angles, neg_angles


def _angle_panel(ax, pos_angles, neg_angles, title,
                 pos_color="#2ecc71", neg_color="#e74c3c"):
    """Plot one angular distribution panel."""
    bins = np.linspace(0, 180, 60)
    ax.hist(pos_angles, bins=bins, density=True, alpha=0.6,
            color=pos_color, label="Positive pairs\n(same class)",
            edgecolor="none")
    ax.hist(neg_angles, bins=bins, density=True, alpha=0.6,
            color=neg_color, label="Negative pairs\n(different class)",
            edgecolor="none")

    # Vertical lines at means
    ax.axvline(pos_angles.mean(), color=pos_color, linestyle="--",
               linewidth=1.5, alpha=0.9)
    ax.axvline(neg_angles.mean(), color=neg_color, linestyle="--",
               linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Angle between pair (degrees)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)

    # Annotate means
    ax.text(pos_angles.mean() + 2, ax.get_ylim()[1] * 0.9,
            f"μ={pos_angles.mean():.1f}°", fontsize=8,
            color=pos_color, fontweight="bold")
    ax.text(neg_angles.mean() + 2, ax.get_ylim()[1] * 0.75,
            f"μ={neg_angles.mean():.1f}°", fontsize=8,
            color=neg_color, fontweight="bold")


def figure2_angles(bl_lat, has_lat, bl_lbl, has_lbl,
                   class_names, n_per_class=250):
    """Angle distributions — replicates Figure 4 of the paper."""
    print("  Figure 2: angular distributions …")

    # Baseline: L2-normalise
    bl_unit = l2_norm(bl_lat)

    print("    Computing Baseline angles …")
    bl_pos,  bl_neg  = _compute_angles(bl_unit,  bl_lbl,  n_per_class)
    print("    Computing HAS angles …")
    has_pos, has_neg = _compute_angles(has_lat, has_lbl, n_per_class)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    _angle_panel(axes[0], bl_pos,  bl_neg,  "Baseline (Softmax)")
    _angle_panel(axes[1], has_pos, has_neg, "HASeparator")

    # Compute Wasserstein distance (same metric as paper)
    from scipy.stats import wasserstein_distance
    bl_wd  = wasserstein_distance(bl_pos,  bl_neg)
    has_wd = wasserstein_distance(has_pos, has_neg)
    improvement = (has_wd - bl_wd) / bl_wd * 100

    fig.suptitle(
        "Figure 2 — Angular Distributions of Positive and Negative Pairs\n"
        f"Wasserstein distance: Baseline = {bl_wd:.2f}°   "
        f"HAS = {has_wd:.2f}°   (Δ = {improvement:+.1f}%)\n"
        "Larger separation between positive/negative distributions = "
        "better feature discrimination.",
        fontsize=10, y=1.01, va="bottom", color="#333"
    )
    plt.tight_layout()
    save(fig, "fig2_angle_distributions.png")

    print(f"    Wasserstein distance — Baseline: {bl_wd:.2f}°  "
          f"HAS: {has_wd:.2f}°  (Δ={improvement:+.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Cosine similarity heatmap
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_matrix(latents, labels, class_names, n_per_class=500, seed=42):
    """Compute 5×5 matrix of mean cosine similarities between class pairs.

    matrix[i, j] = mean cosine similarity between all pairs of
    samples from class i and class j.
    Diagonal = intra-class similarity (compactness).
    Off-diagonal = inter-class similarity (separation).
    """
    unit = l2_norm(latents)
    n = len(class_names)
    matrix = np.zeros((n, n))
    rng = np.random.default_rng(seed)

    for i in range(n):
        for j in range(i, n):
            ci = np.where(labels == i)[0]
            cj = np.where(labels == j)[0]
            if len(ci) == 0 or len(cj) == 0:
                continue

            # Subsample for efficiency
            ai = rng.choice(ci, size=min(n_per_class, len(ci)), replace=False)
            aj = rng.choice(cj, size=min(n_per_class, len(cj)), replace=False)

            cos_m = unit[ai] @ unit[aj].T   # (len_ai, len_aj)
            if i == j:
                # Exclude self-similarity (diagonal of cos_m)
                mask = ~np.eye(len(ai), dtype=bool)
                val  = cos_m[mask].mean() if mask.any() else 0.0
            else:
                val = cos_m.mean()

            matrix[i, j] = val
            matrix[j, i] = val

    return matrix


def _heatmap_panel(ax, matrix, class_names, title,
                   vmin=None, vmax=None, cmap="RdYlGn"):
    """Plot one cosine similarity heatmap."""
    n = len(class_names)
    if vmin is None: vmin = matrix.min()
    if vmax is None: vmax = matrix.max()

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            color = "white" if v < (vmin + (vmax - vmin) * 0.4) else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    return im


def figure3_heatmap(bl_lat, has_lat, bl_lbl, has_lbl,
                    class_names, n_per_class=500):
    """Cosine similarity heatmap Baseline vs HAS."""
    print("  Figure 3: cosine similarity heatmap …")

    bl_unit = l2_norm(bl_lat)

    print("    Computing Baseline cosine matrix …")
    bl_mat  = _cosine_matrix(bl_unit,  bl_lbl,  class_names, n_per_class)
    print("    Computing HAS cosine matrix …")
    has_mat = _cosine_matrix(has_lat, has_lbl, class_names, n_per_class)

    # Use shared colour scale so panels are directly comparable
    vmin = min(bl_mat.min(), has_mat.min())
    vmax = max(bl_mat.max(), has_mat.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = _heatmap_panel(axes[0], bl_mat,  class_names,
                          "Baseline (Softmax)", vmin, vmax)
    im2 = _heatmap_panel(axes[1], has_mat, class_names,
                          "HASeparator",        vmin, vmax)

    # Shared colorbar
    cbar = fig.colorbar(im2, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Mean cosine similarity", fontsize=10)

    # Compute summary stats
    n = len(class_names)
    bl_intra  = np.mean([bl_mat[i, i]  for i in range(n)])
    has_intra = np.mean([has_mat[i, i] for i in range(n)])
    bl_inter  = np.mean([bl_mat[i, j]  for i in range(n)
                          for j in range(n) if i != j])
    has_inter = np.mean([has_mat[i, j] for i in range(n)
                          for j in range(n) if i != j])

    fig.suptitle(
        "Figure 3 — Mean Cosine Similarity Between Class Pairs\n"
        f"Diagonal = intra-class (↑ better):  Baseline {bl_intra:.3f}  →  "
        f"HAS {has_intra:.3f}\n"
        f"Off-diagonal = inter-class (↓ better): Baseline {bl_inter:.3f}  →  "
        f"HAS {has_inter:.3f}",
        fontsize=10, y=1.01, va="bottom", color="#333"
    )
    plt.tight_layout()
    save(fig, "fig3_cosine_heatmap.png")

    print(f"    Intra-class  — Baseline: {bl_intra:.4f}  HAS: {has_intra:.4f}")
    print(f"    Inter-class  — Baseline: {bl_inter:.4f}  HAS: {has_inter:.4f}")
    gap_bl  = bl_intra  - bl_inter
    gap_has = has_intra - has_inter
    print(f"    Intra-inter gap — Baseline: {gap_bl:.4f}  "
          f"HAS: {gap_has:.4f}  (Δ={gap_has-gap_bl:+.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Three spherical embedding figures: "
                    "3-D sphere, angle distributions, cosine heatmap.")
    parser.add_argument("--n-samples", type=int, default=400,
                        help="Samples per class for sphere plot (default 400)")
    parser.add_argument("--n-angle",   type=int, default=250,
                        help="Samples per class for angle distributions (default 250)")
    parser.add_argument("--n-heat",    type=int, default=500,
                        help="Samples per class for heatmap (default 500)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading features …")
    bl_lat,  bl_lbl  = load_npz("bl_train")
    has_lat, has_lbl = load_npz("has_train")
    print(f"  Baseline : {len(bl_lat)} samples  |  "
          f"HAS: {len(has_lat)} samples\n")

    # ── Generate figures ──────────────────────────────────────────────────────
    figure1_sphere(bl_lat, has_lat, bl_lbl, has_lbl,
                   LANDSCAPE_CLASSES, n_per_class=args.n_samples)

    figure2_angles(bl_lat, has_lat, bl_lbl, has_lbl,
                   LANDSCAPE_CLASSES, n_per_class=args.n_angle)

    figure3_heatmap(bl_lat, has_lat, bl_lbl, has_lbl,
                    LANDSCAPE_CLASSES, n_per_class=args.n_heat)

    print("\nAll figures saved to results/")
    print("  results/fig1_sphere_3d.png")
    print("  results/fig2_angle_distributions.png")
    print("  results/fig3_cosine_heatmap.png")


if __name__ == "__main__":
    main()
