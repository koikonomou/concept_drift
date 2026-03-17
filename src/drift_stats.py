"""
Data drift and concept drift detection.

DATA DRIFT  = "Are the custom images different from training images?"
  Measured in the 64-D latent space produced by the model.

  • KS test on latent norms    — quick check: did feature magnitudes shift?
  • Centroid shift              — did the center of the feature cloud move?
  • MMD (RBF kernel)            — gold standard: are the two 64-D clouds
                                  from different distributions?
  • Per-feature KS              — which of the 64 dimensions shifted?

CONCEPT DRIFT = "Is the model confused by the custom images?"
  Measured from the classifier's outputs.

  • Chi² on predictions         — did the predicted class distribution change?
  • KS on confidence            — did the model become less certain?
"""

import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import cdist


# ═════════════════════════════════════════════════════════════════════════════
# DATA DRIFT
# ═════════════════════════════════════════════════════════════════════════════

def ks_test(ref_latents, cur_latents, alpha=0.05):
    """KS test on latent vector norms.

    If the norms are distributed differently, the custom data lives
    in a different region of feature space.
    """
    ref_norms = np.linalg.norm(ref_latents, axis=1)
    cur_norms = np.linalg.norm(cur_latents, axis=1)
    stat, p = ks_2samp(ref_norms, cur_norms)
    return dict(test="KS (latent norms)", statistic=stat,
                p_value=p, drifted=p < alpha)


def centroid_shift(ref_latents, cur_latents):
    """How far did the center of the feature cloud move?

    Returns the Euclidean distance between centroids and a z-score
    (shift / spread of the training cloud).  z > 2 ≈ significant.
    """
    c_ref  = ref_latents.mean(axis=0)
    c_cur  = cur_latents.mean(axis=0)
    shift  = float(np.linalg.norm(c_ref - c_cur))
    spread = float(np.linalg.norm(ref_latents - c_ref, axis=1).std())
    z = shift / (spread + 1e-8)
    return dict(test="Centroid Shift", statistic=shift, z_score=z,
                p_value=None, drifted=z > 2.0)


def mmd_rbf(ref_latents, cur_latents, gamma=None, alpha=0.05, n_perm=300):
    """Maximum Mean Discrepancy with RBF kernel.

    The most reliable test: compares the full 64-D distributions,
    not just a summary statistic.  Uses a permutation test for
    the p-value (no distributional assumptions).
    """
    # Subsample if datasets are large (MMD is O(n²))
    max_n = 2000
    if len(ref_latents) > max_n:
        idx = np.random.choice(len(ref_latents), max_n, replace=False)
        ref_latents = ref_latents[idx]
    if len(cur_latents) > max_n:
        idx = np.random.choice(len(cur_latents), max_n, replace=False)
        cur_latents = cur_latents[idx]

    # RBF bandwidth: median heuristic
    if gamma is None:
        combined = np.vstack([ref_latents, cur_latents])
        dists = cdist(combined, combined, "sqeuclidean")
        gamma = 1.0 / np.median(dists[dists > 0])

    def _rbf(X, Y):
        return np.exp(-gamma * cdist(X, Y, "sqeuclidean"))

    def _mmd2(X, Y):
        return _rbf(X, X).mean() + _rbf(Y, Y).mean() - 2 * _rbf(X, Y).mean()

    observed = _mmd2(ref_latents, cur_latents)

    # Permutation p-value
    pooled = np.vstack([ref_latents, cur_latents])
    n = len(ref_latents)
    count = 0
    for _ in range(n_perm):
        idx = np.random.permutation(len(pooled))
        if _mmd2(pooled[idx[:n]], pooled[idx[n:]]) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)

    return dict(test="MMD (RBF kernel)", statistic=float(observed),
                p_value=p, drifted=p < alpha)


def per_feature_ks(ref_latents, cur_latents, alpha=0.05):
    """KS test on each of the 64 latent dimensions independently.

    Shows *which* dimensions shifted — useful for understanding
    what kind of visual features changed.
    """
    D = ref_latents.shape[1]
    stats = np.zeros(D)
    pvals = np.zeros(D)
    for d in range(D):
        stats[d], pvals[d] = ks_2samp(ref_latents[:, d], cur_latents[:, d])
    n_drifted = int((pvals < alpha).sum())
    return dict(test="Per-Feature KS", n_features=D, n_drifted=n_drifted,
                fraction_drifted=n_drifted / D,
                stats=stats, p_values=pvals)


# ═════════════════════════════════════════════════════════════════════════════
# CONCEPT DRIFT
# ═════════════════════════════════════════════════════════════════════════════

def prediction_distribution_test(ref_preds, cur_preds, n_classes=5, alpha=0.05):
    """Chi² test: did the model's class predictions shift?

    If the model predicts "Forest" 40% of the time on training data
    but only 5% on custom data, that's concept drift.
    """
    ref_counts = np.bincount(ref_preds.astype(int), minlength=n_classes)
    cur_counts = np.bincount(cur_preds.astype(int), minlength=n_classes)
    table = np.vstack([ref_counts, cur_counts]) + 1  # +1 avoids zero cells
    chi2, p, dof, _ = chi2_contingency(table)
    return dict(test="Chi² (predictions)", statistic=chi2,
                p_value=p, drifted=p < alpha)


def confidence_drift_test(ref_conf, cur_conf, alpha=0.05):
    """KS test: did the model's confidence drop?

    If the model was 90%+ confident on training data but only
    50–60% on custom data, it's struggling with the new inputs.
    """
    stat, p = ks_2samp(ref_conf, cur_conf)
    return dict(test="KS (confidence)", statistic=stat,
                p_value=p, drifted=p < alpha)


# ═════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ═════════════════════════════════════════════════════════════════════════════

def full_drift_report(ref_latents, cur_latents,
                      ref_preds=None, cur_preds=None,
                      ref_conf=None, cur_conf=None,
                      alpha=0.05, mmd_perms=300):
    """Run all drift tests, return list of result dicts."""

    results = []

    # Data drift
    results.append(ks_test(ref_latents, cur_latents, alpha))
    results.append(centroid_shift(ref_latents, cur_latents))
    results.append(mmd_rbf(ref_latents, cur_latents, alpha=alpha, n_perm=mmd_perms))
    results.append(per_feature_ks(ref_latents, cur_latents, alpha))

    # Concept drift
    if ref_preds is not None and cur_preds is not None:
        results.append(prediction_distribution_test(ref_preds, cur_preds, alpha=alpha))
    if ref_conf is not None and cur_conf is not None:
        results.append(confidence_drift_test(ref_conf, cur_conf, alpha))

    return results


def print_report(results):
    """Pretty-print drift results."""
    print(f"\n  {'Test':<24} {'Statistic':>12} {'p-value':>12} {'Drifted?':>10}")
    print("  " + "-" * 60)

    for r in results:
        name = r["test"]

        # Per-Feature KS has a different shape — handle separately
        if name == "Per-Feature KS":
            stat = f"{r['n_drifted']}/{r['n_features']}"
            p    = f"{r['fraction_drifted']:.0%} of dims"
            flag = "YES ⚠" if r["fraction_drifted"] > 0.3 else "no"
            print(f"  {name:<24} {stat:>12} {p:>12} {flag:>10}")
            continue

        stat = f"{r['statistic']:.6f}" if r.get("statistic") is not None else "—"
        p    = f"{r['p_value']:.2e}" if r.get("p_value") is not None else "—"
        flag = "YES ⚠" if r.get("drifted") else (
               "no" if r.get("drifted") is not None else "—")

        extra = ""
        if "z_score" in r:
            extra = f"  (z={r['z_score']:.2f})"

        print(f"  {name:<24} {stat:>12} {p:>12} {flag:>10}{extra}")

    print()
