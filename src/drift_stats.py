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
# ADDITION 1 — HAS margin drift tests
# ═════════════════════════════════════════════════════════════════════════════

def has_margin_drift_test(train_margins, custom_margins, alpha=0.05):
    """KS test on the angular margin distributions (train vs custom).

    Under distribution shift the HAS angular margin erodes.  This test
    detects that erosion using a KS test on the per-sample margin scalars.

    Parameters
    ----------
    train_margins  : 1-D array of shape (N_train,) — margins from training set
    custom_margins : 1-D array of shape (N_custom,) — margins from custom set
    alpha          : significance level

    Returns
    -------
    dict with keys:
        test, statistic, p_value, drifted,
        train_margin_mean, custom_margin_mean, margin_drop
    """
    stat, p = ks_2samp(train_margins, custom_margins)
    train_mean  = float(np.mean(train_margins))
    custom_mean = float(np.mean(custom_margins))
    return dict(
        test="KS (HAS margin)",
        statistic=float(stat),
        p_value=float(p),
        drifted=p < alpha,
        train_margin_mean=train_mean,
        custom_margin_mean=custom_mean,
        margin_drop=train_mean - custom_mean,
    )


# ADDITION 1 — drift direction: which class boundary do drifted samples approach?
def has_boundary_direction_test(train_boundaries, custom_boundaries,
                                class_names, n_classes, alpha=0.05):
    """Chi² test on the distribution of closest-boundary class indices.

    Identifies the direction in which drifted samples move on the unit sphere
    by testing whether the distribution of second-best class indices changed.

    Parameters
    ----------
    train_boundaries  : 1-D int array — closest boundary indices (training)
    custom_boundaries : 1-D int array — closest boundary indices (custom)
    class_names       : list of class name strings, length == n_classes
    n_classes         : number of classes
    alpha             : significance level

    Returns
    -------
    dict with keys:
        test, statistic, p_value, drifted,
        dominant_drift_direction  (human-readable string),
        custom_boundary_counts    (array, one count per class)
    """
    train_counts  = np.bincount(train_boundaries.astype(int),  minlength=n_classes)
    custom_counts = np.bincount(custom_boundaries.astype(int), minlength=n_classes)
    table = np.vstack([train_counts, custom_counts]) + 1  # avoid zero cells
    chi2, p, dof, _ = chi2_contingency(table)

    # Dominant boundary in custom set
    dominant_idx  = int(np.argmax(custom_counts))
    dominant_pct  = 100.0 * custom_counts[dominant_idx] / max(custom_counts.sum(), 1)
    dominant_name = class_names[dominant_idx] if dominant_idx < len(class_names) else str(dominant_idx)
    direction_str = f"→ {dominant_name} ({dominant_pct:.1f}% of samples)"

    return dict(
        test="Chi² (HAS boundary direction)",
        statistic=float(chi2),
        p_value=float(p),
        drifted=p < alpha,
        dominant_drift_direction=direction_str,
        custom_boundary_counts=custom_counts,
    )


# ADDITION 1 — Drift direction matrix: true_class × closest_boundary_class
def has_drift_direction_matrix(true_labels, closest_boundaries,
                               class_names, normalise=True):
    """Build the 5×5 drift direction matrix.

    Rows    = true class of the sample (where the image CAME FROM)
    Columns = closest boundary class   (where the image is DRIFTING TOWARD)

    This is the only way to answer "Mountain → Glacier?" specifically:
    look at row 'Mountain', column 'Glacier'.

    Parameters
    ----------
    true_labels       : 1-D int array  (N,) — ground-truth class indices
    closest_boundaries: 1-D int array  (N,) — second-best cosine class index
    class_names       : list of str, length == n_classes
    normalise         : if True, each row sums to 1 (fraction; easier to read)

    Returns
    -------
    matrix : np.ndarray shape (n_classes, n_classes)
    class_names : list of str  (row/column labels, same order)
    """
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=float)
    for lbl, cb in zip(true_labels.astype(int), closest_boundaries.astype(int)):
        if 0 <= lbl < n and 0 <= cb < n:
            matrix[lbl, cb] += 1.0

    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0          # avoid divide-by-zero
        matrix = matrix / row_sums

    return matrix, class_names


def print_direction_matrix(matrix, class_names, title="Drift Direction Matrix"):
    """Pretty-print the n×n drift direction matrix to stdout."""
    n = len(class_names)
    col_w = 10
    name_w = 12
    print(f"\n  {title}")
    print(f"  Rows = True Class   |   Columns = Closest Boundary Class")
    header = " " * (name_w + 2) + "".join(f"{c:>{col_w}}" for c in class_names)
    print("  " + header)
    print("  " + "-" * len(header))
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:<{name_w}}  "
        for j in range(n):
            val = matrix[i, j]
            mark = " ◄" if i != j and val == matrix[i].max() and val > 0.15 else ""
            row_str += f"{val:>{col_w - 2}.1%}  "
        # Add dominant off-diagonal arrow annotation
        off_diag = matrix[i].copy(); off_diag[i] = 0
        dominant_j = int(off_diag.argmax())
        if off_diag[dominant_j] > 0.15:
            row_str += f"  → {class_names[dominant_j]} ({off_diag[dominant_j]:.1%})"
        print("  " + row_str)
    print()


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


# ═════════════════════════════════════════════════════════════════════════════
# ADDITION 2 — Hierarchical subclass drift detection
# ═════════════════════════════════════════════════════════════════════════════

def hierarchical_drift_report(latents_by_key, confs_by_key,
                               ref_latents, ref_confs,
                               sigma, class_names, alpha=0.05):
    """Three-level hierarchical drift analysis.

    Runs KS tests (latent norms) and confidence KS tests for each subclass
    against the full training reference, then aggregates to the class level
    using a majority-vote consensus rule (>50% of subclasses must drift for
    the class to be flagged as drifted).

    Parameters
    ----------
    latents_by_key : dict mapping "ClassName/subname" → np.ndarray (N_sub, 64)
    confs_by_key   : dict mapping "ClassName/subname" → np.ndarray (N_sub,)
    ref_latents    : training latents (N_train, 64)
    ref_confs      : training confidences (N_train,)
    sigma          : unused here (kept for consistent caller signature)
    class_names    : list of top-level class name strings
    alpha          : significance level for KS tests

    Returns
    -------
    Nested dict:
        {
          "ClassName": {
            "class_drifted": bool,
            "subclass_fraction_drifted": float,
            "subclasses": {
              "subname": {
                "data_drifted": bool,
                "concept_drifted": bool,
                "n_samples": int,
                "ks_stat": float,
                "ks_p": float,
                "conf_mean": float,
              }
            }
          }
        }
    """
    ref_norms = np.linalg.norm(ref_latents, axis=1)

    # Group subclass keys by top-level class name
    class_to_subs = {cn: {} for cn in class_names}
    for key in latents_by_key:
        parts = key.split("/", 1)
        cls_name = parts[0]
        sub_name = parts[1] if len(parts) > 1 else parts[0]
        if cls_name in class_to_subs:
            class_to_subs[cls_name][sub_name] = key

    result = {}
    for cls_name in class_names:
        sub_map = class_to_subs.get(cls_name, {})
        subclass_results = {}

        n_drifted_subs = 0
        n_total_subs   = 0

        for sub_name, key in sub_map.items():
            sub_latents = latents_by_key[key]
            sub_confs   = confs_by_key[key]

            if len(sub_latents) < 2:
                # Not enough samples for a test — skip
                continue

            n_total_subs += 1
            sub_norms = np.linalg.norm(sub_latents, axis=1)

            # Data drift: KS on latent norms
            ks_stat, ks_p = ks_2samp(ref_norms, sub_norms)
            data_drifted = ks_p < alpha

            # Concept drift: KS on confidence
            _, conf_p = ks_2samp(ref_confs, sub_confs)
            concept_drifted = conf_p < alpha

            if data_drifted or concept_drifted:
                n_drifted_subs += 1

            subclass_results[sub_name] = dict(
                data_drifted=bool(data_drifted),
                concept_drifted=bool(concept_drifted),
                n_samples=int(len(sub_latents)),
                ks_stat=float(ks_stat),
                ks_p=float(ks_p),
                conf_mean=float(np.mean(sub_confs)),
            )

        frac = n_drifted_subs / n_total_subs if n_total_subs > 0 else 0.0
        class_drifted = frac > 0.5

        result[cls_name] = dict(
            class_drifted=bool(class_drifted),
            subclass_fraction_drifted=float(frac),
            subclasses=subclass_results,
        )

    return result
