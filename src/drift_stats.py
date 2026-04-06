import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# DATA DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def mmd_rbf(ref_latents, cur_latents, gamma=None, alpha=0.05, n_perm=300):
    """MMD with RBF kernel — permutation test on the full 64-D distributions."""
    max_n = 2000
    if len(ref_latents) > max_n:
        ref_latents = ref_latents[np.random.choice(len(ref_latents), max_n, replace=False)]
    if len(cur_latents) > max_n:
        cur_latents = cur_latents[np.random.choice(len(cur_latents), max_n, replace=False)]

    if gamma is None:
        combined = np.vstack([ref_latents, cur_latents])
        dists = cdist(combined, combined, "sqeuclidean")
        gamma = 1.0 / np.median(dists[dists > 0])

    def _rbf(X, Y):
        return np.exp(-gamma * cdist(X, Y, "sqeuclidean"))

    def _mmd2(X, Y):
        return _rbf(X, X).mean() + _rbf(Y, Y).mean() - 2 * _rbf(X, Y).mean()

    observed = _mmd2(ref_latents, cur_latents)
    pooled = np.vstack([ref_latents, cur_latents])
    n = len(ref_latents)
    count = 0
    for _ in range(n_perm):
        idx = np.random.permutation(len(pooled))
        if _mmd2(pooled[idx[:n]], pooled[idx[n:]]) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return dict(test="MMD (RBF)", statistic=float(observed),
                p_value=float(p), drifted=bool(p < alpha))


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT DRIFT — confidence baseline
# ─────────────────────────────────────────────────────────────────────────────

def confidence_drift_test(ref_conf, cur_conf, alpha=0.05):
    """KS test on softmax confidence — used for both models as baseline signal."""
    stat, p = ks_2samp(ref_conf, cur_conf)
    return dict(test="KS (confidence)", statistic=float(stat),
                p_value=float(p), drifted=bool(p < alpha))


# ─────────────────────────────────────────────────────────────────────────────
# HAS-NATIVE TESTS — novel contribution
# ─────────────────────────────────────────────────────────────────────────────

def has_margin_drift_test(train_margins, custom_margins, alpha=0.05):
    """KS test on angular margin distributions.

    margin = cos_best − cos_second_best.
    HASeparator maximises this during training; its erosion is concept drift.
    Scale-independent: unaffected by HAS_SCALE unlike softmax confidence.
    """
    stat, p = ks_2samp(train_margins, custom_margins)
    return dict(
        test="KS (HAS margin)",
        statistic=float(stat),
        p_value=float(p),
        drifted=bool(p < alpha),
        train_margin_mean=float(np.mean(train_margins)),
        custom_margin_mean=float(np.mean(custom_margins)),
        margin_drop=float(np.mean(train_margins) - np.mean(custom_margins)),
    )


def has_boundary_direction_test(train_boundaries, custom_boundaries,
                                class_names, n_classes, alpha=0.05):
    """Chi² on second-best class distribution — tests directional drift.

    If the distribution of closest-boundary classes shifts, samples are
    drifting toward a specific class boundary.
    """
    tc = np.bincount(train_boundaries.astype(int),  minlength=n_classes)
    cc = np.bincount(custom_boundaries.astype(int), minlength=n_classes)
    chi2, p, _, _ = chi2_contingency(np.vstack([tc, cc]) + 1)
    dom = int(np.argmax(cc))
    dom_pct = 100.0 * cc[dom] / max(cc.sum(), 1)
    return dict(
        test="Chi² (boundary direction)",
        statistic=float(chi2),
        p_value=float(p),
        drifted=bool(p < alpha),
        dominant_drift_direction=(
            f"→ {class_names[dom] if dom < len(class_names) else dom} "
            f"({dom_pct:.1f}%)"
        ),
        custom_boundary_counts=cc,
    )


def has_drift_direction_matrix(true_labels, closest_boundaries,
                               class_names, normalise=True):
    """N×N drift direction matrix.

    matrix[i, j] = fraction of true-class-i samples whose closest HAS
    boundary is class j.  Row = origin, Column = destination.
    Diagonal = self-boundary (not a drift signal).
    """
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=float)
    for lbl, cb in zip(true_labels.astype(int), closest_boundaries.astype(int)):
        if 0 <= lbl < n and 0 <= cb < n:
            matrix[lbl, cb] += 1.0
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix /= row_sums
    return matrix, class_names


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL DRIFT
# ─────────────────────────────────────────────────────────────────────────────

def hierarchical_drift_report(latents_by_key, confs_by_key,
                               ref_latents, ref_confs,
                               class_names, alpha=0.05):
    """Three-level drift: dataset → class → subclass.

    Consensus rule: a class is drifted only if >50 % of its subclasses drift.
    Returns nested dict {class: {class_drifted, subclass_fraction_drifted,
                                  subclasses: {sub: {data_drifted,
                                  concept_drifted, n_samples,
                                  ks_stat, ks_p, conf_mean}}}}
    """
    ref_norms = np.linalg.norm(ref_latents, axis=1)
    class_to_subs = {cn: {} for cn in class_names}
    for key in latents_by_key:
        parts = key.split("/", 1)
        cls, sub = parts[0], (parts[1] if len(parts) > 1 else parts[0])
        if cls in class_to_subs:
            class_to_subs[cls][sub] = key

    result = {}
    for cls in class_names:
        sub_results = {}
        n_drifted = n_total = 0
        for sub, key in class_to_subs.get(cls, {}).items():
            slat, sconf = latents_by_key[key], confs_by_key[key]
            if len(slat) < 2:
                continue
            n_total += 1
            ks_stat, ks_p  = ks_2samp(ref_norms, np.linalg.norm(slat, axis=1))
            _,       conf_p = ks_2samp(ref_confs, sconf)
            dd, cd = bool(ks_p < alpha), bool(conf_p < alpha)
            if dd or cd:
                n_drifted += 1
            sub_results[sub] = dict(
                data_drifted=dd, concept_drifted=cd,
                n_samples=int(len(slat)),
                ks_stat=float(ks_stat), ks_p=float(ks_p),
                conf_mean=float(np.mean(sconf)),
            )
        frac = n_drifted / n_total if n_total > 0 else 0.0
        result[cls] = dict(
            class_drifted=bool(frac > 0.5),
            subclass_fraction_drifted=float(frac),
            subclasses=sub_results,
        )
    return result
