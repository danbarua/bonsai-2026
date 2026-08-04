"""
Stage 2A's confirmatory statistical machinery -- the paired
class-stratified bootstrap, per-image log-loss class indexing, exact
McNemar's test, and the Holm-Bonferroni step-down correction used by
the post hoc graph-to-graph comparison. Factored out of
`run_confirmatory_evaluation.py` (which previously defined these
functions itself, then `run_posthoc_graph_pairwise.py` duplicated the
bootstrap function verbatim rather than importing it) into one shared,
unit-tested module -- both scripts now import from here, so there is
exactly one implementation of each statistic to verify and no risk of
the two copies silently drifting apart.

Per the reproducibility gaps flagged in external review (FINDINGS.md's
"Reproducibility gaps" section): "add unit tests for the paired
class-stratified bootstrap, the per-image log-loss class indexing, and
the McNemar contingency-count construction" -- see
`tests/test_stage2a_stats.py`.
"""
import numpy as np
from scipy.stats import binomtest

N_RESAMPLES = 20000
BOOTSTRAP_SEED = 42


def per_image_log_loss(y_true, proba, classes):
    """ell_i for each image -- sklearn's log_loss gives only the mean;
    the locked test needs the per-image value d_i is built from."""
    class_to_col = {c: j for j, c in enumerate(classes)}
    cols = np.array([class_to_col[y] for y in y_true])
    p_true = proba[np.arange(len(y_true)), cols]
    eps = 1e-15
    p_true = np.clip(p_true, eps, 1 - eps)
    return -np.log(p_true)


def paired_class_stratified_bootstrap(d, y, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """DESIGN.md's locked primary test: 20,000 paired, class-stratified
    bootstrap resamples (each resample preserves each class's original
    count, drawn with replacement within class), mean per-image d_i on
    each resample, two-sided 95% percentile interval. Vectorized per
    class rather than materializing a full resampled index array per
    draw."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    total_n = len(d)
    sums = np.zeros(n_resamples)
    for c in classes:
        idx_c = np.where(y == c)[0]
        n_c = len(idx_c)
        d_c = d[idx_c]
        draws = rng.integers(0, n_c, size=(n_resamples, n_c))
        sums += d_c[draws].sum(axis=1)
    means = sums / total_n
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "resampled_means": means, "ci_low": float(lo), "ci_high": float(hi),
        "observed_mean": float(np.mean(d)),
    }


def mcnemar_exact(y_true, pred_a, pred_b, label_a, label_b):
    """Exact McNemar's test on the discordant pairs (A wrong/B right vs.
    A right/B wrong), via a two-sided exact binomial test on the
    discordant counts -- the standard 'exact McNemar' construction,
    implemented directly via scipy.stats.binomtest rather than adding a
    new dependency for it."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    n_b_only = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n_a_only = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    n_discordant = n_a_only + n_b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        k = min(n_a_only, n_b_only)
        p_value = binomtest(k, n_discordant, 0.5, alternative="two-sided").pvalue
    return {
        f"n_{label_a}_only_correct": n_a_only, f"n_{label_b}_only_correct": n_b_only,
        "n_discordant": n_discordant, "p_value": float(p_value),
    }


def bootstrap_two_sided_p(resampled_means, n_resamples):
    """Two-sided p-value from a percentile bootstrap distribution, via the
    double-the-smaller-tail method, with the Monte Carlo floor convention
    (CLAUDE.md principle 6) applied to each one-sided tail so a p-value is
    never reported as exactly zero when zero resamples cross zero."""
    n_below = int(np.sum(resampled_means < 0))
    n_above = int(np.sum(resampled_means > 0))
    p_low = (1 + n_above) / (n_resamples + 1)   # H0: mean_d >= 0
    p_high = (1 + n_below) / (n_resamples + 1)  # H0: mean_d <= 0
    return min(1.0, 2 * min(p_low, p_high))


def holm_bonferroni(raw_p, alpha=0.05):
    """Standard step-down Holm-Bonferroni. `raw_p` is a dict of
    key -> p-value. Returns (adjusted_p, rejected) dicts keyed the same
    way, adjusted_p monotone non-decreasing in sorted-p order, rejected =
    whether that comparison survives at alpha after correction."""
    m = len(raw_p)
    order = sorted(raw_p.keys(), key=lambda k: raw_p[k])
    adjusted = {}
    rejected = {}
    running_max = 0.0
    still_rejecting = True
    for i, key in enumerate(order):  # i = 0-indexed rank
        factor = m - i
        adj = min(1.0, factor * raw_p[key])
        running_max = max(running_max, adj)
        adjusted[key] = running_max
        if still_rejecting and raw_p[key] <= alpha / factor:
            rejected[key] = True
        else:
            still_rejecting = False
            rejected[key] = False
    return adjusted, rejected
