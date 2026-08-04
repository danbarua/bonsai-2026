"""
Post hoc, exploratory graph-to-graph pairwise comparison among the four
evolved conditions (T, lattice, rewired, curr_random), per external
review's own explicit recommendation in FINDINGS.md's secondary-
comparisons section: "these could be computed from the already-saved
per-image losses, but any such comparison would now be explicitly post
hoc and should use multiplicity correction across however many pairwise
tests it involved."

NO NEW SIMULATION, NO NEW GPU TIME: reuses the per-image test-set losses
already computed and saved by run_confirmatory_evaluation.py
(results/stage4_confirmatory_results.pkl). This is a new bootstrap
computation on existing data only.

Identical statistical procedure to the locked primary/secondary
comparisons (DESIGN.md's "Confirmatory endpoint and test"): for each
pair, d_i = ell_i(graph_A) - ell_i(graph_B), 20,000 paired
class-stratified bootstrap resamples, two-sided 95% percentile interval
on mean d_i. This is new, un-pre-registered multiple-comparison
territory (DESIGN.md only locked the four graph-vs-pre-evolution tests)
-- Holm-Bonferroni correction across all six comparisons, as one family,
is applied here, not optional.
"""
import os
import pickle

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

N_RESAMPLES = 20000
BOOTSTRAP_SEED = 42
EVOLVED_CONDITIONS = ["evolved_T", "evolved_lattice", "evolved_rewired", "evolved_curr_random"]


def paired_class_stratified_bootstrap(d, y, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Identical procedure to run_confirmatory_evaluation.py's function --
    copied verbatim, not reimplemented differently, so this new analysis
    uses exactly the same machinery as the locked primary/secondary tests."""
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
    return {"resampled_means": means, "ci_low": float(lo), "ci_high": float(hi),
            "observed_mean": float(np.mean(d))}


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
    """Standard step-down Holm-Bonferroni. Returns (adjusted_p, rejected)
    dicts keyed the same way raw_p is, adjusted_p monotone non-decreasing
    in sorted-p order, rejected = whether that comparison survives at
    alpha after correction."""
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


def main():
    print("Loading already-saved per-image test-set losses (no new simulation)...")
    with open(os.path.join(RESULTS_DIR, "stage4_confirmatory_results.pkl"), "rb") as f:
        d = pickle.load(f)
    ell_i = d["condition_ell_i"]
    y_test = d["y_test"]

    pairs = []
    for i in range(len(EVOLVED_CONDITIONS)):
        for j in range(i + 1, len(EVOLVED_CONDITIONS)):
            pairs.append((EVOLVED_CONDITIONS[i], EVOLVED_CONDITIONS[j]))
    assert len(pairs) == 6, f"expected 6 pairs, got {len(pairs)}"

    print(f"\nRunning all {len(pairs)} pairwise comparisons, "
          f"{N_RESAMPLES} paired class-stratified bootstrap resamples each...\n")
    results = {}
    for a, b in pairs:
        d_ab = ell_i[a] - ell_i[b]
        boot = paired_class_stratified_bootstrap(d_ab, y_test)
        p = bootstrap_two_sided_p(boot["resampled_means"], N_RESAMPLES)
        if boot["ci_high"] < 0:
            verdict = f"{a} IMPROVES over {b}"
        elif boot["ci_low"] > 0:
            verdict = f"{b} IMPROVES over {a}"
        else:
            verdict = "NULL (straddles zero)"
        results[(a, b)] = {"observed_mean": boot["observed_mean"], "ci_low": boot["ci_low"],
                            "ci_high": boot["ci_high"], "raw_p": p, "verdict": verdict}
        print(f"[{a} vs {b}] mean_d={boot['observed_mean']:+.4f}, "
              f"95% CI=[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], raw_p={p:.4e} -> {verdict}")

    raw_p = {k: v["raw_p"] for k, v in results.items()}
    adjusted, rejected = holm_bonferroni(raw_p, alpha=0.05)

    print(f"\n{'='*70}\nHOLM-BONFERRONI CORRECTION, family of {len(pairs)}, alpha=0.05\n{'='*70}")
    order = sorted(raw_p.keys(), key=lambda k: raw_p[k])
    for rank, key in enumerate(order, start=1):
        a, b = key
        print(f"  rank {rank}: [{a} vs {b}] raw_p={raw_p[key]:.4e}, "
              f"holm_adj_p={adjusted[key]:.4e}, survives={rejected[key]}")

    n_survive = sum(rejected.values())
    print(f"\n{n_survive} of {len(pairs)} pairwise comparisons survive Holm correction at alpha=0.05.")

    with open(os.path.join(RESULTS_DIR, "stage4_posthoc_pairwise_results.pkl"), "wb") as f:
        pickle.dump({"results": results, "raw_p": raw_p, "holm_adjusted_p": adjusted,
                     "holm_rejected": rejected}, f)
    print(f"\nSaved results/stage4_posthoc_pairwise_results.pkl")


if __name__ == "__main__":
    main()
