"""
Full analysis for the Stage 1A re-verification: primary Wilcoxon test (4
comparisons, Holm-corrected) plus the three robustness analyses and the
tertiary mixed model, exactly as specified in DESIGN.md.

Prints a full report and saves a summary dict to
results/stage1a_reverification_analysis.pkl.
"""
import itertools
import pickle

import numpy as np
from scipy.stats import wilcoxon

from reverification_core import RESULTS_PATH, STOCHASTIC_CONTROLS, N_SEEDS

ANALYSIS_PATH = RESULTS_PATH.replace(
    "stage1a_reverification_results.pkl", "stage1a_reverification_analysis.pkl")

# Comparison order fixed here and used consistently for Holm correction indexing.
COMPARISONS = ["hist_random", "curr_random", "rewired", "lattice"]
COMPARISON_LABELS = {
    "hist_random": "T vs. historical half-edge random, coupling-budget normalized",
    "curr_random": "T vs. current edge-count-matched random",
    "rewired": "T vs. degree-preserving rewiring",
    "lattice": "T vs. lattice",
}

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42
SIGNIFICANCE_ALPHA = 0.05


def load_results():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    assert len(results) == 770, f"expected 770 results, got {len(results)}"
    return results


def hodges_lehmann(d):
    """Median of all pairwise Walsh averages (i<=j) of the paired differences."""
    d = np.asarray(d)
    n = len(d)
    walsh = [(d[i] + d[j]) / 2 for i, j in itertools.combinations_with_replacement(range(n), 2)]
    return float(np.median(walsh))


def holm_correct(p_values):
    """Standard Holm step-down correction. Returns adjusted p-values in the
    SAME order as the input (not sorted)."""
    p_values = np.asarray(p_values)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def exact_sign_flip_test(d):
    """Exact 2^n sign-flip test on the mean of the n class-level
    differences. Two-sided p-value: fraction of the 2^n sign patterns
    whose |mean| >= |observed mean|."""
    d = np.asarray(d)
    n = len(d)
    observed = np.mean(d)
    count = 0
    total = 0
    for signs in itertools.product([1, -1], repeat=n):
        flipped_mean = np.mean(np.asarray(signs) * d)
        if abs(flipped_mean) >= abs(observed) - 1e-12:
            count += 1
        total += 1
    assert total == 2 ** n
    return observed, count / total


def build_class_level_arrays(results, construction):
    """Returns A_cT (10,), and either Ā_cg/A_cgs_matrix (stochastic) or
    A_c_lattice (10,) (deterministic)."""
    A_cT = np.array([results[(c, "T", None)] for c in range(10)])
    if construction == "lattice":
        A_c = np.array([results[(c, "lattice", None)] for c in range(10)])
        return A_cT, A_c, None
    A_cgs = np.array([[results[(c, construction, s)] for s in range(N_SEEDS)] for c in range(10)])
    return A_cT, A_cgs.mean(axis=1), A_cgs


def seed_stability_diagnostic(A_cgs):
    """Descriptive only: class-level mean using the first 5/10/15/20/25
    seeds, per class. Returns dict of {k: (10,) array}."""
    out = {}
    for k in (5, 10, 15, 20, 25):
        out[k] = A_cgs[:, :k].mean(axis=1)
    return out


def within_class_mcse(A_cgs):
    return A_cgs.std(axis=1, ddof=1) / np.sqrt(A_cgs.shape[1])


def primary_wilcoxon(d):
    stat, p = wilcoxon(d, alternative="two-sided", mode="exact")
    return {
        "diffs": d.tolist(),
        "median_diff": float(np.median(d)),
        "hodges_lehmann": hodges_lehmann(d),
        "sign_positive": int(np.sum(d > 0)),
        "n": len(d),
        "W": float(stat),
        "p_exact": float(p),
    }


def hierarchical_bootstrap_stochastic(A_cT, A_cgs, rng):
    n_classes, n_seeds = A_cgs.shape
    boot_means = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        class_idx = rng.integers(0, n_classes, size=n_classes)
        class_diffs = np.empty(n_classes)
        for i, c in enumerate(class_idx):
            seed_idx = rng.integers(0, n_seeds, size=n_seeds)
            class_diffs[i] = A_cT[c] - A_cgs[c, seed_idx].mean()
        boot_means[b] = class_diffs.mean()
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return ci, boot_means


def bootstrap_deterministic(d, rng):
    """One-level (class-only) bootstrap, for lattice: there is no seed
    axis to resample since the construction is deterministic. This is a
    deliberate reduction of DESIGN.md's two-level scheme, forced by
    lattice having no stochastic axis -- documented here rather than
    silently applying a no-op inner resample."""
    n_classes = len(d)
    boot_means = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        class_idx = rng.integers(0, n_classes, size=n_classes)
        boot_means[b] = d[class_idx].mean()
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return ci, boot_means


def mixed_model_stochastic(A_cT, A_cgs):
    """Tertiary check: d_cgs = A_cT - A_cgs = mu + u_c + e_cgs, class
    random intercept, via statsmodels MixedLM (REML)."""
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM

    n_classes, n_seeds = A_cgs.shape
    rows = []
    for c in range(n_classes):
        for s in range(n_seeds):
            rows.append({"d": A_cT[c] - A_cgs[c, s], "class": c})
    df = pd.DataFrame(rows)
    model = MixedLM.from_formula("d ~ 1", groups="class", data=df)
    fit = model.fit(reml=True)
    mu = float(fit.params["Intercept"])
    ci_low, ci_high = fit.conf_int().loc["Intercept"]
    return {"mu": mu, "ci": (float(ci_low), float(ci_high)), "converged": bool(fit.converged)}


def report_comparison(name, results):
    print(f"\n{'=' * 70}\n{COMPARISON_LABELS[name]}\n{'=' * 70}")
    A_cT, A_c_agg, A_cgs = build_class_level_arrays(results, name)
    d_mean = A_cT - A_c_agg

    print(f"Class-level differences (T - {name}, mean-aggregated):")
    for c in range(10):
        print(f"  class {c}: A_T={A_cT[c]:.4f}, agg={A_c_agg[c]:.4f}, d={d_mean[c]:+.4f}")

    primary = primary_wilcoxon(d_mean)
    print(f"Primary (mean-aggregated) Wilcoxon: W={primary['W']:.4f}, "
          f"p_exact={primary['p_exact']:.5f}, median_d={primary['median_diff']:+.4f}, "
          f"HL={primary['hodges_lehmann']:+.4f}, sign+={primary['sign_positive']}/10")

    out = {"comparison": name, "A_cT": A_cT.tolist(), "A_c_agg_mean": A_c_agg.tolist(),
           "d_mean": d_mean.tolist(), "primary": primary}

    if A_cgs is not None:
        stability = seed_stability_diagnostic(A_cgs)
        mcse = within_class_mcse(A_cgs)
        print("Seed-count stability diagnostic (class-level mean AUC at k seeds), descriptive only:")
        for c in range(10):
            vals = ", ".join(f"k={k}:{stability[k][c]:.3f}" for k in (5, 10, 15, 20, 25))
            print(f"  class {c}: {vals}")
        print("Within-class MCSE (SD_s / sqrt(25)):")
        for c in range(10):
            print(f"  class {c}: MCSE={mcse[c]:.4f} (|d|={abs(d_mean[c]):.4f})")
        out["stability_diagnostic"] = {k: v.tolist() for k, v in stability.items()}
        out["mcse"] = mcse.tolist()

        # Robustness 1: median seed aggregation
        A_c_median = np.median(A_cgs, axis=1)
        d_median = A_cT - A_c_median
        median_test = primary_wilcoxon(d_median)
        print(f"Robustness 1 (median-aggregated) Wilcoxon: W={median_test['W']:.4f}, "
              f"p_exact={median_test['p_exact']:.5f}, sign+={median_test['sign_positive']}/10")
        out["median_aggregated"] = {"A_c_agg": A_c_median.tolist(), "d": d_median.tolist(),
                                     "test": median_test}
    else:
        d_median = d_mean  # deterministic; mean == median == the only value
        median_test = primary
        out["median_aggregated"] = {"A_c_agg": A_c_agg.tolist(), "d": d_median.tolist(),
                                     "test": median_test}

    # Robustness 2: exact sign-flip test (on the mean-aggregated differences)
    observed_stat, signflip_p = exact_sign_flip_test(d_mean)
    print(f"Robustness 2 (exact sign-flip, 1024 flips): observed mean d={observed_stat:+.4f}, "
          f"two-sided p={signflip_p:.5f}")
    out["sign_flip"] = {"observed_mean": float(observed_stat), "p": float(signflip_p)}

    # Robustness 3: hierarchical bootstrap
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if A_cgs is not None:
        ci, boot_means = hierarchical_bootstrap_stochastic(A_cT, A_cgs, rng)
    else:
        ci, boot_means = bootstrap_deterministic(d_mean, rng)
    zero_outside = ci[0] > 0 or ci[1] < 0
    print(f"Robustness 3 (hierarchical bootstrap, B={BOOTSTRAP_B}): "
          f"95% CI for mean class-level difference = [{ci[0]:+.4f}, {ci[1]:+.4f}], "
          f"zero {'OUTSIDE' if zero_outside else 'inside'} the interval")
    out["bootstrap"] = {"ci": ci, "zero_outside": bool(zero_outside)}

    # Consistency check across primary / median / sign-flip, gating the tertiary mixed model
    primary_sig = primary["p_exact"] < SIGNIFICANCE_ALPHA
    median_sig = median_test["p_exact"] < SIGNIFICANCE_ALPHA
    signflip_sig = signflip_p < SIGNIFICANCE_ALPHA
    same_sign = (np.sign(primary["median_diff"]) == np.sign(median_test["median_diff"]) == np.sign(observed_stat))
    consistent = (primary_sig == median_sig == signflip_sig) and same_sign
    print(f"Consistency across primary/median/sign-flip: significance flags "
          f"({primary_sig}, {median_sig}, {signflip_sig}), same sign={same_sign} "
          f"-> {'CONSISTENT' if consistent else 'INCONSISTENT'}")
    out["consistent"] = bool(consistent)

    # Robustness 4: mixed model (tertiary), gated on consistency, and only meaningful
    # when there's a seed axis to attribute residual variance to.
    if consistent and A_cgs is not None:
        mm = mixed_model_stochastic(A_cT, A_cgs)
        print(f"Tertiary mixed model: mu={mm['mu']:+.4f}, 95% CI=[{mm['ci'][0]:+.4f}, {mm['ci'][1]:+.4f}], "
              f"converged={mm['converged']}")
        out["mixed_model"] = mm
    elif not consistent:
        print("Tertiary mixed model: SKIPPED (robustness checks disagree; see DESIGN.md's decision rule)")
        out["mixed_model"] = None
    else:
        print("Tertiary mixed model: SKIPPED (lattice is deterministic -- no per-seed residual "
              "variance for a class random-intercept model to separate from)")
        out["mixed_model"] = None

    return out


def main():
    results = load_results()
    all_out = {}
    p_values_for_holm = []
    for name in COMPARISONS:
        out = report_comparison(name, results)
        all_out[name] = out
        p_values_for_holm.append(out["primary"]["p_exact"])

    holm_p = holm_correct(p_values_for_holm)
    print(f"\n{'=' * 70}\nHolm correction across the 4 primary comparisons\n{'=' * 70}")
    for name, p_raw, p_holm in zip(COMPARISONS, p_values_for_holm, holm_p):
        sig = "SIGNIFICANT" if p_holm < SIGNIFICANCE_ALPHA else "not significant"
        print(f"  {COMPARISON_LABELS[name]}: p_exact={p_raw:.5f}, p_holm={p_holm:.5f} -> {sig}")
        all_out[name]["primary"]["p_holm"] = float(p_holm)

    with open(ANALYSIS_PATH, "wb") as f:
        pickle.dump(all_out, f)
    print(f"\nSaved full analysis to {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
