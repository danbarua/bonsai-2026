"""
Stage 1A re-verification, log-scale iteration (v2). Per
DESIGN_v2_log_scale.md.

PURE RE-ANALYSIS: reuses v1's already-collected raw per-instance AUC
values directly from results/stage1a_reverification_results.pkl (keyed
(class, construction, seed) -> raw AUC, confirmed to hold all 770
individual instances, not just class-level aggregates -- checked before
writing this script). No new joint_tangent_matrix_response call, no new
seed, no new graph construction happens anywhere in this file.

Reuses v1's transform-agnostic statistical primitives
(holm_correct, hodges_lehmann, exact_sign_flip_test,
hierarchical_bootstrap_stochastic, mixed_model_stochastic) unchanged,
imported from analyze_stage1a_reverification.py -- they operate on
whatever array they're handed and have no built-in assumption about
raw-vs-log scale, so passing them log-transformed arrays instead of raw
AUC arrays is a legitimate reuse, not a repurposing that changes their
behavior.
"""
import pickle

import numpy as np
from scipy.stats import wilcoxon

from reverification_core import RESULTS_PATH, N_SEEDS
from analyze_stage1a_reverification import (
    holm_correct, hodges_lehmann, exact_sign_flip_test,
    hierarchical_bootstrap_stochastic, mixed_model_stochastic,
    SIGNIFICANCE_ALPHA, BOOTSTRAP_SEED,
)

ANALYSIS_PATH = RESULTS_PATH.replace(
    "stage1a_reverification_results.pkl", "stage1a_log_scale_analysis.pkl")

# Lattice excluded per DESIGN_v2_log_scale.md: no seed axis, nothing for
# the log transform to change, v1's result for it already stands as final.
COMPARISONS = ["hist_random", "curr_random", "rewired"]
COMPARISON_LABELS = {
    "hist_random": "T vs. historical half-edge random, coupling-budget normalized",
    "curr_random": "T vs. current edge-count-matched random",
    "rewired": "T vs. degree-preserving rewiring",
}


def load_results():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    assert len(results) == 770, f"expected 770 raw instances from v1, got {len(results)}"
    return results


def build_log_class_level_arrays(results, construction):
    """L_cT: (10,) log(A_cT). L_cgs: (10, 25) log(A_cgs) -- the raw
    per-seed log values (not yet aggregated)."""
    L_cT = np.array([np.log(results[(c, "T", None)]) for c in range(10)])
    L_cgs = np.array([[np.log(results[(c, construction, s)]) for s in range(N_SEEDS)]
                       for c in range(10)])
    return L_cT, L_cgs


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


def add_backtransform(test_dict):
    """Adds exp() back-transformed multiplicative versions of the
    log-scale median and Hodges-Lehmann estimate -- per DESIGN_v2's
    instruction to report both scales wherever a difference is
    estimated."""
    test_dict["median_diff_multiplicative"] = float(np.exp(test_dict["median_diff"]))
    test_dict["hodges_lehmann_multiplicative"] = float(np.exp(test_dict["hodges_lehmann"]))
    return test_dict


def seed_stability_diagnostic_log(L_cgs):
    out = {}
    for k in (5, 10, 15, 20, 25):
        out[k] = L_cgs[:, :k].mean(axis=1)
    return out


def within_class_mcse_log(L_cgs):
    return L_cgs.std(axis=1, ddof=1) / np.sqrt(L_cgs.shape[1])


def report_comparison(name, results):
    print(f"\n{'=' * 70}\n{COMPARISON_LABELS[name]} (log scale)\n{'=' * 70}")
    L_cT, L_cgs = build_log_class_level_arrays(results, name)
    Lbar_cg = L_cgs.mean(axis=1)  # log-mean = log(geometric mean of raw AUC)
    d_log_mean = L_cT - Lbar_cg

    print("Class-level log-differences (log(A_T) - log-mean(A_" + name + ")):")
    for c in range(10):
        print(f"  class {c}: log(A_T)={L_cT[c]:+.4f}, log-mean={Lbar_cg[c]:+.4f}, "
              f"d_log={d_log_mean[c]:+.4f} (multiplicative ratio exp(d)={np.exp(d_log_mean[c]):.4f})")

    primary = add_backtransform(primary_wilcoxon(d_log_mean))
    print(f"Primary (log-mean-aggregated) Wilcoxon: W={primary['W']:.4f}, "
          f"p_exact={primary['p_exact']:.5f}, median_d_log={primary['median_diff']:+.4f} "
          f"(x{primary['median_diff_multiplicative']:.4f}), "
          f"HL_log={primary['hodges_lehmann']:+.4f} (x{primary['hodges_lehmann_multiplicative']:.4f}), "
          f"sign+={primary['sign_positive']}/10")

    stability = seed_stability_diagnostic_log(L_cgs)
    mcse = within_class_mcse_log(L_cgs)
    print("Seed-count stability diagnostic (class-level log-mean at k seeds), descriptive only:")
    for c in range(10):
        vals = ", ".join(f"k={k}:{stability[k][c]:+.3f}" for k in (5, 10, 15, 20, 25))
        print(f"  class {c}: {vals}")
    print("Within-class MCSE on log scale (SD_s(log A) / sqrt(25)):")
    for c in range(10):
        print(f"  class {c}: MCSE_log={mcse[c]:.4f} (|d_log|={abs(d_log_mean[c]):.4f})")

    out = {
        "comparison": name,
        "L_cT": L_cT.tolist(),
        "Lbar_cg": Lbar_cg.tolist(),
        "d_log_mean": d_log_mean.tolist(),
        "primary": primary,
        "stability_diagnostic": {k: v.tolist() for k, v in stability.items()},
        "mcse_log": mcse.tolist(),
    }

    # Robustness 1: median seed aggregation, on log scale
    L_median_cg = np.median(L_cgs, axis=1)
    d_log_median = L_cT - L_median_cg
    median_test = add_backtransform(primary_wilcoxon(d_log_median))
    print(f"Robustness 1 (log-median-aggregated) Wilcoxon: W={median_test['W']:.4f}, "
          f"p_exact={median_test['p_exact']:.5f}, sign+={median_test['sign_positive']}/10")
    out["median_aggregated"] = {"Lmedian_cg": L_median_cg.tolist(), "d_log": d_log_median.tolist(),
                                 "test": median_test}

    # Robustness 2: exact sign-flip test on the log-mean differences
    observed_stat, signflip_p = exact_sign_flip_test(d_log_mean)
    print(f"Robustness 2 (exact sign-flip, 1024 flips): observed mean d_log={observed_stat:+.4f}, "
          f"two-sided p={signflip_p:.5f}")
    out["sign_flip"] = {"observed_mean_log": float(observed_stat), "p": float(signflip_p)}

    # Robustness 3: hierarchical bootstrap, on log scale (reuses v1's function unchanged --
    # it just resamples whatever (A_cT, A_cgs)-shaped arrays it's given).
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ci_log, boot_means_log = hierarchical_bootstrap_stochastic(L_cT, L_cgs, rng)
    ci_multiplicative = (float(np.exp(ci_log[0])), float(np.exp(ci_log[1])))
    zero_outside = ci_log[0] > 0 or ci_log[1] < 0
    print(f"Robustness 3 (hierarchical bootstrap, B=10000): 95% CI for mean log-difference = "
          f"[{ci_log[0]:+.4f}, {ci_log[1]:+.4f}] (multiplicative x[{ci_multiplicative[0]:.4f}, "
          f"{ci_multiplicative[1]:.4f}]), zero {'OUTSIDE' if zero_outside else 'inside'} the interval")
    out["bootstrap"] = {"ci_log": ci_log, "ci_multiplicative": ci_multiplicative,
                         "zero_outside": bool(zero_outside)}

    # Consistency gate, same logic as v1, evaluated in log space
    primary_sig = primary["p_exact"] < SIGNIFICANCE_ALPHA
    median_sig = median_test["p_exact"] < SIGNIFICANCE_ALPHA
    signflip_sig = signflip_p < SIGNIFICANCE_ALPHA
    same_sign = (np.sign(primary["median_diff"]) == np.sign(median_test["median_diff"])
                 == np.sign(observed_stat))
    mcse_small = bool(np.all(mcse < np.abs(d_log_mean)))
    consistent = (primary_sig == median_sig == signflip_sig) and same_sign
    print(f"Consistency across primary/median/sign-flip: significance flags "
          f"({primary_sig}, {median_sig}, {signflip_sig}), same sign={same_sign}, "
          f"MCSE small in every class={mcse_small} -> {'CONSISTENT' if consistent else 'INCONSISTENT'}")
    out["consistent"] = bool(consistent)
    out["mcse_small_in_every_class"] = mcse_small

    # Robustness 4 / tertiary mixed model, gated exactly as v1
    if consistent:
        mm = mixed_model_stochastic(L_cT, L_cgs)
        mm["mu_multiplicative"] = float(np.exp(mm["mu"]))
        mm["ci_multiplicative"] = (float(np.exp(mm["ci"][0])), float(np.exp(mm["ci"][1])))
        print(f"Tertiary mixed model: mu_log={mm['mu']:+.4f} (x{mm['mu_multiplicative']:.4f}), "
              f"95% CI_log=[{mm['ci'][0]:+.4f}, {mm['ci'][1]:+.4f}], converged={mm['converged']}")
        out["mixed_model"] = mm
    else:
        print("Tertiary mixed model: SKIPPED (robustness checks disagree; see DESIGN_v2_log_scale.md's decision rule)")
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
    print(f"\n{'=' * 70}\nHolm correction across the 3 in-scope comparisons (log scale)\n{'=' * 70}")
    for name, p_raw, p_holm in zip(COMPARISONS, p_values_for_holm, holm_p):
        sig = "SIGNIFICANT" if p_holm < SIGNIFICANCE_ALPHA else "not significant"
        print(f"  {COMPARISON_LABELS[name]}: p_exact={p_raw:.5f}, p_holm={p_holm:.5f} -> {sig}")
        all_out[name]["primary"]["p_holm"] = float(p_holm)

    print(f"\n{'=' * 70}\nPer-comparison decision-rule verdict\n{'=' * 70}")
    for name in COMPARISONS:
        out = all_out[name]
        verdict = "RESOLVED, CONSISTENT" if out["consistent"] else "STILL INCONSISTENT"
        print(f"  {COMPARISON_LABELS[name]}: {verdict}")

    with open(ANALYSIS_PATH, "wb") as f:
        pickle.dump(all_out, f)
    print(f"\nSaved full log-scale analysis to {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
