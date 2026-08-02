"""
Stage 1D, Part 2: confirmatory stochastic-control analysis.

Consumes experiments/stage1d_topology_specificity_gpu's confirmatory GPU
run output (results/stage1d_confirmatory_gpu_results.pkl, downloaded from
the GPU session -- keyed by (family, realization_seed, trajectory_seed) ->
{'per_tp_delta_map', 'pooled_delta_map', 'n_valid_trials', 'n_total_trials',
'never_valid_node_labels'}, all computed on the GPU session itself via the
REAL analyze_stage1b2.py functions (load_results_as_arrays,
compute_W_B_deltamap), not recomputed here from raw arrays.

For each of the three stochastic-control families (rewired, hist_random,
curr_random), at the locked (R=25, K=3) confirmatory design
(DESIGN.md, "Locked confirmatory-run allocation"):

    d_grk = Delta_map(T,k) - Delta_map(g,r,k)
    d_bar_gr = mean_k(d_grk)   -- aggregated within realization

Primary test: two-sided one-sample t-test on {d_bar_gr} across the R
realizations. Robustness: Wilcoxon signed-rank, exact sign-flip (all 2^25
sign patterns, computed via iterative distribution-doubling -- vectorized,
not a per-pattern Python loop), and a hierarchical bootstrap resampling
realizations then trajectories within realization.

hist_random gets two additional, clearly separate things (DESIGN.md,
"Historical-random: pre-screening and a conditional estimand"):
- the primary comparison is reported explicitly as the CONDITIONAL
  estimand E[Delta_T - Delta_hist_random | evaluable] -- exactly what the
  above already computes, since only evaluable (non-isolated-node)
  realizations were ever built;
- the UNCONDITIONAL rejection rate from the pre-screening protocol
  (build_stage1d_confirmatory_constructions.py's ledger of rejected
  candidates), reported separately with a Clopper-Pearson exact binomial
  confidence interval, never folded into the primary estimate.

Finally: Holm correction across all four fixed-coordinate comparisons
(lattice, already complete via analyze_stage1d.py's run_lattice_analysis,
plus the three stochastic controls here).
"""
import os
import pickle
import itertools

import numpy as np
from scipy import stats

from analyze_stage1d import load_T_delta_maps

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

GPU_RESULTS_PATH = os.path.join(RESULTS_DIR, "stage1d_confirmatory_gpu_results.pkl")
CONSTRUCTIONS_PATH = os.path.join(RESULTS_DIR, "stage1d_confirmatory_constructions.pkl")
LATTICE_ANALYSIS_PATH = os.path.join(RESULTS_DIR, "stage1d_lattice_analysis.pkl")

TRAJECTORY_SEEDS = [3000, 3010, 3020]
FAMILIES = ["rewired", "hist_random", "curr_random"]
N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 12345


def build_d_grk_matrix(family, gpu_results, accepted_seeds, delta_T):
    seeds = accepted_seeds[family]
    d_grk = np.full((len(seeds), len(TRAJECTORY_SEEDS)), np.nan)
    raw_pooled = {}
    for ri, r_seed in enumerate(seeds):
        for ki, traj_seed in enumerate(TRAJECTORY_SEEDS):
            entry = gpu_results[(family, r_seed, traj_seed)]
            raw_pooled[(r_seed, traj_seed)] = entry["pooled_delta_map"]
            d_grk[ri, ki] = delta_T[traj_seed] - entry["pooled_delta_map"]
    return d_grk, seeds, raw_pooled


def exact_sign_flip_test(values):
    """All 2^n sign patterns via iterative distribution-doubling (vectorized,
    not a per-pattern Python loop) -- exact for n=25 (2^25 = 33,554,432
    patterns), matching DESIGN.md's robustness-test requirement."""
    sums = np.array([0.0])
    for x in values:
        sums = np.concatenate([sums + x, sums - x])
    n = len(values)
    observed_sum = np.sum(values)
    p = float(np.mean(np.abs(sums) >= np.abs(observed_sum) - 1e-9))
    return p, len(sums)


def hierarchical_bootstrap(d_grk_matrix, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP):
    """Resamples graph realizations (rows), then matched trajectory seeds
    (columns) within each resampled realization, per DESIGN.md's locked
    robustness analysis (analogous to Stage 1A's hierarchical bootstrap)."""
    rng = np.random.default_rng(seed)
    R, K = d_grk_matrix.shape
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        row_idx = rng.integers(0, R, size=R)
        resampled_rows = d_grk_matrix[row_idx, :]
        col_idx = rng.integers(0, K, size=(R, K))
        resampled = np.take_along_axis(resampled_rows, col_idx, axis=1)
        boot_means[b] = np.mean(resampled)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {"ci_95": (float(ci_lo), float(ci_hi)), "boot_mean": float(np.mean(boot_means)),
            "boot_sd": float(np.std(boot_means, ddof=1)), "n_boot": n_boot}


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def holm_correction(named_pvalues):
    """named_pvalues: list of (name, p). Returns list of
    (name, raw_p, adjusted_p, rank) sorted by raw p ascending, with the
    standard Holm step-down monotonic adjustment."""
    m = len(named_pvalues)
    order = sorted(range(m), key=lambda i: named_pvalues[i][1])
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):  # rank = 0-indexed position in sorted order
        name, p = named_pvalues[i]
        raw_adj = (m - rank) * p
        running_max = max(running_max, raw_adj)
        adjusted[i] = min(1.0, running_max)
    return [
        {"name": named_pvalues[i][0], "raw_p": named_pvalues[i][1],
         "holm_adjusted_p": adjusted[i], "rank": order.index(i) + 1}
        for i in range(m)
    ]


def analyze_family(family, gpu_results, accepted_seeds, delta_T):
    d_grk, seeds, raw_pooled = build_d_grk_matrix(family, gpu_results, accepted_seeds, delta_T)

    row_is_nan = np.all(np.isnan(d_grk), axis=1)
    valid_seeds = [seeds[i] for i in range(len(seeds)) if not row_is_nan[i]]
    excluded_seeds = [seeds[i] for i in range(len(seeds)) if row_is_nan[i]]
    d_grk_valid = d_grk[~row_is_nan, :]

    d_bar_gr = np.nanmean(d_grk_valid, axis=1)
    R = len(d_bar_gr)

    t_stat, t_p = stats.ttest_1samp(d_bar_gr, popmean=0.0, alternative="two-sided")
    sign_flip_p, n_patterns = exact_sign_flip_test(d_bar_gr)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(d_bar_gr, alternative="two-sided")
    except ValueError as e:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        print(f"  [{family}] Wilcoxon signed-rank failed ({e})")

    bootstrap = hierarchical_bootstrap(d_grk_valid)

    return {
        "family": family,
        "d_grk_full": dict(zip(seeds, d_grk.tolist())),
        "raw_pooled_delta_map": raw_pooled,
        "valid_seeds": valid_seeds,
        "excluded_seeds": excluded_seeds,
        "R_used": R,
        "d_bar_gr": dict(zip(valid_seeds, d_bar_gr.tolist())),
        "mean_d_bar_gr": float(np.mean(d_bar_gr)),
        "sd_d_bar_gr": float(np.std(d_bar_gr, ddof=1)),
        "n_realizations_outperforming_T": int(np.sum(d_bar_gr < 0)),
        "primary_t_test": {"t_stat": float(t_stat), "p_value": float(t_p), "df": R - 1},
        "sign_flip_test": {"p_value": sign_flip_p, "n_sign_patterns": n_patterns},
        "wilcoxon_signed_rank": {"stat": float(wilcoxon_stat), "p_value": float(wilcoxon_p)},
        "hierarchical_bootstrap": bootstrap,
    }


def main():
    with open(GPU_RESULTS_PATH, "rb") as f:
        gpu_results = pickle.load(f)
    with open(CONSTRUCTIONS_PATH, "rb") as f:
        constructions_meta = pickle.load(f)
    with open(LATTICE_ANALYSIS_PATH, "rb") as f:
        lattice_result = pickle.load(f)

    accepted_seeds = constructions_meta["accepted_seeds"]
    delta_T = load_T_delta_maps()

    per_family = {}
    for family in FAMILIES:
        print(f"\n{'='*70}\nCONFIRMATORY family: {family}\n{'='*70}")
        res = analyze_family(family, gpu_results, accepted_seeds, delta_T)
        per_family[family] = res
        print(f"R used: {res['R_used']} (excluded: {res['excluded_seeds']})")
        print(f"mean d_bar_gr = {res['mean_d_bar_gr']:.4f}, SD = {res['sd_d_bar_gr']:.4f}")
        print(f"realizations where control outperforms T: {res['n_realizations_outperforming_T']}/{res['R_used']}")
        print(f"Primary t-test: t={res['primary_t_test']['t_stat']:.4f}, p={res['primary_t_test']['p_value']:.5f}")
        print(f"Sign-flip: p={res['sign_flip_test']['p_value']:.6f} (n={res['sign_flip_test']['n_sign_patterns']})")
        print(f"Wilcoxon: p={res['wilcoxon_signed_rank']['p_value']:.5f}")
        print(f"Bootstrap 95% CI: {res['hierarchical_bootstrap']['ci_95']}")

    # hist_random-specific: unconditional rejection-rate disclosure
    rejected = constructions_meta["hist_random_rejected_candidates"]
    n_accepted = len(accepted_seeds["hist_random"])
    n_rejected = len(rejected)
    n_drawn = n_accepted + n_rejected
    ci_lo, ci_hi = clopper_pearson(n_rejected, n_drawn)
    hist_random_rejection = {
        "n_rejected": n_rejected, "n_accepted": n_accepted, "n_candidates_drawn": n_drawn,
        "rejection_rate": n_rejected / n_drawn,
        "clopper_pearson_95_ci": (ci_lo, ci_hi),
        "rejected_candidates": rejected,
    }
    print(f"\nhist_random UNCONDITIONAL rejection rate: {n_rejected}/{n_drawn} = "
          f"{100*n_rejected/n_drawn:.1f}% (95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")
    print("(This is the separate, secondary characteristic of the hist_random family --")
    print(" the primary comparison above is the CONDITIONAL estimand E[Delta_T - Delta_hist_random | evaluable].)")

    # Holm correction across the 4-way fixed-coordinate family
    named_pvalues = [
        ("lattice", lattice_result["paired_t_test"]["p_value"]),
        ("rewired", per_family["rewired"]["primary_t_test"]["p_value"]),
        ("hist_random", per_family["hist_random"]["primary_t_test"]["p_value"]),
        ("curr_random", per_family["curr_random"]["primary_t_test"]["p_value"]),
    ]
    holm_result = holm_correction(named_pvalues)
    print(f"\n{'='*70}\nHOLM-ADJUSTED 4-WAY FIXED-COORDINATE FAMILY\n{'='*70}")
    for entry in sorted(holm_result, key=lambda e: e["rank"]):
        print(f"  rank {entry['rank']}: {entry['name']}: raw p={entry['raw_p']:.5f}, "
              f"Holm-adjusted p={entry['holm_adjusted_p']:.5f}")

    result = {
        "per_family": per_family,
        "hist_random_rejection": hist_random_rejection,
        "holm_4way": holm_result,
        "lattice_reference": {
            "paired_t_test": lattice_result["paired_t_test"],
            "mean_d_k": lattice_result["mean_d_k"],
        },
        "trajectory_seeds": TRAJECTORY_SEEDS,
        "R_design": 25, "K_design": 3,
    }
    with open(os.path.join(RESULTS_DIR, "stage1d_confirmatory_analysis.pkl"), "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved to {os.path.join(RESULTS_DIR, 'stage1d_confirmatory_analysis.pkl')}")
    return result


if __name__ == "__main__":
    main()
