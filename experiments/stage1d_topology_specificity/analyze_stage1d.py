"""
Stage 1D analysis. Two clearly separate parts, per DESIGN.md:

1. LATTICE (confirmatory, complete): d_k = Delta_map(T,k) - Delta_map(lattice,k)
   for the 10 matched trajectory seeds. Primary test, locked: two-sided
   paired t-test on the 10 d_k values. Robustness: exact sign-flip and
   Wilcoxon signed-rank on the same 10 values.

2. PILOT (explicitly non-confirmatory -- see DESIGN.md's "Pilot vs.
   confirmatory" section): for each of the three stochastic controls,
   3 realizations x 3 matched trajectories. Fits the crossed variance
   decomposition d_grk = mu_g + b_gr + tau_k + epsilon_grk via a balanced
   two-way ANOVA method-of-moments estimator (ANOVA is exact/unbiased for
   a balanced crossed design with one observation per cell -- no need for
   an iterative mixed-model solver here), then applies DESIGN.md's locked
   pilot-to-confirmatory allocation rule to select one common (R, K).

Delta_map(T, k) for all 10 matched seeds is read from Stage 1C's own
already-committed final analysis (stage1c_final_analysis.pkl, which
itself sources k=3000 read-only from Stage 1B2's own frozen results) --
never recomputed, keeping Stage 1B2/1C genuinely frozen.

No confirmatory topology-specificity conclusion is drawn from the pilot
data anywhere in this file -- the pilot section computes only the
variance components and the resulting (R, K), never a p-value or an
effect-direction claim about the stochastic controls themselves.
"""
import sys
import os
import pickle
import itertools

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "stage1b2_structured_transformation"))
from analyze_stage1b2 import load_results_as_arrays, run_permutation_test

from run_stage1d import (
    LATTICE_TRAJECTORY_SEEDS, PILOT_TRAJECTORY_SEEDS, PILOT_FAMILIES,
    lattice_checkpoint_path, pilot_checkpoint_path,
)
from build_stage1d_constructions import build_all, PILOT_REALIZATION_SEEDS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

STAGE1C_FINAL_ANALYSIS_PATH = os.path.join(
    _THIS_DIR, "..", "stage1c_trajectory_generalization", "results", "stage1c_final_analysis.pkl")

# ---- Locked design parameters (DESIGN.md, Round 4) ----
DELTA_MIN = 0.05
POWER_TARGET = 0.80
FAMILYWISE_ALPHA = 0.05
N_COMPARISONS_PRIMARY_FAMILY = 4  # rewired, hist_random, curr_random, lattice
HOLM_APPROX_ALPHA = FAMILYWISE_ALPHA / N_COMPARISONS_PRIMARY_FAMILY  # 0.0125, option (b)
R_GRID = [10, 15, 20, 25]
K_GRID = [3, 5, 7, 10]
RELIABLE_DF_THRESHOLD = 3  # our own operational cutoff for "reliably estimable" (see DESIGN.md's
# "where it isn't reliably estimable at that sample size, use... a named conservative fallback")


def load_T_delta_maps():
    with open(STAGE1C_FINAL_ANALYSIS_PATH, "rb") as f:
        per_traj = pickle.load(f)
    return {seed: result["pooled_delta_map"] for seed, result in per_traj.items()}


def analyze_one_construction_trajectory(results_path, nodes_T, label):
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    assert len(results) == 432, f"{label}: expected 432 trials, got {len(results)}"
    organized = load_results_as_arrays(results, nodes_T, time_key="event_aligned_q")
    return run_permutation_test(organized, nodes_T)


# ============================== LATTICE ==============================

def run_lattice_analysis(nodes_T):
    delta_T = load_T_delta_maps()
    lattice_results = {}
    for seed in LATTICE_TRAJECTORY_SEEDS:
        print(f"\n{'='*60}\nLattice trajectory seed={seed}\n{'='*60}")
        lattice_results[seed] = analyze_one_construction_trajectory(
            lattice_checkpoint_path(seed), nodes_T, f"lattice seed={seed}")

    d_k = np.array([delta_T[seed] - lattice_results[seed]["pooled_delta_map"]
                     for seed in LATTICE_TRAJECTORY_SEEDS])

    # Primary: two-sided paired t-test (equivalently, one-sample t-test on d_k)
    t_stat, t_p = stats.ttest_1samp(d_k, popmean=0.0, alternative="two-sided")

    # Robustness: exact sign-flip test on the 10 d_k values
    n = len(d_k)
    observed_mean = np.mean(d_k)
    all_flip_means = []
    for flips in itertools.product([1, -1], repeat=n):
        all_flip_means.append(np.mean(np.array(flips) * d_k))
    all_flip_means = np.array(all_flip_means)
    sign_flip_p = np.mean(np.abs(all_flip_means) >= np.abs(observed_mean) - 1e-12)

    # Robustness: Wilcoxon signed-rank
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(d_k, alternative="two-sided")
    except ValueError as e:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        print(f"Wilcoxon signed-rank failed ({e}) -- likely a zero or tied difference")

    result = {
        "delta_T": delta_T,
        "lattice_results": lattice_results,
        "d_k": dict(zip(LATTICE_TRAJECTORY_SEEDS, d_k.tolist())),
        "mean_d_k": float(observed_mean),
        "sd_d_k": float(np.std(d_k, ddof=1)),
        "paired_t_test": {"t_stat": float(t_stat), "p_value": float(t_p), "df": n - 1},
        "sign_flip_test": {"p_value": float(sign_flip_p), "n_sign_patterns": len(all_flip_means)},
        "wilcoxon_signed_rank": {"stat": float(wilcoxon_stat), "p_value": float(wilcoxon_p)},
    }
    return result


# ============================== PILOT ==============================

def fit_crossed_variance_decomposition(d_grk_matrix):
    """d_grk_matrix: (R, K) array for one stochastic-control family (rows =
    graph realizations, cols = matched trajectory seeds). Balanced two-way
    ANOVA method-of-moments variance-component estimator (crossed random
    effects, one observation per cell) -- exact/unbiased for this design,
    not an approximation:

        d_grk = mu_g + b_gr + tau_k + epsilon_grk

    Returns point estimates for sigma^2_b (between-realization),
    sigma^2_tau (trajectory-seed block effect), sigma^2_eps (residual),
    plus the mean squares and degrees of freedom needed for the
    conservative-CI step in select_common_RK.
    """
    R, K = d_grk_matrix.shape
    grand_mean = d_grk_matrix.mean()
    row_means = d_grk_matrix.mean(axis=1)  # d_bar_r. , length R
    col_means = d_grk_matrix.mean(axis=0)  # d_bar_.k , length K

    ss_r = K * np.sum((row_means - grand_mean) ** 2)
    ss_k = R * np.sum((col_means - grand_mean) ** 2)
    fitted = row_means[:, None] + col_means[None, :] - grand_mean
    ss_resid = np.sum((d_grk_matrix - fitted) ** 2)

    df_r, df_k, df_resid = R - 1, K - 1, (R - 1) * (K - 1)
    ms_r = ss_r / df_r
    ms_k = ss_k / df_k
    ms_resid = ss_resid / df_resid if df_resid > 0 else np.nan

    sigma2_eps = max(0.0, ms_resid)
    sigma2_b = max(0.0, (ms_r - ms_resid) / K)
    sigma2_tau = max(0.0, (ms_k - ms_resid) / R)

    return {
        "R": R, "K": K, "grand_mean": float(grand_mean),
        "row_means_d_bar_gr": row_means.tolist(), "col_means": col_means.tolist(),
        "ss_r": float(ss_r), "ss_k": float(ss_k), "ss_resid": float(ss_resid),
        "df_r": df_r, "df_k": df_k, "df_resid": df_resid,
        "ms_r": float(ms_r), "ms_k": float(ms_k), "ms_resid": float(ms_resid),
        "sigma2_b_point": float(sigma2_b), "sigma2_tau_point": float(sigma2_tau),
        "sigma2_eps_point": float(sigma2_eps),
    }


def conservative_upper_bound(point_estimate, ss, df, alpha=0.05):
    """95% upper confidence bound via the chi-squared pivot (SS/sigma^2 ~
    chi2(df)), used when df is large enough (>= RELIABLE_DF_THRESHOLD) to
    be 'reliably estimable' per DESIGN.md; otherwise returns None so the
    caller falls back to the named conservative fallback (2x point
    estimate)."""
    if df < RELIABLE_DF_THRESHOLD or ss <= 0:
        return None
    return float(ss / stats.chi2.ppf(alpha, df))


def conservative_variance_estimates(decomp):
    """Implements DESIGN.md's locked rule: 'use a prespecified 95% upper
    confidence bound... where it's estimable from the 3x3 pilot; where it
    isn't reliably estimable at that sample size, use the larger of the
    point estimate and a named conservative fallback (2x the point
    estimate)'. Our own operational threshold for 'reliably estimable':
    df >= 3 (RELIABLE_DF_THRESHOLD) -- sigma^2_eps (df_resid=4 at R=K=3)
    qualifies; sigma^2_b (df_r=2) does not, and uses the 2x fallback."""
    eps_upper = conservative_upper_bound(decomp["sigma2_eps_point"], decomp["ss_resid"], decomp["df_resid"])
    eps_conservative = eps_upper if eps_upper is not None else 2 * decomp["sigma2_eps_point"]
    eps_method = "chi2_upper_CI" if eps_upper is not None else "2x_point_fallback"

    # sigma^2_b's denominator df (=R-1=2 at R=3) is below RELIABLE_DF_THRESHOLD by
    # construction of the 3x3 pilot -- there is no valid chi-squared pivot for a
    # subtracted, non-central quantity like (MS_r - MS_resid)/K at this df, so this
    # always takes the named 2x-point-estimate fallback rather than attempting one.
    if decomp["df_r"] >= RELIABLE_DF_THRESHOLD:
        b_upper = conservative_upper_bound(decomp["sigma2_b_point"], decomp["ss_r"], decomp["df_r"])
        b_conservative = b_upper if b_upper is not None else 2 * decomp["sigma2_b_point"]
        b_method = "chi2_upper_CI" if b_upper is not None else "2x_point_fallback"
    else:
        b_conservative = 2 * decomp["sigma2_b_point"]
        b_method = "2x_point_fallback (df_r below reliability threshold)"

    return {
        "sigma2_eps_conservative": eps_conservative, "sigma2_eps_method": eps_method,
        "sigma2_b_conservative": b_conservative, "sigma2_b_method": b_method,
    }


def one_sample_two_sided_t_power(effect, sd, n, alpha):
    """Power of a two-sided one-sample t-test, effect size `effect`,
    per-unit SD `sd`, sample size `n`, significance level `alpha`, via the
    noncentral t distribution."""
    if sd <= 0:
        return 1.0 if effect != 0 else float(alpha)
    df = n - 1
    ncp = effect * np.sqrt(n) / sd
    crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
    return float(power)


def per_family_min_cost_design(sigma2_b_conservative, sigma2_eps_conservative):
    """Searches R_GRID x K_GRID for the lowest-cost (R, K) achieving
    POWER_TARGET for DELTA_MIN under HOLM_APPROX_ALPHA, using this
    family's conservative variance components. Tie-break: larger R.
    Returns None if no grid candidate reaches the power target."""
    candidates = []
    for R in R_GRID:
        for K in K_GRID:
            var_per_realization = sigma2_b_conservative + sigma2_eps_conservative / K
            sd_per_realization = np.sqrt(var_per_realization)
            power = one_sample_two_sided_t_power(DELTA_MIN, sd_per_realization, R, HOLM_APPROX_ALPHA)
            candidates.append({"R": R, "K": K, "cost": R * K, "power": power})

    feasible = [c for c in candidates if c["power"] >= POWER_TARGET]
    if not feasible:
        return None, candidates
    min_cost = min(c["cost"] for c in feasible)
    tied = [c for c in feasible if c["cost"] == min_cost]
    best = max(tied, key=lambda c: c["R"])
    return best, candidates


def diagnose_node_degeneracy(family, r_seed, results_by_traj_seed, nodes_T):
    """Checks, per fixed-coordinate node label, whether event_aligned_valid
    is ever True across this realization's trials -- a node with weighted
    degree 0 in this specific graph realization typically drives E (the
    tangent-departure diagnostic) below E_min at every t_p except t_p=0,
    since an isolated node's response is nearly perfectly linear (see
    stage1b2_core.py's E_min gate). When 2 of the 3 node labels are
    invalid at a given t_p, B_node has no surviving cross-node-label pair
    at all (every combinations(node_labels, 2) pair includes at least one
    invalid label), making that t_p's Delta_map NaN -- not a bug in this
    driver or analyze_stage1b2.py's machinery, but a real structural
    consequence of the fixed-coordinate protocol landing on a node that a
    sparser/independently-resampled construction happened to isolate."""
    never_valid_labels = []
    for node_label in nodes_T:
        any_valid = any(
            any(r["event_aligned_valid"] for key, r in results.items() if key[2] == node_label)
            for results in results_by_traj_seed.values()
        )
        if not any_valid:
            never_valid_labels.append(node_label)
    return never_valid_labels


def run_pilot_analysis(nodes_T):
    delta_T = load_T_delta_maps()
    data_for_degrees = build_all()
    per_family = {}

    for family in PILOT_FAMILIES:
        print(f"\n{'='*60}\nPILOT family: {family}\n{'='*60}")
        raw_delta_map = {}
        d_grk_full = np.full((len(PILOT_REALIZATION_SEEDS), len(PILOT_TRAJECTORY_SEEDS)), np.nan)
        degeneracy_by_realization = {}

        for ri, r_seed in enumerate(PILOT_REALIZATION_SEEDS):
            W_g_r = data_for_degrees["pilot_constructions"][family][r_seed]
            deg_g_r = W_g_r.sum(axis=1)
            isolated_nodes = [label for label, idx in nodes_T.items() if deg_g_r[idx] < 1e-9]

            results_by_traj_seed = {}
            for ki, traj_seed in enumerate(PILOT_TRAJECTORY_SEEDS):
                label = f"{family} r={r_seed} seed={traj_seed}"
                path = pilot_checkpoint_path(family, r_seed, traj_seed)
                with open(path, "rb") as f:
                    results_by_traj_seed[traj_seed] = pickle.load(f)
                res = analyze_one_construction_trajectory(path, nodes_T, label)
                raw_delta_map[(r_seed, traj_seed)] = res["pooled_delta_map"]
                d_grk_full[ri, ki] = delta_T[traj_seed] - res["pooled_delta_map"]

            never_valid = diagnose_node_degeneracy(family, r_seed, results_by_traj_seed, nodes_T)
            degeneracy_by_realization[r_seed] = {
                "isolated_fixed_coordinate_nodes": isolated_nodes,
                "node_labels_never_event_aligned_valid": never_valid,
                "degrees_at_fixed_coordinates": {label: float(deg_g_r[idx]) for label, idx in nodes_T.items()},
            }

        # A realization whose row is entirely NaN (>=2 node labels never valid, collapsing
        # every cross-node-label B_node pair) is excluded from the crossed-variance fit --
        # NOT imputed or silently dropped from the report. DESIGN.md's own precedent for this
        # (the lattice degenerate-role-matching rule) is to disclose the reduced family size
        # explicitly rather than pad or paper over it; the same principle applies here.
        row_is_nan = np.all(np.isnan(d_grk_full), axis=1)
        valid_realization_seeds = [PILOT_REALIZATION_SEEDS[ri] for ri in range(len(PILOT_REALIZATION_SEEDS))
                                    if not row_is_nan[ri]]
        excluded_realization_seeds = [PILOT_REALIZATION_SEEDS[ri] for ri in range(len(PILOT_REALIZATION_SEEDS))
                                       if row_is_nan[ri]]
        d_grk_valid = d_grk_full[~row_is_nan, :]

        if d_grk_valid.shape[0] >= 2:
            decomp = fit_crossed_variance_decomposition(d_grk_valid)
            conservative = conservative_variance_estimates(decomp)
            best_design, all_candidates = per_family_min_cost_design(
                conservative["sigma2_b_conservative"], conservative["sigma2_eps_conservative"])
            reliable = d_grk_valid.shape[0] == len(PILOT_REALIZATION_SEEDS)  # no exclusions
        else:
            decomp, conservative, best_design, all_candidates, reliable = None, None, None, [], False

        per_family[family] = {
            "raw_delta_map_gr_k": raw_delta_map,
            "d_grk_full_including_nan": d_grk_full.tolist(),
            "degeneracy_by_realization": degeneracy_by_realization,
            "valid_realization_seeds": valid_realization_seeds,
            "excluded_realization_seeds": excluded_realization_seeds,
            "d_bar_gr_valid_only": decomp["row_means_d_bar_gr"] if decomp else None,
            "theta_g_hat_mean_over_valid_realizations":
                float(np.mean(decomp["row_means_d_bar_gr"])) if decomp else None,
            "variance_decomposition": decomp,
            "conservative_variance_estimates": conservative,
            "own_min_cost_design": best_design,
            "all_candidate_designs": all_candidates,
            "estimate_reliable_no_exclusions": reliable,
        }

    # One common (R, K): the most demanding (largest-cost) of the RELIABLE families' own
    # minimal designs (i.e. families with no excluded/degenerate realization) -- tie-break
    # larger R. A family whose pilot data was itself degenerate (an excluded realization) is
    # reported separately as INDETERMINATE, not silently included as if it were a normal
    # infeasible-power case, and not silently excluded from the report either.
    own_designs = {fam: per_family[fam]["own_min_cost_design"] for fam in PILOT_FAMILIES}
    reliable_families = [fam for fam in PILOT_FAMILIES if per_family[fam]["estimate_reliable_no_exclusions"]]
    indeterminate_families = [fam for fam in PILOT_FAMILIES if not per_family[fam]["estimate_reliable_no_exclusions"]]
    infeasible = [fam for fam in reliable_families if own_designs[fam] is None]

    if not reliable_families or any(own_designs[fam] is None for fam in reliable_families):
        common_RK = None
    else:
        max_cost = max(own_designs[fam]["cost"] for fam in reliable_families)
        tied_families = [fam for fam in reliable_families if own_designs[fam]["cost"] == max_cost]
        most_demanding_family = max(tied_families, key=lambda fam: own_designs[fam]["R"])
        common_RK = {
            "R": own_designs[most_demanding_family]["R"],
            "K": own_designs[most_demanding_family]["K"],
            "cost": own_designs[most_demanding_family]["cost"],
            "selected_from_family": most_demanding_family,
            "based_on_families": reliable_families,
            "indeterminate_families_excluded_from_selection": indeterminate_families,
        }

    return {
        "per_family": per_family,
        "own_designs_per_family": own_designs,
        "reliable_families": reliable_families,
        "indeterminate_families": indeterminate_families,
        "infeasible_families": infeasible,
        "common_RK": common_RK,
        "locked_parameters": {
            "delta_min": DELTA_MIN, "power_target": POWER_TARGET,
            "familywise_alpha": FAMILYWISE_ALPHA, "holm_approx_alpha": HOLM_APPROX_ALPHA,
            "holm_method": "(b) simpler conservative approximation: alpha=0.05/4=0.0125 per "
                           "comparison, not (a) the joint Holm simulation -- chosen for "
                           "tractability within the pilot's timeframe",
            "R_grid": R_GRID, "K_grid": K_GRID,
        },
    }


def main():
    data = build_all()
    nodes_T = data["nodes_T"]

    print("\n" + "#" * 70)
    print("# PART 1: LATTICE (confirmatory)")
    print("#" * 70)
    lattice_result = run_lattice_analysis(nodes_T)
    with open(os.path.join(RESULTS_DIR, "stage1d_lattice_analysis.pkl"), "wb") as f:
        pickle.dump(lattice_result, f)
    print(f"\nLattice d_k values: {lattice_result['d_k']}")
    print(f"Mean d_k = {lattice_result['mean_d_k']:.4f}, SD = {lattice_result['sd_d_k']:.4f}")
    print(f"Paired t-test: t={lattice_result['paired_t_test']['t_stat']:.4f}, "
          f"p={lattice_result['paired_t_test']['p_value']:.5f}")
    print(f"Sign-flip test: p={lattice_result['sign_flip_test']['p_value']:.5f}")
    print(f"Wilcoxon signed-rank: p={lattice_result['wilcoxon_signed_rank']['p_value']:.5f}")

    print("\n" + "#" * 70)
    print("# PART 2: PILOT (NON-CONFIRMATORY -- variance allocation only)")
    print("#" * 70)
    pilot_result = run_pilot_analysis(nodes_T)
    with open(os.path.join(RESULTS_DIR, "stage1d_pilot_analysis.pkl"), "wb") as f:
        pickle.dump(pilot_result, f)
    for family, res in pilot_result["per_family"].items():
        print(f"\n{family}: valid_realizations={res['valid_realization_seeds']}, "
              f"excluded={res['excluded_realization_seeds']}, "
              f"d_bar_gr(valid only)={res['d_bar_gr_valid_only']}")
        if res["conservative_variance_estimates"] is not None:
            print(f"  sigma2_b_conservative={res['conservative_variance_estimates']['sigma2_b_conservative']:.6f}, "
                  f"sigma2_eps_conservative={res['conservative_variance_estimates']['sigma2_eps_conservative']:.6f}, "
                  f"own_min_cost_design={res['own_min_cost_design']}, "
                  f"reliable(no exclusions)={res['estimate_reliable_no_exclusions']}")
        for r_seed, diag in res["degeneracy_by_realization"].items():
            if diag["isolated_fixed_coordinate_nodes"]:
                print(f"  r={r_seed}: isolated fixed-coordinate nodes = "
                      f"{diag['isolated_fixed_coordinate_nodes']}, "
                      f"never-valid labels = {diag['node_labels_never_event_aligned_valid']}")
    print(f"\nReliable families: {pilot_result['reliable_families']}")
    print(f"Indeterminate families (degenerate pilot realization): {pilot_result['indeterminate_families']}")
    print(f"Common (R, K): {pilot_result['common_RK']}")

    return lattice_result, pilot_result


if __name__ == "__main__":
    main()
