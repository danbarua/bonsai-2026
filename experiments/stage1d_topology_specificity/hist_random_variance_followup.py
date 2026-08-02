"""
Stage 1D pilot follow-up: hist_random variance re-estimation.

PILOT_RESULTS.md flagged (but did not act on) a recommendation: hist_random's
crossed variance decomposition was fit on only 2 of its 3 pilot realizations
after seed=2 turned out fully degenerate (both of T's fixed 'low'/'high'
node indices landed at weighted degree 0 in that independent random draw),
leaving df_r=1 -- too little to make the conservative variance estimate
anything but the named 2x-point-estimate fallback.

This script draws 2 MORE hist_random graph realizations (seeds 3, 4 --
continuing the original pilot's 0, 1, 2 sequence), using the exact same
construction recipe (build_pilot_realization) and the exact same 3 matched
trajectory seeds (3000, 3010, 3020), full 432-trial simulation, Delta_map
computation, and 10,000-permutation validation as the original pilot.

Before running any simulation, each new realization is checked for the same
isolated-fixed-coordinate-node degeneracy PILOT_RESULTS.md diagnosed for
seed=2 -- reported explicitly either way, not silently redrawn if it
recurs.

Does NOT touch seeds 0, 1, 2's cached results or re-run anything already
in results/stage1d_pilot_*.pkl -- this is additional evidence appended to
hist_random's picture, not a correction of the original pilot's numbers.
"""
import os
import pickle
import time

import numpy as np

from build_stage1d_constructions import build_all, build_pilot_realization
from run_stage1d import run_trajectory, pilot_checkpoint_path, RESULTS_DIR, log
from analyze_stage1d import (
    analyze_one_construction_trajectory, diagnose_node_degeneracy,
    fit_crossed_variance_decomposition, conservative_variance_estimates,
    per_family_min_cost_design, load_T_delta_maps,
)

FOLLOWUP_SEEDS = [3, 4]
TRAJECTORY_SEEDS = [3000, 3010, 3020]
FAMILY = "hist_random"
ORIGINAL_PILOT_ANALYSIS_PATH = os.path.join(RESULTS_DIR, "stage1d_pilot_analysis.pkl")
FOLLOWUP_RESULTS_PATH = os.path.join(RESULTS_DIR, "stage1d_hist_random_followup.pkl")


def check_degeneracy_before_running(data):
    """Pre-simulation check: does the new realization isolate any of T's
    fixed-coordinate nodes, the same way seed=2 was diagnosed? Reported
    regardless of outcome -- a recurrence is informative and must be
    disclosed the same way, not silently redrawn again."""
    nodes_T = data["nodes_T"]
    W_T = data["W_T"]
    ink_mask_active = data["ink_mask_active"]
    report = {}
    for seed in FOLLOWUP_SEEDS:
        W = build_pilot_realization(FAMILY, seed, W_T, ink_mask_active)
        deg = W.sum(axis=1)
        degrees = {label: float(deg[idx]) for label, idx in nodes_T.items()}
        isolated = [label for label, idx in nodes_T.items() if deg[idx] < 1e-9]
        report[seed] = {"degrees": degrees, "isolated": isolated, "W": W}
        log(f"[hist_random followup] seed={seed} PRE-SIMULATION check: "
            f"degrees={degrees}, isolated={isolated}")
    return report


def run_new_realizations(data, pre_check):
    nodes_T = data["nodes_T"]
    n = data["n_active"]
    for seed in FOLLOWUP_SEEDS:
        W = pre_check[seed]["W"]
        for traj_seed in TRAJECTORY_SEEDS:
            t0 = time.time()
            out_path = pilot_checkpoint_path(FAMILY, seed, traj_seed)
            label = f"pilot {FAMILY} r={seed} seed={traj_seed} [FOLLOWUP]"
            run_trajectory(W, n, traj_seed, nodes_T, out_path, label)
            log(f"=== {label} total wall time: {time.time()-t0:.1f}s ===")


def analyze_new_realizations(data, pre_check):
    nodes_T = data["nodes_T"]
    delta_T = load_T_delta_maps()
    results = {}
    for seed in FOLLOWUP_SEEDS:
        results_by_traj_seed = {}
        d_row = []
        raw_delta_map = {}
        for traj_seed in TRAJECTORY_SEEDS:
            path = pilot_checkpoint_path(FAMILY, seed, traj_seed)
            with open(path, "rb") as f:
                results_by_traj_seed[traj_seed] = pickle.load(f)
            res = analyze_one_construction_trajectory(path, nodes_T, f"{FAMILY} r={seed} seed={traj_seed}")
            raw_delta_map[traj_seed] = res["pooled_delta_map"]
            d_row.append(delta_T[traj_seed] - res["pooled_delta_map"])
        never_valid = diagnose_node_degeneracy(FAMILY, seed, results_by_traj_seed, nodes_T)
        results[seed] = {
            "raw_delta_map": raw_delta_map,
            "d_grk_row": d_row,
            "isolated_fixed_coordinate_nodes": pre_check[seed]["isolated"],
            "node_labels_never_event_aligned_valid": never_valid,
            "degrees_at_fixed_coordinates": pre_check[seed]["degrees"],
            "is_degenerate": bool(np.all(np.isnan(d_row))),
        }
    return results


def main():
    data = build_all()
    nodes_T = data["nodes_T"]

    print("\n" + "#" * 70)
    print("# PRE-SIMULATION degeneracy check, hist_random seeds 3, 4")
    print("#" * 70)
    pre_check = check_degeneracy_before_running(data)
    for seed, info in pre_check.items():
        flag = "DEGENERATE (isolated node)" if info["isolated"] else "not isolated"
        print(f"seed={seed}: degrees={info['degrees']} -- {flag}")

    print("\n" + "#" * 70)
    print("# Running 6 new trajectories (2 realizations x 3 matched seeds)")
    print("#" * 70)
    run_new_realizations(data, pre_check)

    print("\n" + "#" * 70)
    print("# Analyzing new realizations")
    print("#" * 70)
    new_results = analyze_new_realizations(data, pre_check)
    for seed, res in new_results.items():
        print(f"seed={seed}: d_grk_row={res['d_grk_row']}, "
              f"is_degenerate={res['is_degenerate']}, "
              f"never_valid_labels={res['node_labels_never_event_aligned_valid']}")

    # Combine with the ORIGINAL pilot's hist_random rows (seeds 0, 1, 2) -- read-only,
    # never rewritten. Validity criterion matches the original pilot's exactly: a
    # realization is excluded only if EVERY trajectory in its row is NaN (a graph-level
    # degeneracy knocks out all K trajectories together, not just one).
    with open(ORIGINAL_PILOT_ANALYSIS_PATH, "rb") as f:
        original_pilot = pickle.load(f)
    original_hist = original_pilot["per_family"][FAMILY]
    original_d_grk_full = original_hist["d_grk_full_including_nan"]  # rows in seed order [0, 1, 2]

    combined_rows = {0: original_d_grk_full[0], 1: original_d_grk_full[1], 2: original_d_grk_full[2]}
    for seed in FOLLOWUP_SEEDS:
        combined_rows[seed] = new_results[seed]["d_grk_row"]

    all_seeds = [0, 1, 2, 3, 4]
    valid_seeds = [s for s in all_seeds if not np.all(np.isnan(combined_rows[s]))]
    excluded_seeds = [s for s in all_seeds if s not in valid_seeds]
    d_grk_valid = np.array([combined_rows[s] for s in valid_seeds])

    decomp = fit_crossed_variance_decomposition(d_grk_valid)
    conservative = conservative_variance_estimates(decomp)
    best_design, all_candidates = per_family_min_cost_design(
        conservative["sigma2_b_conservative"], conservative["sigma2_eps_conservative"])

    result = {
        "followup_seeds": FOLLOWUP_SEEDS,
        "pre_simulation_check": {seed: {"degrees": info["degrees"], "isolated": info["isolated"]}
                                  for seed, info in pre_check.items()},
        "new_realization_results": new_results,
        "combined_rows_all_seeds_0_to_4": combined_rows,
        "valid_seeds": valid_seeds,
        "excluded_seeds": excluded_seeds,
        "d_bar_gr_valid": decomp["row_means_d_bar_gr"],
        "variance_decomposition": decomp,
        "conservative_variance_estimates": conservative,
        "own_min_cost_design_refit": best_design,
        "all_candidate_designs": all_candidates,
    }
    with open(FOLLOWUP_RESULTS_PATH, "wb") as f:
        pickle.dump(result, f)

    print(f"\nValid seeds (used in refit): {valid_seeds}")
    print(f"Excluded seeds: {excluded_seeds}")
    print(f"d_bar_gr (valid only): {decomp['row_means_d_bar_gr']}")
    print(f"Variance decomposition: sigma2_b_point={decomp['sigma2_b_point']:.6f}, "
          f"sigma2_tau_point={decomp['sigma2_tau_point']:.6f}, "
          f"sigma2_eps_point={decomp['sigma2_eps_point']:.6f} "
          f"(df_r={decomp['df_r']}, df_k={decomp['df_k']}, df_resid={decomp['df_resid']})")
    print(f"Conservative estimates: {conservative}")
    print(f"hist_random's own min-cost design (refit): {best_design}")

    return result


if __name__ == "__main__":
    main()
