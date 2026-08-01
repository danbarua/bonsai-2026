"""
Data-generation script for the "frontier" (mechanistic/technical) report
visuals in docs/report_visuals/EXTRA_VISUALS_DESIGN.md -- a separate
audience from docs/report_visuals/generate_report_visuals.py's original 4
plain-language plots. Those plots explain the Level-2 finding to a
non-technical reader; the figures this data feeds explain the routing
*mechanism* itself (J(tau), the tangent-vs-finite gap, non-commutativity)
to the research team, and necessarily use the field's own notation.

**This is genuine new simulation, in the same sense and for the same
reason as analyze_stage1b2_time_resolved_propagator.py** (which this
script imports from and extends): stage1b2_results.pkl only ever cached
the tau=T=2.5 endpoint (plus one event-aligned snapshot) per trial, never
the full time series, and never over all 505 nodes -- only three
representative nodes were kept. Producing the design doc's temporal-
routing, pathway-openness, and early-leader/final-winner figures requires
that fuller time series. Concretely, beyond what
analyze_stage1b2_time_resolved_propagator.py already computed (delta_tau
at 3 selected nodes, for 3 seeds), this script adds:

  1. The SAME tangent solve, but retaining ALL 505 nodes (not 3) and ALL
     10 of Stage 1C's baseline trajectories (not 3) -- needed for the
     early-leader-vs-final-winner scatter (design doc item 6), which
     needs an early-time argmax over the whole node set for every
     trajectory, not just the three already singled out.
  2. theta_base(tau) over the same grid (a free byproduct of the tangent
     solve -- sol.y[:n, :] -- not an extra ODE solve), needed to compute
     J_ij(tau) = W_ij * cos(theta_j(tau) - theta_i(tau)) along chosen
     edges (design doc items 2-3).
  3. For the three illustrative seeds only (3000, 3030, 3090), the
     ACTUAL finite/nonlinear perturbed response q_finite(tau) over all
     505 nodes (design doc item 1's primary curve; the existing script
     only ever computed this at the tau=T endpoint via
     stage1b2_results.pkl, never as a full time series) -- this needs a
     second ODE solve of the nonlinear perturbed-theta system, exactly
     matching run_one_trial's second solve_ivp call, but retaining every
     timepoint.
  4. The tangent solve at every additional non-zero replica where ANY
     trial in the (node=high, t_p=0) cell concentrates (top1>0.5), for
     any of the 5 seeds that concentrate at all (3000, 3010, 3020, 3080,
     3090) -- found programmatically by find_concentrating_non_zero_replicas()
     reading the already-cached result files, not guessed or hand-
     enumerated. Without this, plot10's early-leader-vs-final-winner
     comparison would silently cover only the subset of concentrated
     trials whose replica happens to be 0 (an earlier version of this
     script did exactly that: 14 of 87 actually-concentrated trials in
     this cell family, with seeds 3020/3080 -- which never concentrate
     at replica=0 -- entirely absent). This adds 19 more tangent solves
     (~30s at the per-solve cost already observed above), giving full
     coverage of all 87 concentrated trials across all 5 seeds.

Everything here reuses stage1b2_core.py's and
analyze_stage1b2_time_resolved_propagator.py's own building blocks
(replica-state reconstruction, the locked solver settings) rather than
re-deriving anything. Does not touch or regenerate
results/stage1b2_results.pkl or any Stage 1C cache; output goes to a new,
separate cache in this same results/ directory
(stage1b2_frontier_visuals_data.pkl).
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

_THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _THIS_DIR / "results"
CACHE_PATH = RESULTS_DIR / "stage1b2_frontier_visuals_data.pkl"

sys.path.insert(0, str(_THIS_DIR))
from stage1b2_core import (
    rotation_projector, force_jacobian, get_degree_stratified_nodes,
    RTOL, ATOL, MAX_STEP, T_HORIZON,
)
from analyze_stage1b2_time_resolved_propagator import (
    load_T, reconstruct_replica_state_at_tp0, NODE_LABEL, REPLICA_IDX, K_COUPLING, N_T,
)

# The full 10-trajectory family Stage 1C established (Stage 1B2's own reference,
# seed=3000, plus the 9 independent trajectories Stage 1C added).
ALL_BASELINE_SEEDS = [3000, 3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080, 3090]
ILLUSTRATIVE_SEEDS = [3000, 3030, 3090]  # already-characterized in CONCENTRATION_REGIME_NOTE.md Part 3
SIGN = 1
AMPLITUDE = 0.025
STIM_NODE_LABEL = NODE_LABEL  # "high"

STAGE1B2_RESULTS_PATH = _THIS_DIR / "results" / "stage1b2_results.pkl"
STAGE1C_RESULTS_DIR = _THIS_DIR / ".." / "stage1c_trajectory_generalization" / "results"


def find_concentrating_non_zero_replicas():
    """Every (seed, replica) pair, replica != 0, where at least one trial in
    the (node=high, t_p=0) cell concentrates (top1>0.5) -- read directly
    from the already-cached stage1b2_results.pkl / Stage 1C result files,
    not guessed or hand-enumerated. replica=0 is excluded here because
    main()'s primary per-seed pass already covers it for every seed.

    Exists so plot10 (early-leader-vs-final-winner) can cover ALL
    concentrated trials in this cell, not just the ones whose seed's
    replica=0 happens to concentrate -- an earlier version of that plot
    covered only 14 of the 87 actually-concentrated trials in this cell
    family for exactly that reason (seeds 3020 and 3080 concentrate only
    at replicas 1 and 3, never 0, and were being silently dropped)."""
    pairs = []
    for seed in ALL_BASELINE_SEEDS:
        path = STAGE1B2_RESULTS_PATH if seed == 3000 else STAGE1C_RESULTS_DIR / f"stage1c_results_seed{seed}.pkl"
        with open(path, "rb") as f:
            results = pickle.load(f)
        cell = {k: v for k, v in results.items() if k[0] == 0 and k[2] == "high"}
        replicas = sorted({k[1] for k, t in cell.items()
                            if k[1] != 0 and float(np.max(np.asarray(t["fixed_time_q"]))) > 0.5})
        pairs.extend((seed, r) for r in replicas)
    return pairs


EDGE_PAIRS = [
    (129, 105, "source -> relay"),
    (105, 103, "relay -> destination (103)"),
    (129, 152, "source -> direct alternative (152)"),
]


def run_joint_tangent_full(W, replica_state, node, k_coupling=K_COUPLING, n_t=N_T):
    """Same joint (theta, delta) tangent system as
    analyze_stage1b2_time_resolved_propagator.run_tangent_timeseries, but
    additionally returns theta_base(tau) (a free byproduct of the same
    solve, not a second ODE) alongside the full, all-node delta_tau(tau)."""
    n = len(replica_state)
    P = rotation_projector(n)

    delta0 = np.zeros(n)
    delta0[node] = 1.0
    delta0 = P @ delta0
    delta0 = delta0 / np.linalg.norm(delta0)

    y0 = np.concatenate([replica_state, delta0])

    def rhs_tan(t, y):
        theta = y[:n]
        delta = y[n:]
        diff = theta[None, :] - theta[:, None]
        dtheta = k_coupling * np.sum(W * np.sin(diff), axis=1)
        DF = force_jacobian(W, theta, k_coupling=k_coupling)
        return np.concatenate([dtheta, DF @ delta])

    t_eval = np.linspace(0, T_HORIZON, n_t)
    sol = solve_ivp(rhs_tan, (0, T_HORIZON), y0, method="RK45", rtol=RTOL, atol=ATOL,
                     t_eval=t_eval, max_step=MAX_STEP)
    theta_base_tau = sol.y[:n, :]
    delta_tau = sol.y[n:, :]
    return t_eval, theta_base_tau, delta_tau, P, delta0


def run_finite_perturbed_full(W, replica_state, node, sign, amplitude, theta_base_tau,
                               delta_tau, delta0, P, t_eval, k_coupling=K_COUPLING):
    """Solves run_one_trial's SEPARATE nonlinear perturbed-theta system (same
    equations, same solver settings), retaining every timepoint instead of
    only the event-aligned index and the tau=T endpoint, then reconstructs
    q_finite(tau) over ALL nodes using the identical circular-mean alignment
    procedure run_one_trial uses at each timepoint (shift removal, then
    P-projection) -- copied, not re-derived, from stage1b2_core.py's
    run_one_trial / get_outputs_at."""
    n = len(replica_state)
    epsilon = sign * amplitude

    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return k_coupling * np.sum(W * np.sin(diff), axis=1)

    theta0_pert = replica_state + epsilon * delta0
    sol_pert = solve_ivp(rhs, (0, T_HORIZON), theta0_pert, method="RK45", rtol=RTOL, atol=ATOL,
                          t_eval=t_eval, max_step=MAX_STEP)
    theta_pert_tau = sol_pert.y

    n_t = len(t_eval)
    q_finite = np.full((n, n_t), np.nan)
    for i in range(n_t):
        shift = np.angle(np.mean(np.exp(1j * (theta_pert_tau[:, i] - theta_base_tau[:, i]))))
        actual_disp = P @ np.angle(np.exp(1j * (theta_pert_tau[:, i] - theta_base_tau[:, i] - shift)))
        norm_sq = np.sum(actual_disp ** 2)
        if norm_sq > 1e-12:
            q_finite[:, i] = actual_disp ** 2 / norm_sq
    return q_finite


def q_tangent_from_delta(delta_tau, epsilon, P):
    tangent_disp = epsilon * (P @ delta_tau)
    norm_sq = np.sum(tangent_disp ** 2, axis=0, keepdims=True)
    return tangent_disp ** 2 / norm_sq


def main():
    t0 = time.time()
    W = load_T()
    nodes = get_degree_stratified_nodes(W)
    stim_node = nodes[STIM_NODE_LABEL]
    print(f"Stimulated node: {stim_node}, sign={SIGN}, amplitude={AMPLITUDE}, replica={REPLICA_IDX}")

    cache = {
        "stim_node": stim_node,
        "edge_pairs": EDGE_PAIRS,
        "trial_spec": {"node_label": STIM_NODE_LABEL, "t_p": 0, "sign": SIGN,
                        "amplitude": AMPLITUDE, "replica": REPLICA_IDX},
        "all_baseline_seeds": ALL_BASELINE_SEEDS,
        "illustrative_seeds": ILLUSTRATIVE_SEEDS,
        "per_seed": {},
    }

    for seed in ALL_BASELINE_SEEDS:
        print(f"\n=== seed={seed} ===")
        replica_state = reconstruct_replica_state_at_tp0(W, seed, REPLICA_IDX)
        t_eval, theta_base_tau, delta_tau, P, delta0 = run_joint_tangent_full(W, replica_state, stim_node)
        epsilon = SIGN * AMPLITUDE
        q_tangent_full = q_tangent_from_delta(delta_tau, epsilon, P)
        print(f"  tangent solve done ({time.time()-t0:.1f}s elapsed total)")

        entry = {
            "t_eval": t_eval,
            "theta_base_tau": theta_base_tau,
            "q_tangent_full": q_tangent_full,
            "q_finite_full": None,
        }

        if seed in ILLUSTRATIVE_SEEDS:
            q_finite_full = run_finite_perturbed_full(
                W, replica_state, stim_node, SIGN, AMPLITUDE,
                theta_base_tau, delta_tau, delta0, P, t_eval)
            entry["q_finite_full"] = q_finite_full
            print(f"  finite (nonlinear) solve done ({time.time()-t0:.1f}s elapsed total)")

        cache["per_seed"][seed] = entry

    extra_seed_replicas = find_concentrating_non_zero_replicas()
    print(f"\n{len(extra_seed_replicas)} additional (seed, replica) pairs need a tangent solve "
          f"(non-zero replicas with a concentrated trial): {extra_seed_replicas}")
    cache["extra_tangent_by_seed_replica"] = {}
    for seed, replica in extra_seed_replicas:
        print(f"\n=== seed={seed}, replica={replica} (extra, non-zero replica) ===")
        replica_state = reconstruct_replica_state_at_tp0(W, seed, replica)
        t_eval, theta_base_tau, delta_tau, P, delta0 = run_joint_tangent_full(W, replica_state, stim_node)
        q_tangent_full = q_tangent_from_delta(delta_tau, SIGN * AMPLITUDE, P)
        cache["extra_tangent_by_seed_replica"][(seed, replica)] = {
            "t_eval": t_eval, "q_tangent_full": q_tangent_full,
        }
        print(f"  tangent solve done ({time.time()-t0:.1f}s elapsed total)")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved to {CACHE_PATH} ({time.time()-t0:.1f}s total)")
    return cache


if __name__ == "__main__":
    main()
