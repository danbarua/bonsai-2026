"""
New-simulation follow-up to Sections 10-11 of concentration_regime_notebook.ipynb
and CONCENTRATION_REGIME_NOTE.md. Section 11 found that neither a J(0) snapshot
nor a naive trapezoid time-integral of individual Jacobian entries cleanly
explains which pathway wins at (node=high, t_p=0): several edges flip sign
across [0, 2.5], and seed=3000's direct source->152 edge has the LARGEST
integrated exposure of its three candidate edges, yet 3000 still concentrates
onto 103, not 152.

This script gets one step closer to the actual time-ordered propagator by
retaining the FULL tangent solution delta_tau(t) at all 51 evaluation points
in [0, 2.5] -- not just the tau=T=2.5 endpoint that stage1b2_results.pkl
caches -- for three trials at (node_label='high', t_p=0, sign=+1,
amplitude=0.025, replica=0):
  - seed=3000 (concentrates onto node 103, 24/36 trials in this cell)
  - seed=3030 (no concentration, 0/36 trials in this cell)
  - seed=3090 (concentrates onto node 152, 35/36 trials in this cell)

**This is genuine new simulation** (re-integrating the joint (theta, delta)
tangent ODE and retaining every timepoint, rather than the pure re-analysis
of the already-frozen cache that Sections 1-9 and this file's own
analyze_stage1b2_concentration_regime.py performed) -- disclosed plainly.
Does not touch, modify, or regenerate results/stage1b2_results.pkl or any of
Stage 1C's cached trajectory files; output is written to a new, separate
cache in this same results/ directory
(stage1b2_time_resolved_propagator.pkl).

Note on why sign/amplitude don't matter here: q_tangent(t), as computed in
this script, does not depend on sign or amplitude at all. Inspecting
run_one_trial's rhs_tan in stage1b2_core.py, the tangent ODE for delta has no
epsilon term anywhere -- epsilon only multiplies delta_tau AFTER integration,
to form the reported tangent_disp, and both its scale and sign vanish under
q's normalization (x^2 / sum(x^2) is invariant to x -> c*x for any nonzero
c). So delta_tau(t), and therefore q_tangent(t), depends only on
(baseline_seed, node, replica) -- not on sign or amplitude. This is already
implicit in CONCENTRATION_REGIME_NOTE.md Part 2's finding that q_tangent's
top1 doesn't vary with sign or amplitude (0.644-0.696 across all 36 trials in
the cell). The (sign=+1, amplitude=0.025) trial specified below is used only
to fix one literal, already-identified concrete trial from Part 1's
breakdown table -- the same delta_tau(t) curve is shared by all 6
(sign, amplitude) conditions at this (t_p, node, replica) cell.
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

_THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _THIS_DIR / "results"
NEW_CACHE_PATH = RESULTS_DIR / "stage1b2_time_resolved_propagator.pkl"
PLOT_PATH = RESULTS_DIR / "stage1b2_time_resolved_propagator.png"

sys.path.insert(0, str(_THIS_DIR))
from stage1b2_core import (
    rotation_projector, force_jacobian, generate_reference_baseline,
    generate_fixed_replica_directions, get_degree_stratified_nodes,
    RTOL, ATOL, MAX_STEP, T_HORIZON, NEARBY_SCALE,
)

NODES_OF_INTEREST = [103, 105, 152]
TRIALS = [
    {"baseline_seed": 3000, "outcome": "concentrates to 103 (24/36 trials in this cell)"},
    {"baseline_seed": 3030, "outcome": "no concentration (0/36 trials in this cell)"},
    {"baseline_seed": 3090, "outcome": "concentrates to 152 (35/36 trials in this cell)"},
]
NODE_LABEL = "high"
T_P = 0
SIGN = 1
AMPLITUDE = 0.025
REPLICA_IDX = 0
K_COUPLING = 1.0
N_T = 51


def load_T():
    with open(RESULTS_DIR / "class0_constructions.pkl", "rb") as f:
        data = pickle.load(f)[0]
    return data["constructions"]["T"]


def reconstruct_replica_state_at_tp0(W, baseline_seed, replica_idx=REPLICA_IDX):
    """Reconstructs the exact initial replica state each trial in this cell
    actually started from at t_p=0 -- same construction run_stage1b2.py /
    run_stage1c.py use (BASELINE_SEED for the reference trajectory,
    BASELINE_SEED+1 for replica directions), reused unchanged from
    stage1b2_core's own building blocks. Matches the notebook's Section 11
    reconstruct_theta_at_tp0 exactly. Cheap and deterministic (same seed
    that already produced every cached trial for this baseline), not a new
    simulation of any trial's actual dynamics on its own -- the genuinely
    new part is the tangent integration below, retained at every timepoint."""
    n = W.shape[0]
    replica_direction_seed = baseline_seed + 1
    ref_sol = generate_reference_baseline(W, baseline_seed, T_HORIZON)
    state_at_tp0 = ref_sol.sol(0)
    directions = generate_fixed_replica_directions(n, replica_direction_seed, 6)
    return (state_at_tp0 + NEARBY_SCALE * directions[replica_idx]) % (2 * np.pi)


def run_tangent_timeseries(W, replica_state, node, k_coupling=K_COUPLING, n_t=N_T):
    """Minimal variant of run_one_trial (stage1b2_core.py): solves the SAME
    joint (theta, delta) tangent system run_one_trial's rhs_tan solves, with
    identical solver settings, but returns delta_tau at every evaluated
    timepoint instead of discarding all but the event-aligned index and the
    tau=T endpoint. Deliberately does NOT solve run_one_trial's second,
    separate nonlinear perturbed-theta system -- unneeded here, since
    q_tangent depends only on delta_tau, never on the finite/nonlinear
    response (see module docstring)."""
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
    delta_tau = sol.y[n:, :]  # shape (n_nodes, n_t)
    return t_eval, delta_tau


def q_tangent_timeseries(delta_tau):
    """q_i(t) = delta_tau(t)_i^2 / sum_j delta_tau(t)_j^2, at every timepoint.
    Note: delta_tau is already effectively P-orthogonal throughout (the
    tangent generator DF has each row summing to zero, so the
    rotation-uniform mode is invariant), so no re-projection is needed here
    -- matching run_one_trial's own treatment where P is applied once, to
    delta0, and preserved by the dynamics thereafter."""
    return delta_tau ** 2 / np.sum(delta_tau ** 2, axis=0, keepdims=True)


def main():
    W = load_T()
    nodes = get_degree_stratified_nodes(W)
    stim_node = nodes[NODE_LABEL]
    print(f"High-degree stimulated node index: {stim_node}")
    print(f"Fixed trial spec: node_label={NODE_LABEL}, t_p={T_P}, sign={SIGN}, "
          f"amplitude={AMPLITUDE}, replica={REPLICA_IDX} (see module docstring: "
          f"sign/amplitude don't affect q_tangent(t), only node/replica/seed do)")

    cache = {
        "nodes_of_interest": NODES_OF_INTEREST,
        "stim_node": stim_node,
        "trial_spec": {"node_label": NODE_LABEL, "t_p": T_P, "sign": SIGN,
                        "amplitude": AMPLITUDE, "replica": REPLICA_IDX},
        "trials": {},
    }
    for spec in TRIALS:
        seed = spec["baseline_seed"]
        print(f"\n=== seed={seed} ({spec['outcome']}) ===")
        replica_state = reconstruct_replica_state_at_tp0(W, seed, REPLICA_IDX)
        t_eval, delta_tau = run_tangent_timeseries(W, replica_state, stim_node)
        q_t = q_tangent_timeseries(delta_tau)  # (n_nodes, n_t)

        argmax_final = int(np.argmax(q_t[:, -1]))
        cache["trials"][seed] = {
            "outcome_label": spec["outcome"],
            "t_eval": t_eval,
            "q_t_selected_nodes": {node: q_t[node, :].copy() for node in NODES_OF_INTEREST},
            "argmax_node_final": argmax_final,
            "top1_final": float(q_t[:, -1].max()),
        }

        for node in NODES_OF_INTEREST:
            series = q_t[node, :]
            argmax_t = t_eval[np.argmax(series)]
            argmin_t = t_eval[np.argmin(series)]
            marker = "  <-- stimulated node" if node == stim_node else ""
            print(f"  node {node}{marker}: q(0)={series[0]:.4f}, "
                  f"q(t={t_eval[len(t_eval)//2]:.2f})={series[len(series)//2]:.4f}, "
                  f"q(T)={series[-1]:.4f}, max={series.max():.4f} at t={argmax_t:.3f}, "
                  f"min={series.min():.4f} at t={argmin_t:.3f}")
        print(f"  argmax over ALL nodes at t=T: node {argmax_final}, "
              f"top1={cache['trials'][seed]['top1_final']:.4f}")

    with open(NEW_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved to {NEW_CACHE_PATH}")

    plot_energy_share_timeseries(cache)
    return cache


def plot_energy_share_timeseries(cache):
    seeds = list(cache["trials"].keys())
    fig, axes = plt.subplots(1, len(seeds), figsize=(16, 4.5), sharey=True)
    colors = {103: "tab:blue", 105: "tab:green", 152: "tab:orange"}
    stim_node = cache["stim_node"]

    for ax, seed in zip(axes, seeds):
        d = cache["trials"][seed]
        t = d["t_eval"]
        for node in cache["nodes_of_interest"]:
            q = d["q_t_selected_nodes"][node]
            label = f"node {node}" + (" (stimulated)" if node == stim_node else "")
            ax.plot(t, q, label=label, color=colors[node], linewidth=2)
        ax.set_title(f"seed={seed}\n{d['outcome_label']}", fontsize=10)
        ax.set_xlabel(r"$\tau$ (time since perturbation)")
        ax.set_ylim(-0.02, 0.8)

    axes[0].set_ylabel(r"$q_{\mathrm{tangent}}(\tau)$ (share of tangent energy)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Time-resolved tangent energy share: does the eventual winner "
                 "lead the whole way, or overtake late?\n(new simulation -- full "
                 r"$\delta_\tau(\tau)$ time series, not the cached $\tau{=}T$ endpoint)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
