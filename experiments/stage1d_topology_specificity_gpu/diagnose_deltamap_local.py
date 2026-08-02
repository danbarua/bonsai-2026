"""
Local (CPU, numpy-only) reproduction of the Delta_map discrepancy found on
the A100 GPU pilot run (real_pilot_benchmark.py): for T's own trajectory
(seed=3000), that run reported pooled Delta_map=0.2842, vs. Stage 1C's
already-cached, trusted value of 0.3505 for this exact trajectory.

The GPU session that ran the diagnostic (diagnose_deltamap.py) was
ephemeral and terminated before its findings could be recovered locally
or pushed to the repo -- this script re-derives the diagnosis from
scratch, using the real stage1b2_core.py/analyze_stage1b2.py pipeline
(not the JAX port, which was already verified elsewhere to match numpy
to ~1e-6-1e-8 precision per-trial) to isolate which bug(s) in
real_pilot_benchmark.py's glue code explain the gap.

Two candidate bugs identified by reading real_pilot_benchmark.py's
captured source (from the dead session's exported history) against the
real reference:

1. Direction generation: build_432_batch() drew replica directions via
   `rng.uniform(-1, 1, n)` -- raw, unprojected, unnormalized -- instead
   of calling the real generate_fixed_replica_directions(), which draws
   normal(0,1), projects out the rotation-invariant component, and
   unit-normalizes.

2. Invalid-trial gating: run_one_trial() returns a dict of Nones for the
   whole event_aligned_* block when event_aligned_valid is False (E(tau)
   never exceeded E_MIN). The JAX port can't return None from a batched
   computation, so it always computes event_aligned_q and pushes the
   E_min gating onto the caller -- but real_pilot_benchmark.py stored
   event_aligned_q unconditionally and never checked event_aligned_valid
   before handing results to analyze_stage1b2.py, whose d_q() only
   excludes on `is None`. E_min-invalid trials therefore leaked into the
   Delta_map average instead of being excluded.

This script computes all four combinations (correct/buggy directions x
correct/buggy gating) using the real numpy run_one_trial, to attribute
the observed gap to one, both, or neither bug.
"""
import pickle
import numpy as np
import multiprocessing as mp
from functools import partial

import sys
sys.path.insert(0, '/Users/dan/Code/pycharm/bonsai-2026/experiments/stage1b2_structured_transformation')

from stage1b2_core import (
    run_one_trial, generate_reference_baseline, generate_fixed_replica_directions,
    get_degree_stratified_nodes, T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS,
    NEARBY_SCALE, T_HORIZON,
)
from analyze_stage1b2 import load_results_as_arrays, compute_W_B_deltamap

BASELINE_SEED = 3000
REPLICA_DIRECTION_SEED = BASELINE_SEED + 1
KNOWN_GOOD_DELTAMAP = 0.3505  # Stage 1C's cached figure for T, seed=3000


def run_one_trial_always(W, replica_state, node, sign, amplitude, k_coupling=1.0):
    """Identical to stage1b2_core.run_one_trial, except event_aligned_* is
    ALWAYS computed at tau_star_idx regardless of event_aligned_valid --
    reproducing what the JAX port + real_pilot_benchmark.py's un-gated
    glue code actually did, instead of correctly collapsing to None."""
    from stage1b2_core import (
        rotation_projector, force_jacobian, normalized_energy, signed_direction,
        Q_NORM_THRESHOLD, E_MIN,
    )
    from scipy.integrate import solve_ivp
    from scipy.spatial.distance import jensenshannon

    RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05
    n = len(replica_state)
    P = rotation_projector(n)
    epsilon = sign * amplitude

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

    t_eval = np.linspace(0, T_HORIZON, 51)
    sol_tan = solve_ivp(rhs_tan, (0, T_HORIZON), y0, method='RK45', rtol=RTOL, atol=ATOL,
                         t_eval=t_eval, max_step=MAX_STEP)
    theta_base_tau = sol_tan.y[:n, :]
    delta_tau = sol_tan.y[n:, :]

    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return k_coupling * np.sum(W * np.sin(diff), axis=1)

    theta0_pert = replica_state + epsilon * delta0
    sol_pert = solve_ivp(rhs, (0, T_HORIZON), theta0_pert, method='RK45', rtol=RTOL, atol=ATOL,
                          t_eval=t_eval, max_step=MAX_STEP)
    theta_pert_tau = sol_pert.y

    eta = 1e-10
    E_list = []
    for i in range(len(t_eval)):
        shift = np.angle(np.mean(np.exp(1j * (theta_pert_tau[:, i] - theta_base_tau[:, i]))))
        actual_disp = P @ np.angle(np.exp(1j * (theta_pert_tau[:, i] - theta_base_tau[:, i] - shift)))
        predicted_disp = epsilon * (P @ delta_tau[:, i])
        E = np.linalg.norm(actual_disp - predicted_disp) / (abs(epsilon) * np.linalg.norm(P @ delta_tau[:, i]) + eta)
        E_list.append(E)
    E_arr = np.array(E_list)

    tau_star_idx = int(np.argmax(E_arr))

    def get_outputs_at(idx):
        shift = np.angle(np.mean(np.exp(1j * (theta_pert_tau[:, idx] - theta_base_tau[:, idx]))))
        actual_disp = P @ np.angle(np.exp(1j * (theta_pert_tau[:, idx] - theta_base_tau[:, idx] - shift)))
        q_finite = normalized_energy(actual_disp)
        return {'q': q_finite}

    # BUG: always computed, never collapsed to None when event_aligned_valid is False
    event_aligned = get_outputs_at(tau_star_idx)
    return {'event_aligned_q': event_aligned['q']}


def build_replica_directions(n, seed, n_replicas, buggy):
    if buggy:
        rng = np.random.default_rng(seed)
        return [rng.uniform(-1, 1, n) for _ in range(n_replicas)]
    return generate_fixed_replica_directions(n, seed, n_replicas)


def _compute_one(args, W, buggy_gating):
    key, replica_state, node, sign, amp = args
    if buggy_gating:
        result = run_one_trial_always(W, replica_state, node, sign, amp)
    else:
        result = run_one_trial(W, replica_state, node, sign, amp)
    return key, result['event_aligned_q']


def compute_pooled_deltamap(W, nodes, buggy_directions, buggy_gating, n_workers):
    n = W.shape[0]
    node_labels = list(nodes.keys())

    sol_b = generate_reference_baseline(W, BASELINE_SEED, T_HORIZON)
    directions = build_replica_directions(n, REPLICA_DIRECTION_SEED, N_REPLICAS, buggy_directions)

    jobs = []
    for t_p in T_P_VALUES:
        state_at_tp = sol_b.sol(t_p)
        for r in range(N_REPLICAS):
            replica_state = np.mod(state_at_tp + NEARBY_SCALE * directions[r], 2 * np.pi)
            for node_label in node_labels:
                node = nodes[node_label]
                for sign in SIGNS:
                    for amp in AMPLITUDES:
                        key = (t_p, r, node_label, sign, amp)
                        jobs.append((key, replica_state, node, sign, amp))

    worker = partial(_compute_one, W=W, buggy_gating=buggy_gating)
    with mp.Pool(n_workers) as pool:
        pairs = pool.map(worker, jobs)

    results = {key: {'event_aligned_q': q} for key, q in pairs}
    organized = load_results_as_arrays(results, nodes, time_key='event_aligned_q')
    per_tp = {t_p: compute_W_B_deltamap(organized[t_p], node_labels)['delta_map'] for t_p in T_P_VALUES}
    return np.mean(list(per_tp.values())), per_tp


if __name__ == '__main__':
    with open('/Users/dan/Code/pycharm/bonsai-2026/experiments/stage1b2_structured_transformation/results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W_T = data['constructions']['T']
    nodes_T = get_degree_stratified_nodes(W_T)
    print(f'W_T shape: {W_T.shape}, nodes_T: {nodes_T}')

    n_workers = max(1, mp.cpu_count() - 1)
    print(f'Using {n_workers} worker processes\n')

    configs = [
        ('correct directions, correct gating (should match cached 0.3505)', False, False),
        ('BUGGY directions,   correct gating', True, False),
        ('correct directions, BUGGY gating', False, True),
        ('BUGGY directions,   BUGGY gating (full real_pilot_benchmark.py repro, should match 0.2842)', True, True),
    ]

    print(f'{"="*70}\nKNOWN GOOD (Stage 1C cache): {KNOWN_GOOD_DELTAMAP}')
    print(f'GPU pilot run reported:      0.2842\n{"="*70}\n')

    for label, buggy_dirs, buggy_gate in configs:
        pooled, per_tp = compute_pooled_deltamap(W_T, nodes_T, buggy_dirs, buggy_gate, n_workers)
        print(f'{label}')
        print(f'  per-t_p: {per_tp}')
        print(f'  pooled Delta_map: {pooled:.4f}\n')
