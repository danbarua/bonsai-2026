import sys
sys.path.insert(0, '/content')
import pickle
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stage1b2_core import (
    generate_reference_baseline, generate_fixed_replica_directions, NEARBY_SCALE,
    T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, T_HORIZON, RTOL, ATOL, MAX_STEP,
)
from run_one_trial_jax_faithful import run_one_trial_jax_faithful
from analyze_stage1b2 import load_results_as_arrays, compute_W_B_deltamap
from scipy.integrate import solve_ivp

print("JAX backend:", jax.default_backend(), "| x64:", jax.config.jax_enable_x64)

with open('/content/pilot_constructions.pkl', 'rb') as f:
    pc = pickle.load(f)

W_T = pc['W_T']
W_lattice = pc['W_lattice']
pilot_constructions = pc['pilot_constructions']
nodes_T = pc['nodes_T']
n = W_T.shape[0]
node_labels = list(nodes_T.keys())
nodes_batch = [nodes_T['low'], nodes_T['median'], nodes_T['high']]

print("nodes_T:", nodes_T)

def rhs_theta_only_np(t, theta, Wg):
    diff = theta[None, :] - theta[:, None]
    return np.sum(Wg * np.sin(diff), axis=1)

def build_432_batch(Wg_np, baseline_seed):
    """Builds the full 432-trial (theta0, node, sign, amp) batch for ONE trajectory
    on graph Wg_np, matching the real (t_p, replica, node, sign, amp) design exactly,
    tagged with the (t_p, replica, node_label, sign, amp) key each trial corresponds to."""
    replica_direction_seed = baseline_seed + 1
    rng_b = np.random.default_rng(baseline_seed)
    theta0_b = rng_b.uniform(0, 2 * np.pi, n)
    sol_b = solve_ivp(rhs_theta_only_np, (0, T_HORIZON), theta0_b, args=(Wg_np,), method='RK45',
                       rtol=RTOL, atol=ATOL, max_step=MAX_STEP, dense_output=True)
    directions = generate_fixed_replica_directions(n, replica_direction_seed, N_REPLICAS)

    theta0_list, node_list, sign_list, amp_list, keys = [], [], [], [], []
    for t_p in T_P_VALUES:
        state_at_tp = sol_b.sol(t_p)
        for r in range(N_REPLICAS):
            replica_state = np.mod(state_at_tp + NEARBY_SCALE * directions[r], 2 * np.pi)
            for node_label in node_labels:
                node = nodes_T[node_label]
                for sign in SIGNS:
                    for amp in AMPLITUDES:
                        theta0_list.append(replica_state)
                        node_list.append(node)
                        sign_list.append(float(sign))
                        amp_list.append(amp)
                        keys.append((t_p, r, node_label, sign, amp))
    return (jnp.asarray(np.stack(theta0_list)), jnp.asarray(node_list),
            jnp.asarray(sign_list), jnp.asarray(amp_list), keys)

batched_run = jax.jit(jax.vmap(run_one_trial_jax_faithful, in_axes=(None, 0, 0, 0, 0)))

LATTICE_SEEDS = [3000, 3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080, 3090]
PILOT_TRAJECTORY_SEEDS = [3000, 3010, 3020]

jobs = [('lattice', None, W_lattice, LATTICE_SEEDS)]
for family, realizations in pilot_constructions.items():
    for seed, Wg in realizations.items():
        jobs.append((family, seed, Wg, PILOT_TRAJECTORY_SEEDS))

print(f"{len(jobs)} graphs, {sum(len(j[3]) for j in jobs)} trajectories total (should be 37)")

all_results = {}
total_elapsed = 0.0

for name, seed, Wg_np, baseline_seeds in jobs:
    Wg_jax = jnp.asarray(Wg_np)
    for baseline_seed in baseline_seeds:
        theta0_b, node_b, sign_b, amp_b, keys = build_432_batch(Wg_np, baseline_seed)

        _ = batched_run(Wg_jax, theta0_b, node_b, sign_b, amp_b)
        jax.block_until_ready(_)  # warm-up, excluded from timing

        t0 = time.perf_counter()
        result = batched_run(Wg_jax, theta0_b, node_b, sign_b, amp_b)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        label = f"{name}" if seed is None else f"{name} seed={seed}"
        print(f"{label} traj={baseline_seed}: {elapsed:.2f}s")

        # Reformat into the real analyze_stage1b2.py-compatible structure.
        # event_aligned_q must be None when invalid, matching the numpy
        # run_one_trial contract -- the JAX port can't return None from a
        # batched/jitted computation, so it always returns a real (possibly
        # meaningless) array plus this validity flag; the caller (here) is
        # responsible for gating on it before analyze_stage1b2.py sees it,
        # since d_q() only excludes a pair when it's literally None.
        results_dict = {}
        for i, key in enumerate(keys):
            valid = bool(result['event_aligned_valid'][i])
            results_dict[key] = {
                'event_aligned_q': np.asarray(result['event_aligned_q'][i]) if valid else None,
                'event_aligned_valid': valid,
            }
        all_results[(name, seed, baseline_seed)] = results_dict

print()
print(f"Total JAX GPU simulation time, all 37 trajectories: {total_elapsed:.2f}s")
M1_SECONDS = 61 * 60
print(f"M1 baseline (real, measured, simulation stage only): {M1_SECONDS}s (61 min)")
print(f"Speedup vs real M1 baseline: {M1_SECONDS/total_elapsed:.1f}x")

print()
print("=" * 60)
print("Real Delta_map for T's own trajectory seed=3000, computed from THIS GPU run's")
print("event_aligned_q output via the REAL analyze_stage1b2.py functions --")
print("cross-check against Stage 1C's already-known ~0.35 figure for this exact trajectory.")
print("=" * 60)

# T itself isn't in `jobs` (jobs only covers lattice + stochastic controls per the pilot design),
# so compute it directly here for the cross-check, same 432-trial construction.
theta0_b, node_b, sign_b, amp_b, keys = build_432_batch(W_T, 3000)
result = batched_run(jnp.asarray(W_T), theta0_b, node_b, sign_b, amp_b)
results_dict = {}
for i, key in enumerate(keys):
    valid = bool(result['event_aligned_valid'][i])
    results_dict[key] = {
        'event_aligned_q': np.asarray(result['event_aligned_q'][i]) if valid else None,
        'event_aligned_valid': valid,
    }
organized = load_results_as_arrays(results_dict, nodes_T, time_key='event_aligned_q')
per_tp = {t_p: compute_W_B_deltamap(organized[t_p], node_labels)['delta_map'] for t_p in T_P_VALUES}
pooled = np.mean(list(per_tp.values()))
print(f"Per-t_p Delta_map: {per_tp}")
print(f"Pooled Delta_map (T, seed=3000, from THIS GPU run): {pooled:.4f}")
print("(Stage 1C's own cached figure for T, seed=3000: 0.3505 -- compare)")

with open('/content/gpu_pilot_results.pkl', 'wb') as f:
    pickle.dump({'all_results': all_results, 'total_elapsed': total_elapsed,
                 'T_seed3000_delta_map': pooled}, f)
print()
print("Saved to /content/gpu_pilot_results.pkl")
