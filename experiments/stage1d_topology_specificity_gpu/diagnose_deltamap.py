import sys
sys.path.insert(0, '/content')
import pickle
import numpy as np

from stage1b2_core import (
    run_one_trial, T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, NEARBY_SCALE,
    T_HORIZON, RTOL, ATOL, MAX_STEP,
)
from analyze_stage1b2 import load_results_as_arrays, compute_W_B_deltamap
from scipy.integrate import solve_ivp

with open('/content/pilot_constructions.pkl', 'rb') as f:
    pc = pickle.load(f)
W_T = pc['W_T']
nodes_T = pc['nodes_T']
n = W_T.shape[0]
node_labels = list(nodes_T.keys())

def rhs_theta_only_np(t, theta, Wg):
    diff = theta[None, :] - theta[:, None]
    return np.sum(Wg * np.sin(diff), axis=1)

baseline_seed = 3000
replica_direction_seed = baseline_seed + 1
rng_b = np.random.default_rng(baseline_seed)
theta0_b = rng_b.uniform(0, 2 * np.pi, n)
sol_b = solve_ivp(rhs_theta_only_np, (0, T_HORIZON), theta0_b, args=(W_T,), method='RK45',
                   rtol=RTOL, atol=ATOL, max_step=MAX_STEP, dense_output=True)
rng_r = np.random.default_rng(replica_direction_seed)
directions = [rng_r.uniform(-1, 1, n) for _ in range(N_REPLICAS)]

print(f"Running all 432 trials via the REAL numpy run_one_trial, same batch-construction "
      f"logic as the GPU benchmark script, for T/seed=3000...")

results_dict = {}
n_invalid = 0
for t_p in T_P_VALUES:
    state_at_tp = sol_b.sol(t_p)
    for r in range(N_REPLICAS):
        replica_state = np.mod(state_at_tp + NEARBY_SCALE * directions[r], 2 * np.pi)
        for node_label in node_labels:
            node = nodes_T[node_label]
            for sign in SIGNS:
                for amp in AMPLITUDES:
                    res = run_one_trial(W_T, replica_state, node, sign, amp)
                    if not res['event_aligned_valid']:
                        n_invalid += 1
                    results_dict[(t_p, r, node_label, sign, amp)] = res

print(f"Invalid trials: {n_invalid} of 432")

organized = load_results_as_arrays(results_dict, nodes_T, time_key='event_aligned_q')
per_tp = {t_p: compute_W_B_deltamap(organized[t_p], node_labels)['delta_map'] for t_p in T_P_VALUES}
pooled = np.mean(list(per_tp.values()))
print(f"Per-t_p Delta_map: {per_tp}")
print(f"Pooled Delta_map (numpy, same batch logic as GPU benchmark): {pooled:.4f}")
print(f"Compare: Stage 1C's cached figure = 0.3505, GPU benchmark got 0.2842")

with open('/content/diagnose_results.pkl', 'wb') as f:
    pickle.dump({'results_dict': results_dict, 'pooled_delta_map': pooled}, f)
print("Saved to /content/diagnose_results.pkl")
