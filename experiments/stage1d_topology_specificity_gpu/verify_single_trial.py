import pickle
import numpy as np

from stage1b2_core import (
    run_one_trial, generate_reference_baseline, generate_fixed_replica_directions,
    get_degree_stratified_nodes, NEARBY_SCALE,
)
from run_one_trial_jax_faithful import run_one_trial_jax_faithful
import jax.numpy as jnp

with open("class0_constructions.pkl", "rb") as f:
    data = pickle.load(f)[0]
W = data["constructions"]["T"]
nodes = get_degree_stratified_nodes(W)
print("nodes:", nodes)

# Real replica state, matching the actual pipeline's construction exactly
BASELINE_SEED = 3000
REPLICA_DIRECTION_SEED = 3001
ref_sol = generate_reference_baseline(W, BASELINE_SEED, 2.5)
state_at_tp0 = ref_sol.sol(0.833)  # a real, nontrivial t_p, not the trivial t_p=0 raw draw
directions = generate_fixed_replica_directions(W.shape[0], REPLICA_DIRECTION_SEED, 6)
replica_state = np.mod(state_at_tp0 + NEARBY_SCALE * directions[2], 2 * np.pi)  # replica index 2, arbitrary

test_cases = [
    (nodes['low'], 1, 0.025),
    (nodes['median'], -1, 0.2),
    (nodes['high'], 1, 0.8),
    (nodes['high'], -1, 0.8),  # a case likely to show strong nonlinearity
]

W_jax = jnp.asarray(W)
replica_state_jax = jnp.asarray(replica_state)

max_diffs = {}

for node, sign, amp in test_cases:
    print(f"\n{'='*60}\nnode={node}, sign={sign}, amp={amp}\n{'='*60}")
    ref = run_one_trial(W, replica_state, node, sign, amp)
    jaxr = run_one_trial_jax_faithful(W_jax, replica_state_jax, node, float(sign), float(amp))
    jaxr = {k: (np.asarray(v) if hasattr(v, 'shape') or isinstance(v, (float, bool)) else v)
            for k, v in jaxr.items()}

    print(f"tau_star: ref={ref['tau_star']:.6f}  jax={float(jaxr['tau_star']):.6f}")
    print(f"event_aligned_valid: ref={ref['event_aligned_valid']}  jax={bool(jaxr['event_aligned_valid'])}")
    print(f"E_at_tau_star: ref={ref['E_at_tau_star']:.6e}  jax={float(jaxr['E_at_tau_star']):.6e}")

    fields_to_compare = [
        'event_aligned_q', 'event_aligned_r', 'event_aligned_q_tangent', 'event_aligned_q_residual',
        'fixed_time_q', 'fixed_time_r', 'fixed_time_q_tangent', 'fixed_time_q_residual',
    ]
    for field in fields_to_compare:
        ref_val = ref[field]
        jax_val = jaxr[field]
        if ref_val is None:
            print(f"  {field}: ref=None, jax has nan={np.all(np.isnan(jax_val))}")
            continue
        diff = np.max(np.abs(np.asarray(ref_val) - jax_val))
        max_diffs[field] = max(max_diffs.get(field, 0), diff)
        print(f"  {field}: max abs diff = {diff:.3e}")

    scalar_fields = ['event_aligned_J_tan', 'event_aligned_f_source', 'event_aligned_residual_norm',
                      'fixed_time_J_tan', 'fixed_time_f_source', 'fixed_time_residual_norm',
                      'initial_f_source', 'peak_C']
    for field in scalar_fields:
        ref_val = ref[field]
        jax_val = float(jaxr[field])
        if ref_val is None:
            print(f"  {field}: ref=None, jax={jax_val}")
            continue
        diff = abs(ref_val - jax_val)
        max_diffs[field] = max(max_diffs.get(field, 0), diff)
        print(f"  {field}: ref={ref_val:.6e}  jax={jax_val:.6e}  diff={diff:.3e}")

print(f"\n{'='*60}\nSUMMARY: max abs diff across all tested trials, per field\n{'='*60}")
for field, diff in max_diffs.items():
    print(f"  {field}: {diff:.3e}")
