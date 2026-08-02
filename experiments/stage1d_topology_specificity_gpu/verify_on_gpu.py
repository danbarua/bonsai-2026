import sys
sys.path.insert(0, '/content')
import pickle
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
print("JAX x64 enabled:", jax.config.jax_enable_x64)
print()

from stage1b2_core import (
    run_one_trial, generate_reference_baseline, generate_fixed_replica_directions,
    get_degree_stratified_nodes, NEARBY_SCALE,
)
from run_one_trial_jax_faithful import run_one_trial_jax_faithful

with open("/content/class0_constructions.pkl", "rb") as f:
    data = pickle.load(f)[0]
W = data["constructions"]["T"]
nodes = get_degree_stratified_nodes(W)
print("nodes:", nodes)

BASELINE_SEED = 3000
REPLICA_DIRECTION_SEED = 3001
ref_sol = generate_reference_baseline(W, BASELINE_SEED, 2.5)
state_at_tp = ref_sol.sol(0.833)
directions = generate_fixed_replica_directions(W.shape[0], REPLICA_DIRECTION_SEED, 6)
replica_state = np.mod(state_at_tp + NEARBY_SCALE * directions[2], 2 * np.pi)

test_cases = [
    (nodes['low'], 1, 0.025),
    (nodes['median'], -1, 0.2),
    (nodes['high'], 1, 0.8),
    (nodes['high'], -1, 0.8),
]

W_jax = jnp.asarray(W)
replica_state_jax = jnp.asarray(replica_state)

max_diffs = {}
all_pass = True

for node, sign, amp in test_cases:
    print(f"\n{'='*60}\nnode={node}, sign={sign}, amp={amp}\n{'='*60}")
    ref = run_one_trial(W, replica_state, node, sign, amp)
    jaxr = run_one_trial_jax_faithful(W_jax, replica_state_jax, node, float(sign), float(amp))
    jaxr = {k: np.asarray(v) for k, v in jaxr.items()}

    print(f"tau_star: ref={ref['tau_star']:.6f}  jax={float(jaxr['tau_star']):.6f}")
    tau_match = abs(ref['tau_star'] - float(jaxr['tau_star'])) < 1e-9
    print(f"event_aligned_valid: ref={ref['event_aligned_valid']}  jax={bool(jaxr['event_aligned_valid'])}")

    fields_to_compare = ['event_aligned_q', 'event_aligned_r', 'event_aligned_q_tangent',
                          'event_aligned_q_residual', 'fixed_time_q', 'fixed_time_r',
                          'fixed_time_q_tangent', 'fixed_time_q_residual']
    for field in fields_to_compare:
        ref_val = ref[field]
        jax_val = jaxr[field]
        if ref_val is None:
            continue
        diff = np.max(np.abs(np.asarray(ref_val) - jax_val))
        max_diffs[field] = max(max_diffs.get(field, 0), diff)
        if diff > 1e-4:
            all_pass = False
        print(f"  {field}: max abs diff = {diff:.3e}")

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for field, diff in max_diffs.items():
    print(f"  {field}: {diff:.3e}")

print()
if all_pass:
    print("PASS: GPU JAX port matches numpy reference to expected cross-solver precision.")
    sys.exit(0)
else:
    print("FAIL: a field exceeded the 1e-4 tolerance -- investigate before trusting this port on GPU.")
    sys.exit(1)
