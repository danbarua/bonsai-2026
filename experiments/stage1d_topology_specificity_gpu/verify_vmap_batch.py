import pickle
import numpy as np
import jax
import jax.numpy as jnp

from stage1b2_core import (
    run_one_trial, generate_reference_baseline, generate_fixed_replica_directions,
    get_degree_stratified_nodes, NEARBY_SCALE,
)
from run_one_trial_jax_faithful import run_one_trial_jax_faithful

with open("class0_constructions.pkl", "rb") as f:
    data = pickle.load(f)[0]
W = data["constructions"]["T"]
nodes = get_degree_stratified_nodes(W)

BASELINE_SEED = 3000
ref_sol = generate_reference_baseline(W, BASELINE_SEED, 2.5)
directions = generate_fixed_replica_directions(W.shape[0], 3001, 6)

# Build a small, real batch: 3 t_p values x 2 replicas x node/sign/amp combos
batch = []
for t_p in [0.0, 1.667, 2.5]:
    state = ref_sol.sol(t_p)
    for rep_idx in [0, 3]:
        replica_state = np.mod(state + NEARBY_SCALE * directions[rep_idx], 2 * np.pi)
        for node in [nodes['low'], nodes['high']]:
            for sign, amp in [(1, 0.025), (-1, 0.8)]:
                batch.append((replica_state, node, sign, amp))

print(f"{len(batch)} trials in batch")

# Reference: loop, call run_one_trial individually
ref_results = [run_one_trial(W, rs, node, sign, amp) for rs, node, sign, amp in batch]

# JAX: vmap over the whole batch at once
theta0_batch = jnp.asarray(np.stack([b[0] for b in batch]))
node_batch = jnp.asarray([b[1] for b in batch])
sign_batch = jnp.asarray([float(b[2]) for b in batch])
amp_batch = jnp.asarray([b[3] for b in batch])
W_jax = jnp.asarray(W)

batched_fn = jax.jit(jax.vmap(run_one_trial_jax_faithful, in_axes=(None, 0, 0, 0, 0)))
jax_results = batched_fn(W_jax, theta0_batch, node_batch, sign_batch, amp_batch)

print("\nComparing vmapped-batch JAX output against the single-trial reference loop:")
max_diffs = {}
for i in range(len(batch)):
    ref = ref_results[i]
    for field in ['event_aligned_q', 'fixed_time_q']:
        ref_val = np.asarray(ref[field])
        jax_val = np.asarray(jax_results[field][i])
        diff = np.max(np.abs(ref_val - jax_val))
        max_diffs[field] = max(max_diffs.get(field, 0), diff)
    tau_star_diff = abs(ref['tau_star'] - float(jax_results['tau_star'][i]))
    max_diffs['tau_star'] = max(max_diffs.get('tau_star', 0), tau_star_diff)
    valid_match = ref['event_aligned_valid'] == bool(jax_results['event_aligned_valid'][i])
    if not valid_match:
        print(f"  MISMATCH at trial {i}: event_aligned_valid ref={ref['event_aligned_valid']} "
              f"jax={bool(jax_results['event_aligned_valid'][i])}")

print("\nMax abs diff across all", len(batch), "batched trials, vs single-trial-loop reference:")
for field, diff in max_diffs.items():
    print(f"  {field}: {diff:.3e}")

# Also confirm the vmapped batch matches calling run_one_trial_jax_faithful individually
# (not jitted/vmapped) -- isolates whether vmap itself changes anything vs. just JAX vs numpy
print("\nCross-check: vmapped-batch result vs. individually-called (non-vmapped) JAX result:")
individual_jax = [run_one_trial_jax_faithful(W_jax, jnp.asarray(rs), node, float(sign), amp)
                   for rs, node, sign, amp in batch[:3]]
for i in range(3):
    diff = np.max(np.abs(np.asarray(individual_jax[i]['event_aligned_q']) -
                          np.asarray(jax_results['event_aligned_q'][i])))
    print(f"  trial {i}: max abs diff (vmap vs individual JAX call) = {diff:.3e}")
