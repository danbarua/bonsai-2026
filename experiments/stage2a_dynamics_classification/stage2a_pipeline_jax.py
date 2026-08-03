"""
End-to-end JAX pipeline for Stage 2A, matching
stage2a_pipeline.run_pipeline_multi_topology's exact result-dict
contract (same keys, same per-image/per-topology structure, same idx
ordering) -- built so its full output can be diffed field-by-field
against the numpy pipeline, not just the underlying evolution kernel in
isolation. Per this project's own established lesson (Stage 1D's GPU
episode): a verified kernel can still feed a wrong result if the
calling/batching code around it is subtly different from the real
pipeline -- see verify_stage2a_pipeline_equivalence.py for the actual
check.

Encoding and gauge-feature computation reuse stage2a_core's numpy
functions UNCHANGED (not reimplemented) -- only the graph-evolution ODE
solve itself is replaced with the batched JAX/diffrax kernel.
"""
import numpy as np
import jax.numpy as jnp

import stage2a_core as s2a
from evolve_on_graph_jax import batched_evolve_on_graph_jax


def run_pipeline_multi_topology_jax(images_01, labels, topologies, ref_idx, active_indices,
                                     encoder_seeds=None):
    """JAX-evolution analog of stage2a_pipeline.run_pipeline_multi_topology.
    Same signature, same return shape (list of per-image dicts, sorted by
    idx, each with 'idx', 'R_pre', 'feat_pre', 'raw_feat', 'evolved'
    (dict: topology name -> {'solver_failed', 'R_post', 'feat_post'})."""
    n_images = len(images_01)
    if encoder_seeds is None:
        encoder_seeds = [s2a.ENCODER_SEED] * n_images

    # Encoding: identical numpy call to the real pipeline, not reimplemented.
    theta0_list = [s2a.encode_and_restrict(images_01[i], active_indices, seed=encoder_seeds[i])
                   for i in range(n_images)]
    R_pre_list = [s2a.order_parameter(theta0) for theta0 in theta0_list]
    feat_pre_list = [s2a.reference_node_features(theta0, ref_idx) for theta0 in theta0_list]
    raw_feat_list = [images_01[i].flatten() for i in range(n_images)]

    theta0_batch = jnp.asarray(np.stack(theta0_list))

    evolved_per_topology = {}
    for name, W in topologies.items():
        theta_T_batch, success_batch = batched_evolve_on_graph_jax(theta0_batch, jnp.asarray(W))
        theta_T_batch_np = np.asarray(theta_T_batch)
        success_batch_np = np.asarray(success_batch)

        per_image = []
        for i in range(n_images):
            if not bool(success_batch_np[i]):
                per_image.append({"solver_failed": True, "solver_diag": {"jax_solve_failed": True},
                                   "R_post": None, "feat_post": None})
            else:
                theta_T = theta_T_batch_np[i]
                # Gauge features: the SAME numpy function as the real pipeline, applied
                # to the JAX-evolved state -- not a second, parallel implementation.
                per_image.append({
                    "solver_failed": False, "solver_diag": {"jax_solve_failed": False},
                    "R_post": s2a.order_parameter(theta_T),
                    "feat_post": s2a.reference_node_features(theta_T, ref_idx),
                })
        evolved_per_topology[name] = per_image

    results = []
    for i in range(n_images):
        results.append({
            "idx": i, "R_pre": R_pre_list[i], "feat_pre": feat_pre_list[i],
            "raw_feat": raw_feat_list[i],
            "evolved": {name: evolved_per_topology[name][i] for name in topologies},
        })
    return results
