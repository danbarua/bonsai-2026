"""Analyze split Stage-3 results with batched JAX feature extraction.

The encoder and classifier remain the established NumPy pipeline components.
The GPU-produced ``theta_T`` arrays are post-processed in one JAX batch with
shape ``(topology, image, node)``.  This avoids Python loops over images and
keeps the result-list contract consumed by :mod:`stage2a_pipeline` unchanged.
"""
import os
import pickle
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_pipeline as pipe
import stage2a_core as s2a
from stage2a_paths import train_scratch_dir

SCRATCH_DIR = train_scratch_dir()
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]


def _compute_rpost_featpost_jax(theta_batches, ref_idx):
    """Compute post-evolution diagnostics for all topologies and images.

    Parameters
    ----------
    theta_batches : array, shape (n_topologies, n_images, n_nodes)
    ref_idx : int

    Returns
    -------
    tuple of arrays: ``R`` with shape ``(n_topologies, n_images)`` and
    features with shape ``(n_topologies, n_images, 2*n_nodes-2)``.
    """
    import jax
    import jax.numpy as jnp

    # Enable double precision once, matching NumPy's source calculations.
    jax.config.update("jax_enable_x64", True)
    n_nodes = theta_batches.shape[-1]
    keep = np.concatenate((np.arange(ref_idx), np.arange(ref_idx + 1, n_nodes)))

    def one_image(theta):
        R = jnp.abs(jnp.mean(jnp.exp(1j * theta)))
        shifted = theta - theta[ref_idx]
        return R, jnp.concatenate((jnp.cos(shifted)[keep], jnp.sin(shifted)[keep]))

    # vmap(vmap(...)) performs the topology/image batch without Python work.
    batched = jax.jit(jax.vmap(jax.vmap(one_image)))
    R, features = batched(jnp.asarray(theta_batches, dtype=jnp.float64))
    return np.asarray(R), np.asarray(features)


def _numpy_post(theta_batches, success_batches, ref_idx):
    """Reference implementation used when JAX is not installed."""
    n_topologies, n_images = success_batches.shape
    R = np.full((n_topologies, n_images), np.nan, dtype=np.float64)
    features = [
        [None] * n_images for _ in range(n_topologies)
    ]
    for t in range(n_topologies):
        for i in range(n_images):
            if success_batches[t, i]:
                theta = theta_batches[t, i]
                R[t, i] = s2a.order_parameter(theta)
                features[t][i] = s2a.reference_node_features(theta, ref_idx)
    return R, features


def build_results_structure(local_data, gpu_data, ref_idx):
    """Reconstruct the exact multi-topology results-list contract."""
    n_images = int(local_data["n_images"])
    raw_feat = local_data["raw_feat"]
    feat_pre = local_data["feat_pre"]
    R_pre = local_data["R_pre"]

    theta_batches = np.stack([
        np.asarray(gpu_data["results"][name]["theta_T"])
        for name in TOPOLOGY_NAMES
    ])
    success_batches = np.stack([
        np.asarray(gpu_data["results"][name]["success"], dtype=bool)
        for name in TOPOLOGY_NAMES
    ])

    t0 = time.time()
    try:
        import jax  # noqa: F401
        R_post, feat_post = _compute_rpost_featpost_jax(theta_batches, ref_idx)
        print("JAX available -- used one batched topology/image computation.")
    except ImportError:
        R_post, feat_post = _numpy_post(theta_batches, success_batches, ref_idx)
        print("JAX not available -- used NumPy fallback.")
    print(f"R_post/feat_post computation: {time.time() - t0:.1f}s")

    results = []
    for i in range(n_images):
        evolved = {}
        for t, name in enumerate(TOPOLOGY_NAMES):
            if not success_batches[t, i]:
                evolved[name] = {
                    "solver_failed": True,
                    "solver_diag": {"jax_solve_failed": True},
                    "R_post": None,
                    "feat_post": None,
                }
            else:
                evolved[name] = {
                    "solver_failed": False,
                    "solver_diag": {"jax_solve_failed": False},
                    "R_post": float(R_post[t, i]),
                    "feat_post": feat_post[t][i] if isinstance(feat_post, list)
                    else feat_post[t, i],
                }
        results.append({
            "idx": i,
            "R_pre": float(R_pre[i]),
            "feat_pre": feat_pre[i],
            "raw_feat": raw_feat[i],
            "evolved": evolved,
        })
    return results


def main():
    with open(os.path.join(SCRATCH_DIR, "stage3_encode_local.pkl"), "rb") as f:
        local_data = pickle.load(f)
    labels = local_data["labels"]
    ref_idx = local_data["ref_idx"]

    with open(os.path.join(SCRATCH_DIR, "stage3_gpu_results.pkl"), "rb") as f:
        gpu_data = pickle.load(f)
    results = build_results_structure(local_data, gpu_data, ref_idx)

    print("\n" + "=" * 70)
    print("GO/NO-GO MECHANICAL CHECKS (60,000-image, 4-topology scale)")
    print("=" * 70)
    go_no_go = pipe.check_go_no_go_multi_topology(results, TOPOLOGY_NAMES)
    print(f"n_images: {go_no_go['n_images']}")
    print(f"n_non_finite_shared_feature_vectors: {go_no_go['n_non_finite_shared_feature_vectors']}")
    print(f"non_finite_ok (overall): {go_no_go['non_finite_ok']}")
    print(f"solver_failure_rate_ok (overall): {go_no_go['solver_failure_rate_ok']}")
    for name in TOPOLOGY_NAMES:
        pt = go_no_go["per_topology"][name]
        print(f"[{name}] failed={pt['n_solver_failed']}, "
              f"failure_rate={pt['solver_failure_rate']:.6f}, "
              f"non_finite={pt['n_non_finite_feature_vectors']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "stage3_go_no_go.pkl"), "wb") as f:
        pickle.dump(go_no_go, f)

    t0 = time.time()
    conditions = pipe.run_classifier_conditions_multi_topology(
        results, labels, TOPOLOGY_NAMES, label_prefix="stage3_seed0_")
    elapsed = time.time() - t0
    with open(os.path.join(RESULTS_DIR, "stage3_classifier_conditions.pkl"), "wb") as f:
        pickle.dump({"conditions": conditions,
                     "classifier_elapsed_seconds": elapsed,
                     "encode_elapsed_seconds": local_data["encode_elapsed_seconds"],
                     "gpu_evolve_total_elapsed_seconds": gpu_data["total_elapsed"],
                     "gpu_evolve_per_topology_elapsed": {
                         n: gpu_data["results"][n]["elapsed"] for n in TOPOLOGY_NAMES}}, f)

    all_converged = all(c.get("converged", False) for c in conditions.values())
    all_go = go_no_go["solver_failure_rate_ok"] and go_no_go["non_finite_ok"] and all_converged
    print(f"\nOVERALL: {'GO' if all_go else 'NO-GO'}")
    return go_no_go, conditions


if __name__ == "__main__":
    main()
