"""
Verifies evolve_on_graph_jax against the real numpy evolve_on_graph
(stage2a_core.py) on real encoded states, before trusting the JAX port
for anything -- same standard as every prior GPU port in this project
(Stage 1D's run_one_trial_jax_faithful.py, verify_on_gpu.py): field-by-
field comparison, not a rewrite trusted on inspection alone.

Tests a handful of real images' encoded states (not synthetic random
phases) evolved on all four of Stage 2A's canonical topologies.
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_core as s2a
import stage2a_pipeline as pipe
import stage2a_topologies as topo
from evolve_on_graph_jax import evolve_on_graph_jax, batched_evolve_on_graph_jax

import jax.numpy as jnp

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
N_TEST_IMAGES = 6


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=42, n_per_class=1)  # 10 images, 1/class
    images_01 = images_01[:N_TEST_IMAGES]

    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()

    print(f"Encoding {N_TEST_IMAGES} real images...")
    theta0_list = [s2a.encode_and_restrict(img, active_indices) for img in images_01]

    print("\n" + "=" * 70)
    print("SINGLE-TRIAL VERIFICATION (unbatched, one image x one topology at a time)")
    print("=" * 70)
    max_diffs = []
    for topo_name, W in topologies.items():
        for i, theta0 in enumerate(theta0_list):
            theta_T_numpy, diag = s2a.evolve_on_graph(theta0, W)
            assert not diag.get("failed", False), f"numpy solve failed for {topo_name}, image {i}"

            theta_T_jax_raw, success_jax = evolve_on_graph_jax(jnp.asarray(theta0), jnp.asarray(W))
            assert bool(success_jax), f"JAX solve reported failure for {topo_name}, image {i}"
            theta_T_jax = np.asarray(theta_T_jax_raw)

            # Compare on the circle (mod 2pi), not raw difference, since both are wrapped
            # to [0, 2pi) already but floating-point wrap-around near 0/2pi could otherwise
            # register a spurious large diff.
            raw_diff = theta_T_numpy - theta_T_jax
            circular_diff = np.angle(np.exp(1j * raw_diff))
            max_diff = np.max(np.abs(circular_diff))
            max_diffs.append(max_diff)
            status = "OK" if max_diff < 1e-4 else "FAIL"
            print(f"  [{topo_name}, image {i}] max abs circular diff = {max_diff:.3e}  [{status}]")

    print(f"\nOverall max diff across all {len(max_diffs)} (topology, image) pairs: "
          f"{max(max_diffs):.3e}")
    assert max(max_diffs) < 1e-4, "JAX port disagrees with numpy reference beyond tolerance"
    print("PASS: JAX single-trial evolution matches numpy reference.")

    print("\n" + "=" * 70)
    print("BATCHED (VMAP) VERIFICATION -- same images, same topologies, via batched_evolve_on_graph_jax")
    print("=" * 70)
    theta0_batch = jnp.asarray(np.stack(theta0_list))
    batch_max_diffs = []
    for topo_name, W in topologies.items():
        theta_T_batch_jax_raw, success_batch_jax = batched_evolve_on_graph_jax(theta0_batch, jnp.asarray(W))
        assert bool(np.all(success_batch_jax)), f"batched JAX solve reported a failure for {topo_name}"
        theta_T_batch_jax = np.asarray(theta_T_batch_jax_raw)
        for i, theta0 in enumerate(theta0_list):
            theta_T_numpy, diag = s2a.evolve_on_graph(theta0, W)
            raw_diff = theta_T_numpy - theta_T_batch_jax[i]
            circular_diff = np.angle(np.exp(1j * raw_diff))
            max_diff = np.max(np.abs(circular_diff))
            batch_max_diffs.append(max_diff)
        print(f"  [{topo_name}] max abs circular diff across {len(theta0_list)} images: "
              f"{max(batch_max_diffs[-len(theta0_list):]):.3e}")

    print(f"\nOverall max diff (batched): {max(batch_max_diffs):.3e}")
    assert max(batch_max_diffs) < 1e-4, "Batched JAX port disagrees with numpy reference beyond tolerance"
    print("PASS: batched (vmap) JAX evolution matches numpy reference, and matches the "
          "unbatched JAX result (both compared independently against numpy above).")

    return max(max_diffs), max(batch_max_diffs)


if __name__ == "__main__":
    main()
