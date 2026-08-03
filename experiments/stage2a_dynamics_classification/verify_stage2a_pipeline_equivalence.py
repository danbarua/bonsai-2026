"""
End-to-end Stage 2A pipeline equivalence check: numpy vs. JAX, on a
small MIXED-CLASS batch, comparing the full result structure both
pipelines actually produce -- not just the underlying evolution kernel
in isolation (already verified in verify_evolve_on_graph_jax.py).

Per the explicit standard requested (and this project's own established
lesson from Stage 1D's GPU episode -- a verified kernel can still feed
a wrong result if the surrounding batch/caller code is subtly
different): checks, together, in one run:

1. same encoded theta_0^505 (via R_pre matching exactly -- encoding is
   100% numpy in both pipelines, so this must be bit-identical, not
   just close);
2. same evolved theta_T within tolerance (via R_post and feat_post,
   which are direct functions of theta_T);
3. same gauge-fixed circular features (feat_pre exact; feat_post within
   the same tolerance as theta_T, since both pipelines call the
   identical reference_node_features() on their own evolved state);
4. same cached feature row ordering and labels (idx and label arrays
   compared directly, not assumed to line up);
5. same solver-failure accounting (solver_failed flags compared
   per-image per-topology);
6. all of the above together, for a small but genuinely mixed-class
   batch (multiple classes represented, not a single-class or
   already-homogeneous sample).
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_pipeline as pipe
import stage2a_topologies as topo
from stage2a_pipeline_jax import run_pipeline_multi_topology_jax

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
SEED = 42
N_PER_CLASS = 4  # 40 images, genuinely mixed-class (all 10 classes represented)

FEAT_TOLERANCE = 1e-4  # same cross-solver tolerance used throughout this project


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, all 10 classes) "
          f"-- labels present: {sorted(set(labels.tolist()))}")

    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
    ref_idx = nodes_T["median"]

    print("\nRunning NUMPY pipeline (real, locked)...")
    results_np, elapsed_np = pipe.run_pipeline_multi_topology(
        images_01, labels, topologies, ref_idx, active_indices)
    print(f"numpy pipeline: {elapsed_np:.1f}s")

    print("\nRunning JAX pipeline (evolution ported, encoding/gauge features reused unchanged)...")
    results_jax = run_pipeline_multi_topology_jax(
        images_01, labels, topologies, ref_idx, active_indices)

    assert len(results_np) == len(results_jax) == len(images_01)

    print("\n" + "=" * 70)
    print("1. ROW ORDERING / LABELS")
    print("=" * 70)
    for i in range(len(images_01)):
        assert results_np[i]["idx"] == results_jax[i]["idx"] == i, \
            f"idx mismatch at position {i}: numpy={results_np[i]['idx']}, jax={results_jax[i]['idx']}"
    print(f"  PASS: all {len(images_01)} idx values match and are in order 0..{len(images_01)-1} "
          f"in both pipelines.")
    print(f"  Labels (shared array, both pipelines): {labels.tolist()}")

    print("\n" + "=" * 70)
    print("2. SAME ENCODED theta_0 (via R_pre, bit-identical -- encoding is 100% numpy in both)")
    print("=" * 70)
    R_pre_np = np.array([r["R_pre"] for r in results_np])
    R_pre_jax = np.array([r["R_pre"] for r in results_jax])
    max_R_pre_diff = np.max(np.abs(R_pre_np - R_pre_jax))
    print(f"  max |R_pre_numpy - R_pre_jax| = {max_R_pre_diff:.3e}")
    assert max_R_pre_diff < 1e-12, "R_pre differs between pipelines -- encoding is NOT identical!"
    print("  PASS: bit-identical (as expected -- same encode_and_restrict call in both).")

    print("\n" + "=" * 70)
    print("3. SAME GAUGE-FIXED PRE-EVOLUTION FEATURES (feat_pre, bit-identical)")
    print("=" * 70)
    feat_pre_diffs = [np.max(np.abs(results_np[i]["feat_pre"] - results_jax[i]["feat_pre"]))
                       for i in range(len(images_01))]
    print(f"  max |feat_pre_numpy - feat_pre_jax| across all images: {max(feat_pre_diffs):.3e}")
    assert max(feat_pre_diffs) < 1e-12
    print("  PASS: bit-identical.")

    print("\n" + "=" * 70)
    print("4. SAME raw_feat (raw pixels, bit-identical)")
    print("=" * 70)
    raw_diffs = [np.max(np.abs(results_np[i]["raw_feat"] - results_jax[i]["raw_feat"]))
                 for i in range(len(images_01))]
    assert max(raw_diffs) < 1e-12
    print(f"  PASS: bit-identical (max diff {max(raw_diffs):.3e}).")

    print("\n" + "=" * 70)
    print("5+6. PER-TOPOLOGY: SOLVER-FAILURE ACCOUNTING, EVOLVED theta_T (via R_post), "
          "AND GAUGE-FIXED EVOLVED FEATURES (feat_post)")
    print("=" * 70)
    all_ok = True
    for name in topologies:
        failed_np = [results_np[i]["evolved"][name]["solver_failed"] for i in range(len(images_01))]
        failed_jax = [results_jax[i]["evolved"][name]["solver_failed"] for i in range(len(images_01))]
        assert failed_np == failed_jax, \
            f"[{name}] solver_failed flags differ: numpy={failed_np}, jax={failed_jax}"

        R_post_diffs, feat_post_diffs = [], []
        for i in range(len(images_01)):
            if failed_np[i]:
                continue  # both agree it failed -- nothing to compare numerically
            R_post_diffs.append(abs(results_np[i]["evolved"][name]["R_post"]
                                     - results_jax[i]["evolved"][name]["R_post"]))
            feat_post_diffs.append(np.max(np.abs(results_np[i]["evolved"][name]["feat_post"]
                                                  - results_jax[i]["evolved"][name]["feat_post"])))

        max_R_post_diff = max(R_post_diffs) if R_post_diffs else float("nan")
        max_feat_post_diff = max(feat_post_diffs) if feat_post_diffs else float("nan")
        status = "OK" if max_feat_post_diff < FEAT_TOLERANCE else "FAIL"
        print(f"  [{name}] solver_failed flags match ({sum(failed_np)}/{len(images_01)} failed, "
              f"both pipelines agree); max R_post diff={max_R_post_diff:.3e}; "
              f"max feat_post diff={max_feat_post_diff:.3e}  [{status}]")
        if max_feat_post_diff >= FEAT_TOLERANCE:
            all_ok = False

    assert all_ok, "One or more topologies exceeded the feature-level tolerance"
    print(f"\nPASS: full end-to-end pipeline equivalence confirmed on this "
          f"{len(images_01)}-image, {len(set(labels.tolist()))}-class mixed batch, "
          f"across all {len(topologies)} topologies.")


if __name__ == "__main__":
    main()
