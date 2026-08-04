"""
Verifies analyze_stage3_results_jax.py's batched R_post/feat_post
computation against the numpy reference in analyze_stage3_results.py --
referenced by name from JAX_CLASSIFIER_PORT_FINDINGS.md's "Finding 1"
section, but that verification only ever happened ad hoc, as a side
effect of building a real-data test, in an interactive session. No
standalone, re-runnable script encoding it was ever committed -- the
same category of reproducibility gap already found and closed twice for
stage2a_classifier_jax.py. Closes it here, for this file, the same way.

Synthetic data only -- never touches the real Stage-3 cached artifacts.

Two checks, matching exactly what JAX_CLASSIFIER_PORT_FINDINGS.md
already documents as having been checked (not a new or expanded check):

  1. order_parameter/reference_node_features vs. the stage2a_core numpy
     reference, via analyze_stage3_results_jax's own
     _compute_rpost_featpost_jax, on synthetic phase states.
  2. Full build_results_structure output (including solver-failure
     handling across all four topologies) compared field-by-field
     between analyze_stage3_results.py and analyze_stage3_results_jax.py
     on synthetic mock local_data/gpu_data, with solver failures
     injected.
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import analyze_stage3_results as ref
import analyze_stage3_results_jax as jaxver

N_NODES = 505


def check_1_rpost_featpost(n_trials=20):
    """_compute_rpost_featpost_jax vs. stage2a_core's order_parameter/
    reference_node_features, on synthetic phase states -- 20 random
    trials plus explicit ref_idx in {0, n-1} edge cases, matching
    JAX_CLASSIFIER_PORT_FINDINGS.md's documented methodology.

    Not a single-image function -- _compute_rpost_featpost_jax operates
    on a batch of shape (n_topologies, n_images, n_nodes) via
    vmap(vmap(...)). Uses a batch shaped (1, n_trials, N_NODES) so each
    trial gets a fresh random state, then compares every entry against
    the numpy per-image reference directly."""
    print(f"\n{'='*70}\nCHECK 1: order_parameter/reference_node_features vs. numpy "
          f"reference\n{'='*70}")
    rng = np.random.default_rng(0)

    max_diff_R = 0.0
    max_diff_feat = 0.0
    for ref_idx in [rng.integers(0, N_NODES) for _ in range(n_trials)] + [0, N_NODES - 1]:
        theta = rng.uniform(0, 2 * np.pi, size=N_NODES)
        theta_batch = theta[None, None, :]  # (1 topology, 1 image, n_nodes)

        R_jax, feat_jax = jaxver._compute_rpost_featpost_jax(theta_batch, ref_idx)
        R_ref = s2a.order_parameter(theta)
        feat_ref = s2a.reference_node_features(theta, ref_idx)

        max_diff_R = max(max_diff_R, abs(float(R_jax[0, 0]) - R_ref))
        max_diff_feat = max(max_diff_feat, float(np.max(np.abs(feat_jax[0, 0] - feat_ref))))

    print(f"max |R_ref - R_jax| over {n_trials} trials + 2 edge cases: {max_diff_R:.3e}")
    print(f"max |feat_ref - feat_jax|: {max_diff_feat:.3e}")
    assert max_diff_R < 1e-10, f"order_parameter diverges beyond tolerance: {max_diff_R:.3e}"
    assert max_diff_feat < 1e-10, f"reference_node_features diverges beyond tolerance: {max_diff_feat:.3e}"
    print("PASS: matches numpy reference to float64 precision.")


def _make_synthetic_data(n_images, n_nodes, ref_idx, n_failed_per_topology, seed=0):
    """Synthetic local_data/gpu_data dicts shaped exactly like the real
    pickles build_results_structure consumes, with solver failures
    injected for a subset of (topology, image) pairs -- exercises the
    solver_failed branch in both the numpy and JAX build_results_structure
    implementations, not just the happy path."""
    rng = np.random.default_rng(seed)
    local_data = {
        "n_images": n_images,
        "raw_feat": rng.uniform(size=(n_images, 784)),
        "feat_pre": rng.uniform(size=(n_images, 2 * n_nodes - 2)),
        "R_pre": rng.uniform(size=n_images),
    }
    gpu_data = {"results": {}}
    for name in ref.TOPOLOGY_NAMES:
        theta_T = rng.uniform(0, 2 * np.pi, size=(n_images, n_nodes))
        success = np.ones(n_images, dtype=bool)
        if n_failed_per_topology:
            fail_idx = rng.choice(n_images, size=n_failed_per_topology, replace=False)
            success[fail_idx] = False
            theta_T[fail_idx] = np.nan  # matches real GPU output for a failed solve
        gpu_data["results"][name] = {"theta_T": theta_T, "success": success}
    return local_data, gpu_data, ref_idx


def check_2_build_results_structure():
    """Full build_results_structure comparison, numpy vs. JAX, on
    synthetic mock data with injected solver failures across all four
    topologies. np.asarray() both sides before comparing feat_post --
    it's a JAX array on the JAX path, a Python list on the numpy
    fallback path (the isinstance(feat_post, list) branch)."""
    print(f"\n{'='*70}\nCHECK 2: full build_results_structure, numpy vs. JAX, synthetic "
          f"data with injected solver failures\n{'='*70}")
    n_images, n_nodes = 37, 25
    local_data, gpu_data, ref_idx = _make_synthetic_data(
        n_images, n_nodes, ref_idx=12, n_failed_per_topology=5, seed=1)

    results_ref = ref.build_results_structure(local_data, gpu_data, ref_idx)
    results_jax = jaxver.build_results_structure(local_data, gpu_data, ref_idx)

    assert len(results_ref) == len(results_jax) == n_images

    max_diff = 0.0
    mismatches = []
    for i in range(n_images):
        rr, rj = results_ref[i], results_jax[i]
        if rr["idx"] != rj["idx"]:
            mismatches.append((i, "idx"))
        if abs(rr["R_pre"] - rj["R_pre"]) > 1e-12:
            mismatches.append((i, "R_pre"))
        if not np.allclose(rr["feat_pre"], rj["feat_pre"]):
            mismatches.append((i, "feat_pre"))
        if not np.allclose(rr["raw_feat"], rj["raw_feat"]):
            mismatches.append((i, "raw_feat"))
        for name in ref.TOPOLOGY_NAMES:
            er, ej = rr["evolved"][name], rj["evolved"][name]
            if er["solver_failed"] != ej["solver_failed"]:
                mismatches.append((i, name, "solver_failed"))
                continue
            if er["solver_failed"]:
                if ej["R_post"] is not None or ej["feat_post"] is not None:
                    mismatches.append((i, name, "expected None on failure"))
                continue
            d_r = abs(er["R_post"] - ej["R_post"])
            d_f = np.max(np.abs(np.asarray(er["feat_post"]) - np.asarray(ej["feat_post"])))
            max_diff = max(max_diff, d_r, d_f)

    print(f"mismatches: {mismatches}")
    print(f"max numeric diff across all R_post/feat_post: {max_diff:.3e}")
    assert not mismatches, f"build_results_structure disagreement: {mismatches}"
    assert max_diff < 1e-10, f"R_post/feat_post numeric diff beyond tolerance: {max_diff:.3e}"
    print("PASS: full build_results_structure output matches between numpy and JAX versions.")


def main():
    check_1_rpost_featpost()
    check_2_build_results_structure()


if __name__ == "__main__":
    main()
