"""
Stage 2A feasibility stage 3, phase 2 (local, CPU): combines the local
encode results (raw_feat/feat_pre/R_pre/labels, from
run_feasibility_stage3_encode.py) with the GPU-evolved theta_T/success
arrays (from stage3_gpu_evolve.py, run on a mighty-colab GPU session)
into the SAME results-list contract stage2a_pipeline.py's other
multi-topology functions already expect -- so check_go_no_go_multi_topology
and run_classifier_conditions_multi_topology run completely unchanged,
not reimplemented for this stage.

R_post/feat_post are computed here via stage2a_core's own
order_parameter/reference_node_features (the identical numpy functions
used everywhere else in this pipeline, including the JAX-pipeline path
in stage2a_pipeline_jax.py) -- applied to the GPU-evolved theta_T.
"""
import os
import pickle
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import stage2a_pipeline as pipe

SCRATCH_DIR = "/private/tmp/claude-501/-Users-dan-Code-pycharm-bonsai-2026/54a406a1-f8d0-41df-bc2a-d46e08e68715/scratchpad/stage2a_gpu_stage3"
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]


def build_results_structure(local_data, gpu_data, ref_idx):
    """Reconstructs the exact list-of-dicts contract
    (idx/R_pre/feat_pre/raw_feat/evolved{name: solver_failed/R_post/feat_post})
    that run_pipeline_multi_topology / run_pipeline_multi_topology_jax
    produce, from the split local-encode + GPU-evolve artifacts."""
    n_images = local_data["n_images"]
    raw_feat = local_data["raw_feat"]
    feat_pre = local_data["feat_pre"]
    R_pre = local_data["R_pre"]

    print("Computing R_post/feat_post from GPU-evolved theta_T "
          "(via stage2a_core's own numpy functions, reused unchanged)...")
    t0 = time.time()
    evolved_per_topology = {}
    for name in TOPOLOGY_NAMES:
        theta_T = gpu_data["results"][name]["theta_T"]
        success = gpu_data["results"][name]["success"]
        per_image = []
        for i in range(n_images):
            if not bool(success[i]):
                per_image.append({"solver_failed": True, "solver_diag": {"jax_solve_failed": True},
                                   "R_post": None, "feat_post": None})
            else:
                th = theta_T[i]
                per_image.append({
                    "solver_failed": False, "solver_diag": {"jax_solve_failed": False},
                    "R_post": s2a.order_parameter(th),
                    "feat_post": s2a.reference_node_features(th, ref_idx),
                })
        evolved_per_topology[name] = per_image
        n_failed = sum(1 for r in per_image if r["solver_failed"])
        print(f"  [{name}] R_post/feat_post computed for {n_images} images "
              f"({n_failed} solver failures)")
    print(f"R_post/feat_post computation: {time.time()-t0:.1f}s")

    results = []
    for i in range(n_images):
        results.append({
            "idx": i, "R_pre": float(R_pre[i]), "feat_pre": feat_pre[i],
            "raw_feat": raw_feat[i],
            "evolved": {name: evolved_per_topology[name][i] for name in TOPOLOGY_NAMES},
        })
    return results


def main():
    print("Loading local encode results...")
    with open(os.path.join(SCRATCH_DIR, "stage3_encode_local.pkl"), "rb") as f:
        local_data = pickle.load(f)
    labels = local_data["labels"]
    ref_idx = local_data["ref_idx"]
    n_images = local_data["n_images"]
    print(f"n_images={n_images}, ref_idx={ref_idx}, "
          f"encode_elapsed={local_data['encode_elapsed_seconds']:.1f}s")

    print("\nLoading GPU evolve results...")
    with open(os.path.join(SCRATCH_DIR, "stage3_gpu_results.pkl"), "rb") as f:
        gpu_data = pickle.load(f)
    print(f"GPU total_elapsed={gpu_data['total_elapsed']:.1f}s, "
          f"chunk_size={gpu_data['chunk_size']}")
    for name in TOPOLOGY_NAMES:
        print(f"  [{name}] GPU elapsed={gpu_data['results'][name]['elapsed']:.1f}s, "
              f"n_failed={int(np.sum(~gpu_data['results'][name]['success']))}")

    results = build_results_structure(local_data, gpu_data, ref_idx)

    print("\n" + "=" * 70)
    print("GO/NO-GO MECHANICAL CHECKS (60,000-image, 4-topology scale)")
    print("=" * 70)
    go_no_go = pipe.check_go_no_go_multi_topology(results, TOPOLOGY_NAMES)
    print(f"n_images: {go_no_go['n_images']}")
    print(f"n_non_finite_shared_feature_vectors: {go_no_go['n_non_finite_shared_feature_vectors']}")
    print(f"non_finite_ok (overall): {go_no_go['non_finite_ok']}")
    print(f"solver_failure_rate_ok (overall): {go_no_go['solver_failure_rate_ok']}")
    print(f"\nR_pre distribution (shared, all images):")
    for k, v in go_no_go["R_pre_summary"].items():
        print(f"  {k}: {v}")
    print(f"\nPer-topology:")
    for name in TOPOLOGY_NAMES:
        pt = go_no_go["per_topology"][name]
        print(f"\n  [{name}]")
        print(f"    n_solver_failed: {pt['n_solver_failed']}, "
              f"solver_failure_rate: {pt['solver_failure_rate']:.6f}, ok={pt['solver_failure_rate_ok']}")
        print(f"    n_non_finite_feature_vectors: {pt['n_non_finite_feature_vectors']}, ok={pt['non_finite_ok']}")
        if pt["R_post_summary"]:
            print(f"    R_post distribution:")
            for k, v in pt["R_post_summary"].items():
                print(f"      {k}: {v}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    go_no_go_path = os.path.join(RESULTS_DIR, "stage3_go_no_go.pkl")
    with open(go_no_go_path, "wb") as f:
        pickle.dump(go_no_go, f)
    print(f"\nSaved go/no-go report to {go_no_go_path}")

    print("\n" + "=" * 70)
    print("CLASSIFIER FITTING (6 conditions, seed=0 primary) -- this is the")
    print("expensive step at n=60,000; timed explicitly.")
    print("=" * 70)
    t_clf = time.time()
    conditions_out = pipe.run_classifier_conditions_multi_topology(
        results, labels, TOPOLOGY_NAMES, label_prefix="stage3_seed0_")
    clf_elapsed = time.time() - t_clf
    print(f"\nTotal classifier CV fitting time (6 conditions): {clf_elapsed:.1f}s "
          f"({clf_elapsed/60:.1f} min)")

    conditions_path = os.path.join(RESULTS_DIR, "stage3_classifier_conditions.pkl")
    with open(conditions_path, "wb") as f:
        pickle.dump({
            "conditions": conditions_out,
            "classifier_elapsed_seconds": clf_elapsed,
            "encode_elapsed_seconds": local_data["encode_elapsed_seconds"],
            "gpu_evolve_total_elapsed_seconds": gpu_data["total_elapsed"],
            "gpu_evolve_per_topology_elapsed": {
                name: gpu_data["results"][name]["elapsed"] for name in TOPOLOGY_NAMES},
        }, f)
    print(f"Saved classifier conditions to {conditions_path}")

    all_converged = all(c.get("converged", False) for c in conditions_out.values())
    all_go = (go_no_go["solver_failure_rate_ok"] and go_no_go["non_finite_ok"] and all_converged)
    print(f"\n{'='*70}\nOVERALL: {'GO' if all_go else 'NO-GO'} "
          f"(mechanical criteria only -- not a scientific result)\n{'='*70}")

    if not all_converged:
        print("\nNON-CONVERGED CONDITIONS:")
        for label, c in conditions_out.items():
            if not c.get("converged", False):
                print(f"  {label}: {c.get('error')}")

    return go_no_go, conditions_out


if __name__ == "__main__":
    main()
