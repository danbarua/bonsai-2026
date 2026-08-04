"""
Stage 2A feasibility stage 3, phase 1 (CPU-only, local): encode ALL
60,000 official KMNIST training images (no subsampling -- the full
training set, per DESIGN.md's feasibility ladder), build the 4
confirmatory-expansion topologies, and package theta0 for GPU upload.

Evolution is deliberately NOT done here -- at 60,000 images x 4
topologies, evolution is the dominant cost and is delegated to the
verified JAX/GPU pipeline (stage2a_pipeline_jax.py /
evolve_on_graph_jax.py), per the explicit instruction to use the GPU
pipeline for full-scale data generation. This script produces exactly
what the GPU step needs (theta0_batch + topologies) plus everything
needed locally afterward to reconstruct full results (raw_feat, feat_pre,
R_pre, labels, idx) without re-encoding.
"""
import os
import pickle
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_pipeline as pipe
import stage2a_topologies as topo
from stage2a_paths import train_scratch_dir

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
SCRATCH_DIR = train_scratch_dir()


def main():
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    print("Loading official KMNIST training set (full 60,000 images)...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    n_total = X_train.shape[0]
    print(f"Official training set: {n_total} images (official test set NOT loaded -- "
          f"stage 3 is still feasibility, not confirmatory)")

    images_01 = X_train.astype(np.float64) / 255.0
    labels = y_train.copy()
    idx = np.arange(n_total)

    print("\nBuilding all 4 confirmatory-expansion topologies (T, lattice, rewired, curr_random)...")
    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
    ref_idx = nodes_T["median"]
    assert ref_idx == 363, f"expected T's median node at index 363, got {ref_idx}"
    print(f"n_active={len(active_indices)}, ref_idx={ref_idx}, "
          f"topologies={list(topologies.keys())}")

    print(f"\nEncoding all {n_total} images (CPU, multiprocessing, seed=0 primary)...")
    t0 = time.time()
    results, elapsed = pipe.run_encode_only_multi_topology(images_01, ref_idx, active_indices)
    print(f"Encode complete: {n_total} images in {elapsed:.1f}s "
          f"({elapsed/n_total*1000:.3f} ms/image)")

    theta0_batch = np.stack([r["theta0"] for r in results])
    raw_feat = np.stack([r["raw_feat"] for r in results])
    feat_pre = np.stack([r["feat_pre"] for r in results])
    R_pre = np.array([r["R_pre"] for r in results])
    result_idx = np.array([r["idx"] for r in results])
    assert np.array_equal(result_idx, idx), "encode results not in expected idx order"

    print(f"\ntheta0_batch shape: {theta0_batch.shape}, dtype={theta0_batch.dtype}")
    print(f"raw_feat shape: {raw_feat.shape}, feat_pre shape: {feat_pre.shape}")

    # Package for GPU upload: theta0_batch + topologies ONLY (small --
    # evolution needs nothing else). raw_feat/feat_pre/R_pre/labels stay
    # local, recombined with the downloaded evolved results afterward.
    gpu_upload_path = os.path.join(SCRATCH_DIR, "stage3_gpu_upload.pkl")
    with open(gpu_upload_path, "wb") as f:
        pickle.dump({"theta0_batch": theta0_batch, "topologies": topologies}, f)
    upload_size_mb = os.path.getsize(gpu_upload_path) / 1e6
    print(f"\nSaved GPU upload package to {gpu_upload_path} ({upload_size_mb:.1f} MB)")

    local_path = os.path.join(SCRATCH_DIR, "stage3_encode_local.pkl")
    with open(local_path, "wb") as f:
        pickle.dump({
            "raw_feat": raw_feat, "feat_pre": feat_pre, "R_pre": R_pre,
            "idx": result_idx, "labels": labels,
            "active_indices": active_indices, "ref_idx": ref_idx, "nodes_T": nodes_T,
            "encode_elapsed_seconds": elapsed, "n_images": n_total,
        }, f)
    local_size_mb = os.path.getsize(local_path) / 1e6
    print(f"Saved local encode results to {local_path} ({local_size_mb:.1f} MB)")

    total_elapsed = time.time() - t0
    print(f"\nTotal phase-1 (encode) wall-clock: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
