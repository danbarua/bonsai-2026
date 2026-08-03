"""
Stage 2A confirmatory run, phase 1 (CPU-only, local): encode the
OFFICIAL 10,000-image KMNIST test set. This is the first and only time
in this project the test set's labels/images are used at all -- per
DESIGN.md's locked feasibility ladder ("one locked evaluation on the
untouched 10,000-image official test set... happens exactly once, after
stages 1-3 and the rest of this design are fully settled").

Mirrors run_feasibility_stage3_encode.py exactly (same encode_and_restrict/
order_parameter/reference_node_features calls, same topology build), just
pointed at the test split and a separate scratch directory so as not to
mix with the training-side artifacts.
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

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
SCRATCH_DIR = "/private/tmp/claude-501/-Users-dan-Code-pycharm-bonsai-2026/54a406a1-f8d0-41df-bc2a-d46e08e68715/scratchpad/stage2a_gpu_stage4_test"


def main():
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    print("Loading official KMNIST test set (the untouched 10,000-image split) -- "
          "first and only use of it in this project...")
    _X_train, _y_train, X_test, y_test = load_mnist(KMNIST_DIR, gz=False)
    n_total = X_test.shape[0]
    print(f"Official test set: {n_total} images")

    images_01 = X_test.astype(np.float64) / 255.0
    labels = y_test.copy()

    print("\nBuilding all 4 confirmatory-expansion topologies (T, lattice, rewired, curr_random)...")
    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
    ref_idx = nodes_T["median"]
    assert ref_idx == 363, f"expected T's median node at index 363, got {ref_idx}"

    print(f"\nEncoding all {n_total} test images (CPU, multiprocessing, seed=0 primary)...")
    t0 = time.time()
    results, elapsed = pipe.run_encode_only_multi_topology(images_01, ref_idx, active_indices)
    print(f"Encode complete: {n_total} images in {elapsed:.1f}s "
          f"({elapsed/n_total*1000:.3f} ms/image)")

    theta0_batch = np.stack([r["theta0"] for r in results])
    raw_feat = np.stack([r["raw_feat"] for r in results])
    feat_pre = np.stack([r["feat_pre"] for r in results])
    R_pre = np.array([r["R_pre"] for r in results])
    result_idx = np.array([r["idx"] for r in results])
    assert np.array_equal(result_idx, np.arange(n_total)), "encode results not in expected idx order"

    print(f"\ntheta0_batch shape: {theta0_batch.shape}, dtype={theta0_batch.dtype}")

    gpu_upload_path = os.path.join(SCRATCH_DIR, "stage4_gpu_upload_topologies.pkl")
    with open(gpu_upload_path, "wb") as f:
        pickle.dump({"topologies": topologies}, f)
    print(f"Saved GPU topologies package to {gpu_upload_path}")

    # theta0 for the test set is small enough (10000x505 float64 = 40MB)
    # to upload as a single .npy, no chunking needed at this scale.
    theta0_path = os.path.join(SCRATCH_DIR, "stage4_theta0_test.npy")
    np.save(theta0_path, theta0_batch)
    print(f"Saved theta0_batch to {theta0_path} ({os.path.getsize(theta0_path)/1e6:.1f} MB)")

    local_path = os.path.join(SCRATCH_DIR, "stage4_encode_local.pkl")
    with open(local_path, "wb") as f:
        pickle.dump({
            "raw_feat": raw_feat, "feat_pre": feat_pre, "R_pre": R_pre,
            "idx": result_idx, "labels": labels,
            "active_indices": active_indices, "ref_idx": ref_idx, "nodes_T": nodes_T,
            "encode_elapsed_seconds": elapsed, "n_images": n_total,
        }, f)
    print(f"Saved local encode results to {local_path} "
          f"({os.path.getsize(local_path)/1e6:.1f} MB)")

    print(f"\nTotal phase-1 (test encode) wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
