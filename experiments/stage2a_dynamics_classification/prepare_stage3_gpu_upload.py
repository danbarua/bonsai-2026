"""
Splits run_feasibility_stage3_encode.py's output
(scratch/stage3_train/stage3_gpu_upload.pkl -- theta0_batch + topologies
combined) into the exact files stage3_gpu_evolve.py expects on the
remote GPU session: 12 separate theta0_chunk_NN.npy arrays plus a
standalone stage3_topologies.pkl.

Committed because this step was previously undocumented as a runnable
script -- external review found README.md's GPU-reproduction section
gave generic example filenames (theta0_batch.npy, topologies.pkl) that
don't match either driver's actual expected inputs, and tracing the gap
further found the training-side chunking itself had never been captured
as a committed, re-runnable step at all (only described in prose, "the
working pattern is to shard..."). This script IS that step, not a
description of it.

Why 12 chunks specifically (matching stage3_gpu_evolve.py's own
N_UPLOAD_CHUNKS=12): the original 250MB single-pickle upload
(theta0_batch + topologies together) hit the transfer endpoint's size
limit; splitting theta0_batch (60,000 x 505 float64 = ~242MB) into 12
pieces keeps each chunk to ~20MB, comfortably under the limit that
caused the original failure. np.array_split (not a fixed-size slice)
is used so this doesn't assume 60,000 divides evenly by 12, even though
it happens to (5,000 images/chunk) -- correct either way.
"""
import os
import pickle
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from stage2a_paths import train_scratch_dir

SCRATCH_DIR = train_scratch_dir()
N_UPLOAD_CHUNKS = 12  # must match stage3_gpu_evolve.py's own N_UPLOAD_CHUNKS


def prepare(scratch_dir=SCRATCH_DIR, n_chunks=N_UPLOAD_CHUNKS):
    upload_path = os.path.join(scratch_dir, "stage3_gpu_upload.pkl")
    with open(upload_path, "rb") as f:
        d = pickle.load(f)
    theta0_batch = d["theta0_batch"]
    topologies = d["topologies"]
    n_images = theta0_batch.shape[0]
    print(f"Loaded {upload_path}: theta0_batch.shape={theta0_batch.shape}, "
          f"topologies={list(topologies.keys())}")

    chunks = np.array_split(theta0_batch, n_chunks, axis=0)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        path = os.path.join(scratch_dir, f"theta0_chunk_{i:02d}.npy")
        np.save(path, chunk)
        chunk_paths.append(path)
        print(f"  theta0_chunk_{i:02d}.npy: shape={chunk.shape}, "
              f"{os.path.getsize(path)/1e6:.1f} MB")

    topo_path = os.path.join(scratch_dir, "stage3_topologies.pkl")
    with open(topo_path, "wb") as f:
        pickle.dump(topologies, f)
    print(f"Saved {topo_path} ({os.path.getsize(topo_path)/1e6:.1f} MB)")

    # Round-trip check before declaring success -- reassemble exactly as
    # stage3_gpu_evolve.py does (np.concatenate over range(n_chunks) in
    # order) and confirm it's byte-identical to the original, not just
    # "the right shape".
    reassembled = np.concatenate([np.load(p) for p in chunk_paths], axis=0)
    assert reassembled.shape == theta0_batch.shape
    assert np.array_equal(reassembled, theta0_batch), (
        "Reassembled theta0_batch does not match the original byte-for-byte -- "
        "chunking/reassembly is broken, do not upload these chunks.")
    print(f"\nRound-trip check passed: {n_chunks} chunks reassemble byte-identical "
          f"to the original {n_images}-image theta0_batch.")
    print(f"\nUpload these {n_chunks + 1} files to the GPU session: "
          f"{', '.join(os.path.basename(p) for p in chunk_paths)}, "
          f"{os.path.basename(topo_path)}")
    return chunk_paths, topo_path


if __name__ == "__main__":
    prepare()
