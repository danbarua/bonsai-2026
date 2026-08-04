"""
Prepares the tiny inputs `measure_oscillator_gpu_latency.py` needs on
the remote Colab session for COMPUTE_COST_FINDINGS.md's single-image
GPU-path latency measurement: one representative raw image, the shared
`active_indices` mask, and all 4 topologies. Upload these three files
to `/content/...` (via `mighty-colab upload`) before running
`measure_oscillator_gpu_latency.py` remotely -- see `README.md`'s GPU
session pattern.
"""
import os
import sys
import pickle

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_topologies as topo
from bonsai.data.mnist_loader import load_mnist
from stage2a_paths import scratch_root

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
OUT_DIR = os.path.join(scratch_root(), "latency_prep")
os.makedirs(OUT_DIR, exist_ok=True)

_X_train, _y_train, X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
image_01 = X_test[0].astype(np.float64) / 255.0

active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
ref_idx = nodes_T["median"]

np.save(os.path.join(OUT_DIR, "raw_image.npy"), image_01)
np.save(os.path.join(OUT_DIR, "active_indices.npy"), active_indices)

with open(os.path.join(OUT_DIR, "topologies_for_latency.pkl"), "wb") as f:
    pickle.dump({"topologies": topologies, "ref_idx": ref_idx}, f)

print(f"image_01 shape={image_01.shape}, active_indices shape={active_indices.shape}, ref_idx={ref_idx}")
print("Saved raw_image.npy, active_indices.npy, topologies_for_latency.pkl")
