"""
Two visualizations, per the user's request ("both"):

1. The 4 topology graphs themselves -- nodes + edges overlaid on the
   28x28 pixel grid (static structure, doesn't vary by class).
2. Per-class evolved-dynamics: one representative image per class,
   encoded + evolved on each of the 4 topologies, phase state rendered
   spatially with a cyclic colormap -- visualizes the synchronization
   pattern already measured numerically (near-total for rewired/
   curr_random, much weaker for T/lattice).

All local, CPU-only -- no GPU needed at this scale (10 classes x 4
topologies = 40 single-image evolutions, ~12s total per the just-
measured ~280-320ms/evolution CPU latency).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import stage2a_topologies as topo
from bonsai.data.mnist_loader import load_mnist

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]

print("Loading official KMNIST training set...")
X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)

print("Building all 4 topologies...")
active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
ref_idx = nodes_T["median"]

rows = active_indices // 28
cols = active_indices % 28

# ============================================================
# PART 1: static topology graph structure (nodes + edges)
# ============================================================
print("\nBuilding topology-structure plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, name in zip(axes.flat, TOPOLOGY_NAMES):
    W = topologies[name]
    iu, ju = np.triu_indices_from(W, k=1)
    mask = W[iu, ju] > 0
    iu, ju = iu[mask], ju[mask]
    weights = W[iu, ju]

    segments = np.stack([
        np.stack([cols[iu], rows[iu]], axis=1),
        np.stack([cols[ju], rows[ju]], axis=1),
    ], axis=1)
    lc = LineCollection(segments, colors="steelblue", linewidths=0.4,
                         alpha=np.clip(weights / weights.max(), 0.05, 0.4))
    ax.add_collection(lc)
    ax.scatter(cols, rows, s=8, c="black", zorder=3)
    if name == "T":
        ax.scatter([cols[active_indices == active_indices[ref_idx]][0]],
                    [rows[active_indices == active_indices[ref_idx]][0]],
                    s=60, c="red", marker="*", zorder=4, label="reference node")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-1, 28)
    ax.set_ylim(28, -1)
    ax.set_aspect("equal")
    n_edges = len(iu)
    ax.set_title(f"{name} ({n_edges} edges, {len(active_indices)} nodes)")
plt.tight_layout()
out1 = os.path.join(RESULTS_DIR, "topology_graph_structure.png")
plt.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}")

# ============================================================
# PART 2: per-class evolved-dynamics visualization
# ============================================================
print("\nSelecting one representative image per class...")
rep_images = {}
for c in range(10):
    idx = np.where(y_train == c)[0][0]  # first occurrence, deterministic
    rep_images[c] = X_train[idx].astype(np.float64) / 255.0

print("Encoding + evolving each representative image on each topology "
      "(CPU, ~12s total)...")
phase_grids = {}  # (class, condition) -> (28,28) array, NaN for inactive
for c in range(10):
    theta0 = s2a.encode_and_restrict(rep_images[c], active_indices, seed=s2a.ENCODER_SEED)

    grid = np.full(784, np.nan)
    grid[active_indices] = theta0 % (2 * np.pi)
    phase_grids[(c, "pre_evolution")] = grid.reshape(28, 28)

    for name in TOPOLOGY_NAMES:
        theta_T, diag = s2a.evolve_on_graph(theta0, topologies[name])
        assert theta_T is not None, f"solver failure, class {c}, {name}"
        grid = np.full(784, np.nan)
        grid[active_indices] = theta_T % (2 * np.pi)
        phase_grids[(c, name)] = grid.reshape(28, 28)
    print(f"  class {c} done")

print("\nBuilding per-class evolved-dynamics grid...")
conditions = ["pre_evolution"] + TOPOLOGY_NAMES
fig, axes = plt.subplots(10, 6, figsize=(15.5, 26))
for c in range(10):
    ax_ref = axes[c, 0]
    ax_ref.imshow(rep_images[c], cmap="gray_r", vmin=0, vmax=1)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    if c == 0:
        ax_ref.set_title("image", fontsize=11)
    ax_ref.set_ylabel(f"class {c}", fontsize=11)

    for j, cond in enumerate(conditions):
        ax = axes[c, j + 1]
        im = ax.imshow(phase_grids[(c, cond)], cmap="twilight", vmin=0, vmax=2 * np.pi)
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_title(cond, fontsize=11)
fig.suptitle("Reference image, then phase state per class: pre-evolution vs. evolved on "
             "each topology\n(cyclic colormap; near-uniform color = near-total synchronization)",
             fontsize=13)
fig.colorbar(im, ax=axes, shrink=0.5, label="phase (radians)", pad=0.02)
out2 = os.path.join(RESULTS_DIR, "phase_state_per_class_per_topology.png")
plt.savefig(out2, dpi=130)
plt.close(fig)
print(f"Saved {out2}")

print("\nDone.")
