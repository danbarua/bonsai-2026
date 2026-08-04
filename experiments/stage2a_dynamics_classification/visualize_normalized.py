"""
Companion to phase_state_per_class_per_topology.png: the SAME raw-phase
plot at a fixed 0-2pi scale makes real-but-small-magnitude residual
structure in rewired/curr_random look visually uniform. This plot shows
the gauge-shifted phase (theta - theta_ref, the actual quantity the
classifier's cos/sin features are built from), z-scored PER PANEL
(excluding the trivially-constant reference node itself, matching the
locked 1008-dim feature -- not 1010) -- i.e. the same rescaling
StandardScaler performs before the classifier ever sees the features,
made visible. Does not replace the honest raw-scale plot (real absolute
magnitude matters and is not shown here) -- a companion, not a
correction.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

print("Selecting one representative image per class...")
rep_images = {}
for c in range(10):
    idx = np.where(y_train == c)[0][0]
    rep_images[c] = X_train[idx].astype(np.float64) / 255.0

print("Encoding + evolving, computing gauge-shifted z-scored phase deviation...")
zscore_grids = {}  # (class, condition) -> (28,28) array, NaN for inactive
raw_std = {}  # (class, condition) -> the actual std (radians) before z-scoring, for reporting
non_ref_mask = np.arange(len(active_indices)) != np.where(active_indices == active_indices[ref_idx])[0][0]

for c in range(10):
    theta0 = s2a.encode_and_restrict(rep_images[c], active_indices, seed=s2a.ENCODER_SEED)

    def gauge_zscore(theta):
        shifted = (theta - theta[ref_idx] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        others = shifted[non_ref_mask]
        std = others.std()
        z = np.zeros_like(shifted)
        if std > 1e-12:
            z[non_ref_mask] = (others - others.mean()) / std
        return z, std

    z, std = gauge_zscore(theta0)
    grid = np.full(784, np.nan)
    grid[active_indices] = z
    zscore_grids[(c, "pre_evolution")] = grid.reshape(28, 28)
    raw_std[(c, "pre_evolution")] = std

    for name in TOPOLOGY_NAMES:
        theta_T, diag = s2a.evolve_on_graph(theta0, topologies[name])
        assert theta_T is not None, f"solver failure, class {c}, {name}"
        z, std = gauge_zscore(theta_T)
        grid = np.full(784, np.nan)
        grid[active_indices] = z
        zscore_grids[(c, name)] = grid.reshape(28, 28)
        raw_std[(c, name)] = std
    print(f"  class {c} done")

print("\nRaw (pre-normalization) std of gauge-shifted phase, radians, by condition (mean across classes):")
for cond in ["pre_evolution"] + TOPOLOGY_NAMES:
    vals = [raw_std[(c, cond)] for c in range(10)]
    print(f"  {cond}: mean_std={np.mean(vals):.5f} rad, min={np.min(vals):.5f}, max={np.max(vals):.5f}")

print("\nBuilding normalized (per-panel z-score) grid...")
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
        im = ax.imshow(zscore_grids[(c, cond)], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_title(cond, fontsize=11)
        # annotate the actual raw std this panel's color range represents
        ax.text(0.02, 0.98, f"{raw_std[(c, cond)]:.4f} rad", transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="dimgray")
fig.suptitle("Same pipeline, PER-PANEL z-scored gauge-shifted phase deviation\n"
             "(what StandardScaler shows the classifier -- annotation = actual raw std in radians "
             "before normalization; every panel is rescaled to unit variance regardless of that "
             "raw magnitude)", fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.5, label="z-score (std devs from panel's own mean)", pad=0.02)
out_path = os.path.join(RESULTS_DIR, "phase_state_per_class_per_topology_normalized.png")
plt.savefig(out_path, dpi=130)
plt.close(fig)
print(f"\nSaved {out_path}")
