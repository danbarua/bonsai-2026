"""
Tests the user's hypothesis directly: does the residual z-scored phase
deviation (rewired/curr_random especially) correlate with THIS image's
own ink intensity at each masked pixel -- i.e. is "red = inked here,
blue = not inked here (relative to this pixel's own mean across
conditions/classes)" actually what's driving the speckle pattern?
"""
import os
import sys

import numpy as np
from scipy.stats import pearsonr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import stage2a_topologies as topo
from bonsai.data.mnist_loader import load_mnist

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]

print("Loading data, building topologies...")
X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
ref_idx = nodes_T["median"]
non_ref_mask = np.arange(len(active_indices)) != np.where(active_indices == active_indices[ref_idx])[0][0]

rep_images = {}
for c in range(10):
    idx = np.where(y_train == c)[0][0]
    rep_images[c] = X_train[idx].astype(np.float64) / 255.0

def gauge_zscore(theta):
    shifted = (theta - theta[ref_idx] + np.pi) % (2 * np.pi) - np.pi
    others = shifted[non_ref_mask]
    std = others.std()
    z = np.zeros_like(shifted)
    if std > 1e-12:
        z[non_ref_mask] = (others - others.mean()) / std
    return z

print("\nEncoding + evolving, testing ink-intensity vs. residual-z-score correlation...\n")
active_ink_intensity = {}  # per class: this image's own pixel intensity at each active node
zscores = {}  # (class, condition) -> z-score array over active_indices (in active_indices order)

for c in range(10):
    active_ink_intensity[c] = rep_images[c].flatten()[active_indices]
    theta0 = s2a.encode_and_restrict(rep_images[c], active_indices, seed=s2a.ENCODER_SEED)
    zscores[(c, "pre_evolution")] = gauge_zscore(theta0)
    for name in TOPOLOGY_NAMES:
        theta_T, diag = s2a.evolve_on_graph(theta0, topologies[name])
        assert theta_T is not None
        zscores[(c, name)] = gauge_zscore(theta_T)

print("Pearson correlation: this image's own ink intensity (at each active pixel) "
      "vs. z-scored residual, per class:\n")
print(f"{'class':>6} | {'pre_evol':>18} | {'T':>18} | {'lattice':>18} | {'rewired':>18} | {'curr_random':>18}")
for c in range(10):
    row = [f"{c:>6}"]
    for cond in ["pre_evolution"] + TOPOLOGY_NAMES:
        r, p = pearsonr(active_ink_intensity[c], zscores[(c, cond)][non_ref_mask if False else slice(None)])
        # exclude ref node from correlation too, for consistency
        ink = active_ink_intensity[c][non_ref_mask]
        z = zscores[(c, cond)][non_ref_mask]
        r, p = pearsonr(ink, z)
        row.append(f"r={r:+.3f}(p={p:.1e})")
    print(" | ".join(f"{x:>18}" if i > 0 else x for i, x in enumerate(row)))

print("\nMean |r| across classes, per condition:")
for cond in ["pre_evolution"] + TOPOLOGY_NAMES:
    rs = []
    for c in range(10):
        ink = active_ink_intensity[c][non_ref_mask]
        z = zscores[(c, cond)][non_ref_mask]
        r, p = pearsonr(ink, z)
        rs.append(r)
    print(f"  {cond}: mean_r={np.mean(rs):+.4f}, mean_|r|={np.mean(np.abs(rs)):.4f}, "
          f"min_r={np.min(rs):+.4f}, max_r={np.max(rs):+.4f}")
