"""
Decomposes each active pixel's ink intensity (across the 10 class-
representative images) into a population-COMMON component (the
per-pixel mean across all 10 classes -- shared, not class-discriminatory
by construction) and a per-class DISCRIMINATORY component (that
class's deviation from the population mean at that pixel -- genuinely
class-varying). Tests which component the residual z-scored phase
deviation actually correlates with.
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
CONDITIONS = ["pre_evolution"] + TOPOLOGY_NAMES

X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
ref_idx = nodes_T["median"]
non_ref_mask = np.arange(len(active_indices)) != np.where(active_indices == active_indices[ref_idx])[0][0]

rep_images = {c: X_train[np.where(y_train == c)[0][0]].astype(np.float64) / 255.0 for c in range(10)}

def gauge_zscore(theta):
    shifted = (theta - theta[ref_idx] + np.pi) % (2 * np.pi) - np.pi
    others = shifted[non_ref_mask]
    std = others.std()
    z = np.zeros_like(shifted)
    if std > 1e-12:
        z[non_ref_mask] = (others - others.mean()) / std
    return z

# --- Decompose ink intensity: population-common vs. per-class-discriminatory ---
ink_by_class = np.stack([rep_images[c].flatten()[active_indices] for c in range(10)])  # (10, 505)
population_common = ink_by_class.mean(axis=0)  # (505,) -- same for every class, "red = common"
class_deviation = ink_by_class - population_common[None, :]  # (10, 505) -- "blue = discriminatory"

print(f"Population-common component: mean={population_common.mean():.3f}, "
      f"std across pixels={population_common.std():.3f}")
print(f"Class-deviation component: mean|dev| across (class,pixel)="
      f"{np.abs(class_deviation).mean():.3f}\n")

zscores = {}
for c in range(10):
    theta0 = s2a.encode_and_restrict(rep_images[c], active_indices, seed=s2a.ENCODER_SEED)
    zscores[(c, "pre_evolution")] = gauge_zscore(theta0)
    for name in TOPOLOGY_NAMES:
        theta_T, diag = s2a.evolve_on_graph(theta0, topologies[name])
        assert theta_T is not None
        zscores[(c, name)] = gauge_zscore(theta_T)

print(f"{'condition':>15} | {'r(common)':>12} | {'r(discrim.)':>12} | {'r(raw ink)':>12}")
for cond in CONDITIONS:
    r_common_list, r_discrim_list, r_raw_list = [], [], []
    for c in range(10):
        z = zscores[(c, cond)][non_ref_mask]
        common = population_common[non_ref_mask]
        discrim = class_deviation[c][non_ref_mask]
        raw = ink_by_class[c][non_ref_mask]
        r_common_list.append(pearsonr(common, z)[0])
        r_discrim_list.append(pearsonr(discrim, z)[0])
        r_raw_list.append(pearsonr(raw, z)[0])
    print(f"{cond:>15} | {np.mean(r_common_list):>+12.3f} | {np.mean(r_discrim_list):>+12.3f} | "
          f"{np.mean(r_raw_list):>+12.3f}")

# --- Cross-check: does raw = common + discriminatory make sense additively? ---
# raw_ink = population_common + class_deviation, by construction (exact decomposition).
# If z correlates with BOTH sub-components positively and comparably, the raw
# correlation is a mix of both, not dominated by only one.
print("\n(raw ink = population_common + class_deviation, exactly, by construction --")
print(" comparing r(common) vs r(discriminatory) tells us which part the residual actually tracks)")
