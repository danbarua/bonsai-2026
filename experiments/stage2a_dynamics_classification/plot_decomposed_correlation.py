import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import stage2a_topologies as topo
from bonsai.data.mnist_loader import load_mnist

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
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

ink_by_class = np.stack([rep_images[c].flatten()[active_indices] for c in range(10)])
population_common = ink_by_class.mean(axis=0)
class_deviation = ink_by_class - population_common[None, :]

zscores = {}
for c in range(10):
    theta0 = s2a.encode_and_restrict(rep_images[c], active_indices, seed=s2a.ENCODER_SEED)
    zscores[(c, "pre_evolution")] = gauge_zscore(theta0)
    for name in TOPOLOGY_NAMES:
        theta_T, diag = s2a.evolve_on_graph(theta0, topologies[name])
        zscores[(c, name)] = gauge_zscore(theta_T)

r_common = {cond: [] for cond in CONDITIONS}
r_discrim = {cond: [] for cond in CONDITIONS}
for cond in CONDITIONS:
    for c in range(10):
        z = zscores[(c, cond)][non_ref_mask]
        r_common[cond].append(pearsonr(population_common[non_ref_mask], z)[0])
        r_discrim[cond].append(pearsonr(class_deviation[c][non_ref_mask], z)[0])

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(CONDITIONS))
width = 0.35
means_common = [np.mean(r_common[c]) for c in CONDITIONS]
stds_common = [np.std(r_common[c]) for c in CONDITIONS]
means_discrim = [np.mean(r_discrim[c]) for c in CONDITIONS]
stds_discrim = [np.std(r_discrim[c]) for c in CONDITIONS]

ax.bar(x - width/2, means_common, width, yerr=stds_common, capsize=3,
       label="population-common component", color="lightsteelblue")
ax.bar(x + width/2, means_discrim, width, yerr=stds_discrim, capsize=3,
       label="class-discriminatory component", color="firebrick")
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS, rotation=20)
ax.set_ylabel("Pearson r vs. z-scored residual phase deviation")
ax.set_title("The discriminatory ink component correlates more strongly than the common\n"
             "component with the surviving residual, at every condition")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "ink_correlation_decomposed.png")
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
