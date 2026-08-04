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

r_by_condition = {cond: [] for cond in CONDITIONS}
for c in range(10):
    ink = rep_images[c].flatten()[active_indices][non_ref_mask]
    theta0 = s2a.encode_and_restrict(rep_images[c], active_indices, seed=s2a.ENCODER_SEED)
    z = gauge_zscore(theta0)[non_ref_mask]
    r_by_condition["pre_evolution"].append(pearsonr(ink, z)[0])
    for name in TOPOLOGY_NAMES:
        theta_T, diag = s2a.evolve_on_graph(theta0, topologies[name])
        z = gauge_zscore(theta_T)[non_ref_mask]
        r_by_condition[name].append(pearsonr(ink, z)[0])

means = [np.mean(r_by_condition[c]) for c in CONDITIONS]
stds = [np.std(r_by_condition[c]) for c in CONDITIONS]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(CONDITIONS))
ax.bar(x, means, yerr=stds, capsize=4, color=["gray", "tab:blue", "tab:orange", "tab:green", "tab:red"])
ax.set_xticks(x)
ax.set_xticklabels(CONDITIONS, rotation=20)
ax.set_ylabel("Pearson r (this image's ink intensity vs. z-scored\nresidual phase deviation, per active pixel)")
ax.set_title("Ink-presence correlation survives synchronization, shrinking but never vanishing\n"
             "(mean +/- std across 10 classes, one representative image each)")
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "ink_correlation_decay.png")
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
print("means:", dict(zip(CONDITIONS, means)))
print("stds:", dict(zip(CONDITIONS, stds)))
