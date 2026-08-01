"""
Builds all four matched graph constructions (T, rewired, random, lattice)
for all 10 KMNIST classes, using src/bonsai/dynamics/construction_bundle.py's
build_class_construction_bundle() -- extending the class-0-only scope this
project deliberately held to until now (see docs/PROJECT_MEMORY.md Part 4,
"Construction-recovery effort, open items", item 1).

Uses seed=1 for both rewired and random (matching Stage 1A's original
seed=1, recorded in experiments/stage1a_infinitesimal_response/FINDINGS.md's
"Reproducing these results" section) and the first 200 training images per
class for T (n_per_class=200, the historically recovered hyperparameter --
see src/bonsai/dynamics/learned_topology_construction.py's docstring).

NOTE: 'random' here is matched_sparsity_ablation.py's current
edge-count-matched random (build_class_construction_bundle()'s default),
NOT the historical half-edge random reconstruction -- this driver calls
the bundle function as instructed, without substituting a different
random-construction algorithm. See historical_matched_sparsity_random.py
if the historically-faithful control is needed for future work.

Output format matches the historical stage1a_all_classes.pkl exactly:
{class_int: {'constructions': {'T', 'rewired', 'random', 'lattice'},
'n_active': int}} for class_int in 0..9. Runtime: ~1.5s/class, ~15s total
-- confirmed by timing a single class before running all 10, not assumed.
"""
import os
import pickle
import time

import numpy as np

from bonsai.data.mnist_loader import load_mnist
from bonsai.dynamics.construction_bundle import build_class_construction_bundle

N_PER_CLASS = 200
REWIRED_SEED = 1
RANDOM_SEED = 1

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "stage1a_all_classes.pkl")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train, _, _ = load_mnist(KMNIST_DIR, gz=False)

    all_data = {}
    for class_idx in range(10):
        t0 = time.time()
        idx = np.where(y_train == class_idx)[0][:N_PER_CLASS]
        images = X_train[idx].astype(np.float64) / 255.0
        bundle = build_class_construction_bundle(
            images, rewired_seed=REWIRED_SEED, random_seed=RANDOM_SEED)
        all_data[class_idx] = bundle
        elapsed = time.time() - t0
        print(f"class {class_idx}: n_active={bundle['n_active']}, {elapsed:.2f}s")

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved {len(all_data)} classes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
