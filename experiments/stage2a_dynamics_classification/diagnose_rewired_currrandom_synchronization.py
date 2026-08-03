"""
Diagnostic only -- does not touch the locked pipeline or any stage
driver. Two checks on why rewired/curr_random showed near-total phase
synchronization (R(theta) approx 1) in the stage-3 timing sub-test
(n=400), before committing to the full ~4.2-hour stage-3 run:

1. Information-collapse check: does a classifier trained on rewired/
   curr_random's evolved features actually collapse to predicting one
   or two classes (genuine information loss), or does real per-image
   signal survive despite the high R?
2. Multistability check, mirroring Stage 0's own method exactly
   (`find_equilibrium_lbfgs`, `same_attractor`'s 0.05 dedup threshold,
   5 seeds 0-4, no image encoding at all) -- run on ALL FOUR of this
   stage's canonical topologies (T, lattice, rewired, curr_random) for
   a direct, matched comparison, not reused from Stage 0's own table
   (different specific graph realizations/conventions there).
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
from bonsai.dynamics.graph_oscillator_field import find_equilibrium_lbfgs, gauge_corrected_distance
import stage2a_pipeline as pipe
import stage2a_topologies as topo

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
SEED = 42
N_PER_CLASS = 40  # n=400, identical to the timing sub-test

DEDUP_RESIDUAL_THRESHOLD = 0.05  # matches Stage 0's / stage1b_taxonomy.py's same_attractor() rule


def same_attractor(theta_a, theta_b):
    shift = np.angle(np.mean(np.exp(1j * (theta_a - theta_b))))
    residual = np.angle(np.exp(1j * (theta_a - theta_b - shift)))
    return np.mean(np.abs(residual)) < DEDUP_RESIDUAL_THRESHOLD


def multistability_check(W, name, seeds=(0, 1, 2, 3, 4)):
    equilibria = []
    for seed in seeds:
        theta_eq, _result = find_equilibrium_lbfgs(W, seed=seed)
        equilibria.append(theta_eq)

    distinct = []
    for theta in equilibria:
        if not any(same_attractor(theta, other) for other in distinct):
            distinct.append(theta)

    print(f"  [{name}] {len(distinct)} of {len(seeds)} distinct equilibria "
          f"(dedup threshold={DEDUP_RESIDUAL_THRESHOLD})")
    return {"n_distinct": len(distinct), "n_seeds": len(seeds)}


def classifier_collapse_check(feat_X, labels, name):
    scaler = StandardScaler().fit(feat_X)
    X_s = scaler.transform(feat_X)
    clf = LogisticRegression(C=0.01, solver="lbfgs", tol=1e-4, max_iter=10000,
                              class_weight=None, random_state=SEED)
    with np.errstate(all="ignore"):
        clf.fit(X_s, labels)
    train_acc = clf.score(X_s, labels)
    preds = clf.predict(X_s)
    pred_counts = {int(c): int(np.sum(preds == c)) for c in sorted(set(labels))}
    n_classes_predicted = sum(1 for c in pred_counts.values() if c > 0)
    print(f"  [{name}] train_acc={train_acc:.4f}, n_classes_actually_predicted="
          f"{n_classes_predicted}/10, prediction distribution={pred_counts}")
    return {"train_acc": float(train_acc), "n_classes_predicted": n_classes_predicted,
            "pred_counts": pred_counts}


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, SEED={SEED}) "
          f"-- identical to the stage-3 timing sub-test")

    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
    ref_idx = nodes_T["median"]

    print("\nRunning multi-topology pipeline (encode once, evolve on all 4 topologies)...")
    results, elapsed = pipe.run_pipeline_multi_topology(
        images_01, labels, topologies, ref_idx, active_indices)
    print(f"Pipeline complete: {len(results)} images in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("1. INFORMATION-COLLAPSE CHECK (rewired, curr_random)")
    print("=" * 70)
    collapse_results = {}
    for name in ("rewired", "curr_random"):
        valid_mask = np.array([not r["evolved"][name]["solver_failed"] for r in results])
        X = np.stack([r["evolved"][name]["feat_post"] for r in results
                      if not r["evolved"][name]["solver_failed"]])
        y = labels[valid_mask]
        collapse_results[name] = classifier_collapse_check(X, y, name)

    print("\n(For reference, T and lattice at the same n=400, same C:)")
    for name in ("T", "lattice"):
        valid_mask = np.array([not r["evolved"][name]["solver_failed"] for r in results])
        X = np.stack([r["evolved"][name]["feat_post"] for r in results
                      if not r["evolved"][name]["solver_failed"]])
        y = labels[valid_mask]
        collapse_results[name] = classifier_collapse_check(X, y, name)

    print("\n" + "=" * 70)
    print("2. MULTISTABILITY CHECK (mirrors Stage 0's own method exactly, all 4 topologies)")
    print("=" * 70)
    multistability_results = {}
    for name, W in topologies.items():
        multistability_results[name] = multistability_check(W, name)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name in ("T", "lattice", "rewired", "curr_random"):
        c = collapse_results[name]
        m = multistability_results[name]
        print(f"{name}: train_acc={c['train_acc']:.4f} "
              f"({c['n_classes_predicted']}/10 classes actually predicted), "
              f"multistability={m['n_distinct']}/{m['n_seeds']} distinct equilibria")

    return {"collapse": collapse_results, "multistability": multistability_results}


if __name__ == "__main__":
    main()
