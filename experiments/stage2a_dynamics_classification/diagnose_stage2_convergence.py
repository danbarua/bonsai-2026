"""
Diagnostic-only script (not part of the locked pipeline): characterizes
the extent of the non-convergence found in feasibility stage 2's
primary CV run (evolved_T, fold=0, C=100.0, at 5,000 images). Regenerates
the same seed=0 features for the same 5,000-image subset (deterministic,
SEED=42) and scans every (fold, C) combination for all three conditions
without stopping on the first failure -- DESIGN.md's own framing
("a pattern of non-convergence concentrated in one condition or one C
region is itself a reportable diagnostic") requires this before
deciding how to respond.
"""
import os
import pickle
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_pipeline as pipe
import stage2a_classifier as s2a_clf

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
SEED = 42
N_PER_CLASS = 500


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)

    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, SEED={SEED}) "
          f"-- identical to feasibility stage 2's own subsample (same seed, same procedure)")

    print("\nRunning primary (seed=0) pipeline...")
    results, elapsed, active_indices, nodes_T = pipe.run_pipeline(images_01, labels)
    print(f"Pipeline complete: {len(results)} images in {elapsed:.1f}s")

    raw_X = np.stack([r["raw_feat"] for r in results])
    pre_X = np.stack([r["feat_pre"] for r in results])
    valid_mask = np.array([not r["solver_failed"] for r in results])
    evolved_X = np.stack([r["feat_post"] for r in results if not r["solver_failed"]])
    evolved_y = labels[valid_mask]

    print("\n" + "=" * 70)
    print("DIAGNOSTIC CONVERGENCE SCAN (all 5 folds x 9 C values, no early stop)")
    print("=" * 70)

    tables = {}
    for label, X, y in [
        ("raw_pixels", raw_X, labels),
        ("encoded_pre_evolution", pre_X, labels),
        ("evolved_T", evolved_X, evolved_y),
    ]:
        table = s2a_clf.diagnose_convergence_full_grid(X, y, label)
        tables[label] = table
        failed = sorted([(fold, C) for (fold, C), v in table.items() if not v["converged"]],
                         key=lambda x: (x[1], x[0]))
        if failed:
            print(f"  {label}: non-convergent (fold, C) pairs: {failed}")
            for fold, C in failed:
                print(f"    fold={fold}, C={C}: n_iter={table[(fold, C)]['n_iter']}")
        else:
            print(f"  {label}: all 45 (fold, C) combinations converged.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "stage2_convergence_diagnostic.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"tables": tables, "seed": SEED, "n_per_class": N_PER_CLASS}, f)
    print(f"\nSaved to {out_path}")
    return tables


if __name__ == "__main__":
    main()
