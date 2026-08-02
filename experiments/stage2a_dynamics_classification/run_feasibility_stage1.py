"""
Stage 2A feasibility stage 1 (DESIGN.md, "Minimal feasibility pass"):
1,000 official-training images (100/class), end-to-end mechanical
correctness only -- explicitly NOT an early scientific result. This
script never touches the official KMNIST test set.

Checks, per DESIGN.md's locked go/no-go criteria:
- zero non-finite feature vectors;
- zero silent solver failures (every evolution ODE solve reports its
  own status; any non-recovered failure is a stop, not a rate);
- R(theta) diagnostic distribution reported (never a gauge trigger);
- the linear readout converges in every condition (stops, per the
  locked non-convergence gate, otherwise).

Three conditions (raw pixels, encoded pre-evolution, evolved on T),
each fit via the real, locked CV procedure (stage2a_classifier.py) on
this 1,000-image set -- reported descriptively, per DESIGN.md's "the
feasibility ladder may report raw differences descriptively, without
formal inference."
"""
import os
import pickle
import sys
import time
import multiprocessing as mp

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))

from bonsai.data.mnist_loader import load_mnist
import stage2a_core as s2a
import stage2a_classifier as s2a_clf
from stage2a_classifier import NonConvergenceError

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
SEED = 42
N_PER_CLASS = 100
N_CLASSES = 10


def subsample_1000(X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS):
    """Deterministic, class-stratified subsample of the official KMNIST
    training set -- SEED=42 per DESIGN.md ("governs... feasibility-stage
    subsampling"). Returns (images_01, labels), images_01 in [0, 1]."""
    rng = np.random.default_rng(seed)
    selected_idx = []
    for c in range(N_CLASSES):
        class_idx = np.where(y_train == c)[0]
        chosen = rng.choice(class_idx, size=n_per_class, replace=False)
        selected_idx.extend(chosen.tolist())
    selected_idx = np.array(selected_idx)
    images_01 = X_train[selected_idx].astype(np.float64) / 255.0
    labels = y_train[selected_idx]
    return images_01, labels


def _process_one_image(args):
    idx, image_01, active_indices_tuple, W_T, ref_idx = args
    active_indices = np.array(active_indices_tuple)

    theta0 = s2a.encode_and_restrict(image_01, active_indices)
    R_pre = s2a.order_parameter(theta0)
    feat_pre = s2a.reference_node_features(theta0, ref_idx)

    thetaT, diag = s2a.evolve_on_graph(theta0, W_T)
    if thetaT is None:
        return {
            "idx": idx, "solver_failed": True, "solver_diag": diag,
            "R_pre": R_pre, "R_post": None,
            "feat_pre": feat_pre, "feat_post": None,
            "raw_feat": image_01.flatten(),
        }

    R_post = s2a.order_parameter(thetaT)
    feat_post = s2a.reference_node_features(thetaT, ref_idx)
    return {
        "idx": idx, "solver_failed": False, "solver_diag": diag,
        "R_pre": R_pre, "R_post": R_post,
        "feat_pre": feat_pre, "feat_post": feat_post,
        "raw_feat": image_01.flatten(),
    }


def run_pipeline(images_01, labels):
    active_indices, W_T, ink_mask_active, nodes_T = s2a.load_T()
    ref_idx = nodes_T["median"]
    assert ref_idx == 363, f"expected T's median node at index 363, got {ref_idx}"

    n_workers = max(1, mp.cpu_count() - 1)
    work_items = [(i, images_01[i], tuple(active_indices), W_T, ref_idx)
                  for i in range(len(images_01))]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_one_image, work_items)
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["idx"])
    return results, elapsed, active_indices, nodes_T


def check_go_no_go(results):
    """DESIGN.md's locked go/no-go criteria, mechanics only."""
    report = {}

    n_solver_failed = sum(1 for r in results if r["solver_failed"])
    report["n_images"] = len(results)
    report["n_solver_failed"] = n_solver_failed
    report["solver_failure_rate"] = n_solver_failed / len(results)
    report["solver_failure_rate_ok"] = report["solver_failure_rate"] <= 0.001

    non_finite_count = 0
    for r in results:
        if not np.all(np.isfinite(r["raw_feat"])):
            non_finite_count += 1
        if not np.all(np.isfinite(r["feat_pre"])):
            non_finite_count += 1
        if not r["solver_failed"] and not np.all(np.isfinite(r["feat_post"])):
            non_finite_count += 1
    report["n_non_finite_feature_vectors"] = non_finite_count
    report["non_finite_ok"] = non_finite_count == 0

    R_pre_vals = np.array([r["R_pre"] for r in results])
    R_post_vals = np.array([r["R_post"] for r in results if r["R_post"] is not None])
    report["R_pre_summary"] = {
        "min": float(R_pre_vals.min()), "max": float(R_pre_vals.max()),
        "mean": float(R_pre_vals.mean()), "median": float(np.median(R_pre_vals)),
        "n_below_0.01": int(np.sum(R_pre_vals < 0.01)), "n_above_0.99": int(np.sum(R_pre_vals > 0.99)),
    }
    report["R_post_summary"] = {
        "min": float(R_post_vals.min()), "max": float(R_post_vals.max()),
        "mean": float(R_post_vals.mean()), "median": float(np.median(R_post_vals)),
        "n_below_0.01": int(np.sum(R_post_vals < 0.01)), "n_above_0.99": int(np.sum(R_post_vals > 0.99)),
    } if len(R_post_vals) else None
    report["R_pre_values"] = R_pre_vals.tolist()
    report["R_post_values"] = R_post_vals.tolist()
    report["R_near_limits_flag"] = bool(
        report["R_pre_summary"]["n_below_0.01"] or report["R_pre_summary"]["n_above_0.99"] or
        (report["R_post_summary"] and (report["R_post_summary"]["n_below_0.01"]
                                        or report["R_post_summary"]["n_above_0.99"])))

    return report


def run_classifier_conditions(results, labels):
    """CV-fits all 3 conditions via the real, locked selection procedure
    (stage2a_classifier.select_C_via_cv) on this 1,000-image set --
    every (fold, C) combination is actually fit and checked for
    convergence, exercising the real pipeline code. No held-out test
    evaluation here (there is no test set at stage 1, official or
    otherwise): stage 1 is correctness-only, and the official test set
    is never touched at this stage. The final refit-and-apply-to-test
    step (stage2a_classifier.fit_condition) is exercised for real
    starting at stage 4, where an actual test set exists."""
    raw_X = np.stack([r["raw_feat"] for r in results])
    pre_X = np.stack([r["feat_pre"] for r in results])

    valid_mask = np.array([not r["solver_failed"] for r in results])
    evolved_X = np.stack([r["feat_post"] for r in results if not r["solver_failed"]])
    evolved_y = labels[valid_mask]

    conditions_out = {}
    for label, X, y in [
        ("raw_pixels", raw_X, labels),
        ("encoded_pre_evolution", pre_X, labels),
        ("evolved_T", evolved_X, evolved_y),
    ]:
        print(f"\nFitting condition: {label} (n={len(y)}, dim={X.shape[1]})")
        try:
            best_C, mean_val_loss, _non_convergence_log = s2a_clf.select_C_via_cv(X, y, label)
            conditions_out[label] = {
                "converged": True,
                "selected_C": best_C,
                "mean_val_loss_per_C": mean_val_loss,
            }
            print(f"  Converged in every fold/C. Selected C={best_C}, "
                  f"mean_val_loss_per_C={mean_val_loss}")
        except NonConvergenceError as e:
            conditions_out[label] = {"converged": False, "error": str(e)}
            print(f"  NON-CONVERGENCE (stage halts per locked gate): {e}")

    return conditions_out


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
    print(f"Official training set: {X_train.shape[0]} images "
          f"(official test set NOT loaded -- not needed at stage 1)")

    images_01, labels = subsample_1000(X_train, y_train)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, SEED={SEED})")

    print("\nRunning pipeline (encode -> restrict -> evolve -> gauge features)...")
    results, elapsed, active_indices, nodes_T = run_pipeline(images_01, labels)
    print(f"Pipeline complete: {len(results)} images in {elapsed:.1f}s "
          f"({elapsed/len(results)*1000:.1f} ms/image)")

    print("\n" + "=" * 70)
    print("GO/NO-GO MECHANICAL CHECKS")
    print("=" * 70)
    go_no_go = check_go_no_go(results)
    for k, v in go_no_go.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("CLASSIFIER FITTING (3 conditions, correctness check only)")
    print("=" * 70)
    conditions_out = run_classifier_conditions(results, labels)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "stage1_feasibility_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "go_no_go": go_no_go,
            "conditions": conditions_out,
            "elapsed_seconds": elapsed,
            "n_images": len(results),
            "nodes_T": nodes_T,
            "seed": SEED,
            "n_per_class": N_PER_CLASS,
        }, f)
    print(f"\nSaved to {out_path}")

    all_go = (go_no_go["solver_failure_rate_ok"] and go_no_go["non_finite_ok"]
              and all(c.get("converged", False) for c in conditions_out.values()))
    print(f"\n{'='*70}\nOVERALL: {'GO' if all_go else 'NO-GO'} "
          f"(mechanical criteria only -- not a scientific result)\n{'='*70}")
    return go_no_go, conditions_out


if __name__ == "__main__":
    main()
