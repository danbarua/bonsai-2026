"""
Class-0-support audit, part 2: fits and evaluates the two baseline
conditions external review asked for (raw pixels restricted to the
505-node support vs. the already-known full-784 raw pixels; the full
784-pixel locally-encoded state, gauge-featurized without restriction,
vs. the already-known 505-restricted `encoded_pre_evolution`), using
this project's own real, unmodified `select_C_via_cv` +
`fit_final_at_selected_C` -- no reimplementation.

Not runnable locally as-is in the sense of expecting local GPU
acceleration -- designed to run under `cuml.accel` on a `mighty-colab`
GPU session (verified elsewhere in this project, `CUML_ACCEL_
FINDINGS.md`, to reproduce sklearn's results faithfully once `max_iter`
is calibrated for this backend). Reads the feature sets
`run_class0_support_audit.py` (part 1) already built and saved to
`scratch/class0_support_audit/`.

This is an audit, not a new locked confirmatory comparison -- reported
descriptively, alongside the already-known raw_pixels(784) and
encoded_pre_evolution(505) numbers, to quantify how much test
performance the class-0-derived support projection costs before any
graph evolution happens at all.
"""
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from stage2a_paths import scratch_root
from stage2a_classifier import select_C_via_cv, fit_final_at_selected_C, NonConvergenceError
from stage2a_stats import per_image_log_loss

AUDIT_DIR = os.path.join(scratch_root(), "class0_support_audit")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

# cuml.accel-specific override, disclosed -- see CUML_ACCEL_FINDINGS.md
# for why sklearn's own max_iter=10000 isn't necessarily sufficient
# under this backend's different convergence footprint.
CUML_MAX_ITER = 20000


def main():
    use_cuml = "--cuml" in sys.argv
    if use_cuml:
        import cuml.accel
        cuml.accel.install()
        import stage2a_classifier as s2a_clf
        s2a_clf.CLASSIFIER_KWARGS["max_iter"] = CUML_MAX_ITER
        print(f"cuml.accel active, max_iter={CUML_MAX_ITER}")
    else:
        print("Running under plain CPU sklearn (no --cuml flag given).")

    print("Loading audit feature sets...")
    X_train_raw505 = np.load(os.path.join(AUDIT_DIR, "X_train_raw505.npy")).astype(np.float64)
    X_test_raw505 = np.load(os.path.join(AUDIT_DIR, "X_test_raw505.npy")).astype(np.float64)
    X_train_encoded784 = np.load(os.path.join(AUDIT_DIR, "X_train_encoded784.npy")).astype(np.float64)
    X_test_encoded784 = np.load(os.path.join(AUDIT_DIR, "X_test_encoded784.npy")).astype(np.float64)
    y_train = np.load(os.path.join(AUDIT_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(AUDIT_DIR, "y_test.npy"))
    classes = np.unique(y_train)
    print(f"raw505: train={X_train_raw505.shape}, test={X_test_raw505.shape}")
    print(f"encoded784: train={X_train_encoded784.shape}, test={X_test_encoded784.shape}")

    conditions = {
        "raw_pixels_505restricted": (X_train_raw505, X_test_raw505),
        "encoded_784_unrestricted": (X_train_encoded784, X_test_encoded784),
    }

    results = {}
    for label, (X_train, X_test) in conditions.items():
        print(f"\n{'='*70}\n{label} (dim={X_train.shape[1]})\n{'='*70}")
        t0 = time.time()
        try:
            best_C, mean_val_loss, _log = select_C_via_cv(X_train, y_train, label)
        except NonConvergenceError as e:
            print(f"  NON-CONVERGENCE: {e}")
            results[label] = {"converged": False, "error": str(e)}
            continue
        cv_elapsed = time.time() - t0
        print(f"  CV: best_C={best_C}, elapsed={cv_elapsed:.1f}s")

        t0 = time.time()
        fit = fit_final_at_selected_C(X_train, y_train, X_test, best_C, label)
        fit_elapsed = time.time() - t0
        proba = fit["classifier"].predict_proba(fit["X_test_standardized"])
        y_pred = fit["classifier"].classes_[np.argmax(proba, axis=1)]
        ell_i = per_image_log_loss(y_test, proba, fit["classifier"].classes_)
        acc = float(np.mean(y_pred == y_test))
        mean_ll = float(np.mean(ell_i))
        print(f"  final refit: {fit_elapsed:.1f}s, accuracy={acc:.4f}, log_loss={mean_ll:.4f}")

        results[label] = {
            "converged": True, "selected_C": best_C, "cv_elapsed": cv_elapsed,
            "fit_elapsed": fit_elapsed, "accuracy": acc, "mean_log_loss": mean_ll,
        }

    print(f"\n{'='*70}\nSUMMARY (vs. already-known confirmatory numbers)\n{'='*70}")
    print("raw_pixels (784, full):            C=0.001,  accuracy=0.6960, log_loss=0.9848")
    if results.get("raw_pixels_505restricted", {}).get("converged"):
        r = results["raw_pixels_505restricted"]
        print(f"raw_pixels_505restricted:          C={r['selected_C']}, "
              f"accuracy={r['accuracy']:.4f}, log_loss={r['mean_log_loss']:.4f}")
    print("encoded_pre_evolution (1008, 505-restricted): C=0.01, accuracy=0.7208, log_loss=0.9558")
    if results.get("encoded_784_unrestricted", {}).get("converged"):
        r = results["encoded_784_unrestricted"]
        print(f"encoded_784_unrestricted (1566):   C={r['selected_C']}, "
              f"accuracy={r['accuracy']:.4f}, log_loss={r['mean_log_loss']:.4f}")

    import pickle
    out_path = os.path.join(RESULTS_DIR if not use_cuml else "/content", "class0_support_audit_classify_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
