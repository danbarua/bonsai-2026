"""
The exact remote GPU-session driver that produced the class-0-support
audit's classifier numbers behind FINDINGS.md's "The class-0-support
audit: how much information the projection actually removes" section
(the raw_pixels_505restricted / encoded_784_unrestricted rows of that
section's baseline table).

Committed here for the same reason as `stage3_gpu_evolve.py` /
`stage4_gpu_evolve.py`: it produced numbers that are already reported
in FINDINGS.md, so it belongs in the reproducible record even though it
was originally written and run directly on a Colab kernel via
`mighty-colab exec`, not invoked locally.

Distinct from the committed local script `run_class0_support_audit_classify.py`
(its `--cuml` flag activates `cuml.accel` but still reads inputs via
`stage2a_paths.scratch_root()`, i.e. a local filesystem path -- that
combination, with `STAGE2A_SCRATCH_ROOT` pointed at `/content`, has
never actually been run remotely). This script is the one that *was*
actually run remotely, verified end to end: it downloads its six input
`.npy` files directly from the public GCS mirror
(`gs://bonsai-2026-stage2a-cache/class0_support_audit/`, cloud-to-cloud,
no local upload needed for those) rather than reading through
`stage2a_paths`, matching `stage3_gpu_evolve.py`/`stage4_gpu_evolve.py`'s
own `/content`-hardcoded convention for remote-only drivers. Only
`stage2a_classifier.py` and `stage2a_stats.py` need uploading alongside
it (see `Makefile`'s `stage2a-class0-classify-gpu` target for the exact
sequence) -- both used unmodified, no reimplementation.
"""
import sys
sys.path.insert(0, '/content')
import os
import time
import urllib.request
import pickle

import numpy as np

import cuml.accel
cuml.accel.install()

import stage2a_classifier as s2a_clf
from stage2a_classifier import select_C_via_cv, fit_final_at_selected_C, NonConvergenceError
from stage2a_stats import per_image_log_loss

# cuml.accel-specific override, disclosed -- see CUML_ACCEL_FINDINGS.md
# for why sklearn's own max_iter=10000 isn't necessarily sufficient
# under this backend's different convergence footprint.
CUML_MAX_ITER = 20000
s2a_clf.CLASSIFIER_KWARGS["max_iter"] = CUML_MAX_ITER
print(f"cuml.accel active, max_iter={CUML_MAX_ITER}")

BASE = "https://storage.googleapis.com/bonsai-2026-stage2a-cache/class0_support_audit"
files = ["X_train_raw505.npy", "X_test_raw505.npy", "X_train_encoded784.npy",
         "X_test_encoded784.npy", "y_train.npy", "y_test.npy"]
for f in files:
    if os.path.exists(f"/content/{f}"):
        continue
    t0 = time.perf_counter()
    urllib.request.urlretrieve(f"{BASE}/{f}", f"/content/{f}")
    print(f"Downloaded {f} in {time.perf_counter()-t0:.1f}s")

X_train_raw505 = np.load('/content/X_train_raw505.npy').astype(np.float64)
X_test_raw505 = np.load('/content/X_test_raw505.npy').astype(np.float64)
X_train_encoded784 = np.load('/content/X_train_encoded784.npy').astype(np.float64)
X_test_encoded784 = np.load('/content/X_test_encoded784.npy').astype(np.float64)
y_train = np.load('/content/y_train.npy')
y_test = np.load('/content/y_test.npy')
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
    print(f"  mean_val_loss_per_C={mean_val_loss}")

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
        "mean_val_loss_per_C": mean_val_loss,
    }

print(f"\n{'='*70}\nSUMMARY (vs. already-known confirmatory numbers)\n{'='*70}")
print("raw_pixels (784, full):                       C=0.001, accuracy=0.6960, log_loss=0.9848")
if results.get("raw_pixels_505restricted", {}).get("converged"):
    r = results["raw_pixels_505restricted"]
    print(f"raw_pixels_505restricted:                     C={r['selected_C']}, "
          f"accuracy={r['accuracy']:.4f}, log_loss={r['mean_log_loss']:.4f}")
print("encoded_pre_evolution (1008, 505-restricted): C=0.01,  accuracy=0.7208, log_loss=0.9558")
if results.get("encoded_784_unrestricted", {}).get("converged"):
    r = results["encoded_784_unrestricted"]
    print(f"encoded_784_unrestricted (1566):              C={r['selected_C']}, "
          f"accuracy={r['accuracy']:.4f}, log_loss={r['mean_log_loss']:.4f}")

with open('/content/class0_support_audit_classify_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nSaved /content/class0_support_audit_classify_results.pkl")
