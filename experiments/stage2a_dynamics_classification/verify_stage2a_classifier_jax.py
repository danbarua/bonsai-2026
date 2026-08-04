"""
Verifies stage2a_classifier_jax.select_C_via_cv_jax against
stage2a_classifier.select_C_via_cv, and stress-tests the vmap/
lax.while_loop NaN-guard fix -- both referenced by name from
stage2a_classifier_jax.py's own module docstring and from
JAX_CLASSIFIER_PORT_FINDINGS.md, but not previously committed (a real
reproducibility gap: the docstring pointed at a file that only existed
in a local scratch directory). Synthetic data only -- never touches the
real Stage-3 cached artifacts; see
diagnose_classifier_jax_grad_norm_calibration.py for the real-data
checks that motivated GRAD_NORM_REL's recalibration.

Two checks:
  1. Correctness: per-C mean validation log-loss curves and best_C
     selection, stage2a_classifier_jax vs. stage2a_classifier, on three
     synthetic cases of increasing size/dimensionality.
  2. Robustness: repeated runs of the case that originally exposed the
     vmap/lax.while_loop NaN flakiness (see stage2a_classifier_jax.py's
     _solve_one docstring), asserting zero NonConvergenceError/NaN
     failures -- a smaller, always-run version of the 13/20-run
     CPU/GPU stress test documented in JAX_CLASSIFIER_PORT_FINDINGS.md.
"""
import os
import sys
import time

import numpy as np
from sklearn.datasets import make_classification

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_classifier as ref
import stage2a_classifier_jax as jaxclf

# Reasonable given best_C selection is exact-match in every case checked
# so far; the per-C curve itself is NOT held to this bound (see
# JAX_CLASSIFIER_PORT_FINDINGS.md -- the large-C divergence from
# sklearn is a known, unresolved gap, not something this check asserts
# away).
MAX_LOSS_DIFF_WARN = 1e-1


def make_synthetic(n_samples, n_features, n_classes, seed, class_sep=2.0):
    n_informative = min(n_features, max(n_classes * 2, n_features // 2))
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=0, n_repeated=0, n_classes=n_classes, n_clusters_per_class=1,
        class_sep=class_sep, random_state=seed)
    return X, y


def run_case(name, n_samples, n_features, n_classes, seed=0):
    print(f"\n{'='*70}\n{name}: n={n_samples}, d={n_features}, k={n_classes}\n{'='*70}")
    X, y = make_synthetic(n_samples, n_features, n_classes, seed)

    t0 = time.time()
    best_C_ref, mean_loss_ref, nc_ref = ref.select_C_via_cv(X, y, name)
    t_ref = time.time() - t0
    assert not nc_ref, f"[{name}] unexpected sklearn non-convergence on synthetic data: {nc_ref}"
    print(f"sklearn: best_C={best_C_ref}, elapsed={t_ref:.2f}s")

    t0 = time.time()
    best_C_jax, mean_loss_jax, nc_jax = jaxclf.select_C_via_cv_jax(X, y, name)
    t_jax = time.time() - t0
    assert not nc_jax, f"[{name}] unexpected JAX non-convergence on synthetic data: {nc_jax}"
    print(f"jax:     best_C={best_C_jax}, elapsed={t_jax:.2f}s")

    diffs = {C: abs(mean_loss_ref[C] - mean_loss_jax[C]) for C in mean_loss_ref}
    max_diff = max(diffs.values())
    max_diff_C = max(diffs, key=diffs.get)
    match = best_C_ref == best_C_jax
    print(f"best_C match: {match} (ref={best_C_ref}, jax={best_C_jax})")
    print(f"max |val_loss diff| across C grid: {max_diff:.4e} (at C={max_diff_C})"
          f"{' [ABOVE WARN THRESHOLD]' if max_diff > MAX_LOSS_DIFF_WARN else ''}")

    assert match, (f"[{name}] best_C selection diverged: sklearn={best_C_ref}, "
                    f"jax={best_C_jax} -- this is the one property that has held in "
                    f"every case checked so far; a real divergence here would be new.")
    return {"name": name, "max_loss_diff": max_diff, "best_C_match": match}


def stress_test_nan_guard(n_reps=10):
    """The case that originally exposed the vmap/lax.while_loop NaN
    flakiness (n=3000, d=200, k=10 -- see stage2a_classifier_jax.py's
    _solve_one docstring for the isolation trail). Asserts zero
    failures across n_reps fresh process-level calls; the underlying
    bug was nondeterministic run-to-run, not within a single call, so
    repeated *calls* (not just repeated iterations within one call) are
    what actually exercises it."""
    print(f"\n{'='*70}\nSTRESS TEST: {n_reps}x select_C_via_cv_jax on the case that exposed "
          f"the NaN flakiness (n=3000, d=200, k=10)\n{'='*70}")
    X, y = make_synthetic(3000, 200, 10, seed=0)
    n_failed = 0
    for i in range(n_reps):
        try:
            best_C, _mean_loss, nc = jaxclf.select_C_via_cv_jax(X, y, "stress_test")
            status = "OK" if not nc else f"non_convergence={nc}"
        except jaxclf.NonConvergenceError as e:
            n_failed += 1
            status = f"FAILED: {e}"
        print(f"  run {i}: {status}")
    print(f"\n{n_failed}/{n_reps} failed")
    assert n_failed == 0, (
        f"{n_failed}/{n_reps} runs hit the vmap/lax.while_loop NaN flakiness -- "
        f"the guard fix in stage2a_classifier_jax.py's _solve_one is not holding.")
    print("PASS: zero NaN-guard failures.")


def main():
    results = [
        run_case("small_2class", n_samples=400, n_features=20, n_classes=2),
        run_case("small_10class", n_samples=1200, n_features=40, n_classes=10),
        run_case("medium_10class_wide", n_samples=3000, n_features=200, n_classes=10),
    ]
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        print(f"{r['name']:<25} best_C_match={r['best_C_match']} "
              f"max_loss_diff={r['max_loss_diff']:.4e}")
    print("PASS: best_C selection matches sklearn in every synthetic case checked.")

    stress_test_nan_guard()
    return results


if __name__ == "__main__":
    main()
