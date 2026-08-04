"""
Tests for experiments/stage2a_dynamics_classification/stage2a_classifier.py.

Tier 1 only (self-contained, always run, synthetic data): the locked
C_GRID's exact values/order, the deterministic smaller-C tie-break in
select_C_via_cv, and that NonConvergenceError actually raises (not just
logs) rather than being silently absorbed -- DESIGN.md's locked
stop-gate.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
sys.path.insert(0, str(_STAGE2A_DIR))

import stage2a_classifier as clf  # noqa: E402


def test_c_grid_locked_values_and_order():
    assert clf.C_GRID == [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]


def test_select_c_deterministic_smaller_c_tie_break():
    """Hand-construct a near-tie: patch _fit_one via a wrapper so two
    adjacent C values report identical mean_val_loss, and confirm the
    smaller one wins -- not whichever happens to be visited last."""
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2,
                                n_informative=5, random_state=0)

    real_fit_one = clf._fit_one

    def patched_fit_one(X_tr, y_tr, C):
        # Force C=0.01 and C=0.1 to behave identically by routing both
        # through the same underlying fit (C=0.01's), so their mean
        # validation loss ties exactly -- the only way that can happen
        # without depending on real numerical luck.
        if C == 0.1:
            C = 0.01
        return real_fit_one(X_tr, y_tr, C)

    clf._fit_one = patched_fit_one
    try:
        best_C, mean_val_loss, non_convergence = clf.select_C_via_cv(X, y, "tie_break_test")
    finally:
        clf._fit_one = real_fit_one

    assert not non_convergence
    assert abs(mean_val_loss[0.01] - mean_val_loss[0.1]) < 1e-12
    # 0.01 and 0.1 are now tied for best (or worse than another C, but if
    # either is the minimum, 0.01 -- the smaller -- must be selected).
    min_loss = min(mean_val_loss.values())
    if abs(mean_val_loss[0.01] - min_loss) < 1e-12:
        assert best_C == 0.01, "tie-break did not select the smaller C"


def test_nonconvergence_error_actually_raises():
    """A pathological, tiny, high-dimensional separable dataset at
    max_iter forced very low should fail to converge -- confirms
    NonConvergenceError is a real raise, not a silently-logged event."""
    X, y = make_classification(n_samples=40, n_features=30, n_classes=2,
                                n_informative=20, n_redundant=0,
                                n_clusters_per_class=1, class_sep=5.0,
                                random_state=0)

    original_kwargs = dict(clf.CLASSIFIER_KWARGS)
    clf.CLASSIFIER_KWARGS["max_iter"] = 1
    try:
        with pytest.raises(clf.NonConvergenceError):
            clf.select_C_via_cv(X, y, "forced_nonconvergence_test")
    finally:
        clf.CLASSIFIER_KWARGS.clear()
        clf.CLASSIFIER_KWARGS.update(original_kwargs)
