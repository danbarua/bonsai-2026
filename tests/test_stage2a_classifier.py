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


class _UniformProbaClassifier:
    """Deterministically bad stand-in classifier: predict_proba always
    returns uniform probabilities regardless of input. Used to force
    every non-tied C's validation log-loss to be GUARANTEED worse than
    a real, better-than-chance fit -- not empirically likely to be
    worse, provably so, since log-loss is minimized by the true
    conditional probability and a real fit on separable synthetic data
    has genuine predictive skill. This is what makes the tied pair the
    guaranteed global minimum below, not just probably the minimum."""
    def __init__(self, classes):
        self.classes_ = classes
        self.n_iter_ = np.array([1])

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        return np.full((X.shape[0], n_classes), 1.0 / n_classes)


def test_select_c_deterministic_smaller_c_tie_break():
    """Hand-construct a near-tie THAT IS GUARANTEED TO BE THE GLOBAL
    MINIMUM, not just possibly so: C=0.01 and C=0.1 both route through
    the same real, well-separated-data fit (tied, and genuinely good --
    lower log-loss than chance); every OTHER C in the grid is forced to
    a uniform-probability dummy classifier, whose log-loss is provably
    no better than a real fit's on data with genuine class structure.
    Without this, a tie among two non-minimal C values would let the
    test pass without the tie-break logic (`min(tied)`) ever being
    exercised at all -- caught by external review on the original
    version of this test, which conditioned its own assertion on "if
    the tie happens to be the minimum," silently skipping the check
    otherwise."""
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2,
                                n_informative=8, n_redundant=0, class_sep=3.0,
                                random_state=0)
    classes = np.unique(y)

    real_fit_one = clf._fit_one

    def patched_fit_one(X_tr, y_tr, C):
        if C in (0.01, 0.1):
            # Both route through the SAME real fit (at C=0.01) -- tied,
            # and a genuine well-separated-data fit, not a dummy.
            return real_fit_one(X_tr, y_tr, 0.01)
        # Every other C: deterministically bad, never better than the
        # tied pair above.
        return _UniformProbaClassifier(classes), True, 1

    clf._fit_one = patched_fit_one
    try:
        best_C, mean_val_loss, non_convergence = clf.select_C_via_cv(X, y, "tie_break_test")
    finally:
        clf._fit_one = real_fit_one

    assert not non_convergence
    assert abs(mean_val_loss[0.01] - mean_val_loss[0.1]) < 1e-12
    # Guaranteed by construction, not conditional: the tied pair IS the
    # global minimum (every other C is a uniform-probability dummy,
    # provably no better), so the tie-break logic is actually exercised.
    min_loss = min(mean_val_loss.values())
    assert abs(mean_val_loss[0.01] - min_loss) < 1e-12, (
        "test construction failed to make the tied pair the minimum -- "
        "the dummy classifiers were not actually worse than the real fit")
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
