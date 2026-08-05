"""
Tests for experiments/stage2b_denoising/stage2b_ridge.py -- the
intercept-aware SVD ridge production path, its alpha-selection rule, and
the sklearn verification oracle.

Tier 1 (self-contained, always run) only: Stage 2B has no historical
cached artifact to verify against, so there is nothing for a Tier 2
skip-if-absent test to check yet. Every test here is synthetic.

The equivalence checks are the point of the file. DESIGN.md's gate at
the 1,000- and 5,000-image ladder stages is max abs difference in
clipped validation predictions <= 1e-8 against
`Ridge(solver="svd", fit_intercept=True)`, plus identical alpha
selection; these tests establish that the implementation can meet it on
synthetic data, including the rank-deficient case where the two paths'
formulas genuinely differ (DESIGN's formula takes the SVD of
standardized-but-uncentered X and centers only Y; sklearn centers both
and applies a singular-value mask). The real-data gate is a later
integration step and is not attempted here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_ridge as ridge  # noqa: E402


def _synthetic_regression(n=300, p=40, k=12, seed=0, duplicate_columns=0):
    """A multi-output regression problem whose targets live in [0, 1] --
    the range the locked clipped-MSE selection rule actually operates on.
    Standard-normal targets would make `clip(x_hat, 0, 1)` mangle
    everything and the selection test would exercise nothing real."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if duplicate_columns:
        X = np.hstack([X, X[:, :duplicate_columns]])
    B = rng.normal(size=(X.shape[1], k)) / np.sqrt(X.shape[1])
    Y_raw = X @ B + rng.normal(scale=0.1, size=(n, k))
    # squash into [0, 1] so clipping is meaningful rather than destructive
    Y = (Y_raw - Y_raw.min()) / (Y_raw.max() - Y_raw.min())
    y_strat = rng.integers(0, 5, n)
    return X, Y, y_strat


# ---- float64 discipline ----

def test_x64_is_actually_enabled_not_merely_requested():
    """DESIGN.md's dtype table requires float64 for the ridge SVD. The
    `jax.config.update` call is not self-verifying -- assert the effect."""
    import jax.numpy as jnp
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64


def test_fit_outputs_are_float64():
    X, Y, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    assert fit["W"].dtype == np.float64
    assert fit["b"].dtype == np.float64


# ---- the centering guard ----

def test_scaler_centering_guard_passes_on_standardized_features():
    X, _, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    norm = ridge.assert_scaler_centered(X_scaled)
    assert norm < ridge.MEAN_X_TOL


def test_scaler_centering_guard_raises_on_uncentered_features():
    """Negative test: the guard must be live, not decorative. Deliberately
    un-standardized features (a large constant offset) must raise."""
    X, _, _ = _synthetic_regression()
    with pytest.raises(AssertionError, match="not centered"):
        ridge.assert_scaler_centered(X + 5.0)


def test_svd_ridge_fit_raises_on_uncentered_input_by_default():
    X, Y, _ = _synthetic_regression()
    with pytest.raises(AssertionError, match="not centered"):
        ridge.svd_ridge_fit(X + 5.0, Y)


# ---- the intercept formula, general expression not the shortcut ----

def test_intercept_matches_general_expression_exactly():
    """`b = mean(Y_train) - mean(X_train) @ W` is what is implemented.
    Recomputing it independently from the returned W must reproduce the
    returned b bit-for-bit-close, and (separately) it must sit very near
    mean(Y_train) after standardization -- near, not equal, which is the
    whole reason the general form is used."""
    X, Y, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    for a in range(len(fit["alphas"])):
        expected = fit["y_mean"] - fit["x_mean"] @ fit["W"][a]
        np.testing.assert_allclose(fit["b"][a], expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(fit["b"][a], Y.mean(axis=0), atol=1e-9)


def test_single_svd_reused_matches_per_alpha_independent_solves():
    """All nine alphas come from ONE decomposition. Each must agree with
    a fit done for that alpha alone -- the reuse is not introducing
    cross-alpha contamination."""
    X, Y, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    fit_all = ridge.svd_ridge_fit(X_scaled, Y)
    for a, alpha in enumerate(ridge.ALPHA_GRID):
        fit_one = ridge.svd_ridge_fit(X_scaled, Y, alphas=(alpha,))
        np.testing.assert_allclose(fit_all["W"][a], fit_one["W"][0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(fit_all["b"][a], fit_one["b"][0], rtol=1e-12, atol=1e-14)


def test_larger_alpha_shrinks_coefficients_monotonically():
    X, Y, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    norms = [np.linalg.norm(fit["W"][a]) for a in range(len(fit["alphas"]))]
    assert all(norms[i] > norms[i + 1] for i in range(len(norms) - 1))


# ---- sklearn oracle equivalence ----

def test_matches_sklearn_predictions_all_alphas_well_conditioned():
    X, Y, _ = _synthetic_regression(seed=1)
    X_scaled = StandardScaler().fit_transform(X)
    X_eval = StandardScaler().fit(X).transform(X[:50] + 0.01)
    skl_pred, skl_coef, skl_int = ridge.sklearn_ridge_predict(X_scaled, Y, X_eval)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    for a in range(len(ridge.ALPHA_GRID)):
        np.testing.assert_allclose(ridge.ridge_predict(fit, X_eval, a), skl_pred[a],
                                    rtol=0, atol=1e-10)
        np.testing.assert_allclose(fit["W"][a], skl_coef[a], rtol=0, atol=1e-10)
        np.testing.assert_allclose(fit["b"][a], skl_int[a], rtol=0, atol=1e-10)


def test_matches_sklearn_rank_deficient_all_alphas():
    """The discriminating case. With duplicated feature columns the design
    matrix is rank-deficient, which is exactly where the two paths'
    formulas differ (uncentered-X SVD with an unmasked `s/(s^2+alpha)`
    filter, versus sklearn's centered X and its ~1e-15 singular-value
    mask). Agreement here at every alpha is what makes the well-
    conditioned agreement above meaningful rather than lucky."""
    X, Y, _ = _synthetic_regression(n=120, p=30, k=8, seed=2, duplicate_columns=10)
    X_scaled = StandardScaler().fit_transform(X)
    skl_pred, _skl_coef, _ = ridge.sklearn_ridge_predict(X_scaled, Y, X_scaled)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    assert fit["cond"] > 1e10 or not np.isfinite(fit["cond"])
    for a in range(len(ridge.ALPHA_GRID)):
        np.testing.assert_allclose(ridge.ridge_predict(fit, X_scaled, a), skl_pred[a],
                                    rtol=0, atol=1e-10)


def test_matches_sklearn_wide_problem_more_features_than_samples():
    """p > n: the thin SVD has rank n, not p. Another shape where a
    naive implementation can silently transpose or truncate wrongly."""
    X, Y, _ = _synthetic_regression(n=40, p=120, k=6, seed=3)
    X_scaled = StandardScaler().fit_transform(X)
    skl_pred, _c, _i = ridge.sklearn_ridge_predict(X_scaled, Y, X_scaled)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    for a in range(len(ridge.ALPHA_GRID)):
        np.testing.assert_allclose(ridge.ridge_predict(fit, X_scaled, a), skl_pred[a],
                                    rtol=0, atol=1e-10)


def test_sklearn_coef_transposition_convention():
    """sklearn's multi-output `coef_` is (k, p); this module's W is (p, k).
    A silent transposition here would be invisible on a square problem."""
    X, Y, _ = _synthetic_regression(n=200, p=17, k=5, seed=4)  # p != k deliberately
    X_scaled = StandardScaler().fit_transform(X)
    model = Ridge(alpha=1.0, solver="svd", fit_intercept=True).fit(X_scaled, Y)
    assert model.coef_.shape == (5, 17)
    _pred, coef, _int = ridge.sklearn_ridge_predict(X_scaled, Y, X_scaled, alphas=(1.0,))
    assert coef[0].shape == (17, 5)


# ---- clipping is genuinely in the selection path ----

def test_clipped_and_raw_mse_actually_differ():
    """If clipping were a no-op on this data, every clipped-MSE test below
    would pass vacuously."""
    X, Y, _ = _synthetic_regression(seed=5)
    X_scaled = StandardScaler().fit_transform(X)
    fit = ridge.svd_ridge_fit(X_scaled, Y)
    clipped = ridge.mse_per_alpha(fit, X_scaled, Y, clipped=True)
    raw = ridge.mse_per_alpha(fit, X_scaled, Y, clipped=False)
    assert np.any(np.abs(clipped - raw) > 1e-12)
    assert np.all(clipped <= raw + 1e-12)  # clipping toward an in-range target cannot hurt


def test_clipped_per_image_mse_clips_prediction_not_target():
    pred = np.array([[2.0, -1.0]])
    target = np.array([[1.0, 0.0]])
    assert ridge.clipped_per_image_mse(pred, target)[0] == pytest.approx(0.0)
    assert ridge.per_image_mse(pred, target)[0] == pytest.approx(1.0)


# ---- alpha selection: tested as a pure function ----

def test_select_alpha_picks_plain_minimum_when_no_tie():
    alphas = (1.0, 10.0, 100.0)
    alpha, idx = ridge.select_alpha([0.5, 0.3, 0.4], alphas)
    assert alpha == 10.0 and idx == 1


def test_select_alpha_exact_tie_larger_alpha_wins():
    alphas = (1.0, 10.0, 100.0)
    alpha, _ = ridge.select_alpha([0.5, 0.3, 0.3], alphas)
    assert alpha == 100.0


def test_select_alpha_within_tolerance_tie_larger_alpha_wins():
    alphas = (1.0, 10.0, 100.0)
    alpha, _ = ridge.select_alpha([0.5, 0.3, 0.3 + 1e-12], alphas)
    assert alpha == 100.0


def test_select_alpha_outside_tolerance_smaller_alpha_wins():
    """1e-6 > the locked 1e-10 absolute tolerance: not a tie, so the
    genuine minimum wins even though it is the smaller alpha."""
    alphas = (1.0, 10.0, 100.0)
    alpha, _ = ridge.select_alpha([0.5, 0.3, 0.3 + 1e-6], alphas)
    assert alpha == 10.0


def test_select_alpha_tie_across_three_picks_largest_of_the_tied_set():
    alphas = (1.0, 10.0, 100.0, 1000.0)
    alpha, _ = ridge.select_alpha([0.3, 0.3, 0.3, 0.9], alphas)
    assert alpha == 100.0


def test_select_alpha_uses_locked_grid_and_tolerance():
    assert ridge.ALPHA_GRID == (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
    assert ridge.ALPHA_TIE_TOL == 1e-10


def test_select_alpha_rejects_non_finite_mse():
    with pytest.raises(ValueError):
        ridge.select_alpha([0.1, np.nan, 0.3], (1.0, 10.0, 100.0))


# ---- cross-validation driver ----

def test_cross_validate_alpha_reproducible_and_shapes():
    X, Y, y = _synthetic_regression(n=250, p=25, k=6, seed=6)
    r1 = ridge.cross_validate_alpha(X, Y, y)
    r2 = ridge.cross_validate_alpha(X, Y, y)
    assert r1["alpha"] == r2["alpha"]
    np.testing.assert_array_equal(r1["mean_clipped_val_mse"], r2["mean_clipped_val_mse"])
    assert r1["fold_clipped_val_mse"].shape == (5, 9)
    assert r1["fold_cond"].shape == (5,)
    assert np.all(r1["fold_mean_x_norm"] < ridge.MEAN_X_TOL)


def test_cross_validate_selects_strong_regularization_for_pure_noise():
    """No learnable signal: the most regularized model that can only
    predict the target mean should win, not the least regularized."""
    rng = np.random.default_rng(7)
    n, p, k = 200, 30, 5
    X = rng.normal(size=(n, p))
    Y = rng.uniform(0, 1, size=(n, k))
    y = rng.integers(0, 4, n)
    result = ridge.cross_validate_alpha(X, Y, y)
    assert result["alpha"] >= 1e4


def test_cross_validate_selects_weak_regularization_for_clean_signal():
    """Strong, low-noise signal with n >> p: heavy shrinkage should lose."""
    rng = np.random.default_rng(8)
    n, p, k = 600, 20, 4
    X = rng.normal(size=(n, p))
    B = rng.normal(size=(p, k)) / np.sqrt(p)
    Y_raw = X @ B + rng.normal(scale=0.01, size=(n, k))
    Y = (Y_raw - Y_raw.min()) / (Y_raw.max() - Y_raw.min())
    y = rng.integers(0, 4, n)
    result = ridge.cross_validate_alpha(X, Y, y)
    assert result["alpha"] <= 10.0


def test_fit_final_single_svd_at_selected_alpha():
    X, Y, y = _synthetic_regression(n=200, p=20, k=5, seed=9)
    cv = ridge.cross_validate_alpha(X, Y, y)
    fit, scaler = ridge.fit_final(X, Y, cv["alpha"])
    assert fit["W"].shape == (1, 20, 5)
    np.testing.assert_allclose(fit["alphas"], [cv["alpha"]])
    # the returned scaler is the one the caller must reuse on eval features
    np.testing.assert_allclose(scaler.transform(X).mean(axis=0), 0.0, atol=1e-12)


# ---- the equivalence gate itself ----

def test_ridge_equivalence_check_passes_on_synthetic_data():
    X, Y, y = _synthetic_regression(n=300, p=35, k=10, seed=10)
    result = ridge.ridge_equivalence_check(X, Y, y)
    assert result["max_abs_clipped_pred_diff"] <= ridge.EQUIVALENCE_TOL
    assert result["alpha_jax"] == result["alpha_sklearn"]
    assert result["passed"] is True


def test_ridge_equivalence_check_passes_rank_deficient():
    X, Y, y = _synthetic_regression(n=200, p=25, k=6, seed=11, duplicate_columns=12)
    result = ridge.ridge_equivalence_check(X, Y, y)
    assert result["max_abs_clipped_pred_diff"] <= ridge.EQUIVALENCE_TOL
    assert result["passed"] is True


def test_ridge_equivalence_check_reports_coefficient_diff_as_diagnostic_only():
    """Coefficient agreement is recorded but is explicitly NOT a halt
    rule -- `passed` must depend only on predictions and alpha."""
    X, Y, y = _synthetic_regression(n=200, p=25, k=6, seed=12, duplicate_columns=12)
    result = ridge.ridge_equivalence_check(X, Y, y)
    assert "max_abs_coef_diff" in result
    assert result["passed"] == (result["pred_agrees"] and result["alpha_agrees"])
