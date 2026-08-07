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


def _unequal_fold_regression(n=253, p=25, k=6, seed=21, noise=0.1):
    """The same problem, shaped so the five folds are NOT the same size.

    `n` is not divisible by 5 and the stratification labels are
    deliberately imbalanced, which makes `StratifiedKFold` return
    validation folds of 51, 51, 51, 50, 50 rows. Every out-of-fold test
    below uses this fixture rather than the balanced one, so that
    machinery assuming a rectangular `(n_splits, n // n_splits)` block
    layout fails immediately instead of computing a plausible number.

    What this fixture does NOT add, measured rather than assumed: it is
    not what catches a write addressed by a running positional counter.
    `shuffle=True` already scatters each validation fold's indices across
    the corpus, so that bug is caught on equal folds too. The fixture's
    contribution is the fold-SIZE assumption specifically.
    `test_unequal_fold_fixture_really_has_unequal_folds` pins the size
    difference so it cannot quietly evaporate under a later edit."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    B = rng.normal(size=(p, k)) / np.sqrt(p)
    Y_raw = X @ B + rng.normal(scale=noise, size=(n, k))
    Y = (Y_raw - Y_raw.min()) / (Y_raw.max() - Y_raw.min())
    y_strat = rng.choice(5, size=n, p=[0.40, 0.30, 0.15, 0.10, 0.05])
    return X, Y, y_strat


def _folds(X, y_strat):
    """The locked partition, rebuilt independently of the module's own use
    of it, for tests that recompute a per-fold quantity from scratch."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=ridge.N_SPLITS, shuffle=True,
                          random_state=ridge.FOLD_SEED)
    return list(skf.split(X, y_strat))


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

def test_mean_x_tol_is_the_locked_anchor_and_exponent():
    """The tolerance is a design constant now, not an implementation
    choice: pin the anchor, the anchor's n, and the exponent, so a later
    edit to any of the three has to be a deliberate one."""
    assert ridge.MEAN_X_TOL_ANCHOR == 1e-9
    assert ridge.MEAN_X_TOL_ANCHOR_N == 1000
    assert ridge.MEAN_X_TOL_EXPONENT == 0.5
    assert ridge.mean_x_tol_for(1000) == 1e-9


@pytest.mark.parametrize("n,factor", [
    (250, 0.5), (1000, 1.0), (4000, 2.0), (5000, np.sqrt(5.0)),
    (16000, 4.0), (54000, np.sqrt(54.0)),
])
def test_mean_x_tol_scales_as_sqrt_n(n, factor):
    """sqrt(n), not any other power: a 16x corpus must move the tolerance
    by 4x. A constant tolerance, or a linear one, fails every row here
    except n=1,000."""
    np.testing.assert_allclose(ridge.mean_x_tol_for(n), 1e-9 * factor,
                                rtol=1e-12, atol=0)


def test_mean_x_tol_rejects_an_empty_matrix():
    with pytest.raises(ValueError, match="at least one row"):
        ridge.mean_x_tol_for(0)


def test_scaler_centering_guard_passes_on_standardized_features():
    X, _, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    norm = ridge.assert_scaler_centered(X_scaled)
    assert norm < ridge.mean_x_tol_for(X_scaled.shape[0])


def test_scaler_centering_guard_raises_on_uncentered_features():
    """Negative test: the guard must be live, not decorative. Deliberately
    un-standardized features (a large constant offset) must raise."""
    X, _, _ = _synthetic_regression()
    with pytest.raises(AssertionError, match="not centered"):
        ridge.assert_scaler_centered(X + 5.0)


@pytest.mark.parametrize("n", [200, 1000, 5000, 20000])
def test_guard_still_fires_on_an_uncentered_matrix_at_every_scale(n):
    """Detection power under the n-dependent tolerance, shown rather than
    assumed. The tolerance grows with n; a genuinely uncentered matrix
    grows not at all, so the guard must keep firing at every scale the
    ladder will reach. Offset 5.0 is nine orders above the largest
    tolerance tested here."""
    rng = np.random.default_rng(40)
    X = rng.normal(size=(n, 20)) + 5.0
    with pytest.raises(AssertionError, match="not centered"):
        ridge.assert_scaler_centered(X)


@pytest.mark.parametrize("n", [200, 1000, 5000])
def test_guard_derives_its_tolerance_from_the_matrix_it_is_given(n):
    """The tolerance in the failure message is the one for THIS matrix's
    row count -- the thing a default argument cannot do. An offset placed
    just above `mean_x_tol_for(n)` must fire, and the message must quote
    that same n-dependent value rather than some fixed number."""
    tol = ridge.mean_x_tol_for(n)
    X = np.zeros((n, 4)) + 10.0 * tol
    with pytest.raises(AssertionError, match=f"{tol:.3e}"):
        ridge.assert_scaler_centered(X)
    # and the same matrix passes when handed a tolerance wide enough
    assert ridge.assert_scaler_centered(X, tol=1e3 * tol) > 0.0


def test_guard_tolerance_is_the_folds_row_count_not_the_corpus():
    """A matrix that passes at n rows can fail at fewer, because the
    tolerance shrinks. Constructed at the boundary: an offset between
    `mean_x_tol_for(n_small)` and `mean_x_tol_for(n_large)` fires on the
    smaller matrix and passes on the larger one."""
    n_small, n_large = 1000, 5000
    lo, hi = ridge.mean_x_tol_for(n_small), ridge.mean_x_tol_for(n_large)
    assert lo < hi
    offset = float(np.sqrt(lo * hi))    # strictly between the two tolerances
    assert ridge.assert_scaler_centered(np.zeros((n_large, 1)) + offset) > 0.0
    with pytest.raises(AssertionError, match="not centered"):
        ridge.assert_scaler_centered(np.zeros((n_small, 1)) + offset)


@pytest.mark.parametrize("col_std,expect_pass", [
    (1e-2, True),      # ordinary low-variance column: fine
    (1e-4, True),      # residual amplified to ~1e-11, still inside tolerance
    (1e-9, False),     # near-constant: sklearn does not rescue it, guard fires
    (1e-15, True),     # sklearn declares it constant and sets scale to 1
])
def test_centering_guard_behaviour_on_near_constant_columns(col_std, expect_pass):
    """Characterizes an OPEN ITEM, and pins current behaviour so a later
    decision is measured against a known baseline.

    A column that is nearly-but-not-exactly constant trips the guard: it
    sits above sklearn's constant-feature bound, so it is divided by its
    tiny scale, amplifying the float64 centering residual (~9.7e-7 at
    std 1e-9) some 400x past the n=5,000 tolerance. That is a property of
    the data, not a broken scaler, which is what the guard was specified
    to catch -- so this test asserts what happens, and does not assert
    that what happens is right. The n-dependent tolerance tracks sqrt(n)
    accumulation only and does not address this regime; every verdict
    below is the same one the fixed 1e-10 tolerance gave."""
    rng = np.random.default_rng(30)
    n, p = 5000, 200
    X = rng.normal(size=(n, p))
    X[:, 3] = 1.0 + rng.normal(size=n) * col_std
    X_scaled = StandardScaler().fit_transform(X)
    if expect_pass:
        ridge.assert_scaler_centered(X_scaled)
    else:
        with pytest.raises(AssertionError, match="near-constant"):
            ridge.assert_scaler_centered(X_scaled)


def test_centering_guard_message_names_the_worst_column():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(2000, 50))
    X[:, 17] = 1.0 + rng.normal(size=2000) * 1e-9
    X_scaled = StandardScaler().fit_transform(X)
    with pytest.raises(AssertionError, match=r"worst column 17"):
        ridge.assert_scaler_centered(X_scaled)


# ---- the centering margin: the guard's statistic, returned not asserted ----

def test_margin_norm_is_bit_identical_to_the_guards_own_statistic():
    """The margin function must be the same object as the guard, not a
    parallel computation of something similar. Exact equality, not
    approximate: both take `norm(mean(axis=0))` of the same float64
    array, so any difference at all would mean they compute different
    things."""
    X, _, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    margin = ridge.scaler_centering_margin(X_scaled)
    assert margin["mean_x_norm"] == ridge.assert_scaler_centered(X_scaled)


def test_margin_reports_a_passing_matrix_without_raising():
    X, _, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    margin = ridge.scaler_centering_margin(X_scaled, X)
    tol = ridge.mean_x_tol_for(X.shape[0])
    assert margin["within_tol"] is True
    assert margin["tol"] == tol
    assert 0.0 <= margin["mean_x_norm"] < tol
    assert margin["margin_ratio"] == margin["mean_x_norm"] / tol
    assert 0 <= margin["min_col_std_col"] < X.shape[1]


def test_margin_reports_a_miscentered_matrix_instead_of_raising():
    """The same input that makes `assert_scaler_centered` raise must make
    the margin function return a number -- that difference is the whole
    point of having both."""
    X, _, _ = _synthetic_regression()
    with pytest.raises(AssertionError, match="not centered"):
        ridge.assert_scaler_centered(X + 5.0)
    margin = ridge.scaler_centering_margin(X + 5.0, X)
    assert margin["within_tol"] is False
    assert margin["mean_x_norm"] > ridge.mean_x_tol_for(X.shape[0])
    assert margin["margin_ratio"] > 1.0


@pytest.mark.parametrize("col_std,expect_within_tol", [
    (1e-2, True),
    (1e-9, False),
])
def test_margin_names_the_near_constant_column_in_both_regimes(col_std, expect_within_tol):
    """Same construction as the guard-characterization test above, at the
    column std that passes and the one that fires. `min_col_std_col`
    identifies the planted column either way, which is what makes the
    margin informative where the pass/fail boolean is not: the number
    moves continuously across a boundary the guard only reports as
    crossed or not."""
    rng = np.random.default_rng(30)
    n, p = 5000, 200
    X = rng.normal(size=(n, p))
    X[:, 3] = 1.0 + rng.normal(size=n) * col_std
    X_scaled = StandardScaler().fit_transform(X)
    margin = ridge.scaler_centering_margin(X_scaled, X)
    assert margin["min_col_std_col"] == 3
    # loose order-of-magnitude bound: an exact pin would be brittle
    # against this seed's particular realization
    assert 0.1 * col_std < margin["min_col_std"] < 10.0 * col_std
    assert margin["within_tol"] is expect_within_tol
    if not expect_within_tol:
        assert margin["worst_mean_col"] == 3


def test_margin_min_col_std_matches_the_scalers_own_variance():
    """Principle 16: this recomputes a statistic `StandardScaler` already
    measured, so it must agree with the scaler's own `sqrt(var_)` -- the
    raw std -- and NOT with `scale_`, which substitutes 1.0 for columns
    sklearn declares constant and would therefore hide exactly the
    lowest-variance column this diagnostic exists to name.

    Column 11 is exactly constant, which is the case that separates the
    two: sklearn rescues it to `scale_ = 1.0` while its true std is 0.
    Column 12 is near-constant at std 1e-9, which sklearn does NOT rescue
    -- the regime the centering guard's tension is about."""
    rng = np.random.default_rng(33)
    X = rng.normal(size=(400, 30))
    X[:, 11] = 1.0
    X[:, 12] = 1.0 + rng.normal(size=400) * 1e-9
    scaler = StandardScaler().fit(X)
    margin = ridge.scaler_centering_margin(scaler.transform(X), X)

    np.testing.assert_allclose(margin["min_col_std"], np.sqrt(scaler.var_).min(),
                                rtol=1e-12, atol=0)
    assert margin["min_col_std_col"] == 11
    assert margin["min_col_std"] == 0.0
    # scale_ reports ~1 for that same column and so names a different,
    # higher-variance column as the minimum -- the concrete reason this
    # diagnostic reads var_ rather than scale_
    assert scaler.scale_[11] == 1.0
    assert int(np.argmin(scaler.scale_)) == 12
    # the near-constant column is NOT rescued: scale_ tracks its tiny std
    assert scaler.scale_[12] < 1e-8


def test_margin_omits_raw_column_std_when_unscaled_features_not_supplied():
    """Raw column spread cannot be recovered from a standardized matrix,
    so it is reported as absent rather than as some derived stand-in."""
    X, _, _ = _synthetic_regression()
    margin = ridge.scaler_centering_margin(StandardScaler().fit_transform(X))
    assert np.isnan(margin["min_col_std"])
    assert margin["min_col_std_col"] == -1


def test_margin_rejects_mismatched_raw_and_scaled_shapes():
    X, _, _ = _synthetic_regression()
    X_scaled = StandardScaler().fit_transform(X)
    with pytest.raises(ValueError, match="does not match"):
        ridge.scaler_centering_margin(X_scaled, X[:, :5])


def test_margin_does_not_drop_or_alter_any_column():
    """The near-constant columns this diagnostic names are explicitly kept
    -- they carry small-but-nonzero variance. A reported column count that
    ever differed from the input's would mean feature selection had
    entered a path specified to have none."""
    rng = np.random.default_rng(34)
    X = rng.normal(size=(300, 25))
    X[:, 7] = 1.0 + rng.normal(size=300) * 1e-9
    X_scaled = StandardScaler().fit_transform(X)
    fit = ridge.svd_ridge_fit(X_scaled, rng.uniform(0, 1, size=(300, 4)),
                              check_centered=False)
    assert fit["W"].shape[1] == X.shape[1]


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
    assert np.all(r1["fold_mean_x_norm"] < r1["fold_mean_x_tol"])


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

def test_cross_validate_alpha_records_the_margin_every_fold():
    X, Y, y = _synthetic_regression(n=250, p=25, k=6, seed=6)
    result = ridge.cross_validate_alpha(X, Y, y)
    assert result["fold_min_col_std"].shape == (5,)
    assert result["fold_min_col_std_col"].shape == (5,)
    assert result["fold_worst_mean_col"].shape == (5,)
    assert result["fold_mean_x_tol"].shape == (5,)
    # the tolerance is per-fold, and there is no corpus-level scalar that
    # could be mistaken for the value any fold was checked against
    assert "mean_x_tol" not in result
    # every fold reported a real column, none a placeholder
    assert np.all(result["fold_min_col_std"] > 0)
    assert np.all((result["fold_min_col_std_col"] >= 0)
                  & (result["fold_min_col_std_col"] < X.shape[1]))
    assert np.all((result["fold_worst_mean_col"] >= 0)
                  & (result["fold_worst_mean_col"] < X.shape[1]))


def test_cross_validate_margin_agrees_with_recomputing_it_on_the_same_folds():
    """The recorded per-fold numbers must be the training folds' own, not
    whole-corpus values standing in for them: recompute from an
    independently constructed splitter using the locked fold seed."""
    from sklearn.model_selection import StratifiedKFold

    X, Y, y = _synthetic_regression(n=250, p=25, k=6, seed=6)
    result = ridge.cross_validate_alpha(X, Y, y)
    skf = StratifiedKFold(n_splits=ridge.N_SPLITS, shuffle=True,
                          random_state=ridge.FOLD_SEED)
    for f, (tr, _va) in enumerate(skf.split(X, y)):
        expected = ridge.scaler_centering_margin(
            StandardScaler().fit(X[tr]).transform(X[tr]), X[tr])
        assert result["fold_min_col_std"][f] == expected["min_col_std"]
        assert result["fold_min_col_std_col"][f] == expected["min_col_std_col"]
        assert result["fold_worst_mean_col"][f] == expected["worst_mean_col"]
        assert result["fold_mean_x_norm"][f] == expected["mean_x_norm"]
        assert result["fold_mean_x_tol"][f] == expected["tol"]


def test_cross_validate_tolerance_is_the_training_folds_own_row_count():
    """The tolerance recorded per fold is `mean_x_tol_for(len(tr))` -- the
    rows the guard actually saw -- not `mean_x_tol_for(len(X))`. At n=250
    a training fold is 200 rows, so the two differ by sqrt(5/4) and this
    test separates them; computing the tolerance from the corpus size
    would report a threshold 1.118x looser than the one enforced."""
    from sklearn.model_selection import StratifiedKFold

    X, Y, y = _synthetic_regression(n=250, p=25, k=6, seed=6)
    result = ridge.cross_validate_alpha(X, Y, y)
    corpus_tol = ridge.mean_x_tol_for(len(X))
    skf = StratifiedKFold(n_splits=ridge.N_SPLITS, shuffle=True,
                          random_state=ridge.FOLD_SEED)
    for f, (tr, _va) in enumerate(skf.split(X, y)):
        assert len(tr) < len(X)
        assert result["fold_mean_x_tol"][f] == ridge.mean_x_tol_for(len(tr))
        assert result["fold_mean_x_tol"][f] < corpus_tol


def test_cross_validate_margin_does_not_disturb_selection_or_scores():
    """The load-bearing constraint: the margin is instrumentation. Every
    quantity the selection path depends on must be identical to what a
    margin-free recomputation of the same folds produces."""
    from sklearn.model_selection import StratifiedKFold

    X, Y, y = _synthetic_regression(n=250, p=25, k=6, seed=6)
    result = ridge.cross_validate_alpha(X, Y, y)
    skf = StratifiedKFold(n_splits=ridge.N_SPLITS, shuffle=True,
                          random_state=ridge.FOLD_SEED)
    fold_clipped = np.empty((ridge.N_SPLITS, len(ridge.ALPHA_GRID)))
    for f, (tr, va) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        fit = ridge.svd_ridge_fit(scaler.transform(X[tr]), Y[tr])
        fold_clipped[f] = ridge.mse_per_alpha(fit, scaler.transform(X[va]), Y[va])
    np.testing.assert_array_equal(result["fold_clipped_val_mse"], fold_clipped)
    assert result["alpha"] == ridge.select_alpha(fold_clipped.mean(axis=0))[0]


def test_fit_final_attaches_the_margin_and_keeps_its_two_value_return():
    """A third return value would silently break every existing caller, so
    the margin rides on the fit dict instead."""
    X, Y, y = _synthetic_regression(n=200, p=20, k=5, seed=9)
    cv = ridge.cross_validate_alpha(X, Y, y)
    fit, scaler = ridge.fit_final(X, Y, cv["alpha"])
    margin = fit["centering_margin"]
    assert margin["mean_x_norm"] == fit["mean_x_norm"]
    assert margin["within_tol"] is True
    assert 0 <= margin["min_col_std_col"] < X.shape[1]
    np.testing.assert_allclose(margin["min_col_std"], np.sqrt(scaler.var_).min(),
                                rtol=1e-12, atol=0)


def test_svd_ridge_fit_alone_does_not_claim_a_margin():
    """`svd_ridge_fit` never sees the unscaled features, so it cannot
    report `min_col_std`; the key is absent rather than nan-filled, so a
    caller cannot mistake a direct fit for an instrumented one."""
    X, Y, _ = _synthetic_regression(n=150, p=15, k=4, seed=13)
    fit = ridge.svd_ridge_fit(StandardScaler().fit_transform(X), Y)
    assert "centering_margin" not in fit


@pytest.mark.parametrize("check_centered", [True, False])
def test_svd_ridge_fit_records_the_tolerance_it_was_checked_against(check_centered):
    """The threshold is no longer a greppable module constant, so the fit
    carries it. It is derived from the row count, which is known whether
    or not the guard ran -- so `mean_x_tol` is always a real number, and
    only `mean_x_norm` goes nan when the check is skipped."""
    X, Y, _ = _synthetic_regression(n=150, p=15, k=4, seed=13)
    fit = ridge.svd_ridge_fit(StandardScaler().fit_transform(X), Y,
                              check_centered=check_centered)
    assert fit["mean_x_tol"] == ridge.mean_x_tol_for(150)
    assert bool(np.isnan(fit["mean_x_norm"])) == (not check_centered)


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


# ---- out-of-fold per-image error (AUDIT_PROTOCOL.md's prediction basis) ----

def test_unequal_fold_fixture_really_has_unequal_folds():
    """The anti-vacuity measure the OOF tests below rest on: equal folds
    would let a rectangular `(n_splits, n // n_splits)` block layout be
    assumed anywhere in the out-of-fold path and still produce a
    well-shaped answer."""
    X, _Y, y = _unequal_fold_regression()
    sizes = sorted(len(va) for _tr, va in _folds(X, y))
    assert sizes == [50, 50, 51, 51, 51]
    assert len(set(sizes)) > 1


def test_oof_per_fold_mean_reproduces_the_stored_fold_clipped_val_mse():
    """The required cross-check on the new machinery, and the most
    important test in this file.

    `cross_validate_alpha` already records `fold_clipped_val_mse`, the
    fold-aggregate numbers the ladder's stage-1 and stage-2 artifacts
    carry. Averaging the new per-image out-of-fold values over each
    validation fold must reproduce them -- pinning new code against
    numbers this project already trusts rather than against fresh
    expectations.

    The two paths run independent fold loops over the same leaves, which
    is what gives this agreement content; deriving one from the other
    would make it a tautology.

    Exact equality, not a tolerance: both reduce a contiguous float64
    vector of the same values in the same order, so any difference at all
    would mean the reduction, the fit, or the fold addressing had
    changed."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    oof = ridge.oof_per_image_mse(X, Y, y)
    per_fold = np.array([[np.mean(oof["oof_clipped_mse"][va, a])
                          for a in range(len(ridge.ALPHA_GRID))]
                         for _tr, va in _folds(X, y)])
    np.testing.assert_array_equal(per_fold, cv["fold_clipped_val_mse"])


def test_oof_per_fold_mean_reproduces_the_stored_fold_raw_val_mse():
    """The same cross-check on the unclipped diagnostic. Both are recorded,
    so both are pinned -- a clipped/raw swap in the OOF accumulation would
    otherwise be caught in only one direction."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    oof = ridge.oof_per_image_mse(X, Y, y)
    per_fold = np.array([[np.mean(oof["oof_raw_mse"][va, a])
                          for a in range(len(ridge.ALPHA_GRID))]
                         for _tr, va in _folds(X, y)])
    np.testing.assert_array_equal(per_fold, cv["fold_raw_val_mse"])


def test_oof_covers_every_image_exactly_once():
    """Each image appears in exactly one validation fold, so the recorded
    fold membership must partition `range(n)`: no image counted twice, no
    image left unwritten, every value finite."""
    X, Y, y = _unequal_fold_regression()
    oof = ridge.oof_per_image_mse(X, Y, y)
    n, n_alpha = X.shape[0], len(ridge.ALPHA_GRID)
    assert oof["oof_clipped_mse"].shape == (n, n_alpha)
    assert oof["oof_raw_mse"].shape == (n, n_alpha)
    assert oof["fold_index"].shape == (n,)
    assert np.all(np.isfinite(oof["oof_clipped_mse"]))
    assert np.all(np.isfinite(oof["oof_raw_mse"]))
    counts = np.bincount(oof["fold_index"], minlength=ridge.N_SPLITS)
    assert counts.sum() == n
    assert sorted(counts.tolist()) == [50, 50, 51, 51, 51]
    for f, (_tr, va) in enumerate(_folds(X, y)):
        np.testing.assert_array_equal(np.flatnonzero(oof["fold_index"] == f), va)


def test_oof_partition_guard_fires_when_a_fold_is_missed(monkeypatch):
    """The coverage guard must be live, not decorative: a splitter that
    yields four folds instead of five leaves a block of images with no
    out-of-fold prediction at all, and that must raise rather than reach a
    caller as silent nan."""
    from sklearn.model_selection import StratifiedKFold as _RealSKF

    class _DropsTheLastFold:
        def __init__(self, **kwargs):
            self._inner = _RealSKF(**kwargs)

        def split(self, X, y):
            return iter(list(self._inner.split(X, y))[:-1])

    monkeypatch.setattr(ridge, "StratifiedKFold", _DropsTheLastFold)
    X, Y, y = _unequal_fold_regression()
    with pytest.raises(RuntimeError, match="not a partition"):
        ridge.oof_per_image_mse(X, Y, y)


def test_oof_is_genuinely_out_of_sample_not_a_refit():
    """A fold-rotation bug that fitted on the validation rows would make the
    out-of-fold error match, or beat, the in-sample error of a model
    fitted on everything. At the weakest alpha on this fixture the
    out-of-fold error is ~1.27x the in-sample error, which is the sign
    that the held-out rows were genuinely held out."""
    X, Y, y = _unequal_fold_regression()
    oof = ridge.oof_per_image_mse(X, Y, y)
    X_scaled = StandardScaler().fit_transform(X)
    in_sample = ridge.mse_per_alpha(ridge.svd_ridge_fit(X_scaled, Y),
                                    X_scaled, Y, clipped=True)
    assert float(np.mean(oof["oof_clipped_mse"][:, 0])) > 1.05 * in_sample[0]


def test_oof_clipped_and_raw_actually_differ():
    """If clipping were a no-op on this fixture, the clipped cross-check
    above would be checking the raw quantity under another name."""
    X, Y, y = _unequal_fold_regression()
    oof = ridge.oof_per_image_mse(X, Y, y)
    assert np.max(np.abs(oof["oof_clipped_mse"] - oof["oof_raw_mse"])) > 1e-12
    assert np.all(oof["oof_clipped_mse"] <= oof["oof_raw_mse"] + 1e-12)


def test_oof_at_a_single_alpha_matches_that_column_of_the_grid():
    """AUDIT_PROTOCOL.md's fixed-alpha regime calls with one alpha, so
    nothing may hard-code the nine-element grid. The single-alpha run must
    reproduce the grid run's corresponding column exactly -- same fit,
    same folds, same helpers."""
    X, Y, y = _unequal_fold_regression()
    grid = ridge.oof_per_image_mse(X, Y, y)
    idx = 4
    one = ridge.oof_per_image_mse(X, Y, y, alphas=(ridge.ALPHA_GRID[idx],))
    assert one["oof_clipped_mse"].shape == (X.shape[0], 1)
    np.testing.assert_array_equal(one["oof_clipped_mse"][:, 0],
                                  grid["oof_clipped_mse"][:, idx])
    np.testing.assert_array_equal(one["oof_raw_mse"][:, 0],
                                  grid["oof_raw_mse"][:, idx])
    np.testing.assert_array_equal(one["fold_index"], grid["fold_index"])


def test_oof_clipped_mse_at_alpha_returns_the_bare_vector():
    """The named entry point for the fixed-alpha regime returns the `(n,)`
    vector itself, and it is the same numbers the grid path produces --
    not a separately written fold loop."""
    X, Y, y = _unequal_fold_regression()
    idx = 4
    vec = ridge.oof_clipped_mse_at_alpha(X, Y, y, ridge.ALPHA_GRID[idx])
    assert vec.shape == (X.shape[0],)
    grid = ridge.oof_per_image_mse(X, Y, y)
    np.testing.assert_array_equal(vec, grid["oof_clipped_mse"][:, idx])


def test_oof_records_the_partition_it_is_out_of_fold_with_respect_to():
    X, Y, y = _unequal_fold_regression()
    oof = ridge.oof_per_image_mse(X, Y, y)
    assert oof["n_splits"] == ridge.N_SPLITS
    assert oof["random_state"] == ridge.FOLD_SEED
    np.testing.assert_array_equal(oof["alphas"],
                                  np.asarray(ridge.ALPHA_GRID, dtype=np.float64))


def test_oof_is_reproducible():
    X, Y, y = _unequal_fold_regression()
    a = ridge.oof_per_image_mse(X, Y, y)
    b = ridge.oof_per_image_mse(X, Y, y)
    np.testing.assert_array_equal(a["oof_clipped_mse"], b["oof_clipped_mse"])
    np.testing.assert_array_equal(a["fold_index"], b["fold_index"])


# ---- per-fold conditioning, rank and coefficient-size diagnostics ----

def test_fold_singular_values_reproduce_the_recorded_condition_ratio():
    """`fold_cond` is `s_max / s_min`, and the two magnitudes are now
    recorded separately precisely because that ratio cannot be
    un-divided."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    assert cv["fold_s_max"].shape == (ridge.N_SPLITS,)
    assert cv["fold_s_min"].shape == (ridge.N_SPLITS,)
    np.testing.assert_array_equal(cv["fold_s_max"] / cv["fold_s_min"], cv["fold_cond"])
    assert np.all(cv["fold_s_min"] > 0)


def test_s_max_and_s_min_match_an_independent_svd_of_the_same_fold():
    """The recorded magnitudes are the training fold's own, computed from
    the standardized matrix the production path decomposes -- checked
    against a numpy SVD built independently on the same fold rather than
    against the module's own decomposition."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    for f, (tr, _va) in enumerate(_folds(X, y)):
        s = np.linalg.svd(StandardScaler().fit(X[tr]).transform(X[tr]),
                          compute_uv=False)
        np.testing.assert_allclose(cv["fold_s_max"][f], s.max(), rtol=1e-12, atol=0)
        np.testing.assert_allclose(cv["fold_s_min"][f], s.min(), rtol=1e-12, atol=0)


def test_kappa_alpha_is_the_regularized_conditioning_at_every_alpha():
    """`(s_max^2 + alpha) / (s_min^2 + alpha)` -- the squared magnitudes,
    not the raw ones. Recomputed here from the recorded `s_max`/`s_min`
    for the whole grid, so an un-squared or un-shifted form fails."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    assert cv["fold_kappa_by_alpha"].shape == (ridge.N_SPLITS, len(ridge.ALPHA_GRID))
    alphas = np.asarray(ridge.ALPHA_GRID, dtype=np.float64)
    expected = ((cv["fold_s_max"][:, None] ** 2 + alphas[None, :])
                / (cv["fold_s_min"][:, None] ** 2 + alphas[None, :]))
    np.testing.assert_allclose(cv["fold_kappa_by_alpha"], expected, rtol=1e-13, atol=0)
    # and it is NOT the unregularized ratio the module already recorded
    assert np.all(np.abs(cv["fold_kappa_by_alpha"][:, 0] - cv["fold_cond"]) > 1e-6)


def test_kappa_alpha_collapses_to_cond_squared_at_alpha_zero():
    """The zero-alpha limit, checked directly: with no regularization the
    regularized conditioning is exactly the squared spectral ratio. A
    formula that forgot to square would land on `fold_cond` instead."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y, alphas=(0.0, 1e6))
    np.testing.assert_allclose(cv["fold_kappa_by_alpha"][:, 0], cv["fold_cond"] ** 2,
                               rtol=1e-12, atol=0)
    assert np.all(cv["fold_kappa_by_alpha"][:, 0] > cv["fold_cond"])


def test_kappa_alpha_decreases_toward_one_as_alpha_grows():
    """Heavier regularization can only improve the conditioning of the
    system actually solved, and drives it to 1 in the limit."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    kappa = cv["fold_kappa_by_alpha"]
    assert np.all(np.diff(kappa, axis=1) < 0)
    assert np.all(kappa[:, -1] < 1.001)     # alpha = 1e6 >> s_max^2 ~ 3.6e2
    assert np.all(kappa[:, -1] >= 1.0)


def test_kappa_and_coef_norm_are_column_indexed_at_the_selected_alpha():
    """The selected alpha is not known inside the fold loop, so both
    quantities are computed for all nine alphas and column-indexed after
    selection. This fixture selects an INTERIOR index deliberately: at
    index 0 an off-by-one in that indexing would be invisible, and index 0
    is exactly what `fit_final` hard-codes for its own unrelated
    reason."""
    X, Y, y = _unequal_fold_regression(noise=1.0)
    cv = ridge.cross_validate_alpha(X, Y, y)
    idx = cv["alpha_index"]
    assert 0 < idx < len(ridge.ALPHA_GRID) - 1
    np.testing.assert_array_equal(cv["fold_kappa_alpha"], cv["fold_kappa_by_alpha"][:, idx])
    np.testing.assert_array_equal(cv["fold_coef_norm"], cv["fold_coef_norm_by_alpha"][:, idx])
    # neither is the first column standing in for the selected one
    assert not np.array_equal(cv["fold_kappa_alpha"], cv["fold_kappa_by_alpha"][:, 0])
    assert not np.array_equal(cv["fold_coef_norm"], cv["fold_coef_norm_by_alpha"][:, 0])


def test_coef_norm_is_the_frobenius_norm_of_the_fitted_coefficients():
    """Frobenius `||W||` of the (p, k) coefficient block, per fold and per
    alpha -- recomputed from an independently constructed fit on the same
    folds, since the fits themselves are deliberately not retained."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    assert cv["fold_coef_norm_by_alpha"].shape == (ridge.N_SPLITS, len(ridge.ALPHA_GRID))
    for f, (tr, _va) in enumerate(_folds(X, y)):
        fit = ridge.svd_ridge_fit(StandardScaler().fit(X[tr]).transform(X[tr]), Y[tr])
        for a in range(len(ridge.ALPHA_GRID)):
            assert cv["fold_coef_norm_by_alpha"][f, a] == np.linalg.norm(fit["W"][a])
    # and it shrinks with alpha, so the grid is not a repeated single value
    assert np.all(np.diff(cv["fold_coef_norm_by_alpha"], axis=1) < 0)


def test_numerical_rank_detects_duplicated_columns():
    """The discriminating case: ten duplicated columns make a 40-column
    design matrix of true rank 30, and the scale-aware threshold must
    report 30 -- the same answer `numpy.linalg.matrix_rank` gives, and one
    less than a threshold without the `max(n, p)` factor produces on this
    data."""
    X, Y, y = _synthetic_regression(n=253, p=30, k=8, seed=2, duplicate_columns=10)
    assert X.shape[1] == 40
    cv = ridge.cross_validate_alpha(X, Y, y)
    assert cv["fold_numerical_rank"].dtype == np.int64
    np.testing.assert_array_equal(cv["fold_numerical_rank"], np.full(ridge.N_SPLITS, 30))
    for tr, _va in _folds(X, y):
        X_tr = StandardScaler().fit(X[tr]).transform(X[tr])
        assert np.linalg.matrix_rank(X_tr) == 30


def test_numerical_rank_is_full_on_a_well_conditioned_problem():
    """The other side of the same claim: a full-rank design must report
    `p`, or the threshold is discarding genuine directions."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    np.testing.assert_array_equal(cv["fold_numerical_rank"],
                                  np.full(ridge.N_SPLITS, X.shape[1]))


def test_rank_threshold_is_the_stated_scale_aware_rule():
    """The rule in `rank_threshold_rule` is the rule the code applies, at
    each fold's own `s_max` and training row count."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    assert cv["rank_threshold_rule"] == "s_i > eps * s_max * max(n_train, p)"
    assert cv["fold_rank_threshold"].shape == (ridge.N_SPLITS,)
    eps = float(np.finfo(np.float64).eps)
    for f, (tr, _va) in enumerate(_folds(X, y)):
        expected = eps * cv["fold_s_max"][f] * max(len(tr), X.shape[1])
        assert cv["fold_rank_threshold"][f] == expected


def test_rank_threshold_uses_the_training_folds_row_count_not_the_corpus():
    """`max(n, p)` is the training fold's row count, not the corpus's. At
    n=253 a training fold is 202 or 203 rows, so the two differ and this
    separates them -- the same distinction `fold_mean_x_tol` already
    makes."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    eps = float(np.finfo(np.float64).eps)
    for f, (tr, _va) in enumerate(_folds(X, y)):
        assert len(tr) < len(X)
        corpus = eps * cv["fold_s_max"][f] * max(len(X), X.shape[1])
        assert cv["fold_rank_threshold"][f] < corpus
    assert len(set(cv["fold_rank_threshold"].tolist())) > 1


def test_diagnostics_are_per_fold_scalars_not_raw_spectra():
    """Every array returned is indexed by fold or by alpha. Raw spectra are
    deliberately absent: their length is `min(n_train, p)` and the folds
    differ in size by one, so they do not form a rectangular array and
    could not be serialized as one. Derived from the returned dict rather
    than checked against a hand-written key list."""
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    allowed = {ridge.N_SPLITS, len(ridge.ALPHA_GRID)}
    for key, value in cv.items():
        if isinstance(value, np.ndarray):
            assert value.ndim <= 2, key
            assert value.shape[0] in allowed, (key, value.shape)
            if value.ndim == 2:
                assert value.shape[1] in allowed, (key, value.shape)


def test_new_diagnostics_serialize_through_the_ladder_drivers_json_default(monkeypatch):
    """The per-fold diagnostics land in the ladder's JSON gate artifact via
    `run_ladder_stage1._json_default`, so they must be types it handles --
    checked against the driver's own function, not a local restatement of
    which types those are.

    `BONSAI_COMMIT` is removed first: the driver's entry guard fires on
    `__name__ == "__main__"` OR that variable being set, so importing it
    with the variable present -- which is how the Colab launch path runs
    it -- would start a real ladder run from inside a unit test. This file
    sorts before `test_stage2b_ladder_stage1.py` in the Stage 2B target,
    so this is the import that reaches the module first and the guard has
    to live here.

    The out-of-fold arrays are `(n, n_alpha)` and are deliberately NOT
    part of this return value: at the production corpus they would turn a
    small gate artifact into a ~100 MB JSON document."""
    import json
    import importlib

    monkeypatch.delenv("BONSAI_COMMIT", raising=False)
    driver = importlib.import_module("run_ladder_stage1")
    X, Y, y = _unequal_fold_regression()
    cv = ridge.cross_validate_alpha(X, Y, y)
    loaded = json.loads(json.dumps(cv, default=driver._json_default, sort_keys=True))
    assert loaded["rank_threshold_rule"] == cv["rank_threshold_rule"]
    assert loaded["fold_numerical_rank"] == cv["fold_numerical_rank"].tolist()
    np.testing.assert_allclose(loaded["fold_kappa_alpha"], cv["fold_kappa_alpha"])
    assert "oof_clipped_mse" not in cv
