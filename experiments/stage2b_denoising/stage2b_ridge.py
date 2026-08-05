"""
Stage 2B's readout: the intercept-aware SVD ridge production path (JAX),
its alpha-selection rule, and the sklearn verification oracle --
implementing DESIGN.md's "Readout: multi-output ridge -- JAX SVD
production path, sklearn as oracle" section exactly.

    Y_tilde  = Y - mean(Y_train)
    W_alpha  = V @ diag(s / (s^2 + alpha)) @ U.T @ Y_tilde
    b_alpha  = mean(Y_train) - mean(X_train) @ W_alpha

One thin SVD per (fold, condition) of the standardized training
features; all nine alphas evaluated from that single decomposition. The
general intercept expression is what is implemented -- not the
`b_alpha = mean(Y_train)` shortcut that it normally reduces to after
standardization, which would silently stop being correct if the scaler
ever changed.

`assert_scaler_centered` is the guard that makes that shortcut's absence
meaningful: a broken scaler cannot quietly invent intercept structure,
because `||mean(X_train_scaled)||` is checked against `mean_x_tol_for(n)`
before any solve. `scaler_centering_margin` returns that same statistic as a number,
alongside the smallest raw per-column standard deviation and its column
index, so every fold records how far it sat from the guard and how close
its features came to the near-constant regime the guard also fires in.
Diagnostic only: it never raises, drops nothing, and no fitting
decision reads it.

**sklearn (`Ridge(solver="svd")`) is the verification oracle -- not in
the production path, never deleted.** `ridge_equivalence_check` runs
both paths through the same fold splitter, the same scaler, and the same
`select_alpha`, and reports DESIGN.md's literal gate quantities: max
absolute difference in clipped validation predictions (<= 1e-8) and
identical alpha selection.

Scope note: this module is pure functions over arrays. It loads no
dataset and knows nothing about conditions, corruption, or splits. The
42-SVD accounting in DESIGN.md (35 fold-level + 7 final refits) is a
property of the caller that loops 7 conditions over
`cross_validate_alpha` (5 SVDs each) and then `fit_final` (1 SVD each),
not of anything here.

dtype: float64 throughout, per DESIGN.md's dtype table (Stage 2A
measured ~2e6 condition numbers on evolved-feature design matrices;
float32's ~6e-8 precision at that conditioning gives ~12% worst-case
relative error).
"""
import numpy as np
import jax

# x64 must be enabled BEFORE jax.numpy is imported, per this project's
# established ordering (experiments/stage2a_dynamics_classification/
# evolve_on_graph_jax.py). Reversed, the update silently does not apply.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# The config call above is not self-verifying: assert the effect, not the
# call (CLAUDE.md principle 16 -- the call being right is not evidence
# the effect took).
if jnp.zeros(1, dtype=jnp.float64).dtype != jnp.float64:  # pragma: no cover
    raise RuntimeError(
        "jax_enable_x64 did not take effect -- ridge SVD would run in "
        "float32, which DESIGN.md's dtype table explicitly rules out.")

# ---- Locked constants (DESIGN.md, "Readout") ----
ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
ALPHA_TIE_TOL = 1e-10      # "mean validation MSE within 1e-10 absolute"
N_SPLITS = 5
FOLD_SEED = 42
EQUIVALENCE_TOL = 1e-8     # max abs clipped-validation-prediction difference

# ---- The ||mean(X_train_scaled)|| guard's tolerance (DESIGN.md, "Readout") ----
MEAN_X_TOL_ANCHOR = 1e-9   # tolerance at MEAN_X_TOL_ANCHOR_N rows
MEAN_X_TOL_ANCHOR_N = 1000
MEAN_X_TOL_EXPONENT = 0.5  # sqrt(n) -- float accumulation in a mean


def mean_x_tol_for(n):
    """The centering guard's tolerance at `n` rows:
    `1e-9 * (n / 1000) ** 0.5`.

    `n` is the row count of the matrix actually being checked -- a CV
    fold's training rows, not the ladder rung's nominal corpus size. At a
    5-fold split those differ by sqrt(5/4), and the guard is a statement
    about the matrix in front of it.

    ## Where the anchor and the exponent come from

    Both are read off the GPU-spike measurements in this directory's
    `README.md` -- the production evolution path, features encoded and
    evolved by the same kernels the ladder will use. The anchor is 1e-9
    at n=1,000: a 12.7x margin over the worst value measured there
    (`curr_random`, 7.87e-11).

    The exponent is not fitted to those points, it is the mechanism.
    `||mean(X_scaled)||` grows because the mean of n float64 values
    carries ~sqrt(n) accumulated rounding, amplified by division by a
    small column standard deviation. Taking 0.5 rather than the measured
    growth matters in the direction that protects the guard: 0.5
    upper-bounds `curr_random`'s measured 0.405, so the margin GROWS with
    n instead of eroding -- 14.8x at n=5,000 against the measured
    1.51e-10, and ~18x projected at n=54,000 (projection, from that same
    0.405 exponent; the largest corpus actually measured is 5,000).

    Headroom above, in both directions the guard has to stay useful in.
    A tolerance of 7.35e-9 at n=54,000 is still four or more orders below
    the ~3e-4 level at which `||mean(X)||` starts degrading DESIGN.md's
    1e-8 JAX-vs-sklearn equivalence gate (measured, `README.md`), and
    about nine orders below the O(1) offset a genuinely broken scaler --
    the thing the guard exists to catch -- would produce.

    The exponent 0.66 fitted from `README.md`'s first (CPU-evolved,
    n=300 -> 1,000) table is superseded and not used: it measures a
    different pipeline from the one the anchor comes from, and a slope
    from one table with an anchor from another describes neither."""
    n = int(n)
    if n < 1:
        raise ValueError(f"mean_x_tol_for needs at least one row, got n={n}")
    return MEAN_X_TOL_ANCHOR * (n / MEAN_X_TOL_ANCHOR_N) ** MEAN_X_TOL_EXPONENT


def assert_scaler_centered(X_scaled, tol=None):
    """Guard on the standardized training features: `||mean(X)|| < tol`.

    `tol` defaults to `mean_x_tol_for(X_scaled.shape[0])` -- derived from
    the matrix passed in, because the tolerance is n-dependent and a
    default argument cannot be.

    The intercept formula `b = mean(Y) - mean(X) @ W` is only benign
    because `mean(X)` is numerically zero after standardization. If a
    scaler is misconfigured (fitted on the wrong fold, `with_mean=False`,
    applied to already-scaled data), that term stops being noise and
    starts being real, invented intercept structure -- silently, since
    the ridge still fits and still produces predictions.

    KNOWN TENSION, not resolved by the n-dependent tolerance: a NEARLY-
    but-not-exactly-constant feature column also trips this. sklearn's
    `StandardScaler` only rescues a column as constant when its variance
    falls below roughly `(n * mean * eps)^2`; a column just above that
    bound is divided by its tiny scale, which amplifies the float64
    centering residual from ~1e-16 to many orders past the tolerance.
    Measured at n=5,000 with a unit-mean column: the guard passes at
    column std 1e-4 (norm 1.1e-11) and at 1e-12 and below (sklearn
    declares those constant), and FIRES across roughly
    1e-12 < std < 1e-5 (norm 9.7e-7 at std 1e-9, some 400x above the
    n=5,000 tolerance). Stage 2A's cos/sin features under a
    near-synchronized regime can plausibly land there. `mean_x_tol_for`
    tracks the sqrt(n) accumulation growth and nothing else; a
    near-constant column is a different mechanism, several orders larger,
    and still halts. That is a property of the data rather than a broken
    scaler, so it remains open. The failure message names the worst
    offending column so a halt is diagnosable immediately.

    Returns the measured L2 norm; raises AssertionError on exceedance."""
    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    if tol is None:
        tol = mean_x_tol_for(X_scaled.shape[0])
    mean_vec = X_scaled.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    worst = int(np.argmax(np.abs(mean_vec)))
    assert norm < tol, (
        f"standardized training features are not centered: "
        f"||mean(X_train_scaled)|| = {norm:.6e} >= {tol:.3e} "
        f"(tolerance at n={X_scaled.shape[0]} rows); worst column "
        f"{worst} has mean {mean_vec[worst]:.6e} "
        f"(a near-constant column is a likely cause -- see this function's docstring)")
    return norm


def scaler_centering_margin(X_scaled, X_raw=None, tol=None):
    """`assert_scaler_centered`'s statistic as a returned number, plus the
    raw column-variance context that explains where it comes from.

    Same object as the guard, different interface: `mean_x_norm` here is
    `||mean(X_scaled)||` computed exactly as `assert_scaler_centered`
    computes it, so a passing run records how far it actually sat from
    the tolerance rather than only that it was somewhere below it. This
    function never raises; the halt rule stays entirely in
    `assert_scaler_centered`.

    `min_col_std` is the smallest per-column standard deviation of the
    UNSCALED features (`ddof=0`, matching `StandardScaler`'s own
    `sqrt(var_)`), with the column index alongside it. That is the
    quantity the guard's near-constant-column tension is about: a column
    whose raw std sits in roughly `1e-12 < std < 1e-5` is divided by its
    tiny scale, amplifying the float64 centering residual past the
    tolerance. Recording it every call turns "how close is this condition
    to the regime that trips the guard" into a measured per-fold number.

    Why a phase-feature condition can approach that regime at all: a
    strongly synchronizing graph drives a node's phase nearly
    image-independent, so its cos/sin columns vary little across images.
    Stage 2A measured mean order parameters of 0.997 (`rewired`) and
    0.991 (`curr_random`) -- see
    `experiments/stage2a_dynamics_classification/FINDINGS.md` -- so
    `min_col_std` is expected to be smallest for exactly those two
    conditions and largest for raw pixels.

    Near-constant columns are reported, never dropped -- they carry
    small-but-nonzero variance that Stage 2A's findings give reason to
    treat as signal, unlike the encoding pipeline's reference-node
    columns, which are exactly constant by construction.

    Parameters
    ----------
    X_scaled : (n, p) standardized features -- the matrix the guard sees.
    X_raw    : (n, p) the same features BEFORE standardization. Optional;
               when omitted, `min_col_std` is nan and `min_col_std_col`
               is -1, since raw column spread cannot be recovered from
               the standardized matrix.
    tol      : the tolerance to report the margin against; the guard's
               own `mean_x_tol_for(X_scaled.shape[0])` by default, so the
               reported `tol` and `margin_ratio` are n-dependent exactly
               as the halt rule is.

    Returns a dict. `within_tol` is descriptive, not a gate -- the gate
    is `assert_scaler_centered`.

    Scope limit: this reports on whatever it is called with, but the
    callers below reach it only on the path where the guard passes. If
    `svd_ridge_fit`'s assertion fires, the exception leaves
    `cross_validate_alpha` / `fit_final` with no return value at all, so
    the diagnosable record in that case is the assertion message (which
    names `||mean(X_scaled)||` and the worst-mean column) and not
    `min_col_std`."""
    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    if tol is None:
        tol = mean_x_tol_for(X_scaled.shape[0])
    mean_vec = X_scaled.mean(axis=0)
    norm = float(np.linalg.norm(mean_vec))
    worst = int(np.argmax(np.abs(mean_vec)))

    if X_raw is None:
        min_col_std, min_col_std_col = float("nan"), -1
    else:
        X_raw = np.asarray(X_raw, dtype=np.float64)
        if X_raw.shape != X_scaled.shape:
            raise ValueError(f"X_raw shape {X_raw.shape} does not match "
                             f"X_scaled shape {X_scaled.shape}")
        col_std = X_raw.std(axis=0, ddof=0)
        min_col_std_col = int(np.argmin(col_std))
        min_col_std = float(col_std[min_col_std_col])

    return {
        "mean_x_norm": norm,
        "tol": float(tol),
        "margin_ratio": norm / float(tol),
        "within_tol": bool(norm < tol),
        "worst_mean_col": worst,
        "worst_mean_value": float(mean_vec[worst]),
        "min_col_std": min_col_std,
        "min_col_std_col": min_col_std_col,
    }


def svd_ridge_fit(X_train_scaled, Y_train, alphas=ALPHA_GRID, check_centered=True):
    """One thin SVD of the standardized training features; every alpha in
    `alphas` solved from that single decomposition.

    Parameters
    ----------
    X_train_scaled : (n, p) standardized training features.
    Y_train        : (n, k) UNstandardized targets (DESIGN.md: "targets
                     unstandardized"), e.g. clean intensities on the
                     505-coordinate active support.
    alphas         : ridge penalties, evaluated from the one SVD.

    Returns a dict with `W` (n_alpha, p, k), `b` (n_alpha, k), the
    singular values, the condition number (DESIGN.md's required stage-2
    diagnostic, free from the decomposition already computed), and the
    measured `||mean(X_train_scaled)||` alongside the `mean_x_tol` it was
    checked against. That tolerance is derived from the row count, so it
    is reported whether or not `check_centered` ran; `mean_x_norm` is nan
    when it did not."""
    X = jnp.asarray(np.asarray(X_train_scaled, dtype=np.float64))
    Y = jnp.asarray(np.asarray(Y_train, dtype=np.float64))
    if Y.ndim != 2:
        raise ValueError("Y_train must be 2-D (n, k); multi-output ridge, "
                         "shared alpha across all output columns")
    alphas_arr = jnp.asarray(np.asarray(alphas, dtype=np.float64))

    mean_x_tol = mean_x_tol_for(X.shape[0])
    mean_x_norm = assert_scaler_centered(X_train_scaled) if check_centered else float("nan")

    y_mean = Y.mean(axis=0)                       # (k,)
    x_mean = X.mean(axis=0)                       # (p,)
    Y_tilde = Y - y_mean                          # center targets, training fold only

    U, s, Vt = jnp.linalg.svd(X, full_matrices=False)
    Z = U.T @ Y_tilde                             # (r, k)
    # diag(s / (s^2 + alpha)) applied per alpha, reusing the one decomposition
    filt = s[None, :] / (s[None, :] ** 2 + alphas_arr[:, None])   # (a, r)
    W = jnp.einsum("pr,ar,rk->apk", Vt.T, filt, Z)                # (a, p, k)
    # general expression, not the mean(Y_train) shortcut
    b = y_mean[None, :] - jnp.einsum("p,apk->ak", x_mean, W)      # (a, k)

    s_np = np.asarray(s)
    cond = float(s_np.max() / s_np.min()) if s_np.min() > 0 else float("inf")
    return {
        "W": np.asarray(W), "b": np.asarray(b),
        "alphas": np.asarray(alphas, dtype=np.float64),
        "singular_values": s_np, "cond": cond,
        "y_mean": np.asarray(y_mean), "x_mean": np.asarray(x_mean),
        "mean_x_norm": mean_x_norm, "mean_x_tol": mean_x_tol,
    }


def ridge_predict(fit, X_scaled, alpha_index):
    """Predictions for one alpha: `X @ W_alpha + b_alpha`, unclipped
    (DESIGN.md: "ridge fitted without output clipping")."""
    X = np.asarray(X_scaled, dtype=np.float64)
    return X @ fit["W"][alpha_index] + fit["b"][alpha_index]


def per_image_mse(pred, target):
    """MSE per image, averaged over that image's output coordinates."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.mean((pred - target) ** 2, axis=1)


def clipped_per_image_mse(pred, target):
    """DESIGN.md's primary error: MSE after deterministic clipping of the
    PREDICTION to [0, 1]. The target is untouched."""
    return per_image_mse(np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0), target)


def mse_per_alpha(fit, X_scaled, Y, clipped=True):
    """Mean validation MSE for every alpha in the fit, looping over alphas
    rather than materializing an (n_alpha, n, k) prediction tensor."""
    out = np.empty(len(fit["alphas"]), dtype=np.float64)
    for a in range(len(fit["alphas"])):
        pred = ridge_predict(fit, X_scaled, a)
        errs = clipped_per_image_mse(pred, Y) if clipped else per_image_mse(pred, Y)
        out[a] = float(np.mean(errs))
    return out


def select_alpha(mean_val_mse, alphas=ALPHA_GRID, tol=ALPHA_TIE_TOL):
    """DESIGN.md's locked selection rule: `argmin_alpha MSE(clip(x_hat, 0, 1), x_0)`,
    ties within `tol` ABSOLUTE broken toward the LARGER alpha.

    The tie direction is locked (it was a corrected error in an earlier
    draft): among every alpha whose mean validation MSE is within `tol`
    of the minimum, the largest -- i.e. the most regularized -- wins.

    Returns (alpha, index)."""
    mse = np.asarray(mean_val_mse, dtype=np.float64)
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    if mse.shape != alphas_arr.shape:
        raise ValueError(f"mse shape {mse.shape} does not match alphas {alphas_arr.shape}")
    if not np.all(np.isfinite(mse)):
        raise ValueError("non-finite validation MSE -- alpha selection is undefined")
    best = float(mse.min())
    tied = np.where(mse <= best + tol)[0]
    idx = int(tied[int(np.argmax(alphas_arr[tied]))])
    return float(alphas_arr[idx]), idx


def cross_validate_alpha(X, Y, y_strat, alphas=ALPHA_GRID, n_splits=N_SPLITS,
                          random_state=FOLD_SEED):
    """Five-fold stratified CV over the alpha grid, JAX SVD production path.

    `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
    per-fold `StandardScaler` fitted on the training fold only, one thin
    SVD per fold with all alphas reused from it, alpha chosen by mean
    CLIPPED validation MSE (raw kept as a diagnostic only).

    `X` is passed UNSCALED -- the per-fold scaler is fitted inside, which
    is what makes the fold partition and the scaling identical across
    every condition a caller compares.

    `fold_mean_x_norm` carries the centering guard's own statistic per
    fold, and `fold_mean_x_tol` the tolerance that fold was actually
    checked against -- per fold rather than once, because the tolerance
    is `mean_x_tol_for(n_train_rows)` and a training fold has 4/5 of the
    corpus, not all of it. There is deliberately no corpus-level scalar:
    it would sit sqrt(5/4) above every value the guard actually used.
    `fold_min_col_std` / `fold_min_col_std_col` / `fold_worst_mean_col`
    come from `scaler_centering_margin` and are recorded for every fold
    whether or not the guard was anywhere near firing. They change
    nothing about the fit or the selected alpha."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_alpha = len(alphas)
    fold_clipped = np.empty((n_splits, n_alpha), dtype=np.float64)
    fold_raw = np.empty((n_splits, n_alpha), dtype=np.float64)
    fold_cond = np.empty(n_splits, dtype=np.float64)
    fold_mean_x_norm = np.empty(n_splits, dtype=np.float64)
    fold_mean_x_tol = np.empty(n_splits, dtype=np.float64)
    fold_min_col_std = np.empty(n_splits, dtype=np.float64)
    fold_min_col_std_col = np.empty(n_splits, dtype=np.int64)
    fold_worst_mean_col = np.empty(n_splits, dtype=np.int64)

    for f, (tr, va) in enumerate(skf.split(X, y_strat)):
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_va = scaler.transform(X[tr]), scaler.transform(X[va])
        margin = scaler_centering_margin(X_tr, X[tr])
        fit = svd_ridge_fit(X_tr, Y[tr], alphas=alphas)
        fold_clipped[f] = mse_per_alpha(fit, X_va, Y[va], clipped=True)
        fold_raw[f] = mse_per_alpha(fit, X_va, Y[va], clipped=False)
        fold_cond[f] = fit["cond"]
        fold_mean_x_norm[f] = fit["mean_x_norm"]
        fold_mean_x_tol[f] = margin["tol"]
        fold_min_col_std[f] = margin["min_col_std"]
        fold_min_col_std_col[f] = margin["min_col_std_col"]
        fold_worst_mean_col[f] = margin["worst_mean_col"]

    mean_clipped = fold_clipped.mean(axis=0)
    alpha, idx = select_alpha(mean_clipped, alphas)
    return {
        "alpha": alpha, "alpha_index": idx, "alphas": np.asarray(alphas, dtype=np.float64),
        "mean_clipped_val_mse": mean_clipped,
        "mean_raw_val_mse": fold_raw.mean(axis=0),
        "fold_clipped_val_mse": fold_clipped, "fold_raw_val_mse": fold_raw,
        "fold_cond": fold_cond, "fold_mean_x_norm": fold_mean_x_norm,
        "fold_mean_x_tol": fold_mean_x_tol,
        "fold_min_col_std": fold_min_col_std,
        "fold_min_col_std_col": fold_min_col_std_col,
        "fold_worst_mean_col": fold_worst_mean_col,
    }


def fit_final(X_train, Y_train, alpha):
    """The final full-training refit at a selected alpha -- one further
    thin SVD per condition (DESIGN.md's "SVD count: 42, not 35").

    Returns (fit, scaler); the caller applies the same scaler to
    evaluation features. `fit["centering_margin"]` is
    `scaler_centering_margin`'s dict for this refit -- attached here, not
    by `svd_ridge_fit`, which has no access to the unscaled features, so
    the key is absent from a direct `svd_ridge_fit` call."""
    scaler = StandardScaler().fit(np.asarray(X_train, dtype=np.float64))
    X_raw = np.asarray(X_train, dtype=np.float64)
    X_scaled = scaler.transform(X_raw)
    margin = scaler_centering_margin(X_scaled, X_raw)
    fit = svd_ridge_fit(X_scaled, Y_train, alphas=(float(alpha),))
    fit["centering_margin"] = margin
    return fit, scaler


# ---- sklearn verification oracle (never in the production path) ----

def sklearn_ridge_predict(X_train_scaled, Y_train, X_eval_scaled, alphas=ALPHA_GRID):
    """`Ridge(solver="svd", fit_intercept=True)` per alpha -- the oracle
    DESIGN.md's equivalence gate is measured against. Returns
    (predictions (a, n_eval, k), coefficients (a, p, k), intercepts (a, k)).

    sklearn's `coef_` is (k, p) for multi-output; it is transposed here so
    both paths speak the same (p, k) convention."""
    X_tr = np.asarray(X_train_scaled, dtype=np.float64)
    Y_tr = np.asarray(Y_train, dtype=np.float64)
    X_ev = np.asarray(X_eval_scaled, dtype=np.float64)
    preds, coefs, intercepts = [], [], []
    for a in alphas:
        model = Ridge(alpha=float(a), solver="svd", fit_intercept=True).fit(X_tr, Y_tr)
        preds.append(model.predict(X_ev))
        coefs.append(np.atleast_2d(model.coef_).T)
        intercepts.append(np.atleast_1d(model.intercept_))
    return np.stack(preds), np.stack(coefs), np.stack(intercepts)


def ridge_equivalence_check(X, Y, y_strat, alphas=ALPHA_GRID, n_splits=N_SPLITS,
                             random_state=FOLD_SEED, tol=EQUIVALENCE_TOL):
    """DESIGN.md's literal equivalence gate, runnable at any corpus size.

    Both paths share one fold splitter, one per-fold scaler, and one
    `select_alpha` -- the oracle's selection is NOT computed by a
    separately written argmin, so "identical alpha selection" tests the
    selection rule against the same rule, and only the fitted values
    differ.

    Gate (both required):
      (a) max abs difference in CLIPPED validation predictions <= 1e-8
      (b) identical alpha selection

    Max abs COEFFICIENT difference is recorded as a visibility
    diagnostic at whatever value it takes -- explicitly not a second halt
    rule, per DESIGN.md ("prediction agreement is what matters for the
    endpoint")."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_alpha = len(alphas)
    jax_fold = np.empty((n_splits, n_alpha), dtype=np.float64)
    skl_fold = np.empty((n_splits, n_alpha), dtype=np.float64)
    max_pred_diff = 0.0
    max_coef_diff = 0.0

    for f, (tr, va) in enumerate(skf.split(X, y_strat)):
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_va = scaler.transform(X[tr]), scaler.transform(X[va])

        fit = svd_ridge_fit(X_tr, Y[tr], alphas=alphas)
        skl_pred, skl_coef, _skl_int = sklearn_ridge_predict(X_tr, Y[tr], X_va, alphas)

        for a in range(n_alpha):
            jax_pred_a = ridge_predict(fit, X_va, a)
            jax_fold[f, a] = float(np.mean(clipped_per_image_mse(jax_pred_a, Y[va])))
            skl_fold[f, a] = float(np.mean(clipped_per_image_mse(skl_pred[a], Y[va])))
            # the gate is on CLIPPED validation predictions specifically
            diff = np.max(np.abs(np.clip(jax_pred_a, 0.0, 1.0)
                                  - np.clip(skl_pred[a], 0.0, 1.0)))
            max_pred_diff = max(max_pred_diff, float(diff))
            max_coef_diff = max(max_coef_diff,
                                 float(np.max(np.abs(fit["W"][a] - skl_coef[a]))))

    alpha_jax, _ = select_alpha(jax_fold.mean(axis=0), alphas)
    alpha_skl, _ = select_alpha(skl_fold.mean(axis=0), alphas)
    return {
        "max_abs_clipped_pred_diff": max_pred_diff,
        "max_abs_coef_diff": max_coef_diff,          # diagnostic only
        "alpha_jax": alpha_jax, "alpha_sklearn": alpha_skl,
        "alpha_agrees": alpha_jax == alpha_skl,
        "pred_agrees": max_pred_diff <= tol,
        "passed": bool(max_pred_diff <= tol and alpha_jax == alpha_skl),
        "tol": tol,
        "mean_clipped_val_mse_jax": jax_fold.mean(axis=0),
        "mean_clipped_val_mse_sklearn": skl_fold.mean(axis=0),
    }
