"""
PROTOTYPE, NOT YET VERIFIED FOR USE ON ANY REPORTED RESULT. JAX port of
stage2a_classifier.py's locked select_C_via_cv procedure (multinomial
logistic regression, L2-regularized, 5-fold stratified CV over a fixed
9-value C grid).

Fold splitting (StratifiedKFold) and per-fold standardization
(StandardScaler) are reused UNCHANGED from scikit-learn -- they are
cheap and not the bottleneck, and reusing them keeps fold assignment
and scaling byte-identical to stage2a_classifier.py's reference
pipeline, which is what makes this port checkable against that
reference at all.

What actually changes is the model fit: sklearn's per-(fold, C) serial
lbfgs solver calls are replaced by one JAX objective (softmax
cross-entropy + L2 on the weight matrix, unregularized intercept --
matching sklearn's LogisticRegression(solver="lbfgs") objective exactly)
minimized with optax.lbfgs, and the 9 C-grid problems within a fold are
solved as a single batched optimization via jax.vmap rather than 9
serial sklearn fits.

This is explicitly NOT a drop-in replacement for
stage2a_classifier.py's locked procedure:

- Convergence here is defined as ||grad|| (L2 norm over the full
  (W, b) parameter tree) <= GRAD_NORM_TOL after at most MAX_ITER
  L-BFGS steps. This is a different quantity from sklearn's internal
  lbfgs stopping rule (`n_iter_[0] < max_iter`, itself driven by
  scipy's solver-internal criteria) -- a "converged" flag from this
  module is not directly comparable to one from stage2a_classifier.py.
- Because the objective is the same strictly-convex-in-W problem
  (L2 on W breaks the softmax shift degeneracy; the intercept has one
  flat direction that does not affect predict_proba), a tightly
  converged fit here should land on the same predictions as sklearn's,
  and therefore closely matching mean_val_loss / selected C -- but
  this is a claim to verify (see verify_stage_2a_classifier_jax.py),
  not to assume.

Do not use this for any reported Stage 2A/3 result until that
verification has been run and checked.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

jax.config.update("jax_enable_x64", True)

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)
N_FOLDS = 5
SEED = 42
MAX_ITER = 2000
GRAD_NORM_TOL = 1e-6  # distinct criterion from sklearn's -- see module docstring


class NonConvergenceError(RuntimeError):
    """Raised when a required (fold, C) fit fails to reach GRAD_NORM_TOL
    within MAX_ITER L-BFGS steps. Mirrors stage2a_classifier.py's locked
    stop-gate (halt, do not silently log and continue) but is driven by
    this module's own gradient-norm convergence criterion, not sklearn's."""


def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))


def _make_loss_fn(X, y_onehot, C):
    """C * sum_i NLL_i(W, b) + 0.5 * sum(W**2) -- matches
    sklearn.linear_model.LogisticRegression(solver="lbfgs")'s objective
    exactly: C scales the (summed, not averaged) log-loss, L2 penalizes
    only the weight matrix, the intercept is unregularized."""
    def loss_fn(params):
        W, b = params
        logits = X @ W + b
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll_sum = -jnp.sum(y_onehot * log_probs)
        l2 = 0.5 * jnp.sum(W ** 2)
        return C * nll_sum + l2
    return loss_fn


def _tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.all(jnp.array([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


@partial(jax.jit, static_argnames=("n_features", "n_classes", "max_iter"))
def _solve_one(C, X, y_onehot, n_features, n_classes, max_iter, grad_tol):
    """Minimizes _make_loss_fn(X, y_onehot, C) via optax.lbfgs, stopping
    when ||grad|| <= grad_tol or max_iter steps are exhausted. Returns
    (params, n_iter, grad_norm, converged).

    Empirically flaky under jax.vmap + lax.while_loop: identical code and
    data occasionally produce a non-finite value/grad on some evaluation,
    reproduced even at params=0 (a mathematically benign point) and even
    for a vmap batch of size 1 -- not cross-lane contamination between C
    values. Most likely CPU-backend thread-scheduling nondeterminism in
    the large-reduction matmuls feeding value/grad, occasionally
    perturbing an L-BFGS curvature pair (s_k, y_k) enough to make
    y_k . s_k collapse near zero (classic L-BFGS curvature degeneracy,
    most likely exactly where C is weak enough that the objective is
    nearly pure-quadratic and a step lands very close to the optimum).
    See verify_stage_2a_classifier_jax.py's debug history for the
    isolation trail (batch-of-1 reproduction, order-dependence ruled out,
    sklearn-interleaving ruled out).

    Robustification, applied uniformly to every value/grad evaluation
    including the very first (there is no special-cased, unguarded
    pre-loop computation): if an evaluation is non-finite, discard it,
    leave params/state untouched, and retry on the next while_loop
    iteration -- a fresh XLA dispatch of the identical computation, which
    empirically recovers (consistent with dispatch-time thread-scheduling
    nondeterminism rather than a deterministic bug in the math itself).
    A step that computes a finite value/grad but then produces a
    non-finite optimizer update also resets the L-BFGS memory at the
    last-good point (standard L-BFGS numerical robustification) instead
    of letting the corrupted curvature pair persist."""
    loss_fn = _make_loss_fn(X, y_onehot, C)
    solver = optax.lbfgs()

    params0 = (jnp.zeros((n_features, n_classes), dtype=X.dtype),
               jnp.zeros((n_classes,), dtype=X.dtype))
    state0 = solver.init(params0)

    def cond_fn(carry):
        _params, _state, i, gnorm = carry
        return jnp.logical_and(i < max_iter, gnorm > grad_tol)

    def body_fn(carry):
        params, state, i, gnorm = carry
        value, grad = jax.value_and_grad(loss_fn)(params)
        grad_ok = jnp.isfinite(value) & _tree_all_finite(grad)

        def do_step(_):
            updates, new_state = solver.update(
                grad, state, params, value=value, grad=grad, value_fn=loss_fn)
            new_params = optax.apply_updates(params, updates)
            new_value, new_grad = jax.value_and_grad(loss_fn)(new_params)
            new_gnorm = _tree_l2_norm(new_grad)
            step_ok = jnp.isfinite(new_value) & jnp.isfinite(new_gnorm)

            reset_state = solver.init(params)
            out_params = jax.tree_util.tree_map(
                lambda a, b: jnp.where(step_ok, a, b), new_params, params)
            out_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(step_ok, a, b), new_state, reset_state)
            out_gnorm = jnp.where(step_ok, new_gnorm, _tree_l2_norm(grad))
            return out_params, out_state, out_gnorm

        def skip_step(_):
            # value/grad itself was non-finite -- params/state untouched,
            # carried gnorm kept (already > grad_tol, so cond_fn retries).
            return params, state, gnorm

        out_params, out_state, out_gnorm = jax.lax.cond(
            grad_ok, do_step, skip_step, operand=None)
        return out_params, out_state, i + 1, out_gnorm

    params, _state, n_iter, gnorm = jax.lax.while_loop(
        cond_fn, body_fn, (params0, state0, 0, jnp.asarray(jnp.inf, dtype=X.dtype)))
    converged = gnorm <= grad_tol
    return params, n_iter, gnorm, converged


def _solve_grid_for_fold(C_grid, X, y_onehot, n_features, n_classes,
                          max_iter=MAX_ITER, grad_tol=GRAD_NORM_TOL):
    """vmaps _solve_one over the C grid: one batched optimization solving
    all len(C_grid) regularization strengths for this fold in parallel,
    instead of len(C_grid) serial sklearn fits."""
    solve_batched = jax.vmap(
        _solve_one, in_axes=(0, None, None, None, None, None, None))
    return solve_batched(jnp.asarray(C_grid, dtype=X.dtype), X, y_onehot,
                          n_features, n_classes, max_iter, grad_tol)


def _predict_proba(params, X):
    W, b = params
    logits = X @ W + b
    return jax.nn.softmax(logits, axis=-1)


def select_C_via_cv_jax(X_train, y_train, condition_label, C_grid=C_GRID,
                         n_folds=N_FOLDS, seed=SEED, max_iter=MAX_ITER,
                         grad_tol=GRAD_NORM_TOL):
    """JAX-batched analog of stage2a_classifier.select_C_via_cv. Same
    fold-safe standardization, same fold splits (identical StratifiedKFold
    call), same deterministic smaller-C tie-break, same
    (best_C, mean_val_loss, non_convergence_log) return contract -- but
    every fold's 9-value C grid is fit as one vmapped JAX optimization
    rather than 9 serial sklearn fits, and convergence is judged by this
    module's own gradient-norm criterion (see module docstring)."""
    classes = sorted(set(y_train))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_C_val_losses = {C: [] for C in C_grid}
    non_convergence_log = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = jnp.asarray(scaler.transform(X_tr), dtype=jnp.float64)
        X_val_s = jnp.asarray(scaler.transform(X_val), dtype=jnp.float64)

        y_tr_idx = np.array([class_to_idx[y] for y in y_tr])
        y_tr_onehot = jnp.asarray(np.eye(n_classes)[y_tr_idx], dtype=jnp.float64)

        n_features = X_tr_s.shape[1]
        params, n_iter, gnorm, converged = _solve_grid_for_fold(
            C_grid, X_tr_s, y_tr_onehot, n_features, n_classes,
            max_iter=max_iter, grad_tol=grad_tol)

        for ci, C in enumerate(C_grid):
            if not bool(converged[ci]):
                non_convergence_log.append({
                    "condition": condition_label, "fold": fold_idx, "C": C,
                    "n_iter": int(n_iter[ci]), "grad_norm": float(gnorm[ci]),
                })
                raise NonConvergenceError(
                    f"[{condition_label}] fold={fold_idx} C={C}: did not reach "
                    f"||grad||<={grad_tol} in {int(n_iter[ci])} iterations "
                    f"(max_iter={max_iter}, final ||grad||={float(gnorm[ci]):.3e}). "
                    f"This module's own stop-gate -- halts, not silently logged.")
            W_c, b_c = params[0][ci], params[1][ci]
            val_pred_proba = np.asarray(_predict_proba((W_c, b_c), X_val_s))
            per_C_val_losses[C].append(
                log_loss(y_val, val_pred_proba, labels=classes))

    mean_val_loss = {C: float(np.mean(losses)) for C, losses in per_C_val_losses.items()}
    min_loss = min(mean_val_loss.values())
    tied = [C for C, loss in mean_val_loss.items() if abs(loss - min_loss) < 1e-12]
    best_C = min(tied)

    return best_C, mean_val_loss, non_convergence_log
