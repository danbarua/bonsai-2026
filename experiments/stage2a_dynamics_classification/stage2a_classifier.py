"""
Stage 2A locked classifier procedure (DESIGN.md, "Linear readout:
regularization grid, locked" + the standardization/classifier-
implementation corrections): multinomial logistic regression, per-
feature standardization fit fold-safe during cross-validation and
refit once on the complete training set after C-selection, a fixed
9-value log-spaced C grid, 5-fold stratified CV selecting by mean
validation log-loss with a deterministic smaller-C tie-break, and an
explicit non-convergence stop-gate (raises, does not silently log and
continue) -- shared across every feasibility stage and the eventual
confirmatory run, not reimplemented per stage.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
N_FOLDS = 5
SEED = 42

CLASSIFIER_KWARGS = dict(
    solver="lbfgs",
    tol=1e-4,
    max_iter=10000,  # amended from 1000 (DESIGN.md, "Linear classifier
    # implementation") -- feasibility stage 2's evolved_T non-convergence
    # was diagnosed as severe feature multicollinearity (condition number
    # ~2e6), not sample-size-fixable separability; see FINDINGS.md.
    class_weight=None,
    random_state=SEED,
)  # multinomial is lbfgs's native, default behavior for >2 classes in the
# installed scikit-learn version (>=1.9.0) -- the explicit multi_class=
# "multinomial" kwarg from earlier scikit-learn releases was removed.


class NonConvergenceError(RuntimeError):
    """Raised when a required (fold, C) fit fails to converge --
    DESIGN.md's locked stop-gate: logging alone does not satisfy the
    "converges in every condition" requirement, this must halt
    advancement, not be silently absorbed."""


def _fit_one(X, y, C):
    clf = LogisticRegression(C=C, **CLASSIFIER_KWARGS)
    with np.errstate(all="ignore"):
        clf.fit(X, y)
    converged = clf.n_iter_[0] < CLASSIFIER_KWARGS["max_iter"]
    return clf, converged, int(clf.n_iter_[0])


def select_C_via_cv(X_train, y_train, condition_label, n_folds=N_FOLDS, seed=SEED):
    """Fold-safe 5-fold stratified CV over C_GRID. Returns
    (best_C, per_C_mean_val_logloss, non_convergence_log). Raises
    NonConvergenceError immediately on any non-converged (fold, C) fit --
    per DESIGN.md's locked stop-gate, this is not caught and continued
    past."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_C_val_losses = {C: [] for C in C_GRID}
    non_convergence_log = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler = StandardScaler().fit(X_tr)  # fold-safe: fit on this fold's training partition only
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)

        for C in C_GRID:
            clf, converged, n_iter = _fit_one(X_tr_s, y_tr, C)
            if not converged:
                non_convergence_log.append(
                    {"condition": condition_label, "fold": fold_idx, "C": C, "n_iter": n_iter})
                raise NonConvergenceError(
                    f"[{condition_label}] fold={fold_idx} C={C}: did not converge in "
                    f"{n_iter} iterations (max_iter={CLASSIFIER_KWARGS['max_iter']}). "
                    f"Per DESIGN.md's locked stop-gate, this halts advancement -- "
                    f"not silently logged and continued.")
            val_pred_proba = clf.predict_proba(X_val_s)
            per_C_val_losses[C].append(
                log_loss(y_val, val_pred_proba, labels=sorted(set(y_train))))

    mean_val_loss = {C: float(np.mean(losses)) for C, losses in per_C_val_losses.items()}
    min_loss = min(mean_val_loss.values())
    tied = [C for C, loss in mean_val_loss.items() if abs(loss - min_loss) < 1e-12]
    best_C = min(tied)  # deterministic tie-break: smaller C (stronger regularization)

    return best_C, mean_val_loss, non_convergence_log


def diagnose_convergence_full_grid(X_train, y_train, condition_label, n_folds=N_FOLDS, seed=SEED):
    """Diagnostic only -- NOT part of the locked pipeline procedure and
    never used to select C or report a primary/secondary result. Scans
    every (fold, C) combination WITHOUT stopping on the first failure,
    to characterize whether a non-convergence is an isolated point or a
    wider pattern -- DESIGN.md's own framing ("a pattern of non-
    convergence concentrated in one condition or one C region is itself
    a reportable diagnostic") requires this visibility before deciding
    how to respond to a real stop-gate trigger. Returns a full
    (fold, C) -> {converged, n_iter} table."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    table = {}
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        for C in C_GRID:
            _clf, converged, n_iter = _fit_one(X_tr_s, y_tr, C)
            table[(fold_idx, C)] = {"converged": converged, "n_iter": n_iter}
    n_failed = sum(1 for v in table.values() if not v["converged"])
    print(f"[diagnostic, {condition_label}] {n_failed}/{len(table)} (fold, C) "
          f"combinations failed to converge within max_iter={CLASSIFIER_KWARGS['max_iter']}")
    return table


def fit_final_at_selected_C(X_train, y_train, X_test, best_C, condition_label):
    """The confirmatory-run half of fit_condition() below, split out so a
    C already selected elsewhere (stage 3's full-training-set CV) can be
    reused directly -- no new CV search, no new hyperparameter selection,
    just the single locked final refit (fresh scaler on the complete
    training set, fresh classifier at the given C) and its application to
    test features. Raises NonConvergenceError on the same terms as
    fit_condition(), never silently."""
    final_scaler = StandardScaler().fit(X_train)
    X_train_s = final_scaler.transform(X_train)
    X_test_s = final_scaler.transform(X_test)

    final_clf, converged, n_iter = _fit_one(X_train_s, y_train, best_C)
    if not converged:
        raise NonConvergenceError(
            f"[{condition_label}] final fit at selected C={best_C} did not converge in "
            f"{n_iter} iterations. Halts advancement, per DESIGN.md's locked stop-gate.")

    return {
        "condition": condition_label, "selected_C": best_C,
        "scaler": final_scaler, "classifier": final_clf, "final_n_iter": n_iter,
        "X_test_standardized": X_test_s,
    }


def fit_condition(X_train, y_train, X_test, condition_label, n_folds=N_FOLDS, seed=SEED):
    """Full locked procedure for one feature condition: CV-select C
    (fold-safe standardization throughout), then refit a fresh scaler on
    the complete training set and a fresh classifier at the selected C,
    applied unchanged to X_test. Returns a dict with the fitted
    scaler/classifier, selected C, CV diagnostics, and test-set
    (standardized) features ready for prediction."""
    best_C, mean_val_loss, non_convergence_log = select_C_via_cv(
        X_train, y_train, condition_label, n_folds=n_folds, seed=seed)

    final = fit_final_at_selected_C(X_train, y_train, X_test, best_C, condition_label)
    final["mean_val_loss_per_C"] = mean_val_loss
    return final
