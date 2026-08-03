"""
Diagnostic only -- does not touch the locked pipeline, the driver fix,
or the max_iter decision (still pending, separately). Investigates WHY
evolved_T's fold=0/C=100 fit failed to converge (feasibility stage 2,
5,000 images): near-separability (near-100% training accuracy driving
coefficients toward infinity) vs. ill-conditioning of the standardized
feature matrix itself, or both.

Reuses the exact same 5,000-image subsample, the same seed=0 feature
generation, and the same StratifiedKFold(n_splits=5, shuffle=True,
random_state=42) split as feasibility stage 2 and
diagnose_stage2_convergence.py -- fold index 0 is the identical training
partition the original failure came from.
"""
import os
import sys

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from bonsai.data.mnist_loader import load_mnist
import stage2a_pipeline as pipe
from stage2a_classifier import CLASSIFIER_KWARGS, SEED, N_FOLDS

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
N_PER_CLASS = 500


def fit_and_report(X_tr_s, y_tr, C, label):
    clf = LogisticRegression(C=C, **CLASSIFIER_KWARGS)
    with np.errstate(all="ignore"):
        clf.fit(X_tr_s, y_tr)
    n_iter = int(clf.n_iter_[0])
    converged = n_iter < CLASSIFIER_KWARGS["max_iter"]
    train_pred_proba = clf.predict_proba(X_tr_s)
    train_acc = clf.score(X_tr_s, y_tr)
    train_loss = log_loss(y_tr, train_pred_proba, labels=sorted(set(y_tr)))
    coef_norm = float(np.linalg.norm(clf.coef_))
    print(f"  [{label}] C={C}: n_iter={n_iter} (converged={converged}), "
          f"train_acc={train_acc:.6f}, train_logloss={train_loss:.6f}, "
          f"||coef||={coef_norm:.4f}")
    return {"n_iter": n_iter, "converged": converged, "train_acc": train_acc,
            "train_loss": train_loss, "coef_norm": coef_norm}


def condition_number_report(X_tr_s, label):
    s = np.linalg.svd(X_tr_s, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    print(f"  [{label}] condition number={cond:.3e}, "
          f"singular values: max={s[0]:.3e}, min={s[-1]:.3e}, "
          f"n_near_zero(<1e-8*max)={int(np.sum(s < 1e-8 * s[0]))}")
    return {"condition_number": cond, "s_max": float(s[0]), "s_min": float(s[-1]),
            "n_near_zero": int(np.sum(s < 1e-8 * s[0]))}


def r_correlation_check(X_tr_s, R_values, label):
    """Tests the one link between stage 1's R(theta) diagnostic and this
    ill-conditioning finding that hadn't actually been checked: do images
    that load heavily onto the near-null direction (the smallest singular
    value, the direction responsible for the poor condition number) tend
    to be the highest-R (most phase-synchronized) images? Pearson
    correlation between each training image's R and its |projection| onto
    the smallest right-singular-vector direction."""
    _U, _s, Vt = np.linalg.svd(X_tr_s, full_matrices=False)
    v_min = Vt[-1]
    projection = X_tr_s @ v_min
    corr, p_value = pearsonr(R_values, np.abs(projection))
    print(f"  [{label}] correlation(R, |projection onto smallest-singular-value "
          f"direction|): r={corr:.4f}, p={p_value:.4e} (n={len(R_values)})")
    return {"correlation": float(corr), "p_value": float(p_value)}


def main():
    print("Loading official KMNIST training set...")
    X_train, y_train, _X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)

    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=SEED, n_per_class=N_PER_CLASS)
    print(f"Subsampled {len(images_01)} images ({N_PER_CLASS}/class, SEED={SEED}) "
          f"-- identical to feasibility stage 2's own subsample")

    print("\nRunning primary (seed=0) pipeline...")
    results, elapsed, active_indices, nodes_T = pipe.run_pipeline(images_01, labels)
    print(f"Pipeline complete: {len(results)} images in {elapsed:.1f}s")

    raw_X = np.stack([r["raw_feat"] for r in results])
    pre_X = np.stack([r["feat_pre"] for r in results])
    valid_mask = np.array([not r["solver_failed"] for r in results])
    assert valid_mask.all(), "expected zero solver failures at this scale, per stage 2's own result"
    evolved_X = np.stack([r["feat_post"] for r in results])
    y = labels

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold0_train_idx, fold0_val_idx = next(iter(skf.split(raw_X, y)))
    print(f"\nFold 0: {len(fold0_train_idx)} training images, {len(fold0_val_idx)} validation "
          f"(same split for every condition -- identical y, identical StratifiedKFold call)")

    conditions = {
        "raw_pixels": raw_X,
        "encoded_pre_evolution": pre_X,
        "evolved_T": evolved_X,
    }

    standardized = {}
    for label, X in conditions.items():
        X_tr = X[fold0_train_idx]
        scaler = StandardScaler().fit(X_tr)  # fold-safe, matching the locked procedure
        standardized[label] = scaler.transform(X_tr)

    y_tr = y[fold0_train_idx]

    print("\n" + "=" * 70)
    print("1+2. TRAINING ACCURACY/LOG-LOSS/COEF-NORM AT C=100 (the failing value)")
    print("=" * 70)
    at_100 = {label: fit_and_report(standardized[label], y_tr, 100.0, label) for label in conditions}

    print("\n" + "=" * 70)
    print("2 (cont). COEF-NORM AT C=0.01 (the actually-selected value)")
    print("=" * 70)
    at_001 = {label: fit_and_report(standardized[label], y_tr, 0.01, label) for label in conditions}

    print("\n" + "=" * 70)
    print("3. CONDITION NUMBER / SINGULAR VALUE DECAY (fold 0's standardized training matrix)")
    print("=" * 70)
    cond_report = {label: condition_number_report(standardized[label], label) for label in conditions}

    print("\n" + "=" * 70)
    print("4. R(theta) CORRELATION WITH THE ILL-CONDITIONED DIRECTION (evolved_T only)")
    print("=" * 70)
    R_post_fold0_train = np.array([results[i]["R_post"] for i in fold0_train_idx])
    r_corr = r_correlation_check(standardized["evolved_T"], R_post_fold0_train, "evolved_T")

    print("\n" + "=" * 70)
    print(f"SUMMARY (CLASSIFIER_KWARGS max_iter={CLASSIFIER_KWARGS['max_iter']})")
    print("=" * 70)
    for label in conditions:
        print(f"\n{label}:")
        print(f"  train_acc @ C=100: {at_100[label]['train_acc']:.6f}  "
              f"(converged={at_100[label]['converged']}, n_iter={at_100[label]['n_iter']})")
        print(f"  train_loss @ C=100: {at_100[label]['train_loss']:.6f}")
        print(f"  ||coef|| @ C=0.01: {at_001[label]['coef_norm']:.4f}  "
              f"-> @ C=100: {at_100[label]['coef_norm']:.4f}  "
              f"(ratio: {at_100[label]['coef_norm']/at_001[label]['coef_norm']:.2f}x)")
        print(f"  condition number: {cond_report[label]['condition_number']:.3e}")
    print(f"\nevolved_T: R(theta)-vs-ill-conditioned-direction correlation: "
          f"r={r_corr['correlation']:.4f}, p={r_corr['p_value']:.4e}")

    return {"at_100": at_100, "at_001": at_001, "cond_report": cond_report, "r_corr": r_corr}


if __name__ == "__main__":
    main()
