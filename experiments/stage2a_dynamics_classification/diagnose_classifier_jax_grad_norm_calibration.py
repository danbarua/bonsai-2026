"""
Diagnostic-only (not part of any locked pipeline): reproduces the two
real-data measurements behind stage_2a_classifier_jax.py's GRAD_NORM_REL
recalibration, documented in JAX_CLASSIFIER_PORT_FINDINGS.md's
"Follow-up: convergence-criterion recalibration (evolved_T)" section.
Committed because those numbers were previously only reproducible from
an interactive session's scratch scripts -- a real reproducibility gap,
not because either check is part of Stage 2A's locked pipeline.

Uses the real, cached evolved_T features (via the already-verified
analyze_stage_3_results_jax.build_results_structure -- not
reimplemented), a stratified 6,000-of-60,000-image subsample (seed=0),
matching exactly what JAX_CLASSIFIER_PORT_FINDINGS.md reports. Requires
the real Stage-3 training artifacts locally
(stage2a_paths.train_scratch_dir(), or pulled from the public GCS
bucket -- see README.md's "Public artifact cache (GCS)").

Two checks, run in sequence:

1. **Grad-norm calibration** (Step 1): fits sklearn to convergence at
   each locked C value on evolved_T's fold-0 training partition, then
   recomputes ||grad|| at sklearn's own converged solution using
   stage_2a_classifier_jax's own loss/gradient formula (not
   reimplemented). Reproduces the table showing sklearn's achieved
   ||grad|| spans eight orders of magnitude and is three to eleven
   orders of magnitude looser than the module's original fixed
   GRAD_NORM_TOL=1e-6 -- the finding that motivated the C*n_train-
   normalized GRAD_NORM_REL criterion.
2. **Real-data curve verification** (Step 2): full sklearn-vs-JAX
   per-C validation-loss curve comparison on the same data, using the
   now-recalibrated select_C_via_cv_jax. Reproduces the finding that
   best_C selection matches and non-convergence is fixed, but the
   large-C validation-loss curve still diverges substantially from
   sklearn's -- the sole remaining open item for this port.

Slow: check 2 involves a real 5-fold sklearn CV fit at max_iter=10000
on ill-conditioned 1008-dim data (the real run this reproduces took
~250s for sklearn, ~510s for JAX on CPU). Not intended to run as part
of routine testing.

Note on exact reproduction: this script keeps evolved_T's features at
full float64 precision throughout. The original interactive session
that produced JAX_CLASSIFIER_PORT_FINDINGS.md's numbers round-tripped
them through float32 npz files as a Colab-upload-size workaround, so
re-running this script reproduces the same qualitative findings (same
conclusion: GRAD_NORM_REL retains margin; best_C still matches) but not
byte-identical numbers -- e.g. the ||grad||/(C*n_train) range measured
here is [1.03e-3, 2.56e-3], not the findings doc's [1.34e-3, 2.77e-3].
Both support the same GRAD_NORM_REL=6e-3 calibration.
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_classifier as ref
import stage_2a_classifier_jax as jaxclf
import analyze_stage_3_results_jax as azjax

N_SUBSAMPLE = 6000
SUBSAMPLE_SEED = 0


def load_real_evolved_T_subsample():
    """Real, cached Stage-3 evolved_T features (via the already-verified
    build_results_structure), stratified-subsampled to N_SUBSAMPLE
    images, seed=SUBSAMPLE_SEED -- matching
    JAX_CLASSIFIER_PORT_FINDINGS.md's real-data measurements exactly."""
    import pickle
    print("Loading real Stage-3 artifacts (local_data + gpu_data)...")
    with open(os.path.join(azjax.SCRATCH_DIR, "stage3_encode_local.pkl"), "rb") as f:
        local_data = pickle.load(f)
    with open(os.path.join(azjax.SCRATCH_DIR, "stage3_gpu_results.pkl"), "rb") as f:
        gpu_data = pickle.load(f)
    ref_idx = local_data["ref_idx"]

    print("Building results structure (verified JAX R_post/feat_post path)...")
    results = azjax.build_results_structure(local_data, gpu_data, ref_idx)

    y_full = np.asarray(local_data["labels"])
    valid_mask = np.array([not r["evolved"]["T"]["solver_failed"] for r in results])
    X_full = np.stack([r["evolved"]["T"]["feat_post"] for r in results
                        if not r["evolved"]["T"]["solver_failed"]]).astype(np.float64)
    y_full = y_full[valid_mask]

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    classes = np.unique(y_full)
    per_class = N_SUBSAMPLE // len(classes)
    sub_idx = np.sort(np.concatenate([
        rng.choice(np.where(y_full == c)[0], size=per_class, replace=False)
        for c in classes
    ]))
    print(f"evolved_T real subsample: X.shape={X_full[sub_idx].shape}")
    return X_full[sub_idx], y_full[sub_idx]


def check_1_grad_norm_calibration(X, y):
    print(f"\n{'='*70}\nCHECK 1: sklearn's own converged ||grad||, real evolved_T data, "
          f"fold 0\n{'='*70}")
    classes = sorted(set(y.tolist()))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    skf = StratifiedKFold(n_splits=ref.N_FOLDS, shuffle=True, random_state=ref.SEED)
    train_idx, _val_idx = next(iter(skf.split(X, y)))
    X_tr, y_tr = X[train_idx], y[train_idx]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float64)
    y_tr_idx = np.array([class_to_idx[v] for v in y_tr])
    y_tr_onehot = np.eye(n_classes)[y_tr_idx]

    print(f"fold 0: n_train={X_tr_s.shape[0]}, d={X_tr_s.shape[1]}")
    print(f"{'C':>10} {'sklearn n_iter':>15} {'converged':>11} {'||grad||':>14} "
          f"{'||grad||/(C*n)':>16}")
    rows = []
    for C in ref.C_GRID:
        clf, converged, n_iter = ref._fit_one(X_tr_s, y_tr, C)
        W = jnp.asarray(clf.coef_.T, dtype=jnp.float64)
        b = jnp.asarray(clf.intercept_, dtype=jnp.float64)
        # THIS module's own loss function -- not reimplemented -- so the
        # gradient norm reported is directly comparable to what
        # stage_2a_classifier_jax's own convergence check uses.
        loss_fn = jaxclf._make_loss_fn(jnp.asarray(X_tr_s), jnp.asarray(y_tr_onehot), C)
        grad = jax.grad(loss_fn)((W, b))
        gnorm = float(jaxclf._tree_l2_norm(grad))
        ratio = gnorm / (C * X_tr_s.shape[0])
        rows.append((C, n_iter, converged, gnorm, ratio))
        print(f"{C:>10g} {n_iter:>15d} {str(converged):>11} {gnorm:>14.4e} {ratio:>16.4e}")

    ratios = [r[4] for r in rows]
    print(f"\nmax ||grad|| (raw, unnormalized): {max(r[3] for r in rows):.4e}")
    print(f"min ||grad|| (raw, unnormalized): {min(r[3] for r in rows):.4e}")
    print(f"||grad||/(C*n_train) range: [{min(ratios):.4e}, {max(ratios):.4e}]")
    print(f"Currently configured GRAD_NORM_REL: {jaxclf.GRAD_NORM_REL:.1e} "
          f"({'>' if jaxclf.GRAD_NORM_REL > max(ratios) else '<'} max observed ratio)")
    assert jaxclf.GRAD_NORM_REL > max(ratios), (
        "GRAD_NORM_REL no longer has margin above sklearn's own observed "
        "||grad||/(C*n_train) ratio on this real data -- recalibration needed.")
    print("PASS: GRAD_NORM_REL retains margin above sklearn's measured convergence ratio.")


def check_2_real_data_curve(X, y):
    print(f"\n{'='*70}\nCHECK 2: full sklearn-vs-JAX per-C curve, real evolved_T data\n"
          f"{'='*70}")
    t0 = time.time()
    best_C_ref, mean_loss_ref, nc_ref = ref.select_C_via_cv(X, y, "evolved_T_diagnostic")
    t_ref = time.time() - t0
    assert not nc_ref, f"unexpected sklearn non-convergence: {nc_ref}"
    print(f"sklearn: best_C={best_C_ref}, elapsed={t_ref:.1f}s ({t_ref/60:.1f} min)")

    t0 = time.time()
    best_C_jax, mean_loss_jax, nc_jax = jaxclf.select_C_via_cv_jax(X, y, "evolved_T_diagnostic")
    t_jax = time.time() - t0
    assert not nc_jax, f"unexpected JAX non-convergence: {nc_jax}"
    print(f"jax:     best_C={best_C_jax}, elapsed={t_jax:.1f}s ({t_jax/60:.1f} min)")

    print(f"\n{'C':>8} {'sklearn':>12} {'jax':>12} {'diff':>10}")
    for C in mean_loss_ref:
        diff = abs(mean_loss_ref[C] - mean_loss_jax[C])
        print(f"{C:>8g} {mean_loss_ref[C]:>12.6f} {mean_loss_jax[C]:>12.6f} {diff:>10.4f}")

    max_diff = max(abs(mean_loss_ref[C] - mean_loss_jax[C]) for C in mean_loss_ref)
    print(f"\nmax |val_loss diff| across grid: {max_diff:.4f}")
    print(f"best_C match: {best_C_ref == best_C_jax} (ref={best_C_ref}, jax={best_C_jax})")
    print("\nNOTE: per JAX_CLASSIFIER_PORT_FINDINGS.md, best_C matching and zero "
          "non-convergence are the properties this recalibration fixed; the large-C "
          "loss-curve divergence from sklearn is a known, still-open gap -- this "
          "check reports it, it does not assert it away.")
    assert best_C_ref == best_C_jax, (
        f"best_C selection diverged on real data: sklearn={best_C_ref}, jax={best_C_jax}")
    print("PASS: best_C selection matches sklearn on real evolved_T data.")


def main():
    X, y = load_real_evolved_T_subsample()
    check_1_grad_norm_calibration(X, y)
    check_2_real_data_curve(X, y)


if __name__ == "__main__":
    main()
