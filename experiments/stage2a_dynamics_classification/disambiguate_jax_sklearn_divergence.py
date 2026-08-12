"""Is the JAX/sklearn divergence the stopping rule, or GPU numerics?

The GPU gauge run found the two classifiers interchangeable on well-conditioned
arms (accuracy agreeing to ~1.6e-05) and materially apart on ill-conditioned
ones (~1.0e-02, JAX always the worse side). Two explanations fit that:

  (a) STOPPING RULE. JAX stops on ``GRAD_NORM_REL * C * n_train``, a tolerance
      PROPORTIONAL to C, while sklearn stops on a fixed ``tol=1e-4``. At
      C=1000 the JAX fit is held to a ten-times looser standard than at C=100,
      so exactly the hardest fits stop shortest.

  (b) GPU NUMERICS. Thousands of L-BFGS iterations give floating-point error
      more room to accumulate on GPU than on CPU.

Arguing from the pattern favours (a): the gap tracks conditioning over a ~700x
range and is signed consistently, where rounding would be roughly uniform and
unsigned. But arguing is not isolating, and this project's own history
(Stage 1D's 4-way factorial) says to separate the factors rather than reason
about them.

The separation is one cell. Run the SAME JAX classifier on CPU, on one
ill-conditioned fold, and see which number it lands on:

    near the GPU-JAX value  -> the stopping rule; the platform is irrelevant
    near the sklearn value  -> GPU numerics; the stopping rule is exonerated

x64 is already confirmed on for the GPU run (it printed ``x64: True``, and
every array is cast to float64, so cuBLAS dispatches DGEMM and TF32 does not
apply). This test does not assume that -- it measures the CPU/GPU contrast
directly, which is what the objection actually asks for.

Usage
-----
    JAX_PLATFORMS=cpu python disambiguate_jax_sklearn_divergence.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import stage2a_core as s2a
import stage2a_classifier_jax as clf_jax
from stage2a_paths import train_scratch_dir

# The single most divergent cell: worst-conditioned condition, the C it
# selected, the first fold. Chosen before running, from the completed arms.
CONDITION = "evolved_lattice"
TOPOLOGY = "lattice"
GAUGE = "reference"
C_UNDER_TEST = 1000.0
FOLD = 0

RESULTS = os.path.join(_THIS_DIR, "results", "gauge_comparison_2a")
OUT = os.path.join(RESULTS, "jax_sklearn_disambiguation.json")


def _fold_acc_from(fits, fold, C):
    """Pull one (fold, C) accuracy out of a recorded fits array."""
    for f in fits:
        if f["fold"] == fold and float(f["C"]) == float(C):
            return f["acc"], f.get("n_iter")
    return None, None


def main():
    backend = jax.default_backend()
    print(f"JAX backend: {backend} | x64: {jax.config.jax_enable_x64}")
    if backend != "cpu":
        print("WARNING: expected CPU. Re-run with JAX_PLATFORMS=cpu -- this "
              "test is meaningless on the same platform it is comparing against.")

    scratch = train_scratch_dir()
    with open(os.path.join(scratch, "stage3_encode_local.pkl"), "rb") as fh:
        local = pickle.load(fh)
    with open(os.path.join(scratch, "stage3_gpu_results.pkl"), "rb") as fh:
        gpu = pickle.load(fh)

    y = np.asarray(local["labels"])
    ref_idx = int(local["ref_idx"])
    theta_T = np.asarray(gpu["results"][TOPOLOGY]["theta_T"])
    ok = np.asarray(gpu["results"][TOPOLOGY]["success"], dtype=bool)

    X = np.stack([s2a.reference_node_features(t, ref_idx) for t in theta_T[ok]])
    y = y[ok]
    print(f"{CONDITION}:{GAUGE}  X={X.shape}")

    skf = StratifiedKFold(n_splits=clf_jax.N_FOLDS, shuffle=True,
                          random_state=clf_jax.SEED)
    tr, va = list(skf.split(X, y))[FOLD]
    scaler = StandardScaler().fit(X[tr])
    X_tr = jnp.asarray(scaler.transform(X[tr]), dtype=jnp.float64)
    X_va = jnp.asarray(scaler.transform(X[va]), dtype=jnp.float64)

    classes = sorted(set(y.tolist()))
    c2i = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([c2i[v] for v in y[tr].tolist()])
    y_oh = jnp.asarray(np.eye(len(classes))[y_idx], dtype=jnp.float64)

    print(f"fitting fold={FOLD} C={C_UNDER_TEST:g} on {backend} ...", flush=True)
    t0 = time.perf_counter()
    params, n_iter, gnorm, converged, tol_used = clf_jax._solve_grid_for_fold(
        (C_UNDER_TEST,), X_tr, y_oh, X_tr.shape[1], len(classes),
        max_iter=clf_jax.MAX_ITER)
    jax.block_until_ready(params[0])
    elapsed = time.perf_counter() - t0

    proba = np.asarray(clf_jax._predict_proba((params[0][0], params[1][0]), X_va))
    acc = float(np.mean(np.asarray(classes)[proba.argmax(axis=1)] == y[va]))

    # The two numbers this is being compared against, read back from the
    # recorded runs rather than retyped.
    with open(os.path.join(RESULTS, "fit_diagnostics.json")) as fh:
        sk = json.load(fh)["arms"][f"{CONDITION}__{GAUGE}"]["fits"]
    sk_acc, sk_iter = _fold_acc_from(sk, FOLD, C_UNDER_TEST)

    gpu_acc = gpu_iter = None
    gpu_json = os.path.join(RESULTS, "gpu_run.json")
    if os.path.exists(gpu_json):
        with open(gpu_json) as fh:
            arms = json.load(fh)["arms"]
        key = f"{CONDITION}__{GAUGE}"
        if key in arms and arms[key].get("fits"):
            gpu_acc, gpu_iter = _fold_acc_from(arms[key]["fits"], FOLD, C_UNDER_TEST)

    print(f"\n{'=' * 64}\nfold={FOLD}  C={C_UNDER_TEST:g}  {CONDITION}:{GAUGE}\n{'=' * 64}")
    print(f"  sklearn CPU : acc={sk_acc}  n_iter={sk_iter}")
    print(f"  JAX GPU     : acc={gpu_acc}  n_iter={gpu_iter}")
    print(f"  JAX CPU     : acc={acc:.6f}  n_iter={int(n_iter[0])}  "
          f"grad_norm={float(gnorm[0]):.3e}  tol={float(tol_used[0]):.3e}  "
          f"({elapsed:.1f}s)")

    verdict = "inconclusive"
    if sk_acc is not None and gpu_acc is not None:
        d_gpu, d_sk = abs(acc - gpu_acc), abs(acc - sk_acc)
        verdict = ("stopping rule (JAX CPU matches JAX GPU)" if d_gpu < d_sk
                   else "GPU numerics (JAX CPU matches sklearn)")
        print(f"\n  |JAX_CPU - JAX_GPU| = {d_gpu:.6f}")
        print(f"  |JAX_CPU - sklearn| = {d_sk:.6f}")
        print(f"\n  VERDICT: {verdict}")

    with open(OUT, "w") as fh:
        json.dump({"condition": CONDITION, "gauge": GAUGE, "fold": FOLD,
                   "C": C_UNDER_TEST, "backend": backend,
                   "sklearn_cpu_acc": sk_acc, "sklearn_cpu_n_iter": sk_iter,
                   "jax_gpu_acc": gpu_acc, "jax_gpu_n_iter": gpu_iter,
                   "jax_cpu_acc": acc, "jax_cpu_n_iter": int(n_iter[0]),
                   "jax_cpu_grad_norm": float(gnorm[0]),
                   "jax_cpu_grad_tol": float(tol_used[0]),
                   "jax_cpu_seconds": elapsed, "verdict": verdict}, fh, indent=1)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
