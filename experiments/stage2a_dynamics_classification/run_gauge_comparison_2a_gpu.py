"""The Stage 2A gauge comparison, run end to end on a remote GPU session.

The same ten (condition, gauge) arms as `run_gauge_comparison_2a.py`, but
evolved and fitted on an A100 with the JAX classifier instead of locally on
sklearn CPU. Two things come out of it:

  1. wall-clock for the whole job on GPU, against the 34.5 core-hours the
     local sklearn run cost, and
  2. a JAX-vs-sklearn agreement measurement at n=60,000 -- the number that
     would have justified (or refuted) taking the GPU path in the first
     place, which this project has never measured at full scale.

Runs ON the remote session, not locally. Nothing from the repository is on
disk there, so every path here is an explicit /content path and every import
must be either a package installed on the VM or a file uploaded alongside.

Gauges come from `stage2a_core_colab`, not `stage2a_core`: the latter inserts
sibling experiment directories onto sys.path relative to the repo layout and
imports three repo-local modules at module scope, none of which the gauges
need. Importing it here fails before any function runs. The fork is pinned to
the original by tests/test_stage2a_core_colab.py.

Long enough to need `exec-async` -- the local equivalent took 5.5 hours of
wall time. Prints a heartbeat from every fit so `--timeout` (which bounds
the gap between OUTPUTS, not the run) cannot kill it during a silent stretch,
and prints a sentinel on success.
"""
import sys
sys.path.insert(0, '/content')
import json
import pickle
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import stage2a_core_colab as s2a
import stage2a_classifier_jax as clf_jax
from evolve_on_graph_jax import batched_evolve_on_graph_jax

SENTINEL = "GAUGE2A_GPU_OK"
TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]
GAUGED_CONDITIONS = ["encoded_pre_evolution", "evolved_T", "evolved_lattice",
                     "evolved_rewired", "evolved_curr_random"]
GAUGES = ["reference", "circular_mean"]
CHUNK_SIZE = 1000          # as in stage3_gpu_evolve.py -- memory, not speed
N_UPLOAD_CHUNKS = 12

print("JAX backend:", jax.default_backend(), "| x64:", jax.config.jax_enable_x64,
      flush=True)
print("devices:", jax.devices(), flush=True)

# -------------------------------------------------------------------- inputs
with open('/content/stage3_topologies.pkl', 'rb') as f:
    topologies = pickle.load(f)
theta0 = np.concatenate(
    [np.load(f'/content/theta0_chunk_{i:02d}.npy') for i in range(N_UPLOAD_CHUNKS)],
    axis=0)
labels = np.load('/content/stage3_labels.npy')
ref_idx = int(np.load('/content/stage3_ref_idx.npy'))
n_images = theta0.shape[0]
print(f"n_images={n_images} n_nodes={theta0.shape[1]} ref_idx={ref_idx} "
      f"topologies={list(topologies.keys())}", flush=True)
assert labels.shape[0] == n_images, (labels.shape, n_images)

# ------------------------------------------------------------------- evolve
warmup = jnp.asarray(theta0[:2])
_w, _s = batched_evolve_on_graph_jax(warmup, jnp.asarray(topologies[TOPOLOGY_NAMES[0]]))
jax.block_until_ready(_w)
print("warm-up compile done", flush=True)

theta_T, success = {}, {}
evolve_seconds = {}
for name in TOPOLOGY_NAMES:
    W = jnp.asarray(topologies[name])
    t0 = time.perf_counter()
    tc, sc = [], []
    for start in range(0, n_images, CHUNK_SIZE):
        chunk = jnp.asarray(theta0[start:start + CHUNK_SIZE])
        th, su = batched_evolve_on_graph_jax(chunk, W)
        jax.block_until_ready(th)
        tc.append(np.asarray(th))
        sc.append(np.asarray(su))
        if (start // CHUNK_SIZE) % 10 == 0:
            print(f"  [{name}] {start + len(chunk)}/{n_images}", flush=True)
    theta_T[name] = np.concatenate(tc, axis=0)
    success[name] = np.concatenate(sc, axis=0).astype(bool)
    evolve_seconds[name] = time.perf_counter() - t0
    print(f"[evolve] {name}: {evolve_seconds[name]:.3f}s  "
          f"n_failed={int((~success[name]).sum())}", flush=True)

# -------------------------------------------------------------------- gauges
def gauge_features(theta_batch, gauge):
    """Both gauges come from stage2a_core_colab, the deployable fork pinned
    bit-for-bit to stage2a_core. Not reimplemented here, and in particular the
    inlined jnp gauge in analyze_stage3_results_jax.py is not ported -- that
    is the caller-side glue this project has been bitten by."""
    if gauge == "reference":
        return np.stack([s2a.reference_node_features(t, ref_idx) for t in theta_batch])
    if gauge == "circular_mean":
        return np.stack([s2a.circular_mean_features(t) for t in theta_batch])
    raise ValueError(gauge)


def build_X(condition, gauge):
    if condition == "encoded_pre_evolution":
        return gauge_features(theta0, gauge), labels
    topo = condition[len("evolved_"):]
    m = success[topo]
    return gauge_features(theta_T[topo][m], gauge), labels[m]


# ------------------------------------------------- CV recording accuracy too
def cv_with_accuracy_jax(X, y, label):
    """select_C_via_cv_jax, additionally recording validation accuracy.

    Mirrors that function rather than calling it, because it computes no
    accuracy and discards its fitted parameters. Every solve goes through
    the real _solve_grid_for_fold, and the folds, scaler placement, grid,
    tie-break and convergence gate are the module's own.
    """
    classes = sorted(set(y.tolist()))
    n_classes = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    skf = StratifiedKFold(n_splits=clf_jax.N_FOLDS, shuffle=True,
                          random_state=clf_jax.SEED)
    per_C_loss = {C: [] for C in clf_jax.C_GRID}
    per_C_acc = {C: [] for C in clf_jax.C_GRID}
    fits = []

    for fold_idx, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = jnp.asarray(scaler.transform(X_tr), dtype=jnp.float64)
        X_va_s = jnp.asarray(scaler.transform(X_va), dtype=jnp.float64)
        y_tr_idx = np.array([c2i[v] for v in y_tr.tolist()])
        y_oh = jnp.asarray(np.eye(n_classes)[y_tr_idx], dtype=jnp.float64)

        t0 = time.perf_counter()
        params, n_iter, gnorm, converged, tol_used = clf_jax._solve_grid_for_fold(
            clf_jax.C_GRID, X_tr_s, y_oh, X_tr_s.shape[1], n_classes,
            max_iter=clf_jax.MAX_ITER)
        jax.block_until_ready(params[0])
        fold_seconds = time.perf_counter() - t0

        for ci, C in enumerate(clf_jax.C_GRID):
            if not bool(converged[ci]):
                raise clf_jax.NonConvergenceError(
                    f"[{label}] fold={fold_idx} C={C}: ||grad||="
                    f"{float(gnorm[ci]):.3e} > {float(tol_used[ci]):.3e} after "
                    f"{int(n_iter[ci])} iterations")
            proba = np.asarray(clf_jax._predict_proba(
                (params[0][ci], params[1][ci]), X_va_s))
            per_C_loss[C].append(log_loss(y_va, proba, labels=classes))
            per_C_acc[C].append(float(np.mean(
                np.asarray(classes)[proba.argmax(axis=1)] == y_va)))
            fits.append({"fold": fold_idx, "C": float(C),
                         "loss": per_C_loss[C][-1], "acc": per_C_acc[C][-1],
                         "n_iter": int(n_iter[ci]), "grad_norm": float(gnorm[ci])})
        print(f"    [{label}] fold={fold_idx} whole 9-C grid in "
              f"{fold_seconds:.1f}s", flush=True)

    mean_loss = {C: float(np.mean(v)) for C, v in per_C_loss.items()}
    mean_acc = {C: float(np.mean(v)) for C, v in per_C_acc.items()}
    lo = min(mean_loss.values())
    best_C = min([C for C, v in mean_loss.items() if abs(v - lo) < 1e-12])
    return best_C, mean_loss, mean_acc, fits


# ---------------------------------------------------------------- the arms
out = {"n_images": int(n_images), "ref_idx": ref_idx,
       "evolve_seconds": evolve_seconds,
       "jax_backend": jax.default_backend(), "arms": {}}
job_t0 = time.perf_counter()

for condition in GAUGED_CONDITIONS:
    for gauge in GAUGES:
        key = f"{condition}__{gauge}"
        t0 = time.perf_counter()
        X, y = build_X(condition, gauge)
        t_feat = time.perf_counter() - t0
        print(f"\n[{key}] X={X.shape} n={len(y)} features in {t_feat:.1f}s",
              flush=True)
        t0 = time.perf_counter()
        try:
            best_C, mean_loss, mean_acc, fits = cv_with_accuracy_jax(X, y, key)
            out["arms"][key] = {
                "selected_C": best_C,
                "mean_val_loss_per_C": {str(C): v for C, v in mean_loss.items()},
                "mean_val_acc_per_C": {str(C): v for C, v in mean_acc.items()},
                "accuracy_at_selected_C": mean_acc[best_C],
                "n_valid": int(len(y)), "n_features": int(X.shape[1]),
                "elapsed_seconds": time.perf_counter() - t0,
                "feature_seconds": t_feat, "fits": fits,
                "non_convergence": None,
            }
            print(f"[{key}] selected_C={best_C} acc={mean_acc[best_C]:.6f} "
                  f"loss={mean_loss[best_C]:.6f} in "
                  f"{out['arms'][key]['elapsed_seconds']:.1f}s", flush=True)
        except clf_jax.NonConvergenceError as exc:
            # The module's own stop-gate. Recorded as a result, not patched
            # around by raising max_iter.
            out["arms"][key] = {"non_convergence": str(exc),
                                "elapsed_seconds": time.perf_counter() - t0}
            print(f"[{key}] NON-CONVERGENCE: {exc}", flush=True)

out["total_seconds"] = time.perf_counter() - job_t0
with open('/content/gauge_comparison_2a_gpu.json', 'w') as fh:
    json.dump(out, fh, indent=1)

n_ok = sum(1 for a in out["arms"].values() if a.get("non_convergence") is None)
print(f"\ntotal job time: {out['total_seconds']:.1f}s "
      f"({out['total_seconds'] / 3600:.2f}h) for {len(out['arms'])} arms, "
      f"{n_ok} converged", flush=True)
print(SENTINEL, flush=True)
