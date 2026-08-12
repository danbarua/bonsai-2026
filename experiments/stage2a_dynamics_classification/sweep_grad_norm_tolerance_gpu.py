"""What does the JAX classifier do if it is not allowed to stop early?

The gauge replication found JAX and sklearn interchangeable at low C and up to
1.2e-02 apart at high C, with JAX always the pessimistic side, and a
three-way factorial pinned the cause on the convergence criterion rather than
the platform: `stage2a_classifier_jax` stops at
``GRAD_NORM_REL * C * n_train``, which at C=1000 and n_train=48,000 is
2.88e+05. The fit halted at ``grad_norm = 2.814e+05`` after 2,003 iterations
where sklearn's fixed ``tol=1e-4`` ran 5,309.

GRAD_NORM_REL=6e-3 was calibrated as roughly twice the ``||grad||/(C*n_train)``
sklearn itself achieves, so this is not a bug -- it is a deliberately
scale-free threshold set about a factor of two loose. This asks what the
number would be if it were not.

PHASE 1 (`sweep`) sweeps GRAD_NORM_REL on the single most divergent cell and
watches accuracy against sklearn's. If the divergence is the tolerance, the
curve should walk onto sklearn's value as the threshold tightens, and the
iteration count should climb toward sklearn's. If it plateaus somewhere else,
the tolerance is not the whole story and the factorial's verdict needs
revisiting.

PHASE 2 (`arms`) re-runs all ten gauge-comparison arms at a chosen tolerance.
The question there is different and sharper: the class C verdict in
GAUGE_COMPARISON_2A_RESULT.md rests on a 0.000616 inversion that the default
JAX run did not reproduce. If a tightened JAX *does* reproduce it, the verdict
is classifier-independent after all and only the early stop obscured it.

Runs ON the remote session. Set PHASE=sweep|arms.
"""
import sys
sys.path.insert(0, '/content')
import json
import os
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

SENTINEL = "GRADSWEEP_OK"
PHASE = os.environ.get("PHASE", "sweep")
# Descending, so the first value reproduces the default run and the rest
# tighten from there. 6e-3 is the shipped default.
TOL_GRID = [float(x) for x in os.environ.get(
    "TOL_GRID", "6e-3,1e-3,1e-4,1e-5,1e-6").split(",")]
ARMS_TOL = float(os.environ.get("ARMS_TOL", "1e-5"))
# Which arms this session runs. Empty = all ten. Arms are independent, so
# splitting them across sessions is pure wall-clock win at identical cost.
ARMS_SUBSET = [a for a in os.environ.get("ARMS", "").split(",") if a]
# When set, pull the 250MB input from GCS (4.1s in-cloud) instead of
# reassembling twelve uploaded chunks (205s from the caller).
GCS_OBJECT = os.environ.get("BENCH_OBJECT", "")
GCS_BUCKET = os.environ.get("BONSAI_GCS_BUCKET", "")

TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]
GAUGED_CONDITIONS = ["encoded_pre_evolution", "evolved_T", "evolved_lattice",
                     "evolved_rewired", "evolved_curr_random"]
GAUGES = ["reference", "circular_mean"]
CHUNK_SIZE = 1000
N_UPLOAD_CHUNKS = 12

# The cell phase 1 interrogates, and the sklearn values it is chasing.
SWEEP_TOPOLOGY, SWEEP_C, SWEEP_FOLD = "lattice", 1000.0, 0
SKLEARN_ACC, SKLEARN_ITER = 0.88225, 5309

print("JAX backend:", jax.default_backend(), "| x64:", jax.config.jax_enable_x64,
      flush=True)
print(f"PHASE={PHASE}", flush=True)

# -------------------------------------------------------------------- inputs
if GCS_OBJECT:
    from google.cloud import storage
    t0 = time.perf_counter()
    blob = storage.Client.create_anonymous_client().bucket(GCS_BUCKET).blob(GCS_OBJECT)
    blob.download_to_filename('/content/stage3_gpu_upload.pkl')
    n_bytes = os.path.getsize('/content/stage3_gpu_upload.pkl')
    with open('/content/stage3_gpu_upload.pkl', 'rb') as f:
        payload = pickle.load(f)
    theta0 = np.asarray(payload["theta0_batch"])
    topologies = payload["topologies"]
    print(f"pulled {n_bytes:,} bytes from GCS in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
else:
    with open('/content/stage3_topologies.pkl', 'rb') as f:
        topologies = pickle.load(f)
    theta0 = np.concatenate(
        [np.load(f'/content/theta0_chunk_{i:02d}.npy') for i in range(N_UPLOAD_CHUNKS)],
        axis=0)
labels = np.load('/content/stage3_labels.npy')
ref_idx = int(np.load('/content/stage3_ref_idx.npy'))
n_images = theta0.shape[0]
print(f"n_images={n_images} ref_idx={ref_idx}", flush=True)

# ------------------------------------------------------------------- evolve
warm, _ = batched_evolve_on_graph_jax(jnp.asarray(theta0[:2]),
                                      jnp.asarray(topologies[TOPOLOGY_NAMES[0]]))
jax.block_until_ready(warm)
print("warm-up compile done", flush=True)

if PHASE == "arms":
    wanted = ARMS_SUBSET or [f"{c}__{g}" for c in GAUGED_CONDITIONS for g in GAUGES]
    needed = sorted({k.split("__")[0][len("evolved_"):] for k in wanted
                     if k.startswith("evolved_")})
else:
    needed = [SWEEP_TOPOLOGY]
theta_T, success = {}, {}
for name in needed:
    W = jnp.asarray(topologies[name])
    t0 = time.perf_counter()
    tc, sc = [], []
    for start in range(0, n_images, CHUNK_SIZE):
        th, su = batched_evolve_on_graph_jax(
            jnp.asarray(theta0[start:start + CHUNK_SIZE]), W)
        jax.block_until_ready(th)
        tc.append(np.asarray(th)); sc.append(np.asarray(su))
        if (start // CHUNK_SIZE) % 15 == 0:
            print(f"  [{name}] {start}/{n_images}", flush=True)
    theta_T[name] = np.concatenate(tc, axis=0)
    success[name] = np.concatenate(sc, axis=0).astype(bool)
    print(f"[evolve] {name}: {time.perf_counter() - t0:.1f}s "
          f"n_failed={int((~success[name]).sum())}", flush=True)


def gauge_features(batch, gauge):
    if gauge == "reference":
        return np.stack([s2a.reference_node_features(t, ref_idx) for t in batch])
    return np.stack([s2a.circular_mean_features(t) for t in batch])


def fold_arrays(X, y, fold):
    skf = StratifiedKFold(n_splits=clf_jax.N_FOLDS, shuffle=True,
                          random_state=clf_jax.SEED)
    tr, va = list(skf.split(X, y))[fold]
    scaler = StandardScaler().fit(X[tr])
    classes = sorted(set(y.tolist()))
    c2i = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([c2i[v] for v in y[tr].tolist()])
    return (jnp.asarray(scaler.transform(X[tr]), dtype=jnp.float64),
            jnp.asarray(scaler.transform(X[va]), dtype=jnp.float64),
            jnp.asarray(np.eye(len(classes))[y_idx], dtype=jnp.float64),
            y[va], classes)


def fit(c_grid, X_tr, y_oh, n_classes, tol_rel):
    return clf_jax._solve_grid_for_fold(
        c_grid, X_tr, y_oh, X_tr.shape[1], n_classes,
        max_iter=clf_jax.MAX_ITER, grad_norm_rel=tol_rel)


out = {"phase": PHASE, "backend": jax.default_backend()}

# ------------------------------------------------------------------ phase 1
if PHASE == "sweep":
    m = success[SWEEP_TOPOLOGY]
    X = gauge_features(theta_T[SWEEP_TOPOLOGY][m], "reference")
    X_tr, X_va, y_oh, y_va, classes = fold_arrays(X, labels[m], SWEEP_FOLD)
    print(f"\nsweeping GRAD_NORM_REL at C={SWEEP_C:g} fold={SWEEP_FOLD}; "
          f"sklearn reached acc={SKLEARN_ACC} in {SKLEARN_ITER} iterations\n",
          flush=True)
    rows = []
    for tol_rel in TOL_GRID:
        t0 = time.perf_counter()
        params, n_iter, gnorm, conv, tol_used = fit(
            (SWEEP_C,), X_tr, y_oh, len(classes), tol_rel)
        jax.block_until_ready(params[0])
        el = time.perf_counter() - t0
        proba = np.asarray(clf_jax._predict_proba((params[0][0], params[1][0]), X_va))
        acc = float(np.mean(np.asarray(classes)[proba.argmax(axis=1)] == y_va))
        row = {"grad_norm_rel": tol_rel, "acc": acc, "n_iter": int(n_iter[0]),
               "grad_norm": float(gnorm[0]), "grad_tol": float(tol_used[0]),
               "converged": bool(conv[0]), "seconds": el,
               "delta_vs_sklearn": acc - SKLEARN_ACC,
               "hit_max_iter": int(n_iter[0]) >= clf_jax.MAX_ITER}
        rows.append(row)
        print(f"  rel={tol_rel:<8g} tol={row['grad_tol']:>10.3e} "
              f"iter={row['n_iter']:>6} grad={row['grad_norm']:>10.3e} "
              f"acc={acc:.6f} d_sk={row['delta_vs_sklearn']:>+9.6f} "
              f"conv={row['converged']} ({el:.0f}s)", flush=True)
    out["sklearn_acc"] = SKLEARN_ACC
    out["sklearn_n_iter"] = SKLEARN_ITER
    out["sweep"] = rows

# ------------------------------------------------------------------ phase 2
else:
    print(f"\nre-running all ten arms at GRAD_NORM_REL={ARMS_TOL:g}\n", flush=True)
    arms = {}
    for cond in GAUGED_CONDITIONS:
        for gauge in GAUGES:
            key = f"{cond}__{gauge}"
            if ARMS_SUBSET and key not in ARMS_SUBSET:
                continue
            if cond == "encoded_pre_evolution":
                X, y = gauge_features(theta0, gauge), labels
            else:
                topo = cond[len("evolved_"):]
                m = success[topo]
                X, y = gauge_features(theta_T[topo][m], gauge), labels[m]

            skf = StratifiedKFold(n_splits=clf_jax.N_FOLDS, shuffle=True,
                                  random_state=clf_jax.SEED)
            per_C_loss = {C: [] for C in clf_jax.C_GRID}
            per_C_acc = {C: [] for C in clf_jax.C_GRID}
            t0 = time.perf_counter()
            classes = sorted(set(y.tolist()))
            c2i = {c: i for i, c in enumerate(classes)}
            halted = None
            for fi, (tr, va) in enumerate(skf.split(X, y)):
                scaler = StandardScaler().fit(X[tr])
                X_tr = jnp.asarray(scaler.transform(X[tr]), dtype=jnp.float64)
                X_va = jnp.asarray(scaler.transform(X[va]), dtype=jnp.float64)
                y_idx = np.array([c2i[v] for v in y[tr].tolist()])
                y_oh = jnp.asarray(np.eye(len(classes))[y_idx], dtype=jnp.float64)
                params, n_iter, gnorm, conv, tol_used = fit(
                    clf_jax.C_GRID, X_tr, y_oh, len(classes), ARMS_TOL)
                jax.block_until_ready(params[0])
                for ci, C in enumerate(clf_jax.C_GRID):
                    if not bool(conv[ci]):
                        # At a tight tolerance this is EXPECTED, and it is the
                        # measurement: record it rather than raising, so the
                        # sweep shows where max_iter starts binding.
                        halted = (f"fold={fi} C={C} did not reach "
                                  f"{float(tol_used[ci]):.3e} in "
                                  f"{int(n_iter[ci])} iterations "
                                  f"(grad={float(gnorm[ci]):.3e})")
                    proba = np.asarray(clf_jax._predict_proba(
                        (params[0][ci], params[1][ci]), X_va))
                    per_C_loss[C].append(log_loss(y[va], proba, labels=classes))
                    per_C_acc[C].append(float(np.mean(
                        np.asarray(classes)[proba.argmax(axis=1)] == y[va])))
                print(f"    [{key}] fold={fi} done "
                      f"(max iter {int(max(n_iter))})", flush=True)
            mean_loss = {C: float(np.mean(v)) for C, v in per_C_loss.items()}
            mean_acc = {C: float(np.mean(v)) for C, v in per_C_acc.items()}
            lo = min(mean_loss.values())
            best_C = min([C for C, v in mean_loss.items() if abs(v - lo) < 1e-12])
            arms[key] = {"selected_C": best_C,
                         "accuracy_at_selected_C": mean_acc[best_C],
                         "mean_val_loss_per_C": {str(C): v for C, v in mean_loss.items()},
                         "mean_val_acc_per_C": {str(C): v for C, v in mean_acc.items()},
                         "elapsed_seconds": time.perf_counter() - t0,
                         "non_convergence": halted}
            print(f"[{key}] selected_C={best_C} acc={mean_acc[best_C]:.6f} "
                  f"in {arms[key]['elapsed_seconds']:.0f}s"
                  + (f"  NOTE: {halted}" if halted else ""), flush=True)
    out["grad_norm_rel"] = ARMS_TOL
    out["arms"] = arms

dest = f"/content/grad_sweep_{PHASE}{os.environ.get('OUT_SUFFIX', '')}.json"
with open(dest, "w") as fh:
    json.dump(out, fh, indent=1)
print(f"\nwrote {dest}", flush=True)
print(SENTINEL, flush=True)
