"""
GPU single-image inference-latency measurement feeding
`COMPUTE_COST_FINDINGS.md`'s compute-cost accounting -- not the
confirmatory result. Different provenance from `stage3_gpu_evolve.py`/
`stage4_gpu_evolve.py`: this script's classifiers are synthetic
(shape-only stand-ins, never real predictors, see the inline comment
below), the goal is timing the encode+evolve+classify pipeline's
per-image latency at batch=1, not producing real evolved states.

Not runnable locally as-is -- executes ON the remote GPU session
(uploaded via `mighty-colab upload`, run via `mighty-colab exec`,
paired with `prep_oscillator_latency_gpu_inputs.py`, which stages this
script's `/content/...` inputs); see `README.md`'s "Reproducing the
confirmatory GPU evolution" for the general upload/exec pattern and
`COMPUTE_COST_FINDINGS.md`'s "Code" section for this thread's exact run
order. Reuses `evolve_on_graph_jax.py` (uploaded alongside, unmodified)
-- not a reimplementation. The `local_converged_phases`/
`reference_node_features`/`encode_and_restrict` functions below are
inline verbatim copies of the real numpy implementations (disclosed,
not reimplementations from scratch), to avoid installing the whole
`bonsai` package + other experiment folders on this fresh session for
three small, self-contained functions.
"""
import sys
sys.path.insert(0, '/content')
import pickle
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evolve_on_graph_jax import batched_evolve_on_graph_jax

print("JAX backend:", jax.default_backend())

# Inline copies (verbatim, not reimplemented) of the real numpy functions --
# avoids installing the whole bonsai package + other experiment folders on
# this fresh Colab session just for three small, self-contained functions.

def local_converged_phases(image, steps=150, dt=0.1, k_coupling=1.0, k_bias=1.0,
                            perturbation_std=0.01, seed=0):
    target_phase = image * np.pi
    rng = np.random.default_rng(seed)
    phases = (target_phase + rng.normal(0, perturbation_std, target_phase.shape)) % (2 * np.pi)
    for _ in range(steps):
        coupling = np.zeros_like(phases)
        coupling[1:, :] += np.sin(phases[:-1, :] - phases[1:, :])
        coupling[:-1, :] += np.sin(phases[1:, :] - phases[:-1, :])
        coupling[:, 1:] += np.sin(phases[:, :-1] - phases[:, 1:])
        coupling[:, :-1] += np.sin(phases[:, 1:] - phases[:, :-1])
        bias = np.sin(target_phase - phases)
        dtheta = k_coupling * coupling + k_bias * bias
        phases = (phases + dt * dtheta) % (2 * np.pi)
    return phases


def reference_node_features(theta, ref_idx):
    shifted = theta - theta[ref_idx]
    cos_part = np.cos(shifted)
    sin_part = np.sin(shifted)
    cos_part = np.delete(cos_part, ref_idx)
    sin_part = np.delete(sin_part, ref_idx)
    return np.concatenate([cos_part, sin_part])


def encode_and_restrict(image_01, active_indices, seed=0):
    theta_0_784 = local_converged_phases(image_01, seed=seed).flatten()
    return theta_0_784[active_indices]


print("Loading data...")
image_01 = np.load('/content/raw_image.npy')
active_indices = np.load('/content/active_indices.npy')
with open('/content/topologies_for_latency.pkl', 'rb') as f:
    d = pickle.load(f)
topologies = d["topologies"]
ref_idx = d["ref_idx"]
print(f"image_01={image_01.shape}, active_indices={active_indices.shape}, ref_idx={ref_idx}")

print("Fitting synthetic-data classifiers (shape-only, not real predictors)...")
rng = np.random.default_rng(0)
synthetic_clf = {}
for name in topologies:
    X_synth = rng.standard_normal((200, 1008))
    y_synth = rng.integers(0, 10, size=200)
    scaler = StandardScaler().fit(X_synth)
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    clf.fit(scaler.transform(X_synth), y_synth)
    synthetic_clf[name] = (scaler, clf)

N_REPEATS = 100
W_jax = {name: jnp.asarray(W) for name, W in topologies.items()}

print("\nWarm-up (untimed, excludes JIT compilation)...")
theta0_warmup = encode_and_restrict(image_01, active_indices, seed=0)
for name, W in W_jax.items():
    theta0_batch = jnp.asarray(theta0_warmup[None, :])
    theta_T, success = batched_evolve_on_graph_jax(theta0_batch, W)
    jax.block_until_ready(theta_T)
print("Warm-up done.")

print(f"\nMeasuring GPU single-image (batch=1) pipeline latency, {N_REPEATS} repeats per topology...\n")
results = {}
for name, W in W_jax.items():
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        theta0 = encode_and_restrict(image_01, active_indices, seed=0)
        theta0_batch = jnp.asarray(theta0[None, :])
        theta_T_batch, success = batched_evolve_on_graph_jax(theta0_batch, W)
        theta_T_batch = jax.block_until_ready(theta_T_batch)
        theta_T = np.asarray(theta_T_batch)[0]
        assert bool(np.asarray(success)[0]), "unexpected solver failure during latency measurement"
        feat = reference_node_features(theta_T, ref_idx)
        scaler, clf = synthetic_clf[name]
        proba = clf.predict_proba(scaler.transform(feat.reshape(1, -1)))
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    results[name] = {"mean_ms": float(times.mean() * 1000), "std_ms": float(times.std() * 1000),
                      "min_ms": float(times.min() * 1000), "max_ms": float(times.max() * 1000)}
    print(f"[{name}] mean={results[name]['mean_ms']:.2f}ms, std={results[name]['std_ms']:.2f}ms, "
          f"min={results[name]['min_ms']:.2f}ms, max={results[name]['max_ms']:.2f}ms")

with open('/content/gpu_latency_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nSaved gpu_latency_results.pkl")
