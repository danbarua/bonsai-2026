"""
COMPUTE_COST_DESIGN.md item 3: MLP single-image inference latency, CPU
only (per the check-0/(a) decision -- no GPU counterpart).

Same shape-only-fit rationale as the oscillator latency script: predict_proba's
wall-clock cost depends only on matrix dimensions, not fitted parameter
values, so a quick fit (not the locked, fully-converged one) is sufficient
to time this step honestly.
"""
import os
import time

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
N_REPEATS = 100

rng = np.random.default_rng(0)
X_synth = rng.standard_normal((200, 784))
y_synth = rng.integers(0, 10, size=200)
scaler = StandardScaler().fit(X_synth)
X_synth_s = scaler.transform(X_synth)

one_image = rng.standard_normal((1, 784))
one_image_s = scaler.transform(one_image)

results = {}
for H, label in [(13, "MLP_H13"), (128, "MLP_H128")]:
    clf = MLPClassifier(hidden_layer_sizes=(H,), max_iter=50, random_state=42)
    clf.fit(X_synth_s, y_synth)

    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        proba = clf.predict_proba(one_image_s)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    results[label] = {"mean_ms": float(times.mean() * 1000), "std_ms": float(times.std() * 1000),
                       "min_ms": float(times.min() * 1000), "max_ms": float(times.max() * 1000)}
    print(f"[{label}] mean={results[label]['mean_ms']:.4f}ms, std={results[label]['std_ms']:.4f}ms, "
          f"min={results[label]['min_ms']:.4f}ms, max={results[label]['max_ms']:.4f}ms")

import pickle
out_path = os.path.join(_THIS_DIR, "results", "mlp_cpu_latency_results.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved {out_path}")
