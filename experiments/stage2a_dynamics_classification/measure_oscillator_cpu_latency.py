"""
COMPUTE_COST_DESIGN.md item 2, CPU path: single-image (no batching)
encode + restrict + evolve + gauge-feature + linear-readout latency,
all 4 topologies, 100 repeats each, mean+std reported.

The fitted classifier/scaler used for the final readout step is fit on
SYNTHETIC random data of the correct shape (1008-dim, 10 classes) --
disclosed deliberately: predict_proba's wall-clock cost depends only on
matrix dimensions, not on the specific fitted parameter values or which
C was used, so a real (locked-C, real-data) fit is not needed to time
this step honestly. Only the encode/evolve/gauge steps use real data
and the real topologies.
"""
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2a_core as s2a
import stage2a_topologies as topo
from bonsai.data.mnist_loader import load_mnist

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
N_REPEATS = 100

print("Loading one representative official test image...")
_X_train, _y_train, X_test, _y_test = load_mnist(KMNIST_DIR, gz=False)
image_01 = X_test[0].astype(np.float64) / 255.0

print("Building all 4 topologies...")
active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
ref_idx = nodes_T["median"]
assert ref_idx == 363

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

print(f"\nMeasuring CPU single-image pipeline latency, {N_REPEATS} repeats per topology...\n")
results = {}
for name, W in topologies.items():
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        theta0 = s2a.encode_and_restrict(image_01, active_indices, seed=s2a.ENCODER_SEED)
        theta_T, diag = s2a.evolve_on_graph(theta0, W)
        assert theta_T is not None, "unexpected solver failure during latency measurement"
        feat = s2a.reference_node_features(theta_T, ref_idx)
        scaler, clf = synthetic_clf[name]
        proba = clf.predict_proba(scaler.transform(feat.reshape(1, -1)))
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    results[name] = {"mean_ms": float(times.mean() * 1000), "std_ms": float(times.std() * 1000),
                      "min_ms": float(times.min() * 1000), "max_ms": float(times.max() * 1000)}
    print(f"[{name}] mean={results[name]['mean_ms']:.2f}ms, std={results[name]['std_ms']:.2f}ms, "
          f"min={results[name]['min_ms']:.2f}ms, max={results[name]['max_ms']:.2f}ms")

import pickle
out_path = os.path.join(_THIS_DIR, "results", "cpu_latency_results.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nSaved {out_path}")
