"""
Tests for experiments/stage2a_dynamics_classification/stage2a_pipeline.py
and stage2a_pipeline_jax.py.

Tier 1 only (self-contained, always run, synthetic data):
check_go_no_go_multi_topology's failure/non-finite counting on
hand-built synthetic results with injected failures, subsample_stratified's
per-class counts and seed-determinism, and run_pipeline_multi_topology_jax's
result-dict shape/key contract on a tiny fully-synthetic case (skipped
if jax/diffrax aren't importable). Shape/key parity only, not numeric
equivalence against the numpy pipeline -- that's already covered by
verify_stage2a_pipeline_equivalence.py against real KMNIST data;
duplicating it here would be redundant test surface, not new coverage.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
sys.path.insert(0, str(_STAGE2A_DIR))

import stage2a_pipeline as pipe  # noqa: E402

TOPOLOGY_NAMES = ["T", "lattice", "rewired", "curr_random"]


def _make_synthetic_results(n_images, n_nodes, n_failed_per_topology, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    fail_idx = {
        name: set(rng.choice(n_images, size=n_failed_per_topology, replace=False).tolist())
        for name in TOPOLOGY_NAMES
    }
    for i in range(n_images):
        evolved = {}
        for name in TOPOLOGY_NAMES:
            if i in fail_idx[name]:
                evolved[name] = {"solver_failed": True, "R_post": None, "feat_post": None}
            else:
                evolved[name] = {
                    "solver_failed": False,
                    "R_post": float(rng.uniform(0, 1)),
                    "feat_post": rng.uniform(-1, 1, size=2 * n_nodes - 2),
                }
        results.append({
            "idx": i, "R_pre": float(rng.uniform(0, 1)),
            "feat_pre": rng.uniform(-1, 1, size=2 * n_nodes - 2),
            "raw_feat": rng.uniform(0, 1, size=784),
            "evolved": evolved,
        })
    return results, fail_idx


def test_check_go_no_go_multi_topology_counts_failures_correctly():
    n_images, n_nodes, n_failed = 50, 20, 3
    results, fail_idx = _make_synthetic_results(n_images, n_nodes, n_failed)
    report = pipe.check_go_no_go_multi_topology(results, TOPOLOGY_NAMES)

    assert report["n_images"] == n_images
    for name in TOPOLOGY_NAMES:
        pt = report["per_topology"][name]
        assert pt["n_solver_failed"] == n_failed
        assert pt["solver_failure_rate"] == pytest.approx(n_failed / n_images)
        # 3/50 = 0.06 > the locked 0.001 tolerance -- should read as NOT ok.
        assert pt["solver_failure_rate_ok"] is False
    assert report["solver_failure_rate_ok"] is False


def test_check_go_no_go_multi_topology_flags_non_finite_features():
    n_images, n_nodes = 20, 15
    results, _ = _make_synthetic_results(n_images, n_nodes, n_failed_per_topology=0)
    # Inject a non-finite value into one shared (pre-evolution) feature vector.
    results[5]["feat_pre"][0] = np.nan
    # And one evolved feature vector, for a topology that did not fail.
    results[7]["evolved"]["T"]["feat_post"][0] = np.inf

    report = pipe.check_go_no_go_multi_topology(results, TOPOLOGY_NAMES)
    assert report["n_non_finite_shared_feature_vectors"] == 1
    assert report["per_topology"]["T"]["n_non_finite_feature_vectors"] == 1
    assert report["per_topology"]["T"]["non_finite_ok"] is False
    assert report["per_topology"]["lattice"]["non_finite_ok"] is True
    assert report["non_finite_ok"] is False


def test_check_go_no_go_multi_topology_all_clean_passes():
    n_images, n_nodes = 30, 10
    results, _ = _make_synthetic_results(n_images, n_nodes, n_failed_per_topology=0)
    report = pipe.check_go_no_go_multi_topology(results, TOPOLOGY_NAMES)
    assert report["non_finite_ok"] is True
    assert report["solver_failure_rate_ok"] is True


def test_subsample_stratified_exact_per_class_counts():
    n_per_class = 20
    n_total = 1000
    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 255, size=(n_total, 16)).astype(np.uint8)
    y_train = rng.integers(0, 10, size=n_total)
    # Guarantee every class has at least n_per_class members.
    y_train[:200] = np.repeat(np.arange(10), 20)

    images_01, labels, selected_idx = pipe.subsample_stratified(
        X_train, y_train, seed=42, n_per_class=n_per_class)

    assert len(labels) == 10 * n_per_class
    for c in range(10):
        assert np.sum(labels == c) == n_per_class
    # selected_idx are real indices into the original array, dividing by
    # 255 must reproduce images_01 exactly.
    np.testing.assert_allclose(images_01, X_train[selected_idx].astype(np.float64) / 255.0)


def test_subsample_stratified_seed_determinism():
    n_total = 500
    rng = np.random.default_rng(1)
    X_train = rng.uniform(0, 255, size=(n_total, 8)).astype(np.uint8)
    y_train = np.repeat(np.arange(10), n_total // 10)

    _img1, _lab1, idx1 = pipe.subsample_stratified(X_train, y_train, seed=7, n_per_class=5)
    _img2, _lab2, idx2 = pipe.subsample_stratified(X_train, y_train, seed=7, n_per_class=5)
    _img3, _lab3, idx3 = pipe.subsample_stratified(X_train, y_train, seed=8, n_per_class=5)

    np.testing.assert_array_equal(idx1, idx2)
    assert not np.array_equal(idx1, idx3)


# ---- JAX pipeline shape/key contract, fully synthetic, no real data ----

try:
    import jax  # noqa: F401
    import diffrax  # noqa: F401
    _jax_available = True
except ImportError:
    _jax_available = False


@pytest.mark.skipif(not _jax_available, reason="jax/diffrax not importable")
def test_run_pipeline_multi_topology_jax_result_shape_and_keys():
    import stage2a_pipeline_jax as pipe_jax

    n_active = 12
    rng = np.random.default_rng(0)
    active_indices = np.sort(rng.choice(784, size=n_active, replace=False))
    ref_idx = 0

    # Small synthetic symmetric graph, not a real learned topology --
    # shape/key parity doesn't need real dynamics.
    W_raw = rng.uniform(0, 1, size=(n_active, n_active))
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0.0)
    topologies = {"synthetic": W}

    n_images = 3
    images_01 = rng.uniform(0, 1, size=(n_images, 28, 28))
    labels = rng.integers(0, 10, size=n_images)

    results = pipe_jax.run_pipeline_multi_topology_jax(
        images_01, labels, topologies, ref_idx, active_indices)

    assert len(results) == n_images
    for i, r in enumerate(results):
        assert r["idx"] == i
        assert isinstance(r["R_pre"], float)
        assert r["feat_pre"].shape == (2 * n_active - 2,)
        assert r["raw_feat"].shape == (784,)
        assert set(r["evolved"].keys()) == {"synthetic"}
        ev = r["evolved"]["synthetic"]
        assert set(ev.keys()) >= {"solver_failed", "R_post", "feat_post"}
        if not ev["solver_failed"]:
            assert np.asarray(ev["feat_post"]).shape == (2 * n_active - 2,)
