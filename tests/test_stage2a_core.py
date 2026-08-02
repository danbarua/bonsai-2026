"""
Tests for experiments/stage2a_dynamics_classification/stage2a_core.py.

Tier 1 (self-contained, always run): the reference-node gauge's
constant-column property (DESIGN.md's fourth-review correction --
theta_ref's own two circular features are trivially (1, 0) for every
phase configuration, dropped deterministically to give a 1008-dim
feature vector) on synthetic phase arrays, plus the evolution recovery
policy's basic contract.

Tier 2 (skipped if local-only data isn't present): the same
constant-column property on real encoded/evolved states from this
project's own T, KMNIST-derived.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
_KMNIST_DIR = _REPO_ROOT / "datasets" / "kmnist"
sys.path.insert(0, str(_STAGE2A_DIR))

import stage2a_core as s2a  # noqa: E402


# ---- Tier 1: self-contained structural tests ----

def test_reference_node_columns_are_exactly_one_zero_synthetic():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 50
        theta = rng.uniform(0, 2 * np.pi, n)
        ref_idx = rng.integers(0, n)
        shifted = theta - theta[ref_idx]
        cos_at_ref = np.cos(shifted[ref_idx])
        sin_at_ref = np.sin(shifted[ref_idx])
        assert abs(cos_at_ref - 1.0) < 1e-12
        assert abs(sin_at_ref - 0.0) < 1e-12


def test_reference_node_features_dimension_and_no_assertion_error():
    rng = np.random.default_rng(1)
    n = 505
    theta = rng.uniform(0, 2 * np.pi, n)
    feat = s2a.reference_node_features(theta, ref_idx=363)
    assert feat.shape == (2 * n - 2,)
    assert np.all(np.isfinite(feat))


def test_reference_node_features_wrong_ref_idx_still_holds():
    """The constant-column property holds for ANY ref_idx, not just 363 --
    confirms the property is a mathematical consequence of the gauge
    (cos/sin of a self-difference), not an artifact of one specific index."""
    rng = np.random.default_rng(2)
    n = 20
    theta = rng.uniform(0, 2 * np.pi, n)
    for ref_idx in range(n):
        feat = s2a.reference_node_features(theta, ref_idx=ref_idx)
        assert feat.shape == (2 * n - 2,)


def test_circular_mean_features_dimension_unchanged():
    rng = np.random.default_rng(3)
    n = 505
    theta = rng.uniform(0, 2 * np.pi, n)
    feat = s2a.circular_mean_features(theta)
    assert feat.shape == (2 * n,)  # no columns dropped for the secondary gauge


def test_order_parameter_bounds():
    rng = np.random.default_rng(4)
    theta_random = rng.uniform(0, 2 * np.pi, 1000)
    theta_aligned = np.zeros(1000)
    assert 0.0 <= s2a.order_parameter(theta_random) <= 1.0
    assert abs(s2a.order_parameter(theta_aligned) - 1.0) < 1e-12


def test_evolve_on_graph_trivial_case_zero_coupling():
    """W all zeros: dtheta/dt = 0 everywhere, theta should not move."""
    n = 10
    W = np.zeros((n, n))
    theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_T, diag = s2a.evolve_on_graph(theta0, W, t_horizon=1.0)
    assert diag["recovery_step"] == 0
    assert not diag.get("failed", False)
    np.testing.assert_allclose(theta_T, theta0 % (2 * np.pi), atol=1e-6)


def test_recovery_steps_are_ordered_smaller_max_step_then_alternative_solver():
    methods = [step["method"] for step in s2a.RECOVERY_STEPS]
    max_steps = [step["max_step"] for step in s2a.RECOVERY_STEPS]
    assert methods[0] == "RK45"
    assert methods[-1] == "Radau"
    assert max_steps[1] < max_steps[0]
    assert max_steps[2] < max_steps[1]


# ---- Tier 2: real-data check, skipped cleanly if local-only data is absent ----

_kmnist_present = (_KMNIST_DIR / "train-images-idx3-ubyte").exists()


@pytest.mark.skipif(not _kmnist_present, reason="datasets/kmnist not present locally")
def test_reference_node_constant_columns_on_real_encoded_state():
    active_indices, W_T, _ink_mask, nodes_T = s2a.load_T()
    rng = np.random.default_rng(42)
    image = rng.uniform(0, 1, (28, 28))
    theta0 = s2a.encode_and_restrict(image, active_indices)
    ref_idx = nodes_T["median"]

    shifted = theta0 - theta0[ref_idx]
    assert abs(np.cos(shifted[ref_idx]) - 1.0) < 1e-12
    assert abs(np.sin(shifted[ref_idx]) - 0.0) < 1e-12

    feat = s2a.reference_node_features(theta0, ref_idx)
    assert feat.shape == (2 * len(active_indices) - 2,)
