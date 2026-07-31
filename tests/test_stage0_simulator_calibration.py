"""Verifies the quantitative claims in experiments/stage0_simulator_calibration/FINDINGS.md."""
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from bonsai.dynamics.graph_oscillator_field import (
    GraphOscillatorField, find_equilibrium_lbfgs, joint_tangent_matrix_response,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTRUCTIONS_PATH = (
    REPO_ROOT / "experiments" / "stage1b2_structured_transformation" / "results"
    / "class0_constructions.pkl"
)

# Matches stage1b_taxonomy.py's same_attractor(), whose DEDUP_RESIDUAL_THRESHOLD
# comment states it "matches Stage 0's multistability dedup rule".
DEDUP_RESIDUAL_THRESHOLD = 0.05


def same_attractor(a, b):
    shift = np.angle(np.mean(np.exp(1j * (a - b))))
    residual = np.angle(np.exp(1j * (a - b - shift)))
    return np.mean(np.abs(residual)) < DEDUP_RESIDUAL_THRESHOLD


def dedup_equilibria(equilibria):
    distinct = []
    for eq in equilibria:
        if not any(same_attractor(eq, d) for d in distinct):
            distinct.append(eq)
    return distinct


@pytest.fixture(scope="module")
def kmnist_class0_topology():
    # No stage-0-specific topology cache exists in this checkout; this is the
    # same KMNIST class-0 T construction FINDINGS.md's own "Reproducing these
    # results" section describes, cached here by the later Stage 1B.2 stage.
    if not CONSTRUCTIONS_PATH.exists():
        pytest.skip(f"KMNIST class-0 topology cache not found at {CONSTRUCTIONS_PATH}")
    with open(CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    return data["constructions"]["T"], data["n_active"]


@pytest.fixture(scope="module")
def five_distinct_equilibria(kmnist_class0_topology):
    # Seeds 0-4: arbitrary, not the original run's seeds (unrecorded anywhere
    # in FINDINGS.md or docs/PROJECT_MEMORY.md) -- this reproduces the finding
    # (5 distinct equilibria from 5 initializations), not necessarily the same
    # 5 equilibria as the original run. Same caveat class as Stage 1B's
    # topology-cache substitution.
    W, n = kmnist_class0_topology
    equilibria = [find_equilibrium_lbfgs(W, k_coupling=1.0, seed=s)[0] for s in range(5)]
    return dedup_equilibria(equilibria)


def test_topology_is_single_connected_component(kmnist_class0_topology):
    W, n = kmnist_class0_topology
    n_components, _ = connected_components(W > 0, directed=False)
    assert n_components == 1


def test_five_seeds_give_five_distinct_equilibria(five_distinct_equilibria):
    assert len(five_distinct_equilibria) == 5


def test_equilibria_are_stable_with_small_spectral_gap(kmnist_class0_topology, five_distinct_equilibria):
    """Each recovered equilibrium: exactly one near-zero (rotation) mode,
    zero negative eigenvalues, and a small spectral gap -- matching the
    5.3e-3-5.8e-3 range recorded in FINDINGS.md (against a largest eigenvalue
    of ~13.27)."""
    W, n = kmnist_class0_topology
    field = GraphOscillatorField(W, k_coupling=1.0)

    for eq in five_distinct_equilibria:
        J = field.jacobian_at(eq)
        eigvals = np.linalg.eigvalsh(J)
        n_near_zero = np.sum(np.abs(eigvals) < 1e-6)
        n_negative = np.sum(eigvals < -1e-8)
        positive = eigvals[eigvals >= 1e-6]

        assert n_near_zero == 1
        assert n_negative == 0
        assert 1e-4 < positive.min() < 1e-2  # genuinely small, not degenerate


def test_rk45_and_dop853_agree_to_four_decimal_places(kmnist_class0_topology):
    """joint_tangent_matrix_response's method= parameter has documented this
    cross-validation use case since it was written (see its docstring), but
    a repository-wide search found method='DOP853' had never actually been
    invoked anywhere before this test -- this is the first place it's
    exercised."""
    W, n = kmnist_class0_topology
    rng = np.random.default_rng(2000)
    theta0 = rng.uniform(0, 2 * np.pi, n)
    t_eval = np.linspace(0, 2.5, 51)

    _, S_rk45, ok_rk45, _ = joint_tangent_matrix_response(
        W, theta0, [0], t_span=(0, 2.5), method="RK45",
        rtol=1e-8, atol=1e-10, t_eval=t_eval)
    _, S_dop853, ok_dop853, _ = joint_tangent_matrix_response(
        W, theta0, [0], t_span=(0, 2.5), method="DOP853",
        rtol=1e-8, atol=1e-10, t_eval=t_eval)

    assert ok_rk45
    assert ok_dop853
    assert np.max(np.abs(S_rk45 - S_dop853)) < 5e-5  # agreement to 4 decimal places
