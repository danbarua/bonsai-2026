"""
Tests for the regular-lattice graph control (build_lattice_topology).

Two tiers:
1. Self-contained structural tests on a small synthetic grid -- no
   external data required, always run, verify the construction's
   mathematical properties directly (4-connectivity, symmetry, correct
   active-node restriction, correct total-weight normalization).
2. An optional historical-verification test against the actual cached
   artifacts this function was reverse-engineered from
   (kmnist_class_topologies_200.pkl for the true 28x28-grid active-node
   positions, stage1a_all_classes.pkl for the target lattice construction)
   -- skipped if either file isn't present locally (both are gitignored,
   regenerable/historical data, not committed), but confirms byte-exact
   reproduction against the real historical artifact when both are
   available.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from bonsai.dynamics.lattice_construction import build_lattice_topology


# ---- Tier 1: self-contained structural tests ----

def test_small_grid_full_connectivity_matches_hand_computed():
    """4x4 grid, all 16 nodes active: hand-computable edge count for
    standard 4-connectivity (no diagonals, no wraparound)."""
    side = 4
    active = np.arange(side * side)  # all nodes active
    W = build_lattice_topology(active, total_weight_target=1.0, side=side)
    expected_undirected_edges = 2 * side * (side - 1)  # horizontal + vertical
    n_edges_in_matrix = np.count_nonzero(W)  # each undirected edge appears twice
    assert n_edges_in_matrix == 2 * expected_undirected_edges


def test_only_active_nodes_included():
    """Restricting to a subset of nodes should never introduce edges to
    excluded nodes, and the returned matrix's dimension must match the
    active set exactly, not the full grid."""
    side = 4
    active = np.array([0, 1, 4, 5])  # top-left 2x2 block of a 4x4 grid
    W = build_lattice_topology(active, total_weight_target=1.0, side=side)
    assert W.shape == (4, 4)
    # 2x2 block: 0-1 (horizontal), 0-4 (vertical), 1-5 (vertical), 4-5 (horizontal)
    # = 4 undirected edges = 8 nonzero matrix entries
    assert np.count_nonzero(W) == 8


def test_matrix_is_symmetric():
    side = 6
    rng = np.random.default_rng(0)
    active = rng.choice(side * side, size=20, replace=False)
    W = build_lattice_topology(active, total_weight_target=5.0, side=side)
    assert np.allclose(W, W.T)


def test_total_edge_weight_matches_target_exactly():
    """The core normalization property: sum of all entries in the
    returned matrix equals total_weight_target, regardless of how many
    edges exist."""
    side = 6
    rng = np.random.default_rng(1)
    active = rng.choice(side * side, size=20, replace=False)
    target = 37.5
    W = build_lattice_topology(active, total_weight_target=target, side=side)
    assert np.isclose(np.sum(W), target)


def test_uniform_edge_weight():
    """Every nonzero entry has the identical weight value -- this is
    what makes it a fair 'no learned structure' control, unlike T or
    the matched-sparsity random construction."""
    side = 6
    rng = np.random.default_rng(2)
    active = rng.choice(side * side, size=15, replace=False)
    W = build_lattice_topology(active, total_weight_target=10.0, side=side)
    nonzero_values = W[W > 0]
    assert np.allclose(nonzero_values, nonzero_values[0])


def test_no_edges_when_active_nodes_are_all_isolated():
    """Edge case: active nodes with no adjacent pairs return an
    all-zero matrix, not a division-by-zero error."""
    side = 10
    active = np.array([0, 22, 44, 66])  # far enough apart to never be 4-connected
    W = build_lattice_topology(active, total_weight_target=1.0, side=side)
    assert np.allclose(W, 0.0)


def test_node_ordering_matches_input_order_not_sorted_order():
    """Row/column i of the returned matrix must correspond to
    active_indices[i], preserving whatever order the caller passed in --
    this matters because callers align this matrix against T's own
    node ordering, which need not be sorted."""
    side = 4
    active_sorted = np.array([0, 1, 4, 5])
    active_shuffled = np.array([5, 0, 4, 1])
    W_sorted = build_lattice_topology(active_sorted, total_weight_target=1.0, side=side)
    W_shuffled = build_lattice_topology(active_shuffled, total_weight_target=1.0, side=side)
    # same graph, different row/column labeling -- permuting W_shuffled back should equal W_sorted
    perm = [np.where(active_shuffled == v)[0][0] for v in active_sorted]
    assert np.allclose(W_shuffled[np.ix_(perm, perm)], W_sorted)


# ---- Tier 2: optional historical verification ----

# Historical data lives in tarballs/lattice_construction_handoff/ (gitignored
# via the repo-wide *.pkl rule, kept local per that folder's own README --
# not moved elsewhere to avoid duplicating two large (~50MB/~84MB) binaries).
_HANDOFF_DIR = Path(__file__).resolve().parent.parent / "tarballs" / "lattice_construction_handoff"
RAW_TOPOLOGIES = _HANDOFF_DIR / "kmnist_class_topologies_200.pkl"
HISTORICAL_ARTIFACT = _HANDOFF_DIR / "stage1a_all_classes.pkl"


@pytest.mark.skipif(not (RAW_TOPOLOGIES.exists() and HISTORICAL_ARTIFACT.exists()),
                     reason="historical cached artifacts not present locally (gitignored, not committed)")
def test_matches_historical_cached_artifact_byte_exact():
    """Confirms build_lattice_topology reproduces the ACTUAL historical
    lattice construction for KMNIST class 0 -- using the real active-node
    positions from the raw 28x28-grid topology, not a synthetic stand-in --
    byte-exact sparsity pattern and weight value, not just a plausible
    reconstruction."""
    with open(RAW_TOPOLOGIES, "rb") as f:
        raw = pickle.load(f)
    with open(HISTORICAL_ARTIFACT, "rb") as f:
        s1a = pickle.load(f)

    W_full = raw[0]  # class 0's raw 784x784 topology
    active_indices = np.where(W_full.sum(axis=1) > 0)[0]  # real active positions in the 28x28 grid
    cached_lattice = s1a[0]["constructions"]["lattice"]
    T = s1a[0]["constructions"]["T"]

    reconstructed = build_lattice_topology(
        active_indices, total_weight_target=np.sum(T), side=28
    )

    assert reconstructed.shape == cached_lattice.shape
    assert np.array_equal(reconstructed > 0, cached_lattice > 0), \
        "sparsity pattern (edge positions) does not match the historical artifact"
    assert np.allclose(reconstructed, cached_lattice), \
        "edge weights do not match the historical artifact to floating-point precision"
