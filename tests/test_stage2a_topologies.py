"""
Tests for experiments/stage2a_dynamics_classification/stage2a_topologies.py.

Tier 2 only -- build_all_topologies() calls build_and_verify_T(), which
needs real KMNIST data and a cached cross-directory historical artifact
(class0_constructions.pkl) to reconstruct and verify T against. A Tier 1
synthetic version would be vacuous (there is no "synthetic T" to build
these constructions from), so this file is skipped entirely, not
partially, when that data isn't present locally -- matching
test_stage2a_core.py's own skipif convention.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
_KMNIST_DIR = _REPO_ROOT / "datasets" / "kmnist"
sys.path.insert(0, str(_STAGE2A_DIR))

# Amended by external review: the original skip condition checked only
# KMNIST presence, but build_all_topologies() -> build_and_verify_T()
# ALSO needs the cross-stage historical artifact
# stage1b2_structured_transformation/results/class0_constructions.pkl
# (gitignored, not committed) to verify the reconstructed T against. A
# fresh clone with KMNIST present but that artifact missing would have
# errored inside the fixture instead of skipping cleanly -- checking
# only one of two required preconditions. Both are now checked.
_kmnist_present = (_KMNIST_DIR / "train-images-idx3-ubyte").exists()
_class0_constructions_path = (
    _REPO_ROOT / "experiments" / "stage1b2_structured_transformation"
    / "results" / "class0_constructions.pkl")
_class0_constructions_present = _class0_constructions_path.exists()
_tier2_data_present = _kmnist_present and _class0_constructions_present

if not _kmnist_present:
    _skip_reason = "datasets/kmnist not present locally"
elif not _class0_constructions_present:
    _skip_reason = (
        f"{_class0_constructions_path.relative_to(_REPO_ROOT)} not present "
        f"locally (gitignored historical artifact, needed by "
        f"build_and_verify_T() to verify the reconstructed T against)")
else:
    _skip_reason = ""

pytestmark = pytest.mark.skipif(not _tier2_data_present, reason=_skip_reason)


@pytest.fixture(scope="module")
def topologies_bundle():
    import stage2a_topologies as topo
    active_indices, ink_mask_active, nodes_T, topologies = topo.build_all_topologies()
    return active_indices, ink_mask_active, nodes_T, topologies


def test_all_topologies_share_active_support(topologies_bundle):
    active_indices, _ink_mask, _nodes_T, topologies = topologies_bundle
    n = len(active_indices)
    for name, W in topologies.items():
        assert W.shape == (n, n), f"{name}'s W shape does not match T's active support"


def test_all_topologies_symmetric(topologies_bundle):
    _active_indices, _ink_mask, _nodes_T, topologies = topologies_bundle
    for name, W in topologies.items():
        np.testing.assert_allclose(W, W.T, atol=1e-9, err_msg=f"{name}'s W is not symmetric")


def test_rewired_preserves_t_degree_sequence(topologies_bundle):
    """degree_preserving_rewire is a double-edge-swap: which edges exist
    changes, but each node's edge COUNT (unweighted degree) does not.
    Edge weights move with their (rewired) edges, so WEIGHTED degree
    (sum of edge weights) is only approximately preserved, not exactly --
    confirmed by direct check before writing this test (max diff ~0.11,
    ~1.3% relative), so this test checks unweighted degree specifically,
    not summed edge weight."""
    _active_indices, _ink_mask, _nodes_T, topologies = topologies_bundle
    deg_T = np.sort((topologies["T"] != 0).sum(axis=1))
    deg_rewired = np.sort((topologies["rewired"] != 0).sum(axis=1))
    np.testing.assert_array_equal(deg_T, deg_rewired)


def test_rewired_and_curr_random_match_t_edge_count(topologies_bundle):
    _active_indices, _ink_mask, _nodes_T, topologies = topologies_bundle
    n_edges_T = np.count_nonzero(np.triu(topologies["T"], 1))
    for name in ["rewired", "curr_random"]:
        n_edges = np.count_nonzero(np.triu(topologies[name], 1))
        assert n_edges == n_edges_T, f"{name}'s edge count does not match T's"
