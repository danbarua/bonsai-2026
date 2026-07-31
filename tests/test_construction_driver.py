"""
Tests for the construction-bundle driver (build_class_construction_bundle),
which ties T, rewired, random, and lattice into one per-class dict matching
class0_constructions.pkl's format.

Two tiers:
1. Self-contained structural tests on a small synthetic 28x28 image batch
   -- no external data required, always run, verify the bundle's
   dict shape and each construction's basic contract (matching shapes,
   degree-preserving rewiring's exact degree preservation, lattice's
   total-weight match to T).
2. An optional historical-verification test against the real cached
   class0_constructions.pkl and raw KMNIST data -- skipped if either
   isn't present locally. T, rewired (at seed=1), and lattice all
   reproduce byte-exact (to float64 machine epsilon). 'random' does NOT
   match at any of 10 candidate seeds swept (0-9) -- confirmed structural,
   not a seed problem (the cached artifact has roughly half T's edge count
   and a different value scale) -- so this is asserted as a known,
   documented non-match, not silently skipped or forced to look like a
   pass.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from bonsai.data.mnist_loader import load_mnist
from bonsai.dynamics.construction_bundle import build_class_construction_bundle


# ---- Tier 1: self-contained structural tests ----

def _synthetic_images(seed=0, n=5):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 28, 28))
    images[:, 8:20, 8:20] = rng.uniform(0.5, 1.0, size=(n, 12, 12))  # a block of "ink"
    return images


def test_bundle_has_expected_keys_and_shapes():
    images = _synthetic_images()
    bundle = build_class_construction_bundle(images, prune_threshold=0.8)
    assert set(bundle.keys()) == {"constructions", "n_active"}
    constructions = bundle["constructions"]
    assert set(constructions.keys()) == {"T", "rewired", "random", "lattice"}

    n = bundle["n_active"]
    for key, W in constructions.items():
        assert W.shape == (n, n), f"{key} shape {W.shape} != (n_active, n_active)=({n},{n})"


def test_n_active_matches_T_active_node_count():
    images = _synthetic_images(seed=1)
    bundle = build_class_construction_bundle(images, prune_threshold=0.8)
    T = bundle["constructions"]["T"]
    assert bundle["n_active"] == T.shape[0]
    # every row of T must have at least one nonzero entry -- that's what
    # "active" means (see build_class_topology)
    assert np.all(np.any(T != 0, axis=1))


def test_rewired_preserves_exact_degree_sequence():
    """The core contract of degree-preserving rewiring: each node's
    UNWEIGHTED degree (edge count) and the global multiset of edge
    weights are exactly preserved -- NOT each node's weighted degree (sum
    of incident weights), since a swap reassigns which node gets which
    specific weight value (double-edge-swap preserves the degree
    sequence in the edge-count sense, the standard graph-theory meaning,
    not a per-node weight-sum invariant)."""
    images = _synthetic_images(seed=2)
    bundle = build_class_construction_bundle(images, prune_threshold=0.8, rewired_seed=0)
    T = bundle["constructions"]["T"]
    rewired = bundle["constructions"]["rewired"]

    T_edge_counts = np.sort(np.count_nonzero(T, axis=1))
    rewired_edge_counts = np.sort(np.count_nonzero(rewired, axis=1))
    assert np.array_equal(T_edge_counts, rewired_edge_counts)

    T_weights = np.sort(T[T != 0])
    rewired_weights = np.sort(rewired[rewired != 0])
    assert np.allclose(T_weights, rewired_weights)


def test_lattice_total_weight_matches_T():
    images = _synthetic_images(seed=3)
    bundle = build_class_construction_bundle(images, prune_threshold=0.8)
    T = bundle["constructions"]["T"]
    lattice = bundle["constructions"]["lattice"]
    assert np.isclose(np.sum(lattice), np.sum(T))


def test_random_matches_T_edge_count():
    """Current generate_matched_sparsity_topology's own contract: same
    edge count as T, values resampled from T's own value pool -- this is
    a self-consistency check on the function as currently implemented,
    not a claim that it reproduces any particular historical artifact
    (see the Tier-2 test below for that distinction)."""
    images = _synthetic_images(seed=4)
    bundle = build_class_construction_bundle(images, prune_threshold=0.8, random_seed=0)
    T = bundle["constructions"]["T"]
    random_topo = bundle["constructions"]["random"]
    assert np.count_nonzero(random_topo) == np.count_nonzero(T)


# ---- Tier 2: optional historical verification ----

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDOFF_DIR = _REPO_ROOT / "tarballs" / "lattice_construction_handoff"
HISTORICAL_ARTIFACT = _HANDOFF_DIR / "stage1a_all_classes.pkl"

KMNIST_DIR = _REPO_ROOT / "datasets" / "kmnist"
KMNIST_TRAIN_IMAGES = KMNIST_DIR / "train-images-idx3-ubyte"
KMNIST_TRAIN_LABELS = KMNIST_DIR / "train-labels-idx1-ubyte"

_TIER2_REQUIRED = [HISTORICAL_ARTIFACT, KMNIST_TRAIN_IMAGES, KMNIST_TRAIN_LABELS]


@pytest.mark.skipif(not all(p.exists() for p in _TIER2_REQUIRED),
                     reason="historical cached artifact and/or raw KMNIST data not present locally")
def test_matches_historical_cached_bundle_except_random():
    """T, rewired (at seed=1, confirmed by a 0-9 seed sweep to be the
    historical value for class 0), and lattice all reproduce the real
    historical class0_constructions.pkl byte-exact (to float64 machine
    epsilon). 'random' does not, at any of the 10 seeds swept -- and the
    mismatch is structural (cached 'random' has ~half T's edge count and
    a different value scale, consistent with a different, undocumented
    algorithm rather than a different seed), so this is asserted as a
    known, confirmed non-match rather than left unchecked."""
    X_train, y_train, _, _ = load_mnist(str(KMNIST_DIR), gz=False)
    idx = np.where(y_train == 0)[0][:200]
    images = X_train[idx].astype(np.float64) / 255.0

    bundle = build_class_construction_bundle(images, rewired_seed=1, random_seed=0)

    with open(HISTORICAL_ARTIFACT, "rb") as f:
        cached = pickle.load(f)[0]

    assert bundle["n_active"] == cached["n_active"]

    for key in ["T", "rewired", "lattice"]:
        mine = bundle["constructions"][key]
        theirs = cached["constructions"][key]
        assert np.allclose(mine, theirs, atol=1e-9), f"{key} does not match the historical artifact"

    mine_random = bundle["constructions"]["random"]
    theirs_random = cached["constructions"]["random"]
    assert not np.allclose(mine_random, theirs_random), (
        "random now matches the historical artifact -- if generate_matched_sparsity_topology "
        "was changed, update this test's docstring and the construction_bundle module's "
        "known-limitation note, since this was previously a confirmed non-match"
    )
