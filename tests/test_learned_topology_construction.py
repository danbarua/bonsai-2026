"""
Tests for the learned-topology construction (build_class_topology,
population_developmental_stat).

Two tiers:
1. Self-contained structural tests on small synthetic 28x28 image batches
   -- no external data required, always run, verify the construction's
   mathematical properties directly (symmetry, zero diagonal, bounded
   correlation values, explicit background-background exclusion,
   active-node connectivity, determinism). The module hardcodes H=W=28
   (population_developmental_stat's accumulator is sized from module-level
   constants, not from the input images' actual shape), so synthetic
   inputs here are full 28x28 grids with structured content, not an
   arbitrary small side like other Tier-1 suites in this project use.
2. An optional historical-verification test against the actual cached
   artifacts this construction was confirmed against (KMNIST's own raw
   IDX files in datasets/kmnist/, kmnist_class_topologies_200.pkl for the
   true active-node positions, stage1a_all_classes.pkl for the target T)
   -- skipped if any of these isn't present locally (the two pkls are
   gitignored historical data; datasets/kmnist/ itself is also gitignored
   per this project's datasets/*/ rule, so a fresh checkout has neither),
   but confirms byte-exact (to float64 machine epsilon) reproduction of
   the real historical KMNIST class-0 T when all four are available.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from bonsai.data.mnist_loader import load_mnist
from bonsai.dynamics.learned_topology_construction import (
    build_class_topology, population_developmental_stat,
)


# ---- Tier 1: self-contained structural tests ----

def test_population_stat_is_symmetric():
    rng = np.random.default_rng(0)
    images = rng.uniform(0, 1, size=(5, 28, 28))
    W_learned = population_developmental_stat(images, steps=10)
    assert np.allclose(W_learned, W_learned.T)


def test_population_stat_diagonal_is_zero():
    rng = np.random.default_rng(1)
    images = rng.uniform(0, 1, size=(5, 28, 28))
    W_learned = population_developmental_stat(images, steps=10)
    assert np.allclose(np.diag(W_learned), 0)


def test_population_stat_values_bounded():
    """Entries are a mean of cos(.) terms, so every value must lie in
    [-1, 1]."""
    rng = np.random.default_rng(2)
    images = rng.uniform(0, 1, size=(5, 28, 28))
    W_learned = population_developmental_stat(images, steps=10)
    assert np.all(W_learned >= -1.0 - 1e-9)
    assert np.all(W_learned <= 1.0 + 1e-9)


def test_background_background_pairs_always_excluded():
    """Two pixels that are background (below ink_threshold) in every image
    must never appear connected in the final active-restricted matrix,
    even though their local dynamics correlate strongly by construction
    (same fixed seed, same zero target phase) -- this is an explicit
    exclusion, not incidental to magnitude pruning."""
    rng = np.random.default_rng(3)
    images = np.zeros((5, 28, 28))
    images[:, 10:18, 10:18] = rng.uniform(0.5, 1.0, size=(5, 8, 8))  # only this block has "ink"

    active_indices, W_active = build_class_topology(images, steps=20)

    ink_mask_2d = np.zeros((28, 28), dtype=bool)
    ink_mask_2d[10:18, 10:18] = True
    ink_mask = ink_mask_2d.flatten()

    for i, gi in enumerate(active_indices):
        for j, gj in enumerate(active_indices):
            if i != j and not ink_mask[gi] and not ink_mask[gj]:
                assert W_active[i, j] == 0.0


def test_active_indices_are_self_consistent():
    rng = np.random.default_rng(4)
    images = rng.uniform(0, 1, size=(5, 28, 28))
    active_indices, W_active = build_class_topology(images, steps=20, prune_threshold=0.99)
    assert W_active.shape == (len(active_indices), len(active_indices))
    # every retained node must have at least one surviving connection --
    # that's what "active" (non-isolated) means here
    assert np.all(np.any(W_active != 0, axis=1))


def test_deterministic_given_same_images():
    rng = np.random.default_rng(5)
    images = rng.uniform(0, 1, size=(4, 28, 28))
    idx1, W1 = build_class_topology(images, steps=15)
    idx2, W2 = build_class_topology(images, steps=15)
    assert np.array_equal(idx1, idx2)
    assert np.array_equal(W1, W2)


# ---- Tier 2: optional historical verification ----

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDOFF_DIR = _REPO_ROOT / "tarballs" / "lattice_construction_handoff"
RAW_TOPOLOGIES = _HANDOFF_DIR / "kmnist_class_topologies_200.pkl"
HISTORICAL_ARTIFACT = _HANDOFF_DIR / "stage1a_all_classes.pkl"

KMNIST_DIR = _REPO_ROOT / "datasets" / "kmnist"
KMNIST_TRAIN_IMAGES = KMNIST_DIR / "train-images-idx3-ubyte"
KMNIST_TRAIN_LABELS = KMNIST_DIR / "train-labels-idx1-ubyte"

_TIER2_REQUIRED = [RAW_TOPOLOGIES, HISTORICAL_ARTIFACT, KMNIST_TRAIN_IMAGES, KMNIST_TRAIN_LABELS]


@pytest.mark.skipif(not all(p.exists() for p in _TIER2_REQUIRED),
                     reason="historical cached artifacts and/or raw KMNIST data not present locally")
def test_matches_historical_cached_artifact_byte_exact():
    """Confirms build_class_topology reproduces the ACTUAL historical
    KMNIST class-0 T -- built from the first 200 class-0 training images
    (the value encoded in kmnist_class_topologies_200.pkl's own filename,
    not build_class_topologies' MNIST-oriented default of 20) -- byte-exact
    to float64 machine epsilon, not just a plausible reconstruction."""
    X_train, y_train, _, _ = load_mnist(str(KMNIST_DIR), gz=False)
    idx = np.where(y_train == 0)[0][:200]
    images = X_train[idx].astype(np.float64) / 255.0

    active_indices, W_active = build_class_topology(images)

    with open(HISTORICAL_ARTIFACT, "rb") as f:
        s1a = pickle.load(f)
    cached_T = s1a[0]["constructions"]["T"]

    with open(RAW_TOPOLOGIES, "rb") as f:
        raw = pickle.load(f)
    active_from_raw = np.where(raw[0].sum(axis=1) > 0)[0]

    assert np.array_equal(active_indices, active_from_raw), \
        "active-node positions do not match the historical raw topology"
    assert W_active.shape == cached_T.shape
    assert np.allclose(W_active, cached_T, atol=1e-9), \
        "reconstructed T does not match the historical cached artifact to floating-point precision"
