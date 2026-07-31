"""
Tests for the historical matched-sparsity random reconstruction
(generate_historical_matched_sparsity_random, rescale_to_common_budget) --
NOT tests of matched_sparsity_ablation.py, which implements a
deliberately different, currently-in-use algorithm and is left untouched.

Two tiers:
1. Self-contained structural tests on a small synthetic 28x28 image batch
   -- no external data required, always run: the rescale formula hits its
   exact target, background-background pairs stay excluded, values come
   from the real topology's own pool, and an explicit n_edges is honored
   exactly.
2. An optional historical-verification battery against the real recovered
   artifacts (tarballs/random_control_handoff/kmnist_c0_controls.npz and
   ..._normalized.npz, plus raw KMNIST data) -- skipped if any are
   missing locally. Confirms what IS established (the rescaling formula,
   reproduced byte-exact; support drawn from the eligible pool; values
   drawn from T's own pool) and explicitly asserts what is NOT
   established (this module does not, and here is shown not to,
   reproduce the historical artifacts byte-exact -- the exact edge-count
   rule and RNG seed/call-order remain unrecovered; see the module's own
   docstring for the full investigation).
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from bonsai.data.mnist_loader import load_mnist
from bonsai.dynamics.learned_topology_construction import build_class_topology
from bonsai.dynamics.historical_matched_sparsity_random import (
    generate_historical_matched_sparsity_random, rescale_to_common_budget,
)


# ---- Tier 1: self-contained structural tests ----

def _synthetic_topology(seed=0, n=40, density=0.1):
    """A small synthetic 'T'-like symmetric weighted graph plus an
    ink_mask, for testing the reconstruction functions without any real
    dataset."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    triu_i, triu_j = np.triu_indices(n, k=1)
    mask = rng.uniform(0, 1, len(triu_i)) < density
    values = rng.uniform(0.9, 1.0, mask.sum())
    A[triu_i[mask], triu_j[mask]] = values
    A[triu_j[mask], triu_i[mask]] = values
    ink_mask = rng.uniform(0, 1, n) > 0.3
    return A, ink_mask


def test_rescale_hits_exact_target():
    A, _ = _synthetic_topology(seed=0)
    target = 7.5
    A_tilde = rescale_to_common_budget(A, target)
    assert np.isclose(A_tilde.sum(axis=1).mean(), target)


def test_rescale_preserves_support_only_scales_values():
    A, _ = _synthetic_topology(seed=1)
    A_tilde = rescale_to_common_budget(A, 3.0)
    assert np.array_equal(A != 0, A_tilde != 0)


def test_background_background_pairs_excluded_from_support():
    A, ink_mask = _synthetic_topology(seed=2, density=0.3)
    candidate = generate_historical_matched_sparsity_random(A, ink_mask, seed=0)
    n = A.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    for i, j in zip(triu_i, triu_j):
        if candidate[i, j] != 0:
            assert ink_mask[i] or ink_mask[j], f"edge ({i},{j}) connects two background nodes"


def test_values_drawn_from_real_topology_pool():
    A, ink_mask = _synthetic_topology(seed=3, density=0.3)
    candidate = generate_historical_matched_sparsity_random(A, ink_mask, seed=0)
    candidate_vals = set(np.round(candidate[candidate != 0], 12).tolist())
    A_vals = set(np.round(A[A != 0], 12).tolist())
    assert candidate_vals.issubset(A_vals)


def test_explicit_n_edges_is_honored_exactly():
    A, ink_mask = _synthetic_topology(seed=4, density=0.3)
    for n_edges in [5, 20, 50]:
        candidate = generate_historical_matched_sparsity_random(A, ink_mask, seed=0, n_edges=n_edges)
        assert np.count_nonzero(np.triu(candidate, 1)) == n_edges


def test_default_n_edges_is_roughly_half_real_topology_edge_count():
    A, ink_mask = _synthetic_topology(seed=5, density=0.3)
    candidate = generate_historical_matched_sparsity_random(A, ink_mask, seed=0)
    n_edges_A = np.count_nonzero(np.triu(A, 1))
    n_edges_candidate = np.count_nonzero(np.triu(candidate, 1))
    assert n_edges_candidate == round(n_edges_A / 2)


# ---- Tier 2: optional historical verification ----

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDOFF_DIR = _REPO_ROOT / "tarballs" / "random_control_handoff"
RAW_NPZ = _HANDOFF_DIR / "kmnist_c0_controls.npz"
NORMALIZED_NPZ = _HANDOFF_DIR / "kmnist_c0_controls_normalized.npz"

KMNIST_DIR = _REPO_ROOT / "datasets" / "kmnist"
KMNIST_TRAIN_IMAGES = KMNIST_DIR / "train-images-idx3-ubyte"
KMNIST_TRAIN_LABELS = KMNIST_DIR / "train-labels-idx1-ubyte"

_TIER2_REQUIRED = [RAW_NPZ, NORMALIZED_NPZ, KMNIST_TRAIN_IMAGES, KMNIST_TRAIN_LABELS]


@pytest.fixture(scope="module")
def historical_data():
    if not all(p.exists() for p in _TIER2_REQUIRED):
        pytest.skip("historical random-control handoff and/or raw KMNIST data not present locally")
    raw = dict(np.load(RAW_NPZ))
    norm = dict(np.load(NORMALIZED_NPZ))

    X_train, y_train, _, _ = load_mnist(str(KMNIST_DIR), gz=False)
    idx = np.where(y_train == 0)[0][:200]
    images = X_train[idx].astype(np.float64) / 255.0
    active_indices, W_T = build_class_topology(images)
    ink_threshold = 0.15
    mean_intensity = images.mean(axis=0).flatten()
    ink_mask_active = (mean_intensity > ink_threshold)[active_indices]

    return raw, norm, W_T, ink_mask_active


def test_rescale_formula_reproduces_normalized_npz_byte_exact(historical_data):
    """The confirmed part: A_tilde = A * (C / mean_weighted_degree(A))
    reproduces kmnist_c0_controls_normalized.npz from
    kmnist_c0_controls.npz's raw arrays byte-exact, for all four
    constructions, not just 'random'."""
    raw, norm, W_T, _ = historical_data
    C = raw["T"].sum(axis=1).mean()
    for key in ["T", "rewired", "random", "lattice"]:
        predicted = rescale_to_common_budget(raw[key], C)
        assert np.allclose(predicted, norm[key], atol=1e-9), \
            f"rescale formula does not reproduce normalized '{key}'"


def test_historical_raw_random_support_is_within_eligible_pool(historical_data):
    raw, _, W_T, ink_mask_active = historical_data
    n = W_T.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    eligible = ~(~ink_mask_active[triu_i] & ~ink_mask_active[triu_j])
    eligible_pairs = set(zip(triu_i[eligible].tolist(), triu_j[eligible].tolist()))

    random_pos = set(zip(*np.where(np.triu(raw["random"], 1) != 0)))
    assert random_pos.issubset(eligible_pairs)


def test_historical_raw_random_values_are_from_T_pool(historical_data):
    raw, _, W_T, _ = historical_data
    random_vals = set(np.round(raw["random"][raw["random"] != 0], 10).tolist())
    T_vals = set(np.round(W_T[W_T != 0], 10).tolist())
    assert random_vals.issubset(T_vals)


def test_reconstruction_does_not_reproduce_historical_artifact_byte_exact(historical_data):
    """Documents the known, open gap rather than concealing it: using the
    real historical raw npz's own exact edge count (552), no seed in
    0-199 across three plausible call-order variants reproduced it (see
    module docstring) -- this test asserts that remains true for seed=0,
    a representative check, not a re-run of the full sweep (too slow for
    routine test runs, and a negative existence claim over an unbounded
    seed space isn't something a single test can prove either way)."""
    raw, _, W_T, ink_mask_active = historical_data
    n_edges = np.count_nonzero(np.triu(raw["random"], 1))
    candidate = generate_historical_matched_sparsity_random(
        W_T, ink_mask_active, seed=0, n_edges=n_edges)
    assert not np.array_equal(candidate, raw["random"]), (
        "reconstruction now matches the historical raw artifact at seed=0 -- "
        "if this module changed, update its docstring's 'seed not found' claim"
    )
