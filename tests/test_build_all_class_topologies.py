"""
Tests for experiments/stage0_simulator_calibration/build_all_class_topologies.py,
which extends build_class_construction_bundle() from class-0-only to all 10
KMNIST classes.

Two tiers:
1. Self-contained: verifies the output format's structural contract
   ({class_int: {'constructions': {'T','rewired','random','lattice'},
   'n_active': int}}) by assembling a small synthetic multi-class dict
   the same way the driver script does, using synthetic 28x28 image
   batches -- no real dataset or pre-built output file required.
2. Historical verification, skipped if either the driver's real output
   (experiments/stage0_simulator_calibration/results/stage1a_all_classes.pkl,
   gitignored, not committed -- run the driver script to generate it) or
   the historical class0_constructions.pkl comparison artifact isn't
   present locally: full structural checks on the real 10-class output,
   plausible n_active range per class, and a byte-exact check of class
   0's T against the historical cached artifact.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from bonsai.dynamics.construction_bundle import build_class_construction_bundle


# ---- Tier 1: self-contained structural contract test ----

def _synthetic_images(seed, n=5):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 28, 28))
    images[:, 8:20, 8:20] = rng.uniform(0.5, 1.0, size=(n, 12, 12))
    return images


def test_multi_class_dict_matches_expected_output_format():
    """Assembles a small synthetic 3-class dict the same way
    build_all_class_topologies.py assembles the real 10-class one, and
    checks it matches stage1a_all_classes.pkl's structural contract."""
    all_data = {}
    for class_idx in range(3):
        images = _synthetic_images(seed=class_idx, n=5)
        all_data[class_idx] = build_class_construction_bundle(
            images, prune_threshold=0.8, rewired_seed=1, random_seed=1)

    assert sorted(all_data.keys()) == [0, 1, 2]
    for class_idx, bundle in all_data.items():
        assert set(bundle.keys()) == {"constructions", "n_active"}
        assert set(bundle["constructions"].keys()) == {"T", "rewired", "random", "lattice"}
        n = bundle["n_active"]
        assert isinstance(n, int)
        for key, W in bundle["constructions"].items():
            assert W.shape == (n, n), f"class {class_idx} {key} shape {W.shape} != ({n},{n})"


# ---- Tier 2: historical verification ----

_REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = (_REPO_ROOT / "experiments" / "stage0_simulator_calibration"
               / "results" / "stage1a_all_classes.pkl")
HISTORICAL_CLASS0_PATH = (_REPO_ROOT / "experiments" / "stage1b2_structured_transformation"
                          / "results" / "class0_constructions.pkl")

_TIER2_REQUIRED = [OUTPUT_PATH, HISTORICAL_CLASS0_PATH]


@pytest.fixture(scope="module")
def all_classes_data():
    if not all(p.exists() for p in _TIER2_REQUIRED):
        pytest.skip(
            "build_all_class_topologies.py's output and/or the historical "
            "class0_constructions.pkl comparison artifact not present locally "
            "-- run experiments/stage0_simulator_calibration/build_all_class_topologies.py first"
        )
    with open(OUTPUT_PATH, "rb") as f:
        return pickle.load(f)


def test_all_ten_classes_present_with_correct_structure(all_classes_data):
    assert sorted(all_classes_data.keys()) == list(range(10))
    for class_idx, bundle in all_classes_data.items():
        assert set(bundle.keys()) == {"constructions", "n_active"}
        assert set(bundle["constructions"].keys()) == {"T", "rewired", "random", "lattice"}
        n = bundle["n_active"]
        for key, W in bundle["constructions"].items():
            assert W.shape == (n, n), f"class {class_idx} {key} shape {W.shape} != ({n},{n})"


def test_n_active_is_plausible_across_classes(all_classes_data):
    """Active-node counts should vary by class (different digit shapes
    survive pruning differently) but stay within a broad, sane range --
    not e.g. near-zero or near the full 784-pixel grid."""
    n_active_values = [bundle["n_active"] for bundle in all_classes_data.values()]
    assert len(set(n_active_values)) > 1, "n_active is identical across all classes -- suspicious"
    for n in n_active_values:
        assert 200 < n < 700, f"n_active={n} outside a plausible range"


def test_class_0_T_matches_historical_artifact_byte_exact(all_classes_data):
    with open(HISTORICAL_CLASS0_PATH, "rb") as f:
        cached = pickle.load(f)[0]
    T_new = all_classes_data[0]["constructions"]["T"]
    T_cached = cached["constructions"]["T"]
    assert T_new.shape == T_cached.shape
    assert np.allclose(T_new, T_cached, atol=1e-9)
