"""
Tests for the Stage 1A re-verification log-scale iteration
(experiments/stage1a_re_verification/DESIGN_v2_log_scale.md).

Two tiers, same convention as test_stage1a_re_verification.py:
1. Self-contained tests on small synthetic data -- always run.
2. An optional check against the real committed log-scale analysis pkl
   -- skipped if not present locally.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENT_DIR = _REPO_ROOT / "experiments" / "stage1a_re_verification"
sys.path.insert(0, str(_EXPERIMENT_DIR))

from analyze_stage1a_log_scale import (  # noqa: E402
    build_log_class_level_arrays, add_backtransform, primary_wilcoxon, COMPARISONS,
)

ANALYSIS_PATH = _EXPERIMENT_DIR / "results" / "stage1a_log_scale_analysis.pkl"


# ---- Tier 1: self-contained tests ----

def test_build_log_class_level_arrays_full_synthetic():
    from reverification_core import N_SEEDS
    rng = np.random.default_rng(0)
    results = {}
    for c in range(10):
        results[(c, "T", None)] = float(rng.uniform(1, 10))
        for s in range(N_SEEDS):
            results[(c, "toy_control", s)] = float(rng.uniform(1, 10))

    L_cT, L_cgs = build_log_class_level_arrays(results, "toy_control")
    assert L_cT.shape == (10,)
    assert L_cgs.shape == (10, N_SEEDS)
    for c in range(10):
        assert np.isclose(L_cT[c], np.log(results[(c, "T", None)]))
        for s in range(N_SEEDS):
            assert np.isclose(L_cgs[c, s], np.log(results[(c, "toy_control", s)]))


def test_add_backtransform_known_values():
    # A trivial "test" dict with known median_diff / hodges_lehmann in log space.
    d = np.array([np.log(2.0)] * 9 + [np.log(2.0)])  # constant -> median=HL=log(2)
    test = primary_wilcoxon(d)
    test = add_backtransform(test)
    assert np.isclose(test["median_diff_multiplicative"], 2.0)
    assert np.isclose(test["hodges_lehmann_multiplicative"], 2.0)


def test_log_mean_equals_log_of_geometric_mean():
    """Sanity check on the core mathematical claim DESIGN_v2 makes:
    mean(log(x)) == log(geometric_mean(x))."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.1, 100, size=25)
    log_mean = np.mean(np.log(x))
    geo_mean = np.exp(np.mean(np.log(x)))  # standard geometric-mean formula
    assert np.isclose(log_mean, np.log(geo_mean))
    # And confirm geometric mean is NOT the same as arithmetic mean in general
    # (this is the entire motivation for v2 -- otherwise there'd be nothing to test).
    assert not np.isclose(geo_mean, np.mean(x), rtol=0.05)


# ---- Tier 2: optional check against the real committed log-scale run ----

@pytest.mark.skipif(not ANALYSIS_PATH.exists(),
                     reason="stage1a_log_scale_analysis.pkl not present locally (run analyze_stage1a_log_scale.py first)")
def test_log_scale_analysis_covers_exactly_three_comparisons_no_lattice():
    with open(ANALYSIS_PATH, "rb") as f:
        analysis = pickle.load(f)
    assert set(analysis.keys()) == {"hist_random", "curr_random", "rewired"}
    assert "lattice" not in analysis, "v2 explicitly excludes lattice (no seed axis) per DESIGN_v2_log_scale.md"
    for name, out in analysis.items():
        assert "p_holm" in out["primary"]
        assert 0.0 <= out["primary"]["p_holm"] <= 1.0
        assert len(out["d_log_mean"]) == 10
        assert "hodges_lehmann_multiplicative" in out["primary"]


@pytest.mark.skipif(not ANALYSIS_PATH.exists(),
                     reason="stage1a_log_scale_analysis.pkl not present locally (run analyze_stage1a_log_scale.py first)")
def test_log_scale_holm_correction_is_never_looser_than_raw_p():
    with open(ANALYSIS_PATH, "rb") as f:
        analysis = pickle.load(f)
    for name in COMPARISONS:
        p_exact = analysis[name]["primary"]["p_exact"]
        p_holm = analysis[name]["primary"]["p_holm"]
        assert p_holm >= p_exact - 1e-12
