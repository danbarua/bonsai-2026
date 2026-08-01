"""
Tests for the Stage 1A re-verification (experiments/stage1a_re_verification/),
per DESIGN.md.

Two tiers:
1. Self-contained structural/statistical tests on small synthetic data --
   no external data required, always run. Covers the analysis primitives
   (Holm correction, Hodges-Lehmann estimate, exact sign-flip test,
   hierarchical bootstrap) against known or deterministic small cases,
   and reverification_core's node-selection and construction-rescaling
   helpers.
2. An optional historical/integrity check against the real committed
   run's results pkl -- skipped if it isn't present locally (a full run
   takes a few minutes and is not reproduced by CI/a fresh checkout by
   default).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENT_DIR = _REPO_ROOT / "experiments" / "stage1a_re_verification"
sys.path.insert(0, str(_EXPERIMENT_DIR))

from reverification_core import get_degree_stratified_nodes, build_stochastic_construction  # noqa: E402
from analyze_stage1a_reverification import (  # noqa: E402
    holm_correct, hodges_lehmann, exact_sign_flip_test,
    hierarchical_bootstrap_stochastic, bootstrap_deterministic,
)

RESULTS_PATH = _EXPERIMENT_DIR / "results" / "stage1a_reverification_results.pkl"
ANALYSIS_PATH = _EXPERIMENT_DIR / "results" / "stage1a_reverification_analysis.pkl"


# ---- Tier 1: self-contained structural/statistical tests ----

def test_holm_correct_known_case():
    # Textbook example: 4 p-values, Holm step-down by hand.
    # sorted: 0.01, 0.02, 0.03, 0.04 with m=4
    # adjusted: 4*0.01=0.04, max(0.04,3*0.02)=0.06, max(0.06,2*0.03)=0.06, max(0.06,1*0.04)=0.06
    p = np.array([0.03, 0.01, 0.04, 0.02])
    adj = holm_correct(p)
    expected = {0.01: 0.04, 0.02: 0.06, 0.03: 0.06, 0.04: 0.06}
    for raw, got in zip(p, adj):
        assert np.isclose(got, expected[raw]), (raw, got)


def test_holm_correct_never_decreases_relative_order_below_raw():
    p = np.array([0.001, 0.2, 0.5, 0.8])
    adj = holm_correct(p)
    assert np.all(adj >= p - 1e-12)
    assert np.all(adj <= 1.0)


def test_hodges_lehmann_tiny_example():
    # d = [1, 3, 5]; Walsh averages: (1+1)/2=1, (1+3)/2=2, (1+5)/2=3,
    # (3+3)/2=3, (3+5)/2=4, (5+5)/2=5 -> sorted [1,2,3,3,4,5], median=3.0
    d = [1, 3, 5]
    assert np.isclose(hodges_lehmann(d), 3.0)


def test_exact_sign_flip_constant_magnitude_gives_minimum_pvalue():
    """When all |d_i| are equal, only the all-positive and all-negative
    sign patterns achieve the maximum |mean| -- so the two-sided exact
    p-value is exactly 2 / 2^n, the smallest attainable value."""
    d = np.ones(10)
    observed, p = exact_sign_flip_test(d)
    assert np.isclose(observed, 1.0)
    assert np.isclose(p, 2 / 1024)


def test_exact_sign_flip_symmetric_case_gives_pvalue_one():
    """d symmetric around 0 (mean already 0) -- every sign pattern's
    |mean| >= |observed|=0 trivially, so p=1.0."""
    d = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    observed, p = exact_sign_flip_test(d)
    assert np.isclose(observed, 0.0)
    assert np.isclose(p, 1.0)


def test_hierarchical_bootstrap_degenerate_case_is_a_point_mass():
    """If A_cT - A_cgs is identical (=5.0) for every class and every
    seed, every bootstrap draw must also average to exactly 5.0 --
    a deterministic check on the resampling logic itself, not a
    statistical property that could be flaky."""
    A_cT = np.full(10, 5.0)
    A_cgs = np.zeros((10, 25))  # so A_cT - A_cgs == 5.0 everywhere
    rng = np.random.default_rng(0)
    ci, boot_means = hierarchical_bootstrap_stochastic(A_cT, A_cgs, rng)
    assert np.allclose(boot_means, 5.0)
    assert np.isclose(ci[0], 5.0) and np.isclose(ci[1], 5.0)


def test_bootstrap_deterministic_degenerate_case_is_a_point_mass():
    d = np.full(10, -2.5)
    rng = np.random.default_rng(0)
    ci, boot_means = bootstrap_deterministic(d, rng)
    assert np.allclose(boot_means, -2.5)
    assert np.isclose(ci[0], -2.5) and np.isclose(ci[1], -2.5)


def test_get_degree_stratified_nodes_picks_expected_ranks():
    # 10 nodes, degree = their own index (0..9), sorted ascending already.
    W = np.diag(np.zeros(10))  # placeholder, degree comes from row sums below
    W = np.zeros((10, 10))
    for i in range(10):
        W[i, :] = i / 9.0  # row sum ~ i (not exact but strictly increasing in i)
    nodes = get_degree_stratified_nodes(W)
    # n=10: low=order[1], median=order[5], high=order[-1]=order[9]
    assert nodes["low"] == 1
    assert nodes["median"] == 5
    assert nodes["high"] == 9


def test_build_stochastic_construction_rescales_to_target_budget():
    """Smoke test on a small synthetic topology: whatever raw mean
    weighted degree a stochastic construction comes out with, the
    rescaled result must hit the target exactly (this is
    rescale_to_common_budget's job, exercised here through
    build_stochastic_construction's actual call path)."""
    rng = np.random.default_rng(0)
    N = 20
    W_T = np.zeros((N, N))
    triu_i, triu_j = np.triu_indices(N, k=1)
    chosen = rng.choice(len(triu_i), size=30, replace=False)
    vals = rng.uniform(0.1, 1.0, size=30)
    W_T[triu_i[chosen], triu_j[chosen]] = vals
    W_T[triu_j[chosen], triu_i[chosen]] = vals
    ink_mask = np.ones(N, dtype=bool)
    target_c = 7.0

    for name in ["rewired", "hist_random", "curr_random"]:
        W = build_stochastic_construction(name, W_T, ink_mask, seed=1, target_c=target_c)
        mean_weighted_degree = W.sum(axis=1).mean()
        assert np.isclose(mean_weighted_degree, target_c, atol=1e-9), name


# ---- Tier 2: optional check against the real committed run ----

@pytest.mark.skipif(not RESULTS_PATH.exists(),
                     reason="stage1a_reverification_results.pkl not present locally (full run not reproduced by default)")
def test_results_pkl_has_all_770_instances_and_finite_positive_aucs():
    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)
    assert len(results) == 770
    for key, auc in results.items():
        assert np.isfinite(auc), key
        assert auc > 0, key

    for c in range(10):
        assert (c, "T", None) in results
        assert (c, "lattice", None) in results
        for construction in ["rewired", "hist_random", "curr_random"]:
            for s in range(25):
                assert (c, construction, s) in results


@pytest.mark.skipif(not ANALYSIS_PATH.exists(),
                     reason="stage1a_reverification_analysis.pkl not present locally (run analyze script first)")
def test_analysis_pkl_has_all_four_comparisons_with_holm_corrected_p():
    with open(ANALYSIS_PATH, "rb") as f:
        analysis = pickle.load(f)
    assert set(analysis.keys()) == {"hist_random", "curr_random", "rewired", "lattice"}
    for name, out in analysis.items():
        assert "p_holm" in out["primary"]
        assert 0.0 <= out["primary"]["p_holm"] <= 1.0
        assert 0.0 <= out["primary"]["p_exact"] <= 1.0
        assert len(out["d_mean"]) == 10
