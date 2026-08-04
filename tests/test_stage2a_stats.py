"""
Tests for experiments/stage2a_dynamics_classification/stage2a_stats.py
-- the paired class-stratified bootstrap, per-image log-loss class
indexing, exact McNemar's test, and Holm-Bonferroni correction behind
Stage 2A's locked confirmatory result and its post hoc graph-to-graph
follow-up.

Tier 1 (self-contained, always run): synthetic-data correctness checks
for each statistic, including the deterministic-stratification and
step-down-stopping edge cases that are easy to get subtly wrong.

Tier 2 (skipped if the confirmatory artifact isn't present locally): an
artifact-backed regression test asserting the frozen primary effect
(FINDINGS.md's `d_i = -0.2491`, CI `[-0.2721, -0.2266]`) against the
already-saved confirmatory results, so a future refactor that silently
changes the numbers is caught mechanically.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import log_loss

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
sys.path.insert(0, str(_STAGE2A_DIR))

import stage2a_stats as stats  # noqa: E402


# ---- Tier 1: per_image_log_loss ----

def test_per_image_log_loss_mean_matches_sklearn():
    rng = np.random.default_rng(0)
    n, k = 200, 5
    classes = np.arange(k)
    y_true = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    ell_i = stats.per_image_log_loss(y_true, proba, classes)
    assert ell_i.shape == (n,)
    assert np.all(np.isfinite(ell_i))
    np.testing.assert_allclose(ell_i.mean(), log_loss(y_true, proba, labels=classes), rtol=1e-10)


def test_per_image_log_loss_correct_with_nondefault_class_ordering():
    """Classes not 0..k-1 in order -- confirms class_to_col indexes by
    class VALUE, not by position, since a positional bug would silently
    still run without error but give wrong per-image losses."""
    classes = np.array([7, 3, 9])  # deliberately out of numeric order
    proba = np.array([
        [0.1, 0.7, 0.2],  # column order matches `classes`: col0->class7, col1->class3, col2->class9
        [0.5, 0.3, 0.2],
    ])
    y_true = np.array([3, 9])  # image 0 is class 3 (col 1, p=0.7); image 1 is class 9 (col 2, p=0.2)
    ell_i = stats.per_image_log_loss(y_true, proba, classes)
    expected = -np.log(np.array([0.7, 0.2]))
    np.testing.assert_allclose(ell_i, expected, rtol=1e-10)


def test_per_image_log_loss_clips_extreme_probabilities():
    classes = np.array([0, 1])
    proba = np.array([[1.0, 0.0]])
    y_true = np.array([1])  # true class assigned exactly zero probability
    ell_i = stats.per_image_log_loss(y_true, proba, classes)
    assert np.isfinite(ell_i[0])  # would be inf without clipping


# ---- Tier 1: paired_class_stratified_bootstrap ----

def test_bootstrap_all_positive_gives_entirely_positive_ci():
    rng = np.random.default_rng(1)
    d = rng.uniform(0.01, 1.0, 500)
    y = rng.integers(0, 10, 500)
    result = stats.paired_class_stratified_bootstrap(d, y, n_resamples=2000, seed=1)
    assert result["ci_low"] > 0
    assert result["ci_high"] > 0


def test_bootstrap_all_negative_gives_entirely_negative_ci():
    rng = np.random.default_rng(2)
    d = rng.uniform(-1.0, -0.01, 500)
    y = rng.integers(0, 10, 500)
    result = stats.paired_class_stratified_bootstrap(d, y, n_resamples=2000, seed=2)
    assert result["ci_low"] < 0
    assert result["ci_high"] < 0


def test_bootstrap_deterministic_within_class_gives_exact_point_estimate():
    """The sharpest test of genuine per-class stratification: two classes,
    each with a CONSTANT d_i (90 samples at +1.0, 10 samples at -1.0). If
    the resample truly preserves each class's own count every draw, every
    possible resample must reduce to the identical weighted mean --
    (90*1.0 + 10*(-1.0))/100 = 0.8 -- exactly, with zero spread. A bug that
    pooled across classes instead of stratifying (or resampled class
    membership itself) would not produce this exact, zero-variance
    result."""
    d = np.concatenate([np.full(90, 1.0), np.full(10, -1.0)])
    y = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
    result = stats.paired_class_stratified_bootstrap(d, y, n_resamples=1000, seed=3)
    assert abs(result["observed_mean"] - 0.8) < 1e-12
    assert abs(result["ci_low"] - 0.8) < 1e-9
    assert abs(result["ci_high"] - 0.8) < 1e-9
    assert result["resampled_means"].std() < 1e-9


def test_bootstrap_straddling_case_gives_ci_containing_zero():
    rng = np.random.default_rng(4)
    d = rng.normal(0.0, 1.0, 2000)  # centered at zero, no true effect
    y = rng.integers(0, 10, 2000)
    result = stats.paired_class_stratified_bootstrap(d, y, n_resamples=5000, seed=4)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_bootstrap_reproducible_with_same_seed():
    rng = np.random.default_rng(5)
    d = rng.normal(size=300)
    y = rng.integers(0, 10, 300)
    r1 = stats.paired_class_stratified_bootstrap(d, y, n_resamples=500, seed=99)
    r2 = stats.paired_class_stratified_bootstrap(d, y, n_resamples=500, seed=99)
    np.testing.assert_array_equal(r1["resampled_means"], r2["resampled_means"])


# ---- Tier 1: mcnemar_exact ----

def test_mcnemar_hand_computed_discordant_counts():
    # 4 images: A correct/B correct, A correct/B wrong, A wrong/B correct, A wrong/B wrong
    y_true = np.array([0, 0, 0, 0])
    pred_a = np.array([0, 0, 1, 1])   # correct, correct, wrong, wrong
    pred_b = np.array([0, 1, 0, 1])   # correct, wrong, correct, wrong
    result = stats.mcnemar_exact(y_true, pred_a, pred_b, "a", "b")
    assert result["n_a_only_correct"] == 1  # image 1: a right, b wrong
    assert result["n_b_only_correct"] == 1  # image 2: a wrong, b right
    assert result["n_discordant"] == 2


def test_mcnemar_zero_discordant_gives_p_one():
    y_true = np.array([0, 1, 2])
    pred_a = np.array([0, 1, 2])
    pred_b = np.array([0, 1, 2])  # identical predictions, no discordant pairs
    result = stats.mcnemar_exact(y_true, pred_a, pred_b, "a", "b")
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_extreme_asymmetry_gives_small_p():
    y_true = np.zeros(50, dtype=int)
    pred_a = np.ones(50, dtype=int)   # always wrong
    pred_b = np.zeros(50, dtype=int)  # always correct
    result = stats.mcnemar_exact(y_true, pred_a, pred_b, "a", "b")
    assert result["n_discordant"] == 50
    assert result["n_b_only_correct"] == 50
    assert result["p_value"] < 1e-10


# ---- Tier 1: bootstrap_two_sided_p ----

def test_bootstrap_p_at_floor_when_all_resamples_one_sided():
    resampled_means = np.full(20000, 1.0)  # every resample strictly positive
    p = stats.bootstrap_two_sided_p(resampled_means, 20000)
    assert abs(p - 2 * (1 / 20001)) < 1e-12


def test_bootstrap_p_near_one_when_evenly_split():
    rng = np.random.default_rng(6)
    resampled_means = rng.choice([-1.0, 1.0], size=20000)  # ~50/50 split around zero
    p = stats.bootstrap_two_sided_p(resampled_means, 20000)
    assert p > 0.9


# ---- Tier 1: holm_bonferroni ----

def test_holm_bonferroni_hand_computed_example():
    # 4 tests, raw p = 0.001, 0.01, 0.03, 0.04; alpha=0.05
    # Holm thresholds (ascending): 0.05/4=0.0125, 0.05/3=0.01667, 0.05/2=0.025, 0.05/1=0.05
    # p=0.001 <= 0.0125 -> reject; p=0.01 <= 0.01667 -> reject;
    # p=0.03 <= 0.025? NO -> stop, p=0.03 and p=0.04 both fail (step-down)
    raw_p = {"a": 0.001, "b": 0.01, "c": 0.03, "d": 0.04}
    adjusted, rejected = stats.holm_bonferroni(raw_p, alpha=0.05)
    assert rejected["a"] is True
    assert rejected["b"] is True
    assert rejected["c"] is False
    assert rejected["d"] is False


def test_holm_bonferroni_step_down_stops_even_if_later_raw_p_would_individually_pass():
    """A naive (non-step-down) implementation might let a later comparison
    "pass" its own threshold even after an earlier one failed. Step-down
    Holm must not: once the sequence fails at rank i, every subsequent
    rank is also rejected=False regardless of its own raw p vs. its own
    threshold.

    rank thresholds at m=3 (1-indexed, factor=m-rank+1): rank1 factor=3,
    threshold=0.05/3=0.0167; rank2 factor=2, threshold=0.05/2=0.025;
    rank3 factor=1, threshold=0.05/1=0.05. p=0.001 passes rank1
    (<=0.0167). p=0.03 FAILS rank2 (0.03 > 0.025) -> step-down stops
    here. p=0.04 would pass rank3's own threshold in isolation
    (0.04 <= 0.05), but must still be rejected=False since rank2 already
    failed."""
    raw_p = {"x": 0.001, "y": 0.03, "z": 0.04}
    adjusted, rejected = stats.holm_bonferroni(raw_p, alpha=0.05)
    assert rejected["x"] is True
    assert rejected["y"] is False
    assert rejected["z"] is False


def test_holm_bonferroni_adjusted_p_monotone_non_decreasing_in_rank_order():
    raw_p = {"a": 0.001, "b": 0.005, "c": 0.02, "d": 0.048}
    adjusted, _rejected = stats.holm_bonferroni(raw_p, alpha=0.05)
    order = sorted(raw_p.keys(), key=lambda k: raw_p[k])
    adj_in_order = [adjusted[k] for k in order]
    assert all(adj_in_order[i] <= adj_in_order[i + 1] for i in range(len(adj_in_order) - 1))
    assert all(0.0 <= v <= 1.0 for v in adj_in_order)


# ---- Tier 2: artifact-backed regression on the frozen primary effect ----

_RESULTS_DIR = _STAGE2A_DIR / "results"
_CONFIRMATORY_PATH = _RESULTS_DIR / "stage4_confirmatory_results.pkl"


@pytest.mark.skipif(not _CONFIRMATORY_PATH.exists(),
                     reason="stage4_confirmatory_results.pkl not present locally")
def test_frozen_primary_effect_matches_findings_md():
    """FINDINGS.md's locked confirmatory result states: 'Observed mean
    d_i = -0.2491. 95% percentile interval: [-0.2721, -0.2266].' Recomputes
    the primary bootstrap directly from the already-saved per-image losses
    (not from a hard-coded copy of the numbers) and checks it still lands
    at those values -- catches a future refactor silently changing the
    statistic, not just a copy-paste of the current output."""
    with open(_CONFIRMATORY_PATH, "rb") as f:
        d = pickle.load(f)
    ell_i = d["condition_ell_i"]
    y_test = d["y_test"]

    d_primary = ell_i["evolved_T"] - ell_i["encoded_pre_evolution"]
    result = stats.paired_class_stratified_bootstrap(d_primary, y_test)

    assert abs(result["observed_mean"] - (-0.2491)) < 1e-3
    assert abs(result["ci_low"] - (-0.2721)) < 1e-3
    assert abs(result["ci_high"] - (-0.2266)) < 1e-3
    assert result["ci_high"] < 0  # the primary success criterion itself
