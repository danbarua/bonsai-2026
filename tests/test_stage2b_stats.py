"""
Tests for experiments/stage2b_denoising/stage2b_stats.py -- the primary
paired class-stratified bootstrap wiring, the paired t-test, the
studentized chunked sign-flip robustness test, the two Holm families,
the identity-baseline gate, and the branched "one graph wins" rule.

Tier 1 (self-contained, always run) only: Stage 2B has no historical
cached artifact to verify against, so there is nothing for a Tier 2
skip-if-absent test to check yet. Every test here is synthetic.

The sign-flip tests are the point of the file. CLAUDE.md principle 10
requires a NEW permutation scheme to be unit-tested on synthetic data
before it reaches inference: "identical input maps should give the test
statistic ~0; maximally separated, reproducible maps should give a
clearly positive, significant result." Those two extremes are covered
below, and three further properties are checked that neither extreme
would catch:

  - agreement with a slow, transparent per-resample reference written
    here in the test file. The reference shares nothing with production
    but the input `d` and the seed: its own statistic, its own
    comparison operator, its own loop. Because both draw signs from
    float64 uniforms in the same stream order, agreement is EXACT
    equality of the p-value, not a statistical approximation.
  - agreement with an independent algebraic path for the same statistic
    (the flip-invariance identity `sum((s*d)^2) == sum(d^2)`), which
    exercises the vectorized chunk reduction against a formula that
    cannot share an axis bug with it.
  - `chunk_size`-invariance of the p-value itself, across chunk sizes
    that do and do not divide `n_flips` evenly. Sign-matrix equality
    alone would not catch an accumulator bug that recomputes the
    observed statistic per chunk.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ttest_1samp

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
_STAGE2A_DIR = _REPO_ROOT / "experiments" / "stage2a_dynamics_classification"
sys.path.insert(0, str(_STAGE2B_DIR))
sys.path.insert(0, str(_STAGE2A_DIR))

import stage2a_stats as stats2a  # noqa: E402
import stage2b_stats as stats  # noqa: E402


# ---- the slow, obviously-correct reference (oracle, never the production path) ----

def _slow_reference_sign_flip_p(d, n_flips, seed):
    """A transparent per-resample loop: one flip at a time, the statistic
    written out directly, no chunking and no vectorization across flips.

    A per-resample Python loop is forbidden in the production path
    (DESIGN.md) precisely because it is slow; it is the right thing here
    because being obviously correct matters more than being fast in an
    oracle.

    This shares no code with production. It draws its own signs, computes
    its own statistic, and writes its own `>=` comparison. It does draw
    from the same float64-uniform stream at the same seed, so on the same
    input its p-value must match production EXACTLY -- one call of
    `rng.random(n)` per flip consumes the same stream as one
    `rng.random((chunk, n))` call covering those flips."""
    d = np.asarray(d, dtype=np.float64).ravel()
    n = d.size

    def t_of(v):
        m = float(np.mean(v))
        s = float(np.std(v, ddof=1))
        if s == 0.0:
            return 0.0 if m == 0.0 else float(np.inf) * np.sign(m)
        return m / (s / np.sqrt(n))

    t_obs = t_of(d)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_flips):
        u = rng.random(n)
        signs = np.where(u < 0.5, -1.0, 1.0)
        if abs(t_of(signs * d)) >= abs(t_obs):
            count += 1
    return (1 + count) / (n_flips + 1)


def _identity_path_sign_flip_p(d, n_flips, seed, chunk_size):
    """The same p-value via an independent algebraic route.

    `(s_i * d_i)^2 == d_i^2` for any sign, so the sum of squares is
    flip-invariant and the per-flip sample variance reduces to
    `(sum(d^2) - n * mean_b^2) / (n - 1)`, needing only the matrix-vector
    product `S @ d` rather than a materialized `S * d`. Production
    deliberately computes the literal product instead; this route exists
    to check that reduction, since a wrong axis in the vectorized version
    cannot produce the same answer as a formula that never reduces over
    an axis at all."""
    d = np.asarray(d, dtype=np.float64).ravel()
    n = d.size
    sum_sq = float(np.sum(d * d))

    def t_from_mean(mean):
        var = (sum_sq - n * mean ** 2) / (n - 1)
        var = np.maximum(var, 0.0)          # guard the cancellation floor
        sd = np.sqrt(var)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = mean / (sd / np.sqrt(n))
        return np.where(np.isnan(t), 0.0, t)

    t_obs = float(t_from_mean(np.array([float(np.mean(d))]))[0])
    rng = np.random.default_rng(seed)
    count = 0
    remaining = n_flips
    while remaining > 0:
        b = min(chunk_size, remaining)
        u = rng.random((b, n))
        signs = np.where(u < 0.5, -1.0, 1.0)
        means = (signs @ d) / n
        count += int(np.count_nonzero(np.abs(t_from_mean(means)) >= abs(t_obs)))
        remaining -= b
    return (1 + count) / (n_flips + 1)


# ---- synthetic condition fixtures ----

def _mse_conditions(n=400, seed=0, means=None):
    """Per-image clipped MSE vectors for every Stage 2B condition, all
    positive and in a plausible reconstruction-error range."""
    rng = np.random.default_rng(seed)
    if means is None:
        means = {"pre_evolution": 0.070, "T": 0.060, "lattice": 0.062,
                 "rewired": 0.061, "curr_random": 0.059}
    # a shared per-image difficulty term makes the conditions genuinely
    # PAIRED, as real per-image MSEs are
    difficulty = rng.gamma(2.0, 0.005, n)
    out = {}
    for key, mu in means.items():
        out[key] = mu + difficulty + rng.normal(0.0, 0.002, n)
    y = rng.integers(0, 10, n)
    return out, y


# ---- principle 10: the two required extremes ----

def test_sign_flip_p_near_one_for_zero_difference():
    """CLAUDE.md principle 10, first requirement: identical inputs (zero
    difference everywhere) mean the null is exactly true, so p must be
    near 1, not small.

    This is also the degenerate case the statistic is undefined on: every
    flip of an identically-zero vector has SD exactly zero, so 0/0 would
    be NaN and every `>=` comparison would be False, returning the Monte
    Carlo FLOOR -- the exact opposite of correct. The 0/0 -> 0 convention
    makes every flip tie instead, giving p = 1 exactly."""
    d = np.zeros(500)
    result = stats.studentized_sign_flip_p(d, n_flips=2000, seed=1, chunk_size=512)
    assert result["p_value"] == 1.0
    assert result["degenerate_observed"] is True
    assert result["n_zero_sd_flips"] == 2000
    assert result["t_observed"] == 0.0


def test_sign_flip_p_at_floor_for_large_constant_difference():
    """CLAUDE.md principle 10, second requirement: a maximally separated,
    reproducible input -- a large, constant, one-directional difference --
    must give a clearly significant result at the Monte Carlo floor
    (principle 6's `(1 + count) / (N + 1)` convention)."""
    d = np.full(500, 10.0)
    n_flips = 2000
    result = stats.studentized_sign_flip_p(d, n_flips=n_flips, seed=1, chunk_size=512)
    assert result["n_as_extreme"] == 0
    assert result["at_monte_carlo_floor"] is True
    assert result["p_value"] == pytest.approx(1.0 / (n_flips + 1), abs=1e-15)
    assert np.isinf(result["t_observed"])


def test_sign_flip_p_at_floor_for_strong_realistic_effect():
    """The same floor result on a NON-degenerate input, so the floor
    behaviour above is not an artifact of the SD=0 branch."""
    rng = np.random.default_rng(7)
    d = rng.normal(-0.5, 1.0, 500)
    result = stats.studentized_sign_flip_p(d, n_flips=2000, seed=1, chunk_size=512)
    assert result["degenerate_observed"] is False
    assert result["at_monte_carlo_floor"] is True
    assert result["p_value"] == pytest.approx(1.0 / 2001, abs=1e-15)


def test_sign_flip_p_intermediate_for_moderate_effect():
    """Neither extreme exercises the tie boundary. A modest effect
    relative to noise must land genuinely intermediate, confirming the
    test discriminates rather than saturating."""
    rng = np.random.default_rng(2)
    d = rng.normal(0.06, 1.0, 300)
    result = stats.studentized_sign_flip_p(d, n_flips=5000, seed=3, chunk_size=512)
    assert 0.01 < result["p_value"] < 0.99


# ---- agreement with the slow reference oracle ----

@pytest.mark.parametrize("n,n_flips,chunk_size,seed", [
    (37, 500, 64, 11),      # chunk_size != n, chunk does not divide n_flips
    (64, 300, 64, 12),      # chunk_size == n exactly: an axis bug would survive
    (11, 257, 256, 13),     # final chunk of size 1
    (50, 400, 400, 14),     # a single chunk covering every flip
    (23, 199, 7, 15),       # many small chunks, ragged final chunk
])
def test_sign_flip_matches_slow_reference_exactly(n, n_flips, chunk_size, seed):
    """The chunked, vectorized production path against a transparent
    per-resample loop. Same seed and same uniform stream, so the two must
    agree EXACTLY -- not approximately.

    `chunk_size == n` is included deliberately: on a square chunk a
    reduction over the wrong axis still runs and still returns the right
    shape, so only a case where the two differ can prove the axis is
    right, and only a case where they are equal can prove a passing
    non-square test was not a coincidence."""
    rng = np.random.default_rng(seed)
    d = rng.normal(0.15, 1.0, n)
    produced = stats.studentized_sign_flip_p(
        d, n_flips=n_flips, seed=seed, chunk_size=chunk_size)["p_value"]
    reference = _slow_reference_sign_flip_p(d, n_flips=n_flips, seed=seed)
    assert produced == reference


def test_sign_flip_matches_slow_reference_on_skewed_input():
    """Real per-image MSE differences are skewed, not normal. The
    reference agreement must hold there too."""
    rng = np.random.default_rng(21)
    d = rng.gamma(1.2, 0.03, 60) - 0.02
    produced = stats.studentized_sign_flip_p(
        d, n_flips=600, seed=5, chunk_size=128)["p_value"]
    assert produced == _slow_reference_sign_flip_p(d, n_flips=600, seed=5)


def test_sign_flip_matches_independent_algebraic_path():
    """The literal `SD(s*d)` reduction against the flip-invariance
    identity route, which never reduces over the flip axis at all."""
    rng = np.random.default_rng(31)
    d = rng.normal(0.1, 1.0, 80)
    produced = stats.studentized_sign_flip_p(
        d, n_flips=1000, seed=9, chunk_size=256)["p_value"]
    assert produced == _identity_path_sign_flip_p(d, n_flips=1000, seed=9, chunk_size=256)


# ---- chunking must not be load-bearing ----

@pytest.mark.parametrize("chunk_size", [1, 7, 64, 333, 512, 1000, 2048])
def test_sign_flip_p_identical_across_chunk_sizes(chunk_size):
    """The same seed must give the identical p-value at every chunk size,
    including sizes that do not divide `n_flips` evenly (where an
    off-by-one in the final partial chunk would live) and a size larger
    than `n_flips` (a single chunk).

    This asserts the p-value, not the sign matrix. Matching signs are
    necessary but not sufficient: an accumulator that recomputed the
    observed statistic from each chunk instead of from `d` would preserve
    sign equality and still give a chunk-dependent p."""
    rng = np.random.default_rng(41)
    d = rng.normal(0.1, 1.0, 55)
    n_flips = 1001   # prime-ish: divides evenly by almost none of the above
    baseline = stats.studentized_sign_flip_p(
        d, n_flips=n_flips, seed=17, chunk_size=1)["p_value"]
    result = stats.studentized_sign_flip_p(
        d, n_flips=n_flips, seed=17, chunk_size=chunk_size)
    assert result["p_value"] == baseline
    assert result["n_chunks"] == int(np.ceil(n_flips / chunk_size))


def test_sign_flip_uniform_threshold_route_is_chunk_stream_invariant():
    """The property the sign-generation route is chosen for, asserted
    directly rather than inferred from the p-value.

    `Generator.random` at float64 width consumes exactly one 64-bit draw
    per element, so chunked draws concatenate to the unchunked stream.
    `Generator.integers(0, 2, dtype=uint8)` buffers bits within a call and
    discards the remainder at the call boundary, so it does NOT -- which
    is why production does not use it. Both halves are asserted, so a
    future "simplification" to `integers` fails here with the reason
    visible."""
    n = 29
    full = np.random.default_rng(5).random((60, n))
    for cs in (1, 7, 60):
        rng = np.random.default_rng(5)
        parts, left = [], 60
        while left > 0:
            b = min(cs, left)
            parts.append(rng.random((b, n)))
            left -= b
        assert np.array_equal(np.vstack(parts), full)

    full_i = np.random.default_rng(5).integers(0, 2, size=(60, n), dtype=np.uint8)
    rng = np.random.default_rng(5)
    chunked_i = np.vstack([rng.integers(0, 2, size=(7, n), dtype=np.uint8)
                           for _ in range(2)])
    assert not np.array_equal(chunked_i, full_i[:14])


def test_sign_flip_signs_are_int8():
    """DESIGN.md's dtype table locks int8 sign matrices. Asserted on the
    generation expression production uses, since the matrices themselves
    are transient."""
    u = np.random.default_rng(0).random((4, 5))
    signs = np.where(u < 0.5, np.int8(-1), np.int8(1))
    assert signs.dtype == np.int8
    assert set(np.unique(signs)) <= {-1, 1}


def test_sign_flip_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.1, 1.0, 200)
    a = stats.studentized_sign_flip_p(d, n_flips=1000, seed=1, chunk_size=256)
    b = stats.studentized_sign_flip_p(d, n_flips=1000, seed=1, chunk_size=256)
    assert a["p_value"] == b["p_value"]
    assert a["n_as_extreme"] == b["n_as_extreme"]


# ---- the statistic is the t-test's, and the caveat travels ----

def test_sign_flip_observed_statistic_equals_the_paired_t_statistic():
    """DESIGN.md says the studentized statistic matches "the t-test's
    form". With every sign +1 the flipped statistic reduces to the
    observed one, so `t_observed` must equal scipy's paired t-statistic
    exactly. This converts the ddof=1 reading of "matching the t-test's
    form" from a comment into a checked property."""
    rng = np.random.default_rng(6)
    d = rng.normal(0.2, 1.3, 250)
    observed = stats.studentized_sign_flip_p(
        d, n_flips=10, seed=0, chunk_size=8)["t_observed"]
    assert observed == pytest.approx(float(ttest_1samp(d, 0.0).statistic), rel=1e-12)
    assert observed == pytest.approx(stats.paired_t_test(d)["t"], rel=1e-12)


def test_sign_flip_result_carries_the_exchangeability_caveat():
    """The task's requirement that the assumption travel with the number:
    a caller cannot read `p_value` out of this dict without the caveat
    being in the same dict."""
    d = np.random.default_rng(0).normal(size=50)
    result = stats.studentized_sign_flip_p(d, n_flips=100, seed=0, chunk_size=32)
    assert result["assumption"] == stats.SIGN_EXCHANGEABILITY_CAVEAT
    assert "exchangeab" in result["assumption"]


def test_every_family2_sign_flip_entry_carries_the_caveat():
    """Not just the family-level wrapper -- each of the six per-pair
    entries individually."""
    mse, _ = _mse_conditions()
    fam2 = stats.family2_pairwise(mse, run_sign_flip=True, n_flips=200, chunk_size=64)
    assert len(fam2["per_test"]) == 6
    for key, record in fam2["per_test"].items():
        assert record["sign_flip"]["assumption"] == stats.SIGN_EXCHANGEABILITY_CAVEAT, key
    assert fam2["sign_flip_assumption"] == stats.SIGN_EXCHANGEABILITY_CAVEAT


def test_sign_flip_records_design_chunk_range_compliance():
    d = np.random.default_rng(0).normal(size=50)
    inside = stats.studentized_sign_flip_p(d, n_flips=100, seed=0, chunk_size=1024)
    outside = stats.studentized_sign_flip_p(d, n_flips=100, seed=0, chunk_size=8)
    assert inside["chunk_size_in_design_range"] is True
    assert outside["chunk_size_in_design_range"] is False


def test_locked_sign_flip_constants_match_design():
    assert stats.N_SIGN_FLIPS == 100000
    assert stats.SIGN_FLIP_SEED == 42
    assert stats.SIGN_FLIP_CHUNK_RANGE == (512, 4096)
    assert stats.SIGN_FLIP_CHUNK in range(512, 4097)
    assert stats.N_BOOTSTRAP_RESAMPLES == 20000
    assert stats.BOOTSTRAP_SEED == 42


def test_sign_flip_default_denominator_is_the_locked_one():
    """`p = (1 + count) / 100001` at the locked flip count."""
    d = np.full(100, 5.0)
    result = stats.studentized_sign_flip_p(d, n_flips=stats.N_SIGN_FLIPS,
                                           chunk_size=4096)
    assert result["p_value"] == pytest.approx(1.0 / 100001, abs=1e-15)
    assert result["n_flips"] == 100000
    assert result["seed"] == 42


# ---- paired t-test ----

def test_paired_t_test_matches_scipy():
    rng = np.random.default_rng(8)
    d = rng.normal(0.1, 1.0, 300)
    result = stats.paired_t_test(d)
    ref = ttest_1samp(d, 0.0)
    assert result["t"] == pytest.approx(float(ref.statistic), rel=1e-12)
    assert result["p_value"] == pytest.approx(float(ref.pvalue), rel=1e-12)
    assert result["df"] == 299
    assert result["sd"] == pytest.approx(float(np.std(d, ddof=1)), rel=1e-12)


def test_paired_t_test_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        stats.paired_t_test(np.array([1.0, np.nan, 2.0]))


def test_paired_t_test_flags_p_value_underflow():
    """A large effect at the locked corpus size drives the analytic
    two-sided tail below float64's smallest positive value, and scipy
    returns exactly 0.0. Reporting "p = 0" is never correct (CLAUDE.md
    principle 6), so the flag must fire and the caller must be able to say
    "below float64 representable" instead.

    This is reachable at Stage 2B's real scale, not a contrived edge: a
    per-image MSE difference of -0.01 with SD 0.003 over 10,000 images
    gives |t| > 300."""
    d = np.random.default_rng(0).normal(-0.01, 0.003, 10000)
    result = stats.paired_t_test(d)
    assert result["p_value"] == 0.0
    assert result["p_underflow"] is True
    assert abs(result["t"]) > 100

    moderate = np.random.default_rng(1).normal(0.001, 1.0, 300)
    assert stats.paired_t_test(moderate)["p_underflow"] is False


def test_paired_t_test_carries_the_clt_note():
    d = np.random.default_rng(0).normal(size=50)
    assert stats.paired_t_test(d)["assumption"] == stats.T_TEST_CLT_NOTE


# ---- the reference bootstrap is reused unmodified ----

def test_primary_uses_the_stage2a_bootstrap_object_itself():
    """The reuse boundary as a checked property: Stage 2B calls Stage 2A's
    bootstrap, it does not carry a copy that could drift from it."""
    assert stats.paired_class_stratified_bootstrap is stats2a.paired_class_stratified_bootstrap
    assert stats.holm_bonferroni is stats2a.holm_bonferroni


def test_primary_bootstrap_reproduces_the_reference_exactly():
    """The primary wiring must not perturb the oracle: same contrast, same
    seed, same resample count => bit-identical resampled means."""
    mse, y = _mse_conditions(n=300, seed=3)
    result = stats.primary_bootstrap_test(mse, y, n_resamples=500, seed=42)
    d = mse["T"] - mse["pre_evolution"]
    reference = stats2a.paired_class_stratified_bootstrap(d, y, n_resamples=500, seed=42)
    np.testing.assert_array_equal(result["resampled_means"], reference["resampled_means"])
    assert result["ci_low"] == reference["ci_low"]
    assert result["ci_high"] == reference["ci_high"]


# ---- the primary test is structurally outside both families ----

def test_primary_result_carries_no_p_value_of_any_kind():
    """The primary decision rule is CI-based. A p-value in this dict could
    be swept into a Holm family by a caller assembling one from
    p-values, so there must not be one under any key."""
    mse, y = _mse_conditions(n=200, seed=4)
    result = stats.primary_bootstrap_test(mse, y, n_resamples=300)
    assert not any("p_value" in k or k == "p" for k in result)
    assert result["multiplicity_family"] is None


def test_primary_verdict_branches_on_the_locked_rule():
    n = 400
    y = np.tile(np.arange(10), n // 10)
    base = np.full(n, 0.05)
    # T clearly better -> interval entirely below zero
    better = {"pre_evolution": base + 0.02, "T": base,
              "lattice": base, "rewired": base, "curr_random": base}
    assert stats.primary_bootstrap_test(better, y, n_resamples=500)["verdict"] == \
        "evolution_improves"
    # T clearly worse -> interval entirely above zero
    worse = dict(better, pre_evolution=base - 0.02)
    assert stats.primary_bootstrap_test(worse, y, n_resamples=500)["verdict"] == \
        "pre_evolution_wins"
    # no systematic difference -> straddling
    rng = np.random.default_rng(0)
    noise = {k: base + rng.normal(0, 0.01, n) for k in better}
    assert stats.primary_bootstrap_test(noise, y, n_resamples=2000)["verdict"] == "null"


# ---- Family 1 ----

def test_family1_has_exactly_the_three_prespecified_members():
    mse, y = _mse_conditions(n=200, seed=5)
    fam1 = stats.family1_vs_pre_evolution(mse, y, n_resamples=300)
    assert set(fam1["raw_p"]) == {"lattice", "rewired", "curr_random"}
    assert "T" not in fam1["raw_p"]      # T's contrast is the primary test
    assert fam1["n_tests"] == 3


def test_family1_holm_is_the_imported_step_down_over_its_own_three():
    mse, y = _mse_conditions(n=300, seed=6)
    fam1 = stats.family1_vs_pre_evolution(mse, y, n_resamples=300)
    expected_adj, expected_rej = stats2a.holm_bonferroni(fam1["raw_p"], alpha=0.05)
    assert fam1["holm_adjusted_p"] == expected_adj
    assert fam1["holm_rejected"] == expected_rej


def test_family1_contrast_direction_is_control_minus_pre_evolution():
    """`mean < 0` must mean the CONTROL has lower error. Built so the
    controls are unambiguously better."""
    n = 200
    rng = np.random.default_rng(12)
    y = np.tile(np.arange(10), n // 10)
    base = np.full(n, 0.05)
    # per-condition noise, so each difference genuinely varies across
    # images rather than being an exactly constant (degenerate) vector
    mse = {"pre_evolution": base + 0.03 + rng.normal(0, 0.0005, n)}
    for graph in stats.EVOLVED_GRAPHS:
        mse[graph] = base + rng.normal(0, 0.0005, n)
    fam1 = stats.family1_vs_pre_evolution(mse, y, n_resamples=300)
    for graph in stats.CONTROL_GRAPHS:
        assert fam1["per_test"][graph]["t_test"]["mean"] == pytest.approx(-0.03, abs=5e-4)
        assert fam1["per_test"][graph]["favorable"] is True


# ---- Family 2 ----

def test_family2_has_exactly_the_six_prespecified_pairs():
    mse, _ = _mse_conditions(n=200, seed=7)
    fam2 = stats.family2_pairwise(mse, run_sign_flip=False)
    assert len(fam2["raw_p"]) == 6
    assert set(fam2["raw_p"]) == {"T_vs_lattice", "T_vs_rewired", "T_vs_curr_random",
                                  "lattice_vs_rewired", "lattice_vs_curr_random",
                                  "rewired_vs_curr_random"}


def test_family2_holm_corrects_only_the_t_test_p_values():
    """Sign-flip p-values are robustness only and must not enter Holm."""
    mse, _ = _mse_conditions(n=200, seed=8)
    fam2 = stats.family2_pairwise(mse, run_sign_flip=True, n_flips=200, chunk_size=64)
    for key, record in fam2["per_test"].items():
        assert fam2["raw_p"][key] == record["t_test"]["p_value"]
    expected_adj, _ = stats2a.holm_bonferroni(fam2["raw_p"], alpha=0.05)
    assert fam2["holm_adjusted_p"] == expected_adj


def test_family_level_underflow_is_visible_without_reaching_into_per_test():
    """A raw p of 0.0 Holm-adjusts to 0.0, so the underflow propagates
    into the corrected value with nothing at that layer to mark it. A
    caller reading `raw_p` / `holm_adjusted_p` to write up a result must
    be able to see the underflow from the family dict itself -- the same
    "the caveat travels with the number" requirement the sign-flip's
    assumption string satisfies.

    Effect sizes here are the realistic-scale ones that actually underflow
    (|t| in the hundreds over 10,000 images), not a contrived edge."""
    n = 10000
    rng = np.random.default_rng(15)
    y = np.tile(np.arange(10), n // 10)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base + 0.02 + rng.normal(0, 0.001, n)}
    for graph in stats.EVOLVED_GRAPHS:
        mse[graph] = base + rng.normal(0, 0.001, n)

    fam1 = stats.family1_vs_pre_evolution(mse, y, n_resamples=200)
    assert all(fam1["raw_p"][g] == 0.0 for g in stats.CONTROL_GRAPHS)
    assert set(fam1["p_underflowed"]) == set(stats.CONTROL_GRAPHS)
    assert fam1["n_p_underflowed"] == 3
    assert "principle 6" in fam1["underflow_note"]

    # and the clean case reports no underflow and no note
    mse_null = {k: base + rng.normal(0, 0.01, n)
                for k in ("pre_evolution", *stats.EVOLVED_GRAPHS)}
    fam1_null = stats.family1_vs_pre_evolution(mse_null, y, n_resamples=200)
    assert fam1_null["n_p_underflowed"] == 0
    assert fam1_null["underflow_note"] is None


def test_family2_sign_flip_is_intermediate_on_near_null_differences():
    """The family-level sign-flip path on a case that does NOT saturate.

    Every other family-level check uses effects large enough to floor the
    sign-flip at `1/(n_flips+1)`, which would also be the result of a
    broken statistic that never counts an exceedance. Near-null
    differences are the only place the family wiring, the chunk defaults,
    and a non-saturated statistic meet."""
    n = 600
    rng = np.random.default_rng(16)
    base = np.full(n, 0.05)
    mse = {k: base + rng.normal(0, 0.01, n)
           for k in ("pre_evolution", *stats.EVOLVED_GRAPHS)}
    fam2 = stats.family2_pairwise(mse, run_sign_flip=True, n_flips=4000,
                                  chunk_size=512)
    p_values = [r["sign_flip"]["p_value"] for r in fam2["per_test"].values()]
    assert all(0.001 < p <= 1.0 for p in p_values), p_values
    assert any(p > 0.05 for p in p_values), p_values      # genuinely non-floor
    assert not any(r["sign_flip"]["at_monte_carlo_floor"]
                   for r in fam2["per_test"].values())


def test_family2_pair_direction_is_first_minus_second():
    n = 200
    rng = np.random.default_rng(13)
    base = np.full(n, 0.05)
    mse = {k: base + rng.normal(0, 0.0005, n)
           for k in ("pre_evolution", "T", "rewired", "curr_random")}
    mse["lattice"] = base + 0.01 + rng.normal(0, 0.0005, n)
    fam2 = stats.family2_pairwise(mse, run_sign_flip=False)
    # MSE(T) - MSE(lattice) ~ -0.01: T is better, T is first
    assert fam2["per_test"]["T_vs_lattice"]["t_test"]["mean"] == pytest.approx(-0.01, abs=5e-4)
    assert fam2["per_test"]["T_vs_lattice"]["a_favored"] is True
    # MSE(lattice) - MSE(rewired) ~ +0.01: lattice is worse, lattice is first
    assert fam2["per_test"]["lattice_vs_rewired"]["t_test"]["mean"] == pytest.approx(0.01, abs=5e-4)
    assert fam2["per_test"]["lattice_vs_rewired"]["a_favored"] is False


# ---- pairwise_outcome: the direction lookup that flips half the time ----

def test_pairwise_outcome_negates_when_the_graph_is_second_in_its_pair():
    """`curr_random` is last in `EVOLVED_GRAPHS`, so it is the SECOND
    element of all three of its pairs. A lookup that read the stored mean
    without negating would report its direction backwards in every one of
    them -- and a test where `T` (always first) wins would pass under
    that bug.

    Here `curr_random` is the best graph by construction."""
    n = 200
    rng = np.random.default_rng(14)
    base = np.full(n, 0.05)
    offsets = {"pre_evolution": 0.05, "T": 0.03, "lattice": 0.02,
               "rewired": 0.01, "curr_random": 0.0}
    mse = {k: base + v + rng.normal(0, 0.0005, n) for k, v in offsets.items()}
    fam2 = stats.family2_pairwise(mse, run_sign_flip=False)

    for other in ("T", "lattice", "rewired"):
        out = stats.pairwise_outcome(fam2, "curr_random", other)
        assert out["graph_is_first_in_pair"] is False
        assert out["favorable"] is True, f"curr_random should beat {other}"
        assert out["mean_diff"] < 0
        # the mirrored lookup must be the exact negation
        mirror = stats.pairwise_outcome(fam2, other, "curr_random")
        assert mirror["mean_diff"] == pytest.approx(-out["mean_diff"])
        assert mirror["favorable"] is False
        assert mirror["holm_adjusted_p"] == out["holm_adjusted_p"]


def test_pairwise_outcome_raises_on_an_unknown_pair():
    mse, _ = _mse_conditions(n=100, seed=9)
    fam2 = stats.family2_pairwise(mse, run_sign_flip=False)
    with pytest.raises(KeyError):
        stats.pairwise_outcome(fam2, "T", "pre_evolution")


# ---- "one graph wins", branched ----

def _run_all(mse, y, n_resamples=500, alpha=0.05):
    primary = stats.primary_bootstrap_test(mse, y, n_resamples=n_resamples)
    fam1 = stats.family1_vs_pre_evolution(mse, y, alpha=alpha, n_resamples=n_resamples)
    fam2 = stats.family2_pairwise(mse, alpha=alpha, run_sign_flip=False)
    return primary, fam1, fam2, stats.one_graph_wins(primary, fam1, fam2, alpha=alpha)


def _graded(n=400, seed=0, order=("T", "lattice", "rewired", "curr_random"),
            gap=0.01, noise=0.001):
    """Conditions where `order[0]` is best and each next is worse, all
    better than `pre_evolution`."""
    rng = np.random.default_rng(seed)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base + gap * (len(order) + 1) + rng.normal(0, noise, n)}
    for i, g in enumerate(order):
        mse[g] = base + gap * i + rng.normal(0, noise, n)
    y = np.tile(np.arange(10), n // 10)
    return mse, y


def test_one_graph_wins_t_qualifies_by_the_primary_interval_not_family1():
    """T's qualification rule reads the PRIMARY bootstrap interval. T's
    contrast against `pre_evolution` is not in Family 1 at all, so a rule
    that looked for T's Family-1 Holm p would raise, not silently pass."""
    mse, y = _graded(order=("T", "lattice", "rewired", "curr_random"))
    primary, _fam1, _fam2, verdict = _run_all(mse, y)
    t_qual = verdict["qualification"]["T"]
    assert t_qual["rule_source"].startswith("primary test")
    assert t_qual["qualifies"] is True
    assert t_qual["ci_high"] == primary["ci_high"]
    assert "holm_adjusted_p" not in t_qual
    assert verdict["unique_winner"] == "T"


def test_one_graph_wins_control_qualifies_by_family1_holm_not_the_primary():
    """A control graph's rule reads its FAMILY-1 Holm-adjusted p. Here
    `curr_random` -- second element in all three of its Family-2 pairs --
    is the winner, which is the case an inverted direction lookup fails."""
    mse, y = _graded(order=("curr_random", "rewired", "lattice", "T"))
    _primary, fam1, _fam2, verdict = _run_all(mse, y)
    cr = verdict["qualification"]["curr_random"]
    assert cr["rule_source"].startswith("Family 1")
    assert cr["favorable"] is True
    assert cr["holm_adjusted_p"] == fam1["holm_adjusted_p"]["curr_random"]
    assert cr["qualifies"] is True
    assert verdict["unique_winner"] == "curr_random"
    assert verdict["beats_others"]["curr_random"]["beats_all"] is True


def test_one_graph_wins_no_winner_when_the_leader_fails_to_separate():
    """All four beat pre_evolution but are indistinguishable from each
    other: DESIGN.md's named watched-for outcome 1. No unique winner,
    even though every graph qualifies."""
    n = 400
    rng = np.random.default_rng(0)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base + 0.03 + rng.normal(0, 0.001, n)}
    for g in stats.EVOLVED_GRAPHS:
        mse[g] = base + rng.normal(0, 0.001, n)
    y = np.tile(np.arange(10), n // 10)
    _primary, _fam1, _fam2, verdict = _run_all(mse, y)
    assert all(verdict["qualification"][g]["qualifies"] for g in stats.EVOLVED_GRAPHS)
    assert verdict["unique_winner"] is None


def test_one_graph_wins_no_winner_when_the_best_graph_fails_qualification():
    """A graph that beats all three others but does NOT qualify against
    `pre_evolution` is not a winner. Here every graph is WORSE than
    pre_evolution, so nothing qualifies, even though T beats the rest."""
    n = 400
    rng = np.random.default_rng(1)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base + rng.normal(0, 0.0005, n)}
    for i, g in enumerate(stats.EVOLVED_GRAPHS):
        mse[g] = base + 0.02 + 0.01 * i + rng.normal(0, 0.0005, n)
    y = np.tile(np.arange(10), n // 10)
    _primary, _fam1, _fam2, verdict = _run_all(mse, y)
    assert verdict["beats_others"]["T"]["beats_all"] is True
    assert verdict["qualification"]["T"]["qualifies"] is False
    assert verdict["unique_winner"] is None


def test_one_graph_wins_reports_every_graph_and_every_opponent():
    mse, y = _graded()
    _p, _f1, _f2, verdict = _run_all(mse, y)
    assert set(verdict["qualification"]) == set(stats.EVOLVED_GRAPHS)
    for graph in stats.EVOLVED_GRAPHS:
        opponents = verdict["beats_others"][graph]["per_opponent"]
        assert set(opponents) == set(stats.EVOLVED_GRAPHS) - {graph}


# ---- the alpha comparison: strict `<` in BOTH families ----
#
# DESIGN.md words Family 1's threshold explicitly ("Holm-adjusted
# p < 0.05") and leaves Family 2's unworded ("after Family-2 Holm
# correction"). The module applies the Family-1 rule to both. The only
# input that can tell strict `<` apart from the step-down rule's `<=` is
# an adjusted p landing exactly on alpha, so that is what these tests
# construct -- by running the real pipeline and then overwriting one
# adjusted p, since no synthetic MSE fixture lands a t-test tail on
# 0.05 exactly. `holm_rejected` is left True in each case, which is what
# the unmodified `stage2a_stats.holm_bonferroni` returns at `p == alpha`;
# the assertions therefore pin the module's OWN comparison, not the
# imported function's.

_ALPHA = 0.05
_JUST_BELOW_ALPHA = np.nextafter(_ALPHA, 0.0)


def test_family1_qualification_boundary_is_strict_at_exactly_alpha():
    """Family 1: an adjusted p exactly equal to alpha does NOT qualify,
    and the next representable double below alpha does."""
    mse, y = _graded(order=("lattice", "T", "rewired", "curr_random"))
    _primary, fam1, fam2, _v = _run_all(mse, y, alpha=_ALPHA)

    # sanity: lattice is favorable, so only the p comparison decides
    assert fam1["per_test"]["lattice"]["favorable"] is True

    fam1["holm_rejected"]["lattice"] = True          # what `<=` would say
    for adj_p, expected in ((_ALPHA, False), (_JUST_BELOW_ALPHA, True)):
        fam1["holm_adjusted_p"]["lattice"] = adj_p
        verdict = stats.one_graph_wins(_primary, fam1, fam2, alpha=_ALPHA)
        qual = verdict["qualification"]["lattice"]
        assert qual["holm_significant"] is expected, f"adjusted p = {adj_p!r}"
        assert qual["qualifies"] is expected
        # the imported step-down decision is recorded, unchanged, alongside
        assert qual["holm_rejected"] is True


def test_family2_pairwise_boundary_is_strict_at_exactly_alpha():
    """Family 2: same boundary, same direction. `pairwise_outcome` reads
    alpha from the family dict, so both orientations of the pair agree."""
    mse, _y = _graded()
    fam2 = stats.family2_pairwise(mse, alpha=_ALPHA, run_sign_flip=False)
    key = "T_vs_lattice"
    fam2["holm_rejected"][key] = True                # what `<=` would say

    for adj_p, expected in ((_ALPHA, False), (_JUST_BELOW_ALPHA, True)):
        fam2["holm_adjusted_p"][key] = adj_p
        out = stats.pairwise_outcome(fam2, "T", "lattice")
        assert out["holm_significant"] is expected, f"adjusted p = {adj_p!r}"
        assert out["holm_rejected"] is True
        assert out["alpha"] == _ALPHA
        # reading the pair from the other side must not change significance
        mirror = stats.pairwise_outcome(fam2, "lattice", "T")
        assert mirror["holm_significant"] is expected


def test_beats_all_uses_the_strict_comparison_not_the_step_down_flag():
    """The boundary propagates into `beats_all`, and so into
    `unique_winner`: T sweeps its three pairs, but one of them adjusts to
    exactly alpha, so it does not beat all three."""
    mse, y = _graded(order=("T", "lattice", "rewired", "curr_random"))
    primary, fam1, fam2, verdict = _run_all(mse, y, alpha=_ALPHA)
    assert verdict["unique_winner"] == "T"           # before the boundary edit

    key = "T_vs_curr_random"
    fam2["holm_adjusted_p"][key] = _ALPHA
    fam2["holm_rejected"][key] = True
    edged = stats.one_graph_wins(primary, fam1, fam2, alpha=_ALPHA)
    assert edged["beats_others"]["T"]["beats_all"] is False
    assert edged["unique_winner"] is None
    # T still qualifies -- only the Family-2 comparison moved
    assert edged["qualification"]["T"]["qualifies"] is True


def test_alpha_comparison_note_is_carried_in_the_verdict():
    """A reporting caller must see why the threshold is strict without
    reading the source."""
    mse, y = _graded()
    _p, _f1, _f2, verdict = _run_all(mse, y)
    note = verdict["alpha_comparison"]
    assert note == stats.ALPHA_COMPARISON_NOTE
    assert "strict" in note.lower()
    assert "holm_rejected" in note


# ---- identity baselines: the hierarchical gate ----

def _with_identity(mse, identity_mse):
    out = dict(mse)
    out["identity"] = identity_mse
    return out


def test_identity_gate_not_evaluated_when_the_primary_fails():
    """"Only if primary succeeds ... never rescues a failed primary."
    Enforced structurally: when the primary fails there is no gate number
    at all, not a computed-but-ignored one."""
    n = 200
    y = np.tile(np.arange(10), n // 10)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base, "T": base + 0.02, "lattice": base,
           "rewired": base, "curr_random": base}
    mse = _with_identity(mse, base + 0.10)   # T would easily beat identity
    primary = stats.primary_bootstrap_test(mse, y, n_resamples=300)
    assert primary["entirely_below_zero"] is False
    result = stats.identity_baseline_gate(primary, mse, y, "identity", n_resamples=300)
    assert result["gate_evaluated"] is False
    assert result["gate"] is None
    assert result["gate_passed"] is False


def test_identity_gate_evaluated_and_context_always_reported():
    n = 200
    y = np.tile(np.arange(10), n // 10)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base + 0.02, "T": base, "lattice": base,
           "rewired": base, "curr_random": base}
    mse = _with_identity(mse, base + 0.10)
    primary = stats.primary_bootstrap_test(mse, y, n_resamples=300)
    result = stats.identity_baseline_gate(primary, mse, y, "identity", n_resamples=300)
    assert result["gate_evaluated"] is True
    assert result["gate_passed"] is True
    assert result["gate"]["contrast"] == "MSE(T) - MSE(identity)"
    # context is reported independently, outside the gate
    assert result["pre_evolution_vs_identity"]["entirely_below_zero"] is True


def test_identity_context_reported_even_when_the_primary_fails():
    n = 200
    y = np.tile(np.arange(10), n // 10)
    base = np.full(n, 0.05)
    mse = {"pre_evolution": base, "T": base + 0.02, "lattice": base,
           "rewired": base, "curr_random": base}
    mse = _with_identity(mse, base + 0.10)
    primary = stats.primary_bootstrap_test(mse, y, n_resamples=300)
    result = stats.identity_baseline_gate(primary, mse, y, "identity", n_resamples=300)
    assert result["gate"] is None
    assert result["pre_evolution_vs_identity"] is not None


# ---- orchestrator and validation ----

def test_orchestrator_keeps_the_three_levels_separate():
    mse, y = _graded()
    out = stats.run_stage2b_inference(mse, y, n_resamples=300, run_sign_flip=False)
    assert set(out["primary"]).isdisjoint({"raw_p", "holm_adjusted_p"})
    assert out["family1"]["n_tests"] == 3
    assert out["family2"]["n_tests"] == 6
    assert set(out["family1"]["raw_p"]).isdisjoint(set(out["family2"]["raw_p"]))
    assert out["one_graph_wins"]["unique_winner"] == "T"


def test_orchestrator_includes_identity_baselines_only_when_asked():
    mse, y = _graded()
    without = stats.run_stage2b_inference(mse, y, n_resamples=300, run_sign_flip=False)
    assert "identity_baselines" not in without
    mse2 = _with_identity(mse, mse["pre_evolution"] + 0.05)
    with_id = stats.run_stage2b_inference(mse2, y, n_resamples=300, run_sign_flip=False,
                                          identity_key="identity")
    assert with_id["identity_baselines"]["gate_evaluated"] is True


def test_validate_conditions_catches_missing_and_mismatched_inputs():
    mse, y = _mse_conditions(n=100, seed=10)
    assert stats.validate_conditions(mse, y) is True
    with pytest.raises(KeyError, match="missing conditions"):
        stats.validate_conditions({k: v for k, v in mse.items() if k != "rewired"}, y)
    with pytest.raises(ValueError, match="values but y has"):
        stats.validate_conditions(dict(mse, T=mse["T"][:50]), y)
    with pytest.raises(ValueError, match="non-finite"):
        bad = dict(mse, T=mse["T"].copy())
        bad["T"][0] = np.nan
        stats.validate_conditions(bad, y)
