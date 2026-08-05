"""
Tests for experiments/stage2b_denoising/stage2b_encoder_gate.py -- the
encoder-on-noisy-inputs gate (DESIGN.md, feasibility stage 1).

Tier 1 (self-contained, always run) only. The gate's decision logic is
tested as a pure function on synthetic final-Delta sequences with known
median and tail behaviour; the final-Delta measurement itself is tested
against a one-step oracle written here in the test file.

That oracle deliberately reimplements one iteration of
`_local_converged_phases`' update. That is a test-only verification
device, the same pattern as sklearn-as-oracle for the ridge -- CLAUDE.md
principle 16 is about PRODUCTION glue reimplementing a helper instead of
calling it, which `stage2b_encoder_gate` does not do (it calls the
unmodified encoder twice and wraps the difference).

No KMNIST data of any split is touched by this file. The synthetic
"images" are random arrays in [0, 1].
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_encoder_gate as gate  # noqa: E402
from bonsai.dynamics.learned_topology_construction import _local_converged_phases  # noqa: E402

_ALL_784 = np.arange(784)


def _one_update_oracle(phases, target_phase, dt=0.1, k_coupling=1.0, k_bias=1.0):
    """TEST-ONLY oracle: one iteration of the encoder's update law, written
    out independently so the two-call prefix trick can be checked against
    it rather than assumed correct. Returns the applied update dt*dtheta."""
    coupling = np.zeros_like(phases)
    coupling[1:, :] += np.sin(phases[:-1, :] - phases[1:, :])
    coupling[:-1, :] += np.sin(phases[1:, :] - phases[:-1, :])
    coupling[:, 1:] += np.sin(phases[:, :-1] - phases[:, 1:])
    coupling[:, :-1] += np.sin(phases[:, 1:] - phases[:, :-1])
    bias = np.sin(target_phase - phases)
    return dt * (k_coupling * coupling + k_bias * bias)


# ---- final-Delta measurement ----

def test_encoder_prefix_is_byte_identical_across_step_counts():
    """The two-call trick is only exact if the `steps=149` field is a
    byte-identical prefix of the `steps=150` trajectory -- i.e. the
    initial random draw does not depend on `steps`."""
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (28, 28))
    a = _local_converged_phases(image, steps=100, seed=0)
    b = _local_converged_phases(image, steps=100, seed=0)
    np.testing.assert_array_equal(a, b)
    # extend a shorter run by hand and land exactly on the longer run
    short = _local_converged_phases(image, steps=99, seed=0)
    stepped = (short + _one_update_oracle(short, image * np.pi)) % (2 * np.pi)
    np.testing.assert_allclose(stepped, a, rtol=0, atol=1e-12)


def test_final_delta_matches_one_step_oracle():
    """final_delta must equal max|dt*dtheta| computed directly from the
    step-149 state -- the quantity the module claims to measure."""
    rng = np.random.default_rng(1)
    for _ in range(3):
        image = rng.uniform(0, 1, (28, 28))
        prev = _local_converged_phases(image, steps=gate.ENCODER_STEPS - 1, seed=0)
        expected = float(np.max(np.abs(_one_update_oracle(prev, image * np.pi))))
        assert gate.final_delta(image) == pytest.approx(expected, rel=0, abs=1e-14)


def test_final_delta_matches_oracle_on_pure_noise_image():
    """The bound |dt*dtheta| <= 0.1*(4+1) = 0.5 < pi is input-independent,
    so the wrapped difference stays alias-free on a maximally rough,
    non-smooth input -- the case this gate exists for."""
    rng = np.random.default_rng(2)
    image = rng.uniform(0, 1, (28, 28))  # spatially uncorrelated, unlike a digit
    prev = _local_converged_phases(image, steps=gate.ENCODER_STEPS - 1, seed=0)
    applied = _one_update_oracle(prev, image * np.pi)
    assert np.max(np.abs(applied)) < np.pi  # no aliasing possible
    assert gate.final_delta(image) == pytest.approx(float(np.max(np.abs(applied))), abs=1e-14)


def test_final_delta_is_positive_and_finite_on_smooth_and_rough_inputs():
    rng = np.random.default_rng(3)
    rough = rng.uniform(0, 1, (28, 28))
    smooth = np.tile(np.linspace(0, 1, 28), (28, 1))
    for image in (rough, smooth):
        d = gate.final_delta(image)
        assert np.isfinite(d) and d >= 0.0


def test_final_delta_deterministic_for_fixed_seed():
    rng = np.random.default_rng(4)
    image = rng.uniform(0, 1, (28, 28))
    assert gate.final_delta(image, seed=0) == gate.final_delta(image, seed=0)


def test_final_delta_non_finite_input_propagates_not_absorbed():
    image = np.full((28, 28), 0.5)
    image[3, 3] = np.nan
    assert not np.isfinite(gate.final_delta(image))


# ---- the gate as a pure function on synthetic final-Delta sequences ----

def _sequences(n=200, clean_scale=1e-6, ratio=1.0, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.uniform(0.5, 1.5, n) * clean_scale
    noisy = rng.uniform(0.5, 1.5, n) * clean_scale * ratio
    return clean, noisy


def test_gate_passes_when_noisy_matches_clean():
    clean, noisy = _sequences(ratio=1.0)
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["passed"] is True
    assert result["rho"] == pytest.approx(np.median(noisy) / np.median(clean))


def test_gate_passes_at_moderate_ratio():
    clean = np.full(100, 2e-6)
    noisy = np.full(100, 6e-6)   # rho = 3
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["rho"] == pytest.approx(3.0)
    assert result["passed"] is True


def test_gate_passes_exactly_at_the_threshold():
    """PASS iff rho <= 10 -- the boundary is inclusive.

    The values are chosen so the ratio is exactly 10.0 in float64:
    `1e-5 / 1e-6` is 10.000000000000002, which would fail the comparison
    for a reason that has nothing to do with the gate."""
    clean = np.full(100, 0.5)
    noisy = np.full(100, 5.0)
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["rho"] == 10.0
    assert result["passed"] is True


def test_gate_fails_just_above_the_threshold():
    clean = np.full(100, 1e-6)
    noisy = np.full(100, 1.01e-5)
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["rho"] > gate.RHO_THRESHOLD
    assert result["passed"] is False
    assert any("threshold" in r for r in result["failure_reasons"])


def test_gate_uses_median_not_mean_so_a_few_outliers_do_not_flip_it():
    """A handful of enormous noisy final-Deltas move the mean by orders of
    magnitude but not the median -- the gate must key on the median, with
    the tail visible in p95 rather than driving the decision."""
    clean = np.full(200, 1e-6)
    noisy = np.full(200, 2e-6)
    noisy[:5] = 1e3
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["rho"] == pytest.approx(2.0)
    assert result["passed"] is True
    assert np.mean(noisy) / np.mean(clean) > 1e6   # the mean WOULD have failed


def test_gate_reports_p95_tail_without_gating_on_it():
    """DESIGN.md: the 95th percentile is visibility for a passing-median-
    but-exploding-tail pattern, explicitly not a second gate."""
    clean = np.full(1000, 1e-6)
    noisy = np.full(1000, 2e-6)
    noisy[-100:] = 1.0             # top 10% explodes
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["passed"] is True                      # median-based, still passes
    assert result["p95_delta_noisy"] == pytest.approx(1.0)
    assert result["p95_delta_noisy"] / result["p95_delta_clean"] > 1e5


def test_gate_median_floor_protects_against_zero_clean_median():
    """A degenerate all-zero clean median must hit the 1e-15 floor rather
    than dividing by zero -- and the floor is numerical protection, so the
    result is an honest, enormous rho and a FAIL, not a pass."""
    clean = np.zeros(50)
    noisy = np.full(50, 1e-6)
    result = gate.evaluate_rho_gate(clean, noisy)
    assert np.isfinite(result["rho"])
    assert result["rho"] == pytest.approx(1e-6 / 1e-15)
    assert result["passed"] is False


def test_gate_both_medians_zero_gives_rho_zero_and_passes():
    result = gate.evaluate_rho_gate(np.zeros(20), np.zeros(20))
    assert result["rho"] == 0.0
    assert result["passed"] is True


def test_gate_automatic_failure_on_non_finite_final_delta_despite_small_rho():
    """Automatic failure regardless of rho: the surviving finite values
    give a perfectly acceptable ratio, and the gate must still fail."""
    clean = np.full(100, 1e-6)
    noisy = np.full(100, 1e-6)
    noisy[0] = np.nan
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["automatic_failure"] is True
    assert result["passed"] is False
    assert result["n_nonfinite_delta_noisy"] == 1


def test_gate_automatic_failure_on_infinite_final_delta():
    clean = np.full(100, 1e-6)
    noisy = np.full(100, 1e-6)
    noisy[7] = np.inf
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["automatic_failure"] is True
    assert result["passed"] is False


def test_gate_automatic_failure_on_non_finite_encoded_phase_with_healthy_rho():
    clean = np.full(100, 1e-6)
    noisy = np.full(100, 1e-6)
    thetas = np.zeros((100, 505))
    thetas[4, 11] = np.nan
    result = gate.evaluate_rho_gate(clean, noisy, thetas_noisy=thetas)
    assert result["rho"] == pytest.approx(1.0)          # rho alone would pass
    assert result["automatic_failure"] is True
    assert result["passed"] is False
    assert result["n_nonfinite_phase_noisy"] == 1
    assert any("non-finite encoded phase" in r for r in result["failure_reasons"])


def test_gate_records_all_required_quantities_even_when_failing():
    clean = np.full(100, 1e-6)
    noisy = np.full(100, 1e-3)     # rho = 1000, a clear failure
    result = gate.evaluate_rho_gate(clean, noisy)
    assert result["passed"] is False
    for key in ("median_delta_clean", "median_delta_noisy",
                "p95_delta_clean", "p95_delta_noisy"):
        assert np.isfinite(result[key])
    assert gate.format_gate_log(result).startswith("encoder-on-noisy-inputs gate: FAIL")


def test_gate_locked_constants():
    assert gate.RHO_THRESHOLD == 10.0
    assert gate.MEDIAN_FLOOR == 1e-15
    assert gate.ENCODER_STEPS == 150


def test_gate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        gate.evaluate_rho_gate(np.zeros(10), np.zeros(11))


# ---- end-to-end on synthetic images (no dataset of any split involved) ----

def test_run_encoder_gate_end_to_end_on_synthetic_images():
    """Smooth "clean" images vs. heavily corrupted versions of them, both
    synthetic. The assertion is on mechanics and bookkeeping, not on
    whether the real encoder passes on real KMNIST -- that is the actual
    stage-1 gate and is deliberately not pre-judged here."""
    rng = np.random.default_rng(5)
    xx, yy = np.meshgrid(np.linspace(0, 1, 28), np.linspace(0, 1, 28))
    clean = np.stack([np.clip(xx * a + yy * (1 - a), 0, 1) for a in (0.2, 0.5, 0.8)])
    noisy = np.clip(np.sqrt(0.5) * clean + np.sqrt(0.5) * rng.normal(size=clean.shape), 0, 1)

    result = gate.run_encoder_gate(clean, noisy, _ALL_784)
    assert result["thetas_clean"].shape == (3, 784)
    assert result["delta_noisy"].shape == (3,)
    assert np.all(np.isfinite(result["thetas_noisy"]))
    assert result["n_nonfinite_phase_noisy"] == 0
    assert isinstance(result["passed"], bool)
    assert "rho" in gate.format_gate_log(result)


def test_run_encoder_gate_uses_the_production_encode_path():
    """The phases the gate inspects must be exactly what
    `stage2a_core.encode_and_restrict` produces -- not a parallel
    encoding written for the diagnostic."""
    from stage2a_core import encode_and_restrict
    rng = np.random.default_rng(6)
    images = rng.uniform(0, 1, (2, 28, 28))
    active = np.array([0, 5, 100, 783])
    result = gate.run_encoder_gate(images, images, active)
    for i, img in enumerate(images):
        np.testing.assert_array_equal(result["thetas_clean"][i],
                                       encode_and_restrict(img, active))


def test_run_encoder_gate_identical_inputs_give_rho_exactly_one():
    """Feeding the same images as both "clean" and "noisy" must give
    rho = 1 exactly -- the degenerate control that would catch the two
    arms being swapped, mis-seeded, or silently reusing one encode."""
    rng = np.random.default_rng(7)
    images = rng.uniform(0, 1, (4, 28, 28))
    result = gate.run_encoder_gate(images, images, _ALL_784)
    np.testing.assert_array_equal(result["delta_clean"], result["delta_noisy"])
    assert result["rho"] == pytest.approx(1.0)
    assert result["passed"] is True


def test_encode_with_final_delta_batch_parallel_matches_serial():
    rng = np.random.default_rng(8)
    images = rng.uniform(0, 1, (4, 28, 28))
    t1, d1 = gate.encode_with_final_delta_batch(images, _ALL_784, n_workers=1)
    t2, d2 = gate.encode_with_final_delta_batch(images, _ALL_784, n_workers=2)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(d1, d2)
