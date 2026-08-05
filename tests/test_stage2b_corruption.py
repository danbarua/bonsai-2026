"""
Tests for experiments/stage2b_denoising/stage2b_corruption.py -- the
locked corruption RNG, the clipped-Gaussian forward process, the
analytical censoring profile, and the corruption diagnostics.

Tier 1 (self-contained, always run) only.

The empirical-vs-analytical clip-rate check is the unit test for the
whole corruption path, not a separate diagnostic script: it exercises
the literal seed derivation, the PCG64 draw, the forward formula, and
the clipping together, and compares the result against DESIGN.md's
precomputed censoring table.

**No KMNIST data of any split is loaded by this file, and no image is
corrupted with `split="test"`.** Two tests call the pure seed-and-draw
layer with `"test"` -- `corruption_seed` and `epsilon_for` take no image
data at all, so those are seed-derivation checks, which is exactly what
DESIGN.md's spec requires be distinguishable from `"train"`. The corpus
layer refuses `split="test"` outright, and there is a test asserting
that refusal.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_corruption as corr  # noqa: E402


# ---- the locked seed derivation ----

def test_seed_matches_the_literal_spec_formula():
    """SHA256(f"{split}:{index}:{MASTER_SEED}"), first 8 bytes
    little-endian -- recomputed independently here as an oracle."""
    import hashlib
    for split, index in (("train", 0), ("train", 59999), ("test", 0)):
        digest = hashlib.sha256(f"{split}:{index}:42".encode()).digest()
        expected = int.from_bytes(digest[:8], "little")
        assert corr.corruption_seed(split, index) == expected


def test_seed_pinned_literal_values():
    """Pinned integers, so a future edit to the formula is caught even if
    the oracle above were edited alongside it."""
    assert corr.corruption_seed("train", 0) == 3738863147873697668
    assert corr.corruption_seed("train", 1) == 12119846103895835675


def test_master_seed_and_alpha_bar_are_locked():
    assert corr.MASTER_SEED == 42
    assert corr.ALPHA_BAR == 0.5
    assert corr.N_PIXELS == 784


def test_same_split_and_index_give_identical_epsilon_on_repeated_calls():
    a = corr.epsilon_for("train", 123)
    b = corr.epsilon_for("train", 123)
    np.testing.assert_array_equal(a, b)


def test_train_and_test_indices_give_different_epsilon():
    """Seed-derivation check only: `epsilon_for` takes a split label and an
    integer, never an image, so no test-side image data is involved. The
    two streams must differ or the corrupted train and test corpora would
    share realizations."""
    assert corr.corruption_seed("train", 0) != corr.corruption_seed("test", 0)
    assert not np.allclose(corr.epsilon_for("train", 0), corr.epsilon_for("test", 0))


def test_different_indices_give_different_epsilon():
    assert not np.allclose(corr.epsilon_for("train", 0), corr.epsilon_for("train", 1))


def test_epsilon_is_reproducible_across_processes():
    """The reason the spec forbids Python's `hash()`: it is salted per
    process. A fresh interpreter (with hash randomization at its default)
    must reproduce the same draw bit-for-bit."""
    code = (
        f"import sys; sys.path.insert(0, {str(_STAGE2B_DIR)!r});"
        "import stage2b_corruption as c;"
        "print(repr(c.epsilon_for('train', 7)[:5].tolist()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert eval(out) == corr.epsilon_for("train", 7)[:5].tolist()


def test_epsilon_dtype_shape_and_distribution():
    eps = corr.epsilon_for("train", 5)
    assert eps.dtype == np.float64
    assert eps.shape == (784,)
    assert np.all(np.isfinite(eps))


def test_seed_rejects_bad_split_and_index():
    with pytest.raises(ValueError):
        corr.corruption_seed("validation", 0)
    with pytest.raises(ValueError):
        corr.corruption_seed("train", -1)


# ---- the test-split lock ----

def test_corpus_layer_refuses_test_split():
    """DESIGN.md: 'no Stage 2B test-side result is accessed during stages
    1-3'. The corpus layer enforces that structurally."""
    images = np.zeros((2, 28, 28))
    with pytest.raises(PermissionError, match="stages 1-3"):
        corr.corrupt_corpus(images, "test", np.arange(2))
    with pytest.raises(PermissionError, match="stages 1-3"):
        corr.corrupt_image(images[0], "test", 0)


def test_corpus_layer_allows_train_split_without_a_flag():
    x_t, x_t_clip = corr.corrupt_corpus(np.zeros((2, 28, 28)), "train", np.arange(2))
    assert x_t.shape == (2, 28, 28) and x_t_clip.shape == (2, 28, 28)


# ---- the forward process ----

def test_forward_formula_matches_spec_by_hand():
    x0 = np.array([0.0, 0.5, 1.0])
    eps = np.array([1.0, -1.0, 0.25])
    x_t, x_t_clip = corr.forward_corrupt(x0, eps)
    expected = np.sqrt(0.5) * x0 + np.sqrt(0.5) * eps
    np.testing.assert_allclose(x_t, expected, rtol=0, atol=0)
    np.testing.assert_allclose(x_t_clip, np.clip(expected, 0, 1))


def test_equal_signal_and_noise_coefficients_at_alpha_bar_half():
    assert corr.signal_coefficient() == corr.noise_coefficient()
    assert corr.noise_coefficient() == pytest.approx(0.7071067811865476)


def test_corrupt_image_dtype_and_shape_preserved():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (28, 28))
    x_t, x_t_clip = corr.corrupt_image(img, "train", 3)
    assert x_t.dtype == np.float64 and x_t_clip.dtype == np.float64
    assert x_t.shape == (28, 28) and x_t_clip.shape == (28, 28)
    flat_t, flat_c = corr.corrupt_image(img.reshape(-1), "train", 3)
    np.testing.assert_array_equal(flat_t, x_t.reshape(-1))
    np.testing.assert_array_equal(flat_c, x_t_clip.reshape(-1))


def test_clipped_output_is_within_unit_interval_and_pre_clip_is_not():
    rng = np.random.default_rng(1)
    images = rng.uniform(0, 1, (20, 28, 28))
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", np.arange(20))
    assert x_t_clip.min() >= 0.0 and x_t_clip.max() <= 1.0
    assert x_t.min() < 0.0 and x_t.max() > 1.0   # clipping is not a no-op


def test_corruption_is_reused_identically_across_calls():
    """'One realization per image, reused identically across every
    condition' -- the same image at the same index must corrupt the same
    way every time it is requested."""
    rng = np.random.default_rng(2)
    img = rng.uniform(0, 1, (28, 28))
    a, _ = corr.corrupt_image(img, "train", 99)
    b, _ = corr.corrupt_image(img, "train", 99)
    np.testing.assert_array_equal(a, b)


def test_corpus_uses_the_supplied_split_indices_not_positions():
    """A feasibility subset must be corrupted with each image's ORIGINAL
    split index. Corrupting the same image at position 0 of a subset and
    at its true index 500 must give different, and index-determined,
    realizations."""
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 1, (28, 28))
    corpus, _ = corr.corrupt_corpus(np.stack([img]), "train", np.array([500]))
    single, _ = corr.corrupt_image(img, "train", 500)
    np.testing.assert_array_equal(corpus[0], single)
    at_zero, _ = corr.corrupt_image(img, "train", 0)
    assert not np.allclose(corpus[0], at_zero)


def test_corpus_pairs_each_image_with_its_own_index():
    """Multi-image, non-monotonic indices -- an off-by-one or a zip/
    enumerate mixup survives the single-image test above but not this."""
    rng = np.random.default_rng(30)
    images = rng.uniform(0, 1, (3, 28, 28))
    indices = np.array([500, 7, 12000])
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", indices)
    for i, idx in enumerate(indices):
        single_t, single_c = corr.corrupt_image(images[i], "train", int(idx))
        np.testing.assert_array_equal(x_t[i], single_t)
        np.testing.assert_array_equal(x_t_clip[i], single_c)
    # and the pairing is not accidentally symmetric across rows
    assert not np.allclose(x_t[0], corr.corrupt_image(images[0], "train", 7)[0])


def test_corrupt_image_rejects_a_restricted_array_rather_than_adapting():
    """Corruption is defined on the full 784-pixel grid. A 505-length
    array must raise, not quietly get a 505-length draw."""
    with pytest.raises(ValueError, match="784"):
        corr.corrupt_image(np.zeros(505), "train", 0)
    with pytest.raises(ValueError, match="784"):
        corr.corrupt_corpus(np.zeros((2, 505)), "train", np.arange(2))


def test_corpus_requires_one_index_per_image_and_rejects_duplicates():
    images = np.zeros((3, 28, 28))
    with pytest.raises(ValueError):
        corr.corrupt_corpus(images, "train", np.arange(2))
    with pytest.raises(ValueError):
        corr.corrupt_corpus(images, "train", np.array([0, 1, 1]))


# ---- input range: [0, 1], asserted rather than adapted to ----

def test_corpus_rejects_uint8_scale_input():
    """`bonsai.data.mnist_loader.load_mnist` returns uint8 0-255. Handed
    straight to the corruption, that is accepted arithmetically and
    produces a corpus in which ~99% of pixels saturate at 1.0 -- no
    error, a plausible-looking result, and every downstream MSE wrong.
    Refused for the same reason the 784-pixel size is refused rather than
    adapted to."""
    images = (np.random.default_rng(0).random((4, 28, 28)) * 255).astype(np.uint8)
    with pytest.raises(ValueError, match="255"):
        corr.corrupt_image(images[0], "train", 0)
    with pytest.raises(ValueError, match="255"):
        corr.corrupt_corpus(images, "train", np.arange(4))


def test_input_range_error_reports_the_observed_range():
    """The message has to say what was actually seen -- a caller holding
    an array of unknown provenance learns the scale from the error
    instead of going back to instrument the loader."""
    images = np.zeros((3, 784))
    images[1, 2] = 7.5
    images[2, 3] = -0.25
    with pytest.raises(ValueError) as excinfo:
        corr.corrupt_corpus(images, "train", np.arange(3))
    message = str(excinfo.value)
    assert "7.5" in message and "-0.25" in message
    assert "255" in message


def test_unit_interval_boundaries_are_accepted():
    """0.0 and 1.0 are legal clean intensities -- DESIGN.md's censoring
    table tabulates both endpoints. A comparison written one step too
    strict would reject the two most common pixel values in the corpus."""
    for value in (0.0, 1.0):
        x_t, x_t_clip = corr.corrupt_image(np.full((28, 28), value), "train", 3)
        assert np.all(np.isfinite(x_t)) and x_t_clip.min() >= 0.0
    images = np.stack([np.zeros((28, 28)), np.ones((28, 28))])
    corr.corrupt_corpus(images, "train", np.arange(2))


def test_valid_unit_interval_input_is_unaffected_by_the_check():
    """The guard rejects; it does not touch the values that pass it."""
    images = np.random.default_rng(1).random((5, 784))
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", np.arange(5))
    for i in range(5):
        expected_t, expected_c = corr.forward_corrupt(
            images[i], corr.epsilon_for("train", i))
        np.testing.assert_array_equal(x_t[i], expected_t)
        np.testing.assert_array_equal(x_t_clip[i], expected_c)


def test_non_finite_input_is_rejected():
    """NaN defeats the range comparison itself: `nan < 0` and `nan > 1`
    are both False, so an unchecked NaN would pass a min/max test and
    propagate silently into every downstream MSE."""
    images = np.zeros((2, 784))
    images[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        corr.corrupt_corpus(images, "train", np.arange(2))
    with pytest.raises(ValueError, match="finite"):
        corr.corrupt_image(np.full(784, np.inf), "train", 0)


# ---- the censoring profile: analytical form and the DESIGN.md table ----

def test_analytical_rates_reproduce_the_design_table():
    """The literal table transcribed from DESIGN.md, to its stated
    3-decimal precision."""
    for x0, p_below, p_above, total in corr.ANALYTICAL_CLIP_TABLE:
        b, a, t = corr.analytical_clip_rates(x0)
        assert float(b) == pytest.approx(p_below, abs=5e-4)
        assert float(a) == pytest.approx(p_above, abs=5e-4)
        assert float(t) == pytest.approx(total, abs=1e-3)


def test_analytical_rates_closed_form_at_alpha_bar_half():
    """At alpha_bar = 0.5 the general expression collapses to
    Phi(-x_0) and 1 - Phi(sqrt(2) - x_0) -- checked so a change to the
    general formula cannot quietly stop reproducing the frozen level."""
    x0 = np.linspace(0, 1, 11)
    b, a, _t = corr.analytical_clip_rates(x0)
    np.testing.assert_allclose(b, norm.cdf(-x0), rtol=1e-12)
    np.testing.assert_allclose(a, 1.0 - norm.cdf(np.sqrt(2.0) - x0), rtol=1e-12)


def test_censoring_is_majority_at_every_clean_intensity():
    """DESIGN.md's stated consequence -- roughly half of all pixel values
    clip at every clean intensity. Stated as fact in the design, so it is
    asserted here rather than rediscovered."""
    _b, _a, total = corr.analytical_clip_rates(np.linspace(0, 1, 21))
    assert np.all(total > 0.47)


def test_empirical_clip_rates_match_the_analytical_table():
    """The unit test for the whole corruption path: real seeds, real
    PCG64 draws, the real forward formula, measured against the
    precomputed table. Constant-intensity synthetic images at each of the
    table's five x_0 levels; ~313,600 draws per level gives a standard
    error near 9e-4, so a 5e-3 tolerance is roughly 5 sigma."""
    n_images = 400
    for level, (x0, p_below, p_above, _total) in enumerate(corr.ANALYTICAL_CLIP_TABLE):
        images = np.full((n_images, 28, 28), x0)
        # distinct index blocks per level, so no two levels share draws
        indices = np.arange(n_images) + level * 10_000
        x_t, _clip = corr.corrupt_corpus(images, "train", indices)
        rates = corr.empirical_clip_rates(x_t)
        assert rates["below_zero"] == pytest.approx(p_below, abs=5e-3)
        assert rates["above_one"] == pytest.approx(p_above, abs=5e-3)
        assert rates["n"] == n_images * 784


def test_predicted_clip_rates_match_the_measured_ones_on_a_mixed_corpus():
    """The table's five rows are constant-intensity predictions; a real
    corpus is a mixture and matches no row. `predicted_clip_rates` is the
    mixture's prediction, and this is the comparison DESIGN.md describes
    as confirmation -- pinned rather than left to a human reading the
    measured rates against the tabulated ones.

    Tolerance is computed here, from the model: each pixel clips
    independently with its own probability, so the standard error of the
    corpus rate is `sqrt(sum p(1-p)) / n`. Four sigma."""
    rng = np.random.default_rng(11)
    n_images = 300
    images = np.clip(rng.beta(0.6, 0.6, size=(n_images, 28, 28)), 0.0, 1.0)
    x_t, _clip = corr.corrupt_corpus(images, "train", np.arange(n_images) + 70_000)

    predicted = corr.predicted_clip_rates(images)
    observed = corr.empirical_clip_rates(x_t)
    p_below, p_above, _t = corr.analytical_clip_rates(images.reshape(-1))
    n = p_below.size
    assert predicted["n"] == observed["n"] == n

    for key, p in (("below_zero", p_below), ("above_one", p_above)):
        se = float(np.sqrt(np.sum(p * (1.0 - p)))) / n
        assert observed[key] == pytest.approx(predicted[key], abs=4.0 * se)
    # the corpus is genuinely a mixture: no table row predicts it
    for _x0, table_below, _table_above, _total in corr.ANALYTICAL_CLIP_TABLE:
        assert abs(predicted["below_zero"] - table_below) > 1e-3


def test_predicted_clip_rates_reduce_to_the_table_at_constant_intensity():
    """A corpus at one intensity has to reproduce that intensity's row --
    the bridge between the mixture prediction and the tabulated one."""
    for x0, p_below, p_above, _total in corr.ANALYTICAL_CLIP_TABLE:
        predicted = corr.predicted_clip_rates(np.full((5, 784), x0))
        assert predicted["below_zero"] == pytest.approx(p_below, abs=5e-4)
        assert predicted["above_one"] == pytest.approx(p_above, abs=5e-4)
        assert predicted["n"] == 5 * 784


def test_empirical_clip_rates_report_the_two_directions_separately():
    x_t = np.array([-1.0, -0.5, 0.5, 1.5, 2.0])
    rates = corr.empirical_clip_rates(x_t)
    assert rates["below_zero"] == pytest.approx(0.4)
    assert rates["above_one"] == pytest.approx(0.4)
    assert rates["total"] == pytest.approx(0.8)


def test_format_analytical_table_reproduces_design_md_rows():
    text = corr.format_analytical_table()
    for x0, p_below, p_above, total in corr.ANALYTICAL_CLIP_TABLE:
        assert f"{x0:.2f}" in text
        assert f"{p_below:.3f}" in text and f"{p_above:.3f}" in text
        assert f"{total:.3f}" in text


# ---- predicted-vs-empirical agreement, with its Monte Carlo tolerance ----

def _naive_clip_probabilities(x0_flat):
    """An oracle for the per-pixel censoring probabilities, written here
    rather than imported: a plain Python loop over the alpha_bar = 0.5
    closed form, `Phi(-x_0)` and `1 - Phi(sqrt(2) - x_0)`. Independent of
    `analytical_clip_rates`, which is the function under test."""
    below, above = [], []
    for value in x0_flat:
        below.append(float(norm.cdf(-value)))
        above.append(float(1.0 - norm.cdf(np.sqrt(2.0) - value)))
    return np.array(below), np.array(above)


def _mixed_corpus(n_images, seed, index_offset):
    """A corpus with mass at both ends of [0, 1], so the below-zero and
    above-one channels both carry signal."""
    rng = np.random.default_rng(seed)
    images = np.clip(rng.beta(0.5, 0.5, size=(n_images, 28, 28)), 0.0, 1.0)
    x_t, x_t_clip = corr.corrupt_corpus(
        images, "train", np.arange(n_images) + index_offset)
    return images, x_t, x_t_clip


def test_clip_rate_agreement_returns_the_documented_keys():
    images, x_t, _clip = _mixed_corpus(40, seed=101, index_offset=200_000)
    out = corr.clip_rate_agreement(images, x_t)
    assert set(out) == {"below_zero", "above_one", "total",
                        "n_sigma", "n_pixels", "alpha_bar"}
    for key in ("below_zero", "above_one", "total"):
        assert set(out[key]) == {"predicted", "empirical", "se", "z",
                                 "within_tolerance"}
        # native Python types: a driver JSON-dumps this
        assert type(out[key]["within_tolerance"]) is bool
        for scalar in ("predicted", "empirical", "se", "z"):
            assert type(out[key][scalar]) is float
    assert out["n_pixels"] == 40 * 784      # total observations, not N_PIXELS
    assert out["alpha_bar"] == 0.5 and out["n_sigma"] == 5.0


def test_clip_rate_agreement_against_an_independent_naive_oracle():
    """Test 1: a corpus whose predicted rates and tolerance are both
    recomputed here by a per-pixel loop over the closed form, with no
    call into the module's own analytical path."""
    images, x_t, _clip = _mixed_corpus(120, seed=102, index_offset=210_000)
    out = corr.clip_rate_agreement(images, x_t)

    p_below, p_above = _naive_clip_probabilities(images.reshape(-1))
    n = float(p_below.size)
    for key, p in (("below_zero", p_below), ("above_one", p_above),
                   ("total", p_below + p_above)):
        assert out[key]["predicted"] == pytest.approx(float(np.mean(p)), abs=1e-12)
        assert out[key]["se"] == pytest.approx(
            float(np.sqrt(np.sum(p * (1.0 - p)))) / n, rel=1e-12)
        assert out[key]["within_tolerance"] is True
        assert abs(out[key]["z"]) <= 5.0

    # the empirical side is the module's measured rate, unmodified
    observed = corr.empirical_clip_rates(x_t)
    for key in ("below_zero", "above_one", "total"):
        assert out[key]["empirical"] == pytest.approx(observed[key], abs=0.0)


def test_clip_rate_agreement_total_variance_matches_the_covariance_identity():
    """The mutual-exclusivity premise behind using `p_total` directly,
    verified rather than asserted. Within a pixel the two events cannot
    both occur, so `Cov(below, above) = -p_b * p_a` and

        Var(below + above) = sum[ p_b(1-p_b) + p_a(1-p_a) - 2 p_b p_a ]

    which must equal `sum[ p_t(1-p_t) ]`. A second derivation, not the
    same formula rewritten."""
    images, x_t, _clip = _mixed_corpus(60, seed=103, index_offset=220_000)
    out = corr.clip_rate_agreement(images, x_t)

    p_below, p_above = _naive_clip_probabilities(images.reshape(-1))
    n = float(p_below.size)
    var_decomposed = float(np.sum(p_below * (1.0 - p_below)
                                  + p_above * (1.0 - p_above)
                                  - 2.0 * p_below * p_above))
    assert out["total"]["se"] == pytest.approx(np.sqrt(var_decomposed) / n, rel=1e-12)
    # and it is genuinely not the sum of the two channel SEs
    assert out["total"]["se"] < out["below_zero"]["se"] + out["above_one"]["se"]


def test_clip_rate_agreement_keeps_the_two_directions_uncollapsed():
    """Test 2: on a corpus asymmetric in x_0 the two channels must differ
    substantially, so an implementation that summed or shared them is
    caught."""
    images = np.full((150, 28, 28), 0.05)      # dark: clips down far more than up
    x_t, _clip = corr.corrupt_corpus(images, "train", np.arange(150) + 230_000)
    out = corr.clip_rate_agreement(images, x_t)

    assert out["below_zero"]["predicted"] > 0.4
    assert out["above_one"]["predicted"] < 0.1
    assert out["below_zero"]["empirical"] != out["above_one"]["empirical"]
    assert out["below_zero"]["se"] != out["above_one"]["se"]
    assert out["total"]["predicted"] == pytest.approx(
        out["below_zero"]["predicted"] + out["above_one"]["predicted"], rel=1e-12)
    for key in ("below_zero", "above_one", "total"):
        assert out[key]["within_tolerance"] is True


def test_clip_rate_agreement_fails_on_a_wrong_alpha_bar():
    """Test 3, the one that matters: a guard never seen to fail is not
    yet a guard (CLAUDE.md principle 21). A corpus corrupted at the
    locked alpha_bar = 0.5 and checked against an alpha_bar = 0.9 profile
    must fall outside tolerance, in a named channel, by a wide margin.

    The above-one channel is the sharp one: `P(clip below 0)` at x_0 = 0
    is 0.5 for every alpha_bar, so a dark corpus can look nearly fine
    below while the upper tail collapses. A mixed corpus is used so both
    channels carry mass."""
    images, x_t, _clip = _mixed_corpus(200, seed=104, index_offset=240_000)
    honest = corr.clip_rate_agreement(images, x_t, alpha_bar=0.5)
    assert honest["above_one"]["within_tolerance"] is True

    wrong = corr.clip_rate_agreement(images, x_t, alpha_bar=0.9)
    assert wrong["above_one"]["within_tolerance"] is False
    assert abs(wrong["above_one"]["z"]) > 20.0
    assert wrong["total"]["within_tolerance"] is False
    assert abs(wrong["total"]["z"]) > 20.0


def test_clip_rate_agreement_fails_on_a_mis_scaled_corpus():
    """Test 3 again, from the other side: a corpus actually corrupted at
    the wrong level, checked against the locked profile. Nothing raises
    -- the agreement statistic is what detects it.

    The corruption is built at alpha_bar = 0.8 (noise std 0.447 instead
    of 0.707) and compared against the frozen 0.5 profile, so both
    channels move. Rescaling a correct `x_t` by a positive constant
    instead would be the weaker probe: it is sign-preserving, so the
    below-zero count is identical by construction and only the above-one
    channel could ever depart."""
    rng = np.random.default_rng(105)
    images = np.clip(rng.beta(0.5, 0.5, size=(200, 28, 28)), 0.0, 1.0)
    indices = np.arange(200) + 250_000
    x_t_wrong, _clip = corr.corrupt_corpus(images, "train", indices, alpha_bar=0.8)
    x_t_right, _clip = corr.corrupt_corpus(images, "train", indices, alpha_bar=0.5)

    assert corr.clip_rate_agreement(images, x_t_right)["total"]["within_tolerance"]

    mis_scaled = corr.clip_rate_agreement(images, x_t_wrong)     # profile stays 0.5
    assert mis_scaled["below_zero"]["within_tolerance"] is False
    assert mis_scaled["above_one"]["within_tolerance"] is False
    assert abs(mis_scaled["total"]["z"]) > 20.0

    # a positive rescale moves the upper channel only -- recorded, not relied on
    rescaled = corr.clip_rate_agreement(images, x_t_right * 1.5)
    assert rescaled["below_zero"]["empirical"] == pytest.approx(
        corr.empirical_clip_rates(x_t_right)["below_zero"], abs=0.0)
    assert rescaled["above_one"]["within_tolerance"] is False


def test_clip_rate_agreement_flags_a_post_clip_array_handed_in_by_mistake():
    """`x_t_clip` in place of `x_t` is the documented footgun: both
    measured rates collapse to zero. Not raised -- a genuinely tiny
    corpus can also clip nowhere -- but it must show up as an enormous
    departure rather than passing quietly."""
    images, _x_t, x_t_clip = _mixed_corpus(100, seed=106, index_offset=260_000)
    out = corr.clip_rate_agreement(images, x_t_clip)
    assert out["below_zero"]["empirical"] == 0.0
    assert out["above_one"]["empirical"] == 0.0
    assert out["below_zero"]["within_tolerance"] is False
    assert abs(out["total"]["z"]) > 50.0


def test_clip_rate_agreement_se_shrinks_as_one_over_sqrt_n():
    """Test 4, made exact. At one constant intensity every p_i is equal,
    so `se = sqrt(p(1-p)/N)` and quadrupling the corpus divides it by
    exactly 2 -- no tolerance band needed."""
    small = np.full((50, 28, 28), 0.5)
    large = np.full((200, 28, 28), 0.5)
    x_t_small, _ = corr.corrupt_corpus(small, "train", np.arange(50) + 270_000)
    x_t_large, _ = corr.corrupt_corpus(large, "train", np.arange(200) + 280_000)
    out_small = corr.clip_rate_agreement(small, x_t_small)
    out_large = corr.clip_rate_agreement(large, x_t_large)

    assert out_large["n_pixels"] == 4 * out_small["n_pixels"]
    for key in ("below_zero", "above_one", "total"):
        assert out_small[key]["se"] / out_large[key]["se"] == pytest.approx(2.0, rel=1e-12)


def test_clip_rate_agreement_tolerance_depends_only_on_the_clean_corpus():
    """"Exact, not fudged" restated as a property: the SE comes from the
    analytical profile alone, so two different noise realizations of the
    same clean corpus give a bit-identical tolerance while their measured
    rates differ."""
    images = np.clip(np.random.default_rng(107).beta(0.5, 0.5, (80, 28, 28)), 0.0, 1.0)
    x_t_a, _ = corr.corrupt_corpus(images, "train", np.arange(80) + 290_000)
    x_t_b, _ = corr.corrupt_corpus(images, "train", np.arange(80) + 300_000)
    out_a = corr.clip_rate_agreement(images, x_t_a)
    out_b = corr.clip_rate_agreement(images, x_t_b)

    assert out_a["below_zero"]["empirical"] != out_b["below_zero"]["empirical"]
    for key in ("below_zero", "above_one", "total"):
        assert out_a[key]["se"] == out_b[key]["se"]
        assert out_a[key]["predicted"] == out_b[key]["predicted"]


def test_clip_rate_agreement_handles_a_zero_variance_channel():
    """The `se == 0` branch, exercised rather than left to inspection.
    At alpha_bar = 0.999999 the noise std is 1e-3, so at x_0 = 1 the
    below-zero probability underflows to exactly 0.0 and that channel has
    no variance to divide by. A matching observation must read z = 0, and
    any departure at all must read infinite rather than raising or
    silently becoming NaN."""
    assert float(corr.analytical_clip_rates(1.0, 0.999999)[0]) == 0.0

    images = np.ones((5, 28, 28))
    indices = np.arange(5) + 330_000
    x_t, _clip = corr.corrupt_corpus(images, "train", indices, alpha_bar=0.999999)
    out = corr.clip_rate_agreement(images, x_t, alpha_bar=0.999999)
    assert out["below_zero"]["se"] == 0.0
    assert out["below_zero"]["predicted"] == 0.0
    assert out["below_zero"]["empirical"] == 0.0
    assert out["below_zero"]["z"] == 0.0
    assert out["below_zero"]["within_tolerance"] is True

    x_t_bad = np.array(x_t, copy=True).reshape(-1)
    x_t_bad[0] = -0.5                       # one clip a zero-variance channel forbids
    bad = corr.clip_rate_agreement(images, x_t_bad, alpha_bar=0.999999)
    assert bad["below_zero"]["se"] == 0.0
    assert bad["below_zero"]["z"] == np.inf
    assert bad["below_zero"]["within_tolerance"] is False


def test_clip_rate_agreement_accepts_a_restricted_scope_and_rejects_a_mismatch():
    """The 505 active support is a legitimate scope for this comparison,
    so nothing may assume 784. A size MISMATCH between x_0 and x_t is
    refused: it would silently mix two censoring populations."""
    images, x_t, _clip = _mixed_corpus(30, seed=108, index_offset=310_000)
    active = np.sort(np.random.default_rng(9).choice(784, 505, replace=False))
    flat0 = images.reshape(30, -1)[:, active]
    flat_t = x_t.reshape(30, -1)[:, active]
    out = corr.clip_rate_agreement(flat0, flat_t)
    assert out["n_pixels"] == 30 * 505
    assert out["total"]["within_tolerance"] is True

    with pytest.raises(ValueError, match="same pixels"):
        corr.clip_rate_agreement(images, flat_t)


def test_clip_rate_agreement_rejects_a_uint8_scale_clean_corpus():
    """Same guard as the corruption path itself: a 0-255 x_0 would be
    handed to the analytical profile, which is tabulated on [0, 1]."""
    images = (np.random.default_rng(0).random((4, 28, 28)) * 255).astype(np.uint8)
    with pytest.raises(ValueError, match="255"):
        corr.clip_rate_agreement(images, np.zeros(4 * 784))


def test_clip_rate_agreement_n_sigma_widens_and_narrows_the_verdict():
    """`n_sigma` is the knob, and it is the only thing that moves the
    verdict for a fixed corpus -- z itself does not depend on it."""
    images, x_t, _clip = _mixed_corpus(200, seed=109, index_offset=320_000)
    strict = corr.clip_rate_agreement(images, x_t, alpha_bar=0.9, n_sigma=1)
    generous = corr.clip_rate_agreement(images, x_t, alpha_bar=0.9, n_sigma=10_000)
    assert strict["above_one"]["z"] == generous["above_one"]["z"]
    assert strict["above_one"]["within_tolerance"] is False
    assert generous["above_one"]["within_tolerance"] is True
    assert generous["n_sigma"] == 10_000.0


# ---- rescaled-identity baseline ----

def test_rescaled_identity_implements_the_literal_spec_expression():
    """`clip(x_t_clip / sqrt(0.5), 0, 1)` -- the already-clipped value is
    divided, then clipped again."""
    x_t_clip = np.array([0.0, 0.3, 0.9, 1.0])
    np.testing.assert_allclose(corr.rescaled_identity(x_t_clip),
                                np.clip(x_t_clip / np.sqrt(0.5), 0, 1))


def test_rescaled_identity_final_clip_is_present():
    """Without the outer clip the baseline would emit values above 1:
    0.9 / sqrt(0.5) = 1.27."""
    assert 0.9 / np.sqrt(0.5) > 1.0
    assert corr.rescaled_identity(np.array([0.9]))[0] == 1.0


def test_rescaled_identity_clip_order_is_provably_immaterial_here():
    """Recorded so the order is neither "fixed" nor worried about later.
    Dividing by `sqrt(alpha_bar) <= 1` is monotone increasing with gain
    >= 1, so `clip(clip(x)/c, 0, 1)` and `clip(x/c, 0, 1)` are the same
    function on all of R -- the inner clip cannot change the outcome.
    The spec's order is implemented as written; it simply is not a place
    a bug can hide, unlike the outer clip above."""
    rng = np.random.default_rng(20)
    x_t = rng.normal(0.5, 2.0, 20000)
    np.testing.assert_array_equal(corr.rescaled_identity(np.clip(x_t, 0, 1)),
                                   np.clip(x_t / np.sqrt(0.5), 0, 1))


def test_rescaled_identity_stays_in_unit_interval():
    rng = np.random.default_rng(4)
    x_t_clip = rng.uniform(0, 1, 1000)
    out = corr.rescaled_identity(x_t_clip)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rescaled_identity_undoes_attenuation_where_no_clipping_occurred():
    """On an uncensored value the rescale exactly inverts the sqrt(0.5)
    signal attenuation -- which is the whole point of the baseline."""
    x0 = 0.4
    x_t_clip = np.array([np.sqrt(0.5) * x0])   # the noiseless attenuated value
    np.testing.assert_allclose(corr.rescaled_identity(x_t_clip), [x0], rtol=1e-12)


# ---- diagnostics ----

def test_corruption_diagnostics_scopes_and_clipping_direction():
    rng = np.random.default_rng(5)
    images = rng.uniform(0, 1, (30, 28, 28))
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", np.arange(30))
    active = np.sort(rng.choice(784, 505, replace=False))
    d = corr.corruption_diagnostics(images, x_t, x_t_clip, active)

    # clipping toward an in-range target cannot increase error
    assert d["mse_postclip_784"] < d["mse_preclip_784"]
    assert d["mse_postclip_505"] < d["mse_preclip_505"]
    assert d["n_active"] == 505 and d["n_images"] == 30
    assert d["per_image_mse_postclip_505"].shape == (30,)
    assert d["clip_rates_784"]["n"] == 30 * 784
    assert d["clip_rates_505"]["n"] == 30 * 505


def test_corruption_diagnostics_505_scope_actually_restricts():
    """A scope bug that silently used all 784 coordinates for the
    'active support' value would be invisible without this."""
    images = np.zeros((10, 28, 28))
    images[:, 10:18, 10:18] = 1.0            # bright block, everything else background
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", np.arange(10))
    active = np.array([r * 28 + c for r in range(10, 18) for c in range(10, 18)])
    d = corr.corruption_diagnostics(images, x_t, x_t_clip, active)
    assert d["mse_postclip_505"] != d["mse_postclip_784"]
    # bright pixels clip upward far more often than the zero background
    assert d["clip_rates_505"]["above_one"] > d["clip_rates_784"]["above_one"]
    assert d["clip_rates_505"]["below_zero"] < d["clip_rates_784"]["below_zero"]


def test_corruption_diagnostics_per_class_breakdown():
    rng = np.random.default_rng(7)
    images = rng.uniform(0, 1, (20, 28, 28))
    labels = np.array([0] * 10 + [1] * 10)
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", np.arange(20))
    d = corr.corruption_diagnostics(images, x_t, x_t_clip, np.arange(784), labels=labels)
    assert set(d["clip_rates_784_by_class"]) == {0, 1}
    assert d["clip_rates_784_by_class"][0]["n"] == 10 * 784


def test_identity_baseline_is_the_active_support_postclip_mse():
    """DESIGN.md: 'the active-support post-clip value is the identity
    baseline used by the hierarchical gate' -- so it must equal the mean
    of the per-image values the paired test consumes."""
    rng = np.random.default_rng(8)
    images = rng.uniform(0, 1, (15, 28, 28))
    x_t, x_t_clip = corr.corrupt_corpus(images, "train", np.arange(15))
    active = np.arange(0, 784, 2)
    d = corr.corruption_diagnostics(images, x_t, x_t_clip, active)
    assert d["mse_postclip_505"] == pytest.approx(
        float(np.mean(d["per_image_mse_postclip_505"])))
