"""Tests for experiments/stage2b_denoising/stage2b_cnn.py -- the locked
residual CNN denoising baseline, its scope-matched masked loss, and the
locked training procedure.

Tier 1 (self-contained, always run) only: Stage 2B has no historical
cached artifact to verify against. Every test here runs on synthetic
arrays; no dataset, no corrupted corpus, no evaluation data of any kind
is touched.

Three groups carry the weight.

**Scope matching.** One masking function serves the training loss and the
checkpoint-selection criterion, so the 279 non-active-support coordinates
can never reach the selection criterion. The tests below discriminate
both failure modes: validation not masked at all, and validation masked
with a different support from training. Targets are perturbed, never
inputs -- an off-support INPUT pixel legitimately reaches on-support
outputs through the convolution, so perturbing one would prove nothing
about loss scope.

**Raw versus clipped.** DESIGN.md locks that training losses are raw and
selection criteria are clipped. The discriminating construction here is a
model with its final convolution zeroed, whose output is therefore
exactly its input: feed an out-of-range input against an in-range target
and the two losses must disagree by a large, exactly predictable amount.

**The early-stopping rule.** Tested as a pure function over synthetic
validation sequences, which is the only way to exercise strict `<` on an
exact plateau, and then tied back to a real run's history.

`stage2b_ridge` is imported first, deliberately: it enables
`jax_enable_x64` at import time, and equinox resolves `dtype=None`
against that flag. Under the real `pytest tests/test_stage2b_*.py`
invocation x64 IS on, so these tests run in the condition that would
otherwise silently change the CNN's numerics.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_ridge as _ridge  # noqa: E402,F401  -- enables x64 before the CNN loads
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import stage2b_cnn as cnn  # noqa: E402

SIDE = cnn.IMAGE_SIDE
N_PIX = cnn.N_PIXELS


def _support(n_active=cnn.N_ACTIVE, seed=0):
    """A synthetic active support of the locked size: the mask, and the
    flat indices inside and outside it."""
    rng = np.random.default_rng(seed)
    active = np.sort(rng.choice(N_PIX, size=n_active, replace=False))
    inactive = np.setdiff1d(np.arange(N_PIX), active)
    return cnn.build_active_support_mask(active, expect_n_active=n_active), active, inactive


def _mask_and_split_coords(n_active=cnn.N_ACTIVE, seed=0):
    """The same support, reduced to one representative index on each side."""
    mask, active, inactive = _support(n_active, seed)
    return mask, int(active[len(active) // 2]), int(inactive[len(inactive) // 2])


def _synthetic_corpus(n=32, n_val=16, seed=0):
    """Corrupted inputs and clean targets in [0, 1], shaped like the real
    task: full 28x28 grids, restriction left entirely to the mask."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, SIDE, SIDE))
    y = np.clip(x + rng.normal(0.0, 0.2, size=x.shape), 0.0, 1.0)
    xv = rng.uniform(0.0, 1.0, size=(n_val, SIDE, SIDE))
    yv = np.clip(xv + rng.normal(0.0, 0.2, size=xv.shape), 0.0, 1.0)
    return x, y, xv, yv


def _params(model):
    return jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))


def _zeroed_output_model(seed=0):
    """The locked model with its final convolution zeroed, so
    `f_psi(x) == 0` and `model(x) == x` exactly. Used to make loss
    behaviour predictable in closed form without changing the
    architecture."""
    model = cnn.make_model(cnn.seed_keys(seed)[0])
    model = eqx.tree_at(lambda m: m.conv3.weight, model,
                        jnp.zeros_like(model.conv3.weight))
    return eqx.tree_at(lambda m: m.conv3.bias, model, jnp.zeros_like(model.conv3.bias))


def _fast(**overrides):
    """Small, explicit training settings. Tests never inherit MAX_EPOCHS."""
    kwargs = dict(batch_size=8, max_epochs=3, patience=2, eval_batch_size=8)
    kwargs.update(overrides)
    return kwargs


# ---- locked constants ----

def test_locked_architecture_constants():
    assert cnn.CONV_CHANNELS == 32
    assert cnn.KERNEL_SIZE == 3
    assert cnn.PADDING == 1
    assert cnn.PADDING_MODE == "ZEROS"
    assert cnn.USE_BIAS is True
    assert cnn.EXPECTED_PARAM_COUNT == 9857
    assert cnn.N_ACTIVE == 505 and cnn.N_INACTIVE == 279
    assert cnn.N_ACTIVE + cnn.N_INACTIVE == cnn.N_PIXELS


def test_locked_optimizer_and_training_constants():
    assert cnn.ADAM_LEARNING_RATE == 1e-3
    assert cnn.ADAM_B1 == 0.9
    assert cnn.ADAM_B2 == 0.999
    assert cnn.ADAM_EPS == 1e-8
    assert cnn.ADAM_EPS_ROOT == 0.0
    assert cnn.BATCH_SIZE == 128
    assert cnn.MAX_EPOCHS == 100
    assert cnn.PATIENCE == 10
    assert cnn.MIN_DELTA == 0.0
    assert cnn.SEEDS == (0, 1, 2)


def test_no_second_deep_learning_framework_is_imported():
    """PyTorch and TensorFlow are DESIGN.md's documented rejections, not
    style preferences. Neither may appear in the CNN module's imports."""
    source = (_STAGE2B_DIR / "stage2b_cnn.py").read_text()
    for banned in ("import torch", "import tensorflow", "from torch", "from tensorflow"):
        assert banned not in source
    assert "torch" not in sys.modules and "tensorflow" not in sys.modules


# ---- architecture ----

def test_parameter_count_is_exactly_9857():
    model = cnn.make_model(cnn.seed_keys(0)[0])
    assert cnn.count_trainable_parameters(model) == 9857


def test_per_layer_parameter_counts_match_the_design_arithmetic():
    """`(1*32*9+32) + (32*32*9+32) + (32*1*9+1)` -- checked term by term,
    so a compensating error in two layers cannot pass on the total."""
    model = cnn.make_model(cnn.seed_keys(0)[0])
    assert model.conv1.weight.size + model.conv1.bias.size == 1 * 32 * 9 + 32 == 320
    assert model.conv2.weight.size + model.conv2.bias.size == 32 * 32 * 9 + 32 == 9248
    assert model.conv3.weight.size + model.conv3.bias.size == 32 * 1 * 9 + 1 == 289


def test_parameter_count_assert_is_live_not_decorative(monkeypatch):
    """Negative test: change the channel count and construction must halt.
    Without this the 9,857 assert could be trivially satisfied by an
    architecture nobody had checked."""
    monkeypatch.setattr(cnn, "CONV_CHANNELS", 16)
    with pytest.raises(AssertionError, match="9857"):
        cnn.make_model(cnn.seed_keys(0)[0])


def test_biases_are_enabled_on_every_convolution():
    model = cnn.make_model(cnn.seed_keys(0)[0])
    for conv in (model.conv1, model.conv2, model.conv3):
        assert conv.use_bias is True
        assert conv.bias is not None


def test_zero_padding_preserves_the_28x28_grid():
    model = cnn.make_model(cnn.seed_keys(0)[0])
    for conv in (model.conv1, model.conv2, model.conv3):
        assert conv.padding_mode == "ZEROS"
        assert tuple(np.ravel(conv.padding)) == (1, 1, 1, 1)
    out = model(jnp.zeros((1, SIDE, SIDE), dtype=cnn.CNN_DTYPE))
    assert out.shape == (1, SIDE, SIDE)


def test_residual_skip_connection_is_real():
    """Zero the final convolution and the model must reproduce its input
    EXACTLY. This is the only check that distinguishes
    `x_hat = x + f_psi(x)` from a plain feed-forward `x_hat = f_psi(x)`."""
    model = _zeroed_output_model()
    x = jnp.asarray(np.random.default_rng(1).uniform(0, 1, (1, SIDE, SIDE)),
                    dtype=cnn.CNN_DTYPE)
    np.testing.assert_array_equal(np.asarray(model(x)), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(model.residual(x)), np.zeros((1, SIDE, SIDE)))


def test_final_layer_is_linear_with_no_sigmoid():
    """A sigmoid or clipped output would confine predictions to [0, 1] and
    make the unclipped diagnostic incomparable to ridge's. Outputs must be
    able to leave the unit interval."""
    model = cnn.make_model(cnn.seed_keys(0)[0])
    x = jnp.asarray(np.random.default_rng(2).uniform(-5.0, 5.0, (1, SIDE, SIDE)),
                    dtype=cnn.CNN_DTYPE)
    out = np.asarray(model(x))
    assert out.max() > 1.0 or out.min() < 0.0

    # Stronger: scaling the final convolution's parameters must scale its
    # output by exactly the same factor. Any squashing nonlinearity after
    # conv3 -- a sigmoid, a tanh, a clip -- saturates and breaks that,
    # whatever the output range happened to be at initialization.
    scaled = eqx.tree_at(lambda m: (m.conv3.weight, m.conv3.bias), model,
                         (model.conv3.weight * 100.0, model.conv3.bias * 100.0))
    np.testing.assert_allclose(np.asarray(scaled.residual(x)),
                               np.asarray(model.residual(x)) * 100.0,
                               rtol=1e-4, atol=1e-4)
    assert np.abs(np.asarray(scaled.residual(x))).max() > 1.0


def test_parameters_are_the_explicit_dtype_even_with_x64_enabled():
    """`stage2b_ridge` turned x64 on at import. equinox resolves
    `dtype=None` to float64 in that state and float32 otherwise, so an
    implicit dtype would make the CNN's numerics depend on module import
    order (verified: equinox 0.13.8 / jax 0.11.0)."""
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64, "x64 is not on"
    implicit = eqx.nn.Conv2d(1, 4, 3, padding=1, key=jax.random.PRNGKey(0))
    assert implicit.weight.dtype == jnp.float64, "the hazard this test guards is gone"
    model = cnn.make_model(cnn.seed_keys(0)[0])
    for leaf in _params(model):
        assert leaf.dtype == cnn.CNN_DTYPE


# ---- the active-support mask ----

def test_mask_has_505_ones_and_279_zeros():
    mask, _, _ = _mask_and_split_coords()
    arr = np.asarray(mask)
    assert arr.shape == (SIDE, SIDE)
    assert arr.sum() == cnn.N_ACTIVE
    assert int((arr == 0).sum()) == cnn.N_INACTIVE


def test_mask_marks_exactly_the_given_flat_indices():
    active = np.array([0, 5, 27, 100, 783])
    arr = np.asarray(cnn.build_active_support_mask(active)).reshape(-1)
    np.testing.assert_array_equal(np.flatnonzero(arr), active)


@pytest.mark.parametrize("bad,match", [
    (np.array([], dtype=int), "empty"),
    (np.array([1, 1, 2]), "duplicates"),
    (np.array([0, 784]), "must lie in"),
    (np.array([-1, 3]), "must lie in"),
])
def test_mask_builder_rejects_malformed_supports(bad, match):
    with pytest.raises(ValueError, match=match):
        cnn.build_active_support_mask(bad)


def test_mask_builder_enforces_an_expected_support_size_when_given_one():
    with pytest.raises(ValueError, match="expected 505"):
        cnn.build_active_support_mask(np.arange(500), expect_n_active=cnn.N_ACTIVE)


# ---- SCOPE MATCHING: one masking function, both call sites ----

def test_train_cnn_takes_exactly_one_mask_argument():
    """Divergence made unrepresentable rather than merely untested: there
    is no second mask parameter for a caller to set differently."""
    for fn in (cnn.train_cnn, cnn.train_cnn_for_seed, cnn.train_best_of_seeds):
        names = list(inspect.signature(fn).parameters)
        assert [n for n in names if "mask" in n] == ["mask"]


def test_masking_primitive_is_the_only_place_the_mask_is_applied():
    """Principle 16: the second call site must call the primitive, not
    reimplement it. Both wrappers are one-liners over
    `masked_per_image_mse`, so a divergent second implementation would
    show up as an extra multiplication by the mask in the source."""
    source = (_STAGE2B_DIR / "stage2b_cnn.py").read_text()
    assert source.count("* flat_mask") == 1
    assert source.count("jnp.sum(flat_mask)") == 1


@pytest.mark.parametrize("clip", [False, True])
def test_masked_mse_ignores_off_support_coordinates_entirely(clip):
    mask, inside, outside = _mask_and_split_coords()
    rng = np.random.default_rng(3)
    pred = jnp.asarray(rng.uniform(0, 1, (4, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    target = jnp.asarray(rng.uniform(0, 1, (4, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    base = float(cnn.masked_mse(pred, target, mask, clip_predictions=clip,
                                reduce_dtype=cnn.CNN_DTYPE))

    moved_out = target.reshape(4, -1).at[:, outside].set(-99.0).reshape(4, SIDE, SIDE)
    assert float(cnn.masked_mse(pred, moved_out, mask, clip_predictions=clip,
                                reduce_dtype=cnn.CNN_DTYPE)) == base

    moved_in = target.reshape(4, -1).at[:, inside].set(-99.0).reshape(4, SIDE, SIDE)
    assert float(cnn.masked_mse(pred, moved_in, mask, clip_predictions=clip,
                                reduce_dtype=cnn.CNN_DTYPE)) != base


def test_masked_mse_denominator_is_the_support_size_not_the_grid_size():
    """A unit error on exactly one active coordinate must give `1/505`, not
    `1/784` -- the difference between masking the numerator only and
    masking both."""
    mask, inside, _ = _mask_and_split_coords()
    pred = jnp.zeros((1, SIDE, SIDE), dtype=cnn.CNN_DTYPE)
    target = jnp.zeros((1, N_PIX), dtype=cnn.CNN_DTYPE).at[:, inside].set(1.0)
    got = float(cnn.masked_mse(pred, target.reshape(1, SIDE, SIDE), mask,
                               clip_predictions=False, reduce_dtype=cnn.CNN_DTYPE))
    assert got == pytest.approx(1.0 / cnn.N_ACTIVE, rel=1e-6)
    assert got != pytest.approx(1.0 / N_PIX, rel=1e-6)


def test_2x2_both_losses_respond_to_the_same_coordinates():
    """The discrimination test the scope-matching requirement needs.

    An in-support coordinate must move BOTH the training loss and the
    validation metric; an off-support coordinate must move NEITHER. That
    pins the two paths to the same mask -- not merely to some mask each,
    which a divergent-support bug would also satisfy."""
    mask, inside, outside = _mask_and_split_coords()
    model = cnn.make_model(cnn.seed_keys(0)[0])
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.uniform(0, 1, (6, 1, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    y = jnp.asarray(rng.uniform(0, 1, (6, 1, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)

    def perturb(arr, flat_index):
        return arr.reshape(6, -1).at[:, flat_index].add(0.5).reshape(6, 1, SIDE, SIDE)

    train_base = float(cnn.training_loss(model, x, y, mask))
    val_base = float(cnn.clipped_validation_mse(model, x, y, mask))

    y_in = perturb(y, inside)
    assert float(cnn.training_loss(model, x, y_in, mask)) != train_base
    assert float(cnn.clipped_validation_mse(model, x, y_in, mask)) != val_base

    y_out = perturb(y, outside)
    assert float(cnn.training_loss(model, x, y_out, mask)) == train_base
    assert float(cnn.clipped_validation_mse(model, x, y_out, mask)) == val_base


def test_both_losses_depend_on_exactly_the_same_784_coordinates():
    """The exhaustive version of the 2x2 test above, and the one that
    actually pins scope matching.

    Probing one in-support and one off-support coordinate leaves 782
    unprobed: a validation mask differing from the training mask at any
    other single coordinate would pass. Differentiating each loss with
    respect to the whole target array tests all 784 at once -- the set of
    coordinates a loss responds to is exactly the support of that
    gradient.

    Both gradient supports must equal the mask's support, and each other.
    Verified by mutation: flipping one coordinate of the mask on the
    validation path alone fails this test."""
    mask, active, inactive = _support()
    model = cnn.make_model(cnn.seed_keys(0)[0])
    rng = np.random.default_rng(14)
    x = cnn.as_image_batch(rng.uniform(0, 1, (5, SIDE, SIDE)))
    y = cnn.as_image_batch(rng.uniform(0, 1, (5, SIDE, SIDE)))

    def support_of(loss_fn):
        grad = jax.grad(lambda t: loss_fn(model, x, t, mask))(y)
        per_coord = np.asarray(jnp.sum(jnp.abs(grad), axis=(0, 1))).reshape(-1)
        return np.flatnonzero(per_coord)

    train_support = support_of(cnn.training_loss)
    val_support = support_of(cnn.clipped_validation_mse)
    np.testing.assert_array_equal(train_support, active)
    np.testing.assert_array_equal(val_support, active)
    np.testing.assert_array_equal(train_support, val_support)
    assert np.intersect1d(val_support, inactive).size == 0


def test_off_support_validation_targets_never_reach_checkpoint_selection():
    """End to end: two runs whose validation targets differ ONLY off the
    active support must be bit-identical in every training-derived
    quantity, including the selected checkpoint's weights.

    If the validation metric were computed on the full 784-coordinate grid
    -- the concrete divergence DESIGN.md's scope-matching rule forbids --
    the validation history would differ and the selected epoch could
    differ with it.

    EVERY off-support coordinate is perturbed, not a representative one,
    so a validation mask that leaked any of the 279 would be caught."""
    mask, _, inactive = _support()
    x, y, xv, yv = _synthetic_corpus()
    yv_perturbed = yv.reshape(len(yv), -1).copy()
    yv_perturbed[:, inactive] = -7.0
    yv_perturbed = yv_perturbed.reshape(yv.shape)

    init_key, shuffle_key = cnn.seed_keys(0)
    a = cnn.train_cnn(x, y, xv, yv, mask, init_key=init_key,
                      shuffle_key=shuffle_key, **_fast())
    b = cnn.train_cnn(x, y, xv, yv_perturbed, mask, init_key=init_key,
                      shuffle_key=shuffle_key, **_fast())

    np.testing.assert_array_equal(a["clipped_val_mse_history"], b["clipped_val_mse_history"])
    assert a["best_epoch"] == b["best_epoch"]
    assert a["best_clipped_val_mse"] == b["best_clipped_val_mse"]
    for pa, pb in zip(_params(a["model"]), _params(b["model"])):
        np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))


def test_on_support_validation_targets_do_change_the_run():
    """The sensitivity control for the test above: without it, a validation
    metric that ignored its targets entirely would also pass."""
    mask, inside, _ = _mask_and_split_coords()
    x, y, xv, yv = _synthetic_corpus()
    yv_perturbed = yv.reshape(len(yv), -1).copy()
    yv_perturbed[:, inside] = 0.0
    yv_perturbed = yv_perturbed.reshape(yv.shape)

    init_key, shuffle_key = cnn.seed_keys(0)
    a = cnn.train_cnn(x, y, xv, yv, mask, init_key=init_key,
                      shuffle_key=shuffle_key, **_fast())
    b = cnn.train_cnn(x, y, xv, yv_perturbed, mask, init_key=init_key,
                      shuffle_key=shuffle_key, **_fast())
    assert not np.array_equal(a["clipped_val_mse_history"], b["clipped_val_mse_history"])


def test_off_support_fit_targets_produce_no_gradient():
    """The training side of the same property: the 279 off-support
    coordinates receive no training signal, so changing all 279 of them
    must leave the fitted weights bit-identical."""
    mask, _, inactive = _support()
    x, y, xv, yv = _synthetic_corpus()
    y_perturbed = y.reshape(len(y), -1).copy()
    y_perturbed[:, inactive] = 3.0
    y_perturbed = y_perturbed.reshape(y.shape)

    init_key, shuffle_key = cnn.seed_keys(0)
    a = cnn.train_cnn(x, y, xv, yv, mask, init_key=init_key,
                      shuffle_key=shuffle_key, **_fast())
    b = cnn.train_cnn(x, y_perturbed, xv, yv, mask, init_key=init_key,
                      shuffle_key=shuffle_key, **_fast())
    np.testing.assert_array_equal(a["raw_train_loss_history"], b["raw_train_loss_history"])
    for pa, pb in zip(_params(a["model"]), _params(b["model"])):
        np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))


# ---- RAW training loss versus CLIPPED selection criterion ----

@pytest.mark.parametrize("param_name", ["clip_predictions", "reduce_dtype"])
def test_primitive_choices_are_required_at_every_call_site(param_name):
    """No default: neither call site can inherit the wrong side of a
    distinction the two call sites genuinely differ on. `clip_predictions`
    is DESIGN.md-locked; `reduce_dtype` is not, but its failure mode is
    the same shape -- a silently inherited precision."""
    for fn in (cnn.masked_per_image_mse, cnn.masked_mse):
        param = inspect.signature(fn).parameters[param_name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty
    mask, _, _ = _mask_and_split_coords()
    z = jnp.zeros((1, SIDE, SIDE), dtype=cnn.CNN_DTYPE)
    kwargs = {"clip_predictions": False, "reduce_dtype": cnn.CNN_DTYPE}
    kwargs.pop(param_name)
    with pytest.raises(TypeError, match=param_name):
        cnn.masked_mse(z, z, mask, **kwargs)


def test_training_loss_is_raw_and_validation_metric_is_clipped():
    """The zeroed-output model makes `model(x) == x`, so with `x = 5` and
    `target = 1` the raw loss is exactly `(5-1)^2 = 16` while the clipped
    metric is exactly 0. A training loss that clipped, or a validation
    metric that did not, would fail on a value computable by hand."""
    mask, _, _ = _mask_and_split_coords()
    model = _zeroed_output_model()
    x = jnp.full((3, 1, SIDE, SIDE), 5.0, dtype=cnn.CNN_DTYPE)
    y = jnp.ones((3, 1, SIDE, SIDE), dtype=cnn.CNN_DTYPE)
    assert float(cnn.training_loss(model, x, y, mask)) == pytest.approx(16.0, rel=1e-5)
    assert float(cnn.clipped_validation_mse(model, x, y, mask)) == pytest.approx(0.0, abs=1e-7)


def test_clipping_touches_the_prediction_and_never_the_target():
    mask, _, _ = _mask_and_split_coords()
    model = _zeroed_output_model()
    x = jnp.ones((2, 1, SIDE, SIDE), dtype=cnn.CNN_DTYPE)
    y_out_of_range = jnp.full((2, 1, SIDE, SIDE), 4.0, dtype=cnn.CNN_DTYPE)
    # prediction 1 (in range, untouched by clipping), target 4 (never clipped)
    assert float(cnn.clipped_validation_mse(model, x, y_out_of_range, mask)) == pytest.approx(9.0, rel=1e-5)


def test_validation_chunking_introduces_no_batch_boundary_weighting():
    """Chunking must not change how images are averaged -- a ragged final
    chunk that got equal weight to a full one would bias the criterion.
    Agreement is to float32 vectorization noise, not bit-exact: `jax.vmap`
    vectorizes differently at different batch widths, which is why
    `EVAL_BATCH_SIZE` is fixed rather than treated as inert."""
    mask, _, _ = _mask_and_split_coords()
    model = cnn.make_model(cnn.seed_keys(0)[0])
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.uniform(0, 1, (17, 1, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    y = jnp.asarray(rng.uniform(0, 1, (17, 1, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    whole = cnn.clipped_validation_per_image_mse(model, x, y, mask, batch_size=64)
    chunked = cnn.clipped_validation_per_image_mse(model, x, y, mask, batch_size=5)
    assert whole.shape == chunked.shape == (17,)
    np.testing.assert_allclose(np.asarray(whole), np.asarray(chunked), rtol=1e-5, atol=0)
    np.testing.assert_allclose(float(jnp.mean(whole)), float(jnp.mean(chunked)),
                               rtol=1e-5, atol=0)


def test_masked_mse_is_the_mean_of_the_per_image_primitive():
    mask, _, _ = _mask_and_split_coords()
    rng = np.random.default_rng(6)
    pred = jnp.asarray(rng.uniform(0, 1, (5, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    target = jnp.asarray(rng.uniform(0, 1, (5, SIDE, SIDE)), dtype=cnn.CNN_DTYPE)
    per_image = cnn.masked_per_image_mse(pred, target, mask, clip_predictions=False,
                                         reduce_dtype=cnn.CNN_DTYPE)
    assert per_image.shape == (5,)
    np.testing.assert_allclose(
        float(cnn.masked_mse(pred, target, mask, clip_predictions=False,
                             reduce_dtype=cnn.CNN_DTYPE)),
        float(jnp.mean(per_image)), rtol=0, atol=0)


@pytest.mark.parametrize("pred_shape,target_shape,match", [
    ((4, SIDE, SIDE), (4, SIDE, SIDE - 1), "does not match"),
    ((SIDE, SIDE), (SIDE, SIDE), "each image has"),
])
def test_masked_primitive_rejects_mismatched_shapes(pred_shape, target_shape, match):
    mask, _, _ = _mask_and_split_coords()
    with pytest.raises(ValueError, match=match):
        cnn.masked_mse(jnp.zeros(pred_shape), jnp.zeros(target_shape), mask,
                       clip_predictions=False, reduce_dtype=cnn.CNN_DTYPE)


# ---- the optimizer ----

def test_optimizer_is_the_literal_adam_configuration():
    """Pinned against a literally-written reference, not against the
    module's own constants -- weight decay, gradient clipping, or a changed
    epsilon would all break the exact equality."""
    reference = optax.adam(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)
    ours = cnn.make_optimizer()
    params = {"w": jnp.asarray(np.random.default_rng(7).normal(size=(6, 6)),
                               dtype=cnn.CNN_DTYPE)}
    grads = {"w": jnp.asarray(np.random.default_rng(8).normal(size=(6, 6)) * 1e3,
                              dtype=cnn.CNN_DTYPE)}
    s_ours, s_ref = ours.init(params), reference.init(params)
    for _ in range(3):
        u_ours, s_ours = ours.update(grads, s_ours, params)
        u_ref, s_ref = reference.update(grads, s_ref, params)
        np.testing.assert_array_equal(np.asarray(u_ours["w"]), np.asarray(u_ref["w"]))


def test_optimizer_has_no_weight_decay():
    """With a zero gradient, adam leaves parameters untouched; any decoupled
    or L2 weight decay would shrink them."""
    opt = cnn.make_optimizer()
    params = {"w": jnp.ones((4, 4), dtype=cnn.CNN_DTYPE)}
    grads = {"w": jnp.zeros((4, 4), dtype=cnn.CNN_DTYPE)}
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)
    np.testing.assert_array_equal(np.asarray(updates["w"]), np.zeros((4, 4)))


# ---- early stopping, as a pure function ----

def test_min_delta_zero_means_strict_improvement_on_an_exact_plateau():
    """The property an end-to-end run cannot reliably produce: an epoch
    that exactly ties the best is NOT an improvement, so the patience
    counter advances through a flat sequence."""
    trace = cnn.early_stopping_trace([1.0] * 20, patience=3)
    assert trace["best_epoch"] == 0
    assert trace["improved"] == [True, False, False, False]
    assert trace["stopped_early"] is True
    assert trace["n_epochs_run"] == 4


def test_monotonically_improving_sequence_never_stops_early():
    trace = cnn.early_stopping_trace(list(np.linspace(1.0, 0.1, 30)), patience=3)
    assert trace["stopped_early"] is False
    assert trace["best_epoch"] == 29
    assert all(trace["improved"])


def test_worsening_run_stops_after_patience_epochs_keeping_the_first_checkpoint():
    """`best_val = inf` makes epoch 0 an improvement, so at the locked
    patience of 10 a run that only ever worsens runs 11 epochs and keeps
    epoch 0."""
    trace = cnn.early_stopping_trace(list(np.linspace(0.1, 1.0, 100)))
    assert trace["patience"] == 10 and trace["min_delta"] == 0.0
    assert trace["best_epoch"] == 0
    assert trace["stop_epoch"] == 10
    assert trace["n_epochs_run"] == 11
    assert trace["stopped_early"] is True


def test_patience_counter_resets_on_a_genuine_improvement():
    seq = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    trace = cnn.early_stopping_trace(seq, patience=3)
    assert trace["best_epoch"] == 3
    assert trace["best_val"] == 0.5
    assert trace["stop_epoch"] == 6
    assert trace["stopped_early"] is True


def test_best_epoch_tracks_the_minimum_not_the_last_epoch():
    trace = cnn.early_stopping_trace([0.9, 0.4, 0.6, 0.5], patience=10)
    assert trace["best_epoch"] == 1 and trace["best_val"] == 0.4


def test_early_stopping_update_reports_improvement_and_stop_separately():
    state = cnn.early_stopping_init()
    state, improved, stop = cnn.early_stopping_update(state, 0.5, patience=1)
    assert improved is True and stop is False and state["best_epoch"] == 0
    state, improved, stop = cnn.early_stopping_update(state, 0.5, patience=1)
    assert improved is False and stop is True and state["best_epoch"] == 0


def test_training_loop_and_the_pure_rule_agree_on_a_real_run():
    """Ties the pure function to the loop: replaying a real run's recorded
    validation history through `early_stopping_trace` must reproduce the
    run's own best epoch and length. If the loop reimplemented the rule,
    the two could drift."""
    mask, _, _ = _mask_and_split_coords()
    x, y, xv, yv = _synthetic_corpus()
    init_key, shuffle_key = cnn.seed_keys(1)
    run = cnn.train_cnn(x, y, xv, yv, mask, init_key=init_key, shuffle_key=shuffle_key,
                        **_fast(max_epochs=6, patience=2))
    trace = cnn.early_stopping_trace(run["clipped_val_mse_history"], patience=2)
    assert trace["best_epoch"] == run["best_epoch"]
    assert trace["n_epochs_run"] == run["n_epochs_run"]
    assert trace["best_val"] == run["best_clipped_val_mse"]
    assert trace["stopped_early"] == run["stopped_early"]


def _diverging_run(max_epochs, patience=10, seed=2):
    """A setup whose validation MSE rises monotonically, so the best
    checkpoint is provably NOT the last one.

    The fit targets are `1 - x` (the model is trained to invert its input)
    while the validation targets are the identity. Training therefore moves
    the model steadily away from what validation rewards, rather than
    relying on a small corpus happening to overfit."""
    mask, _, _ = _mask_and_split_coords()
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 1, (24, SIDE, SIDE))
    xv = rng.uniform(0, 1, (12, SIDE, SIDE))
    init_key, shuffle_key = cnn.seed_keys(seed)
    run = cnn.train_cnn(x, 1.0 - x, xv, xv, mask, init_key=init_key,
                        shuffle_key=shuffle_key,
                        **_fast(max_epochs=max_epochs, patience=patience))
    return run, mask, xv


def test_returned_model_is_the_best_checkpoint_not_the_last():
    """A worsening run must return its BEST epoch's weights, not its final
    ones. Verified two ways, because the scalar check alone is satisfied by
    any model whose score happens to match.

    Verified by mutation: replacing the best-checkpoint condition with an
    unconditional assignment (keep the last model) fails this test."""
    run, mask, xv = _diverging_run(max_epochs=6)
    history = run["clipped_val_mse_history"]

    # the setup must actually diverge, or the whole test is vacuous
    assert run["best_epoch"] < run["n_epochs_run"] - 1
    assert history[-1] > history[0]

    assert run["best_clipped_val_mse"] == float(np.min(history))
    assert run["best_epoch"] == int(np.argmin(history))
    recomputed = float(cnn.clipped_validation_mse(
        run["model"], cnn.as_image_batch(xv), cnn.as_image_batch(xv), mask,
        batch_size=8))
    assert recomputed == pytest.approx(run["best_clipped_val_mse"], rel=1e-6)


def test_restored_checkpoint_is_bit_identical_to_stopping_at_that_epoch():
    """The definitive form: a run truncated at `best_epoch + 1` epochs ends
    on the very weights the longer run selected. Compares parameters, not
    scores, so a different model with a coincidentally equal validation MSE
    would not pass."""
    long_run, _, _ = _diverging_run(max_epochs=6)
    truncated, _, _ = _diverging_run(max_epochs=long_run["best_epoch"] + 1)
    assert truncated["n_epochs_run"] == long_run["best_epoch"] + 1
    for pa, pb in zip(_params(long_run["model"]), _params(truncated["model"])):
        np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))


# ---- seeds ----

def test_seed_keys_are_deterministic_and_seed_dependent():
    a_init, a_shuffle = cnn.seed_keys(0)
    b_init, b_shuffle = cnn.seed_keys(0)
    c_init, c_shuffle = cnn.seed_keys(1)
    np.testing.assert_array_equal(np.asarray(a_init), np.asarray(b_init))
    np.testing.assert_array_equal(np.asarray(a_shuffle), np.asarray(b_shuffle))
    assert not np.array_equal(np.asarray(a_init), np.asarray(c_init))
    assert not np.array_equal(np.asarray(a_shuffle), np.asarray(c_shuffle))
    assert not np.array_equal(np.asarray(a_init), np.asarray(a_shuffle))


def test_epoch_permutation_is_a_permutation_that_reshuffles_each_epoch():
    key = cnn.seed_keys(0)[1]
    p0 = np.asarray(cnn.epoch_permutation(key, 50, 0))
    np.testing.assert_array_equal(np.sort(p0), np.arange(50))
    np.testing.assert_array_equal(p0, np.asarray(cnn.epoch_permutation(key, 50, 0)))
    assert not np.array_equal(p0, np.asarray(cnn.epoch_permutation(key, 50, 1)))
    assert not np.array_equal(p0, np.asarray(cnn.epoch_permutation(cnn.seed_keys(1)[1],
                                                                   50, 0)))


def test_the_seed_governs_minibatch_order_not_only_initialization():
    """The discriminating version. Same-seed-vs-different-seed is vacuous
    here, since initialization alone would explain any difference: hold the
    init key FIXED and vary only the shuffle key, and the run must still
    change."""
    mask, _, _ = _mask_and_split_coords()
    x, y, xv, yv = _synthetic_corpus()
    init_key = cnn.seed_keys(0)[0]
    a = cnn.train_cnn(x, y, xv, yv, mask, init_key=init_key,
                      shuffle_key=cnn.seed_keys(0)[1], **_fast())
    b = cnn.train_cnn(x, y, xv, yv, mask, init_key=init_key,
                      shuffle_key=cnn.seed_keys(5)[1], **_fast())
    assert not np.array_equal(a["clipped_val_mse_history"], b["clipped_val_mse_history"])


def test_the_same_seed_reproduces_a_run_exactly():
    mask, _, _ = _mask_and_split_coords()
    x, y, xv, yv = _synthetic_corpus()
    a = cnn.train_cnn_for_seed(x, y, xv, yv, mask, seed=0, **_fast())
    b = cnn.train_cnn_for_seed(x, y, xv, yv, mask, seed=0, **_fast())
    np.testing.assert_array_equal(a["clipped_val_mse_history"], b["clipped_val_mse_history"])
    np.testing.assert_array_equal(a["raw_train_loss_history"], b["raw_train_loss_history"])
    for pa, pb in zip(_params(a["model"]), _params(b["model"])):
        np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))


# ---- best-of-3 selection ----

def test_select_best_seed_takes_only_training_derived_inputs():
    """"Training-derived only -- official-test performance never inspected
    during selection" made structural: the function is given nothing but
    seeds and their clipped validation MSEs."""
    assert list(inspect.signature(cnn.select_best_seed).parameters) == [
        "seeds", "clipped_val_mses"]


def test_select_best_seed_picks_the_lowest_clipped_validation_mse():
    seed, idx = cnn.select_best_seed([0, 1, 2], [0.31, 0.29, 0.30])
    assert seed == 1 and idx == 1


def test_select_best_seed_breaks_exact_ties_toward_the_lower_seed():
    seed, _ = cnn.select_best_seed([2, 0, 1], [0.25, 0.25, 0.9])
    assert seed == 0


@pytest.mark.parametrize("seeds,mses,match", [
    ([0, 1], [0.1], "does not match"),
    ([], [], "no runs"),
    ([0, 1], [0.1, np.nan], "non-finite"),
])
def test_select_best_seed_rejects_malformed_input(seeds, mses, match):
    with pytest.raises(ValueError, match=match):
        cnn.select_best_seed(seeds, mses)


def test_train_best_of_seeds_runs_every_seed_and_selects_by_the_locked_rule():
    mask, _, _ = _mask_and_split_coords()
    x, y, xv, yv = _synthetic_corpus()
    out = cnn.train_best_of_seeds(x, y, xv, yv, mask, seeds=(0, 1), **_fast(max_epochs=2))
    assert [r["seed"] for r in out["runs"]] == [0, 1]
    expected_seed, expected_idx = cnn.select_best_seed(
        [0, 1], out["clipped_val_mse_per_seed"])
    assert out["best_seed"] == expected_seed
    assert out["best"] is out["runs"][expected_idx]
    assert out["best"]["best_clipped_val_mse"] == min(out["clipped_val_mse_per_seed"])


def test_run_record_contains_no_evaluation_derived_quantity():
    """Every recorded key must be a function of the fit or validation
    partitions only. A stray test-side number in the run record would be
    exactly the leak DESIGN.md's ladder forbids before stage 4."""
    mask, _, _ = _mask_and_split_coords()
    x, y, xv, yv = _synthetic_corpus()
    run = cnn.train_cnn_for_seed(x, y, xv, yv, mask, seed=0, **_fast(max_epochs=2))
    assert set(run) == {
        "model", "best_clipped_val_mse", "best_epoch", "clipped_val_mse_history",
        "raw_train_loss_history", "n_epochs_run", "stopped_early", "n_params",
        "batch_size", "max_epochs", "patience", "min_delta", "seed"}
    assert run["n_params"] == cnn.EXPECTED_PARAM_COUNT


# ---- input shape handling ----

def test_the_partial_last_batch_of_an_epoch_is_trained_on():
    """54,000 / 128 leaves a final batch of 104 images. Those images are
    included, not dropped -- so they must produce gradients.

    Discriminating construction: with n=10 and batch_size=8 the last batch
    holds exactly two images, identified through the same
    `epoch_permutation` the loop uses. Perturbing only those two images'
    on-support targets must change the fitted weights; if the remainder
    batch were skipped they would never be seen and the weights would be
    bit-identical."""
    mask, active, _ = _support()
    rng = np.random.default_rng(15)
    x = rng.uniform(0, 1, (10, SIDE, SIDE))
    y = rng.uniform(0, 1, (10, SIDE, SIDE))
    init_key, shuffle_key = cnn.seed_keys(0)

    order = np.asarray(cnn.epoch_permutation(shuffle_key, 10, 0))
    tail = order[8:]
    assert tail.size == 2, "this test needs a genuinely partial final batch"

    y_perturbed = y.reshape(10, -1).copy()
    y_perturbed[np.ix_(tail, active)] += 0.5
    y_perturbed = y_perturbed.reshape(y.shape)

    kwargs = dict(init_key=init_key, shuffle_key=shuffle_key,
                  **_fast(max_epochs=1, batch_size=8))
    a = cnn.train_cnn(x, y, x, y, mask, **kwargs)
    b = cnn.train_cnn(x, y_perturbed, x, y, mask, **kwargs)
    assert any(not np.array_equal(np.asarray(pa), np.asarray(pb))
               for pa, pb in zip(_params(a["model"]), _params(b["model"])))


@pytest.mark.parametrize("shape", [(10, N_PIX), (10, SIDE, SIDE), (10, 1, SIDE, SIDE)])
def test_train_cnn_accepts_the_three_equivalent_image_layouts(shape):
    mask, _, _ = _mask_and_split_coords()
    rng = np.random.default_rng(11)
    x, y = rng.uniform(0, 1, shape), rng.uniform(0, 1, shape)
    run = cnn.train_cnn(x, y, x, y, mask, init_key=cnn.seed_keys(0)[0],
                        shuffle_key=cnn.seed_keys(0)[1], **_fast(max_epochs=1))
    assert run["n_epochs_run"] == 1


@pytest.mark.parametrize("shape", [(10, N_PIX), (10, SIDE, SIDE), (10, 1, SIDE, SIDE)])
def test_as_image_batch_casts_and_reshapes(shape):
    """The cast is not optional: under x64 a plain numpy array is float64,
    and a float64 input raises inside the float32 convolution rather than
    promoting."""
    arr = np.random.default_rng(13).uniform(0, 1, shape)
    assert arr.dtype == np.float64
    out = cnn.as_image_batch(arr)
    assert out.shape == (10, 1, SIDE, SIDE)
    assert out.dtype == cnn.CNN_DTYPE
    mask, _, _ = _mask_and_split_coords()
    model = cnn.make_model(cnn.seed_keys(0)[0])
    assert np.isfinite(float(cnn.training_loss(model, out, out, mask)))


@pytest.mark.parametrize("bad,match", [
    ((4, 30), "not a square grid"),
    ((4,), "must be"),
])
def test_as_image_batch_rejects_unusable_layouts(bad, match):
    with pytest.raises(ValueError, match=match):
        cnn.as_image_batch(np.zeros(bad), "x")


def test_train_cnn_rejects_mismatched_input_and_target_shapes():
    mask, _, _ = _mask_and_split_coords()
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 1, (8, SIDE, SIDE))
    y = rng.uniform(0, 1, (7, SIDE, SIDE))
    with pytest.raises(ValueError, match="shapes differ"):
        cnn.train_cnn(x, y, x, x, mask, init_key=cnn.seed_keys(0)[0],
                      shuffle_key=cnn.seed_keys(0)[1], **_fast(max_epochs=1))
