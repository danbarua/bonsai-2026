"""Stage 2B's CNN denoising baseline: the locked residual architecture,
its scope-matched masked loss, and the locked training procedure --
DESIGN.md's "CNN: equinox + optax, locked" section.

    x_hat_0 = x_t_clip + f_psi(x_t_clip)
    f_psi:  Conv2d(1,32,k=3,p=1) -> ReLU -> Conv2d(32,32,k=3,p=1) -> ReLU
            -> Conv2d(32,1,k=3,p=1), linear output

9,857 trainable parameters, `(1*32*9+32) + (32*32*9+32) + (32*1*9+1)`,
asserted at construction rather than commented. Biases enabled, zero
padding, no sigmoid.

**Framework**: equinox + optax, both already project dependencies
(diffrax pulls in equinox; optax was added previously). One framework end
to end, one RNG story, zero new dependencies. PyTorch is DESIGN.md's
documented fallback-not-taken -- rejected because a second deep-learning
framework sharing one GPU with JAX starves on JAX's default memory
preallocation. Neither PyTorch nor TensorFlow appears anywhere here.

## Scope matching: one masking function, two call sites

The CNN predicts a full 28x28 image but is scored on the 505-coordinate
active support only, so 279 output coordinates receive no training
signal. DESIGN.md's requirement is that the training loss and the
validation metric are scope-matched -- if the checkpoint-selection
criterion saw those 279 untrained coordinates, it would be selecting on
noise the model was never asked to fit.

`masked_per_image_mse` is the single primitive that applies the mask.
`training_loss` and `clipped_validation_per_image_mse` both call it; the
mask is never applied anywhere else, and it is never reimplemented at the
second call site (CLAUDE.md principle 16). `train_cnn` takes exactly one
`mask` argument, used for both, so a train/validation mask divergence is
unrepresentable rather than merely untested.

## Raw training loss, clipped selection criterion

DESIGN.md, "Prediction-range handling": "Selection criteria (ridge alpha,
CNN checkpoint) use clipped validation predictions; training losses
remain raw -- the distinction is locked, not left to implementation."

The primitive therefore takes `clip_predictions` as a REQUIRED
keyword-only argument with no default: neither call site can inherit the
wrong behaviour from a default, and the distinction is visible at both.
`training_loss` passes False; `clipped_validation_per_image_mse` passes
True.

## dtype

DESIGN.md's per-stage dtype table has no CNN row, so `CNN_DTYPE` is an
implementation choice, not a locked one -- and it is made explicit
because the alternative is worse than either value. `stage2b_ridge`
enables `jax_enable_x64` at import time, and equinox's `dtype=None`
resolves to float64 when x64 is on and float32 when it is off (verified,
equinox 0.13.8 / jax 0.11.0). Left implicit, the CNN's numerics would
depend on whether an unrelated module had been imported first -- the
class of silent state dependency CLAUDE.md principle 7 is about. Every
parameter, input, target, and mask is cast to `CNN_DTYPE` explicitly, and
the cast is asserted.

## Device

Written device-agnostically: no explicit placement, no `jax.device_put`,
nothing GPU-specific. Verified on CPU only.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

# ---- Locked architecture (DESIGN.md, "CNN: equinox + optax, locked") ----
IMAGE_SIDE = 28
N_PIXELS = 784
N_ACTIVE = 505              # the fixed active support the task is defined on
N_INACTIVE = N_PIXELS - N_ACTIVE   # 279 coordinates that receive no training signal
CONV_CHANNELS = 32
KERNEL_SIZE = 3
PADDING = 1
PADDING_MODE = "ZEROS"      # DESIGN.md: "Zero padding for padding=1"
USE_BIAS = True             # "Convolutional biases enabled, consistent with that count"
EXPECTED_PARAM_COUNT = 9857  # (1*32*9+32) + (32*32*9+32) + (32*1*9+1)

# ---- Locked optimizer (DESIGN.md, "Optimizer, literal") ----
ADAM_LEARNING_RATE = 1e-3
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
ADAM_EPS_ROOT = 0.0

# ---- Locked training procedure (DESIGN.md, "Training, literal") ----
BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 0.0             # with strict `<` improvement
SEEDS = (0, 1, 2)           # each jointly governs init, minibatch order, framework RNG

# ---- Implementation choices, not locked by DESIGN.md (see module docstring) ----
CNN_DTYPE = jnp.float32
EVAL_BATCH_SIZE = 512       # validation is evaluated in chunks purely for memory

assert N_ACTIVE + N_INACTIVE == N_PIXELS


# ---- The active-support mask ----

def build_active_support_mask(active_indices, side=IMAGE_SIDE, expect_n_active=None,
                              dtype=CNN_DTYPE):
    """The 28x28 indicator of the active support, as a float mask.

    `active_indices` are positions in the row-major flattened 784-grid --
    the same index space `bonsai.dynamics.learned_topology_construction`
    produces and Stage 2A's pipeline uses throughout.

    `expect_n_active` is checked when given; production callers pass
    `N_ACTIVE` so a support of the wrong size halts here rather than
    producing a plausible-looking model scored on the wrong coordinates.
    It is not defaulted to 505, because the same function has to build the
    small synthetic masks the tests exercise the training loop on.

    Returns a `(side, side)` array of ones on the support and zeros off
    it. Zeros, not a gathered 505-vector, because the CNN is convolutional
    and must see the full grid as input -- restriction happens in the loss,
    not in the data."""
    idx = np.asarray(active_indices, dtype=np.int64).reshape(-1)
    n_pixels = side * side
    if idx.size == 0:
        raise ValueError("active support is empty")
    if np.unique(idx).size != idx.size:
        raise ValueError("active_indices contain duplicates")
    if idx.min() < 0 or idx.max() >= n_pixels:
        raise ValueError(f"active_indices must lie in 0..{n_pixels - 1}, got "
                         f"{idx.min()}..{idx.max()}")
    if expect_n_active is not None and idx.size != expect_n_active:
        raise ValueError(f"expected {expect_n_active} active coordinates, got {idx.size}")
    flat = np.zeros(n_pixels, dtype=np.float64)
    flat[idx] = 1.0
    return jnp.asarray(flat.reshape(side, side), dtype=dtype)


# ---- The masking primitive: ONE function, both call sites ----

def masked_per_image_mse(pred, target, mask, *, clip_predictions):
    """Per-image MSE over the masked coordinates only -- the single place
    the active-support mask is ever applied.

    `pred` and `target` are `(n, ...)` with matching shapes whose
    per-image size equals `mask.size`; both `(n, 28, 28)` and the model's
    own `(n, 1, 28, 28)` channel-first output are accepted. `mask` is any
    shape with that many entries; it is flattened.

    `clip_predictions` is keyword-only and REQUIRED. DESIGN.md locks that
    training losses are raw and selection criteria are clipped, so the
    choice is stated at every call site rather than inherited from a
    default. Only the PREDICTION is clipped; the target is untouched.

    The denominator is `sum(mask)`, not the coordinate count, so the 279
    off-support coordinates contribute neither error nor weight. Averaging
    is per-image first (this function), then across images -- DESIGN.md's
    stated convention."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} does not match target shape "
                         f"{target.shape}")
    if pred.ndim < 2:
        raise ValueError(f"pred must be batched, (n, ...); got shape {pred.shape}")
    n = pred.shape[0]
    flat_pred = jnp.reshape(pred, (n, -1))
    flat_target = jnp.reshape(target, (n, -1))
    flat_mask = jnp.reshape(mask, (-1,))
    if flat_mask.shape[0] != flat_pred.shape[1]:
        raise ValueError(f"mask has {flat_mask.shape[0]} entries but each image has "
                         f"{flat_pred.shape[1]}")
    if clip_predictions:
        flat_pred = jnp.clip(flat_pred, 0.0, 1.0)
    sq = (flat_pred - flat_target) ** 2
    return jnp.sum(sq * flat_mask, axis=1) / jnp.sum(flat_mask)


def masked_mse(pred, target, mask, *, clip_predictions):
    """`masked_per_image_mse` averaged across images -- DESIGN.md's "mean
    over batch images and their 505 active coordinates"."""
    return jnp.mean(masked_per_image_mse(pred, target, mask,
                                         clip_predictions=clip_predictions))


# ---- The locked architecture ----

class ResidualDenoiserCNN(eqx.Module):
    """`x_hat_0 = x_t_clip + f_psi(x_t_clip)`.

    Operates on one image at a time, channel-first `(1, 28, 28)`, the
    equinox convention; batches go through `jax.vmap`. The final layer is
    linear -- no sigmoid -- so the unclipped diagnostic is categorically
    comparable to ridge's, and the shared clipping rule applies at
    evaluation only."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(self, key, dtype=CNN_DTYPE):
        k1, k2, k3 = jr.split(key, 3)
        common = dict(kernel_size=KERNEL_SIZE, padding=PADDING, use_bias=USE_BIAS,
                      padding_mode=PADDING_MODE, dtype=dtype)
        self.conv1 = eqx.nn.Conv2d(1, CONV_CHANNELS, key=k1, **common)
        self.conv2 = eqx.nn.Conv2d(CONV_CHANNELS, CONV_CHANNELS, key=k2, **common)
        self.conv3 = eqx.nn.Conv2d(CONV_CHANNELS, 1, key=k3, **common)

    def residual(self, x):
        """`f_psi(x)` alone -- the correction the skip connection adds."""
        h = jax.nn.relu(self.conv1(x))
        h = jax.nn.relu(self.conv2(h))
        return self.conv3(h)

    def __call__(self, x):
        return x + self.residual(x)


def count_trainable_parameters(model):
    """Total size of every inexact-array leaf -- equinox's own definition
    of what an optimizer updates."""
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(int(leaf.size) for leaf in leaves))


def make_model(key, dtype=CNN_DTYPE):
    """The locked model, with its parameter count and dtype asserted.

    The 9,857 count is a hard check, not a comment: it is the one
    arithmetic statement DESIGN.md makes about the architecture, so a
    changed channel count, a dropped bias, or a different kernel size
    halts here instead of producing a differently-sized model that trains
    perfectly well."""
    model = ResidualDenoiserCNN(key, dtype=dtype)
    n_params = count_trainable_parameters(model)
    assert n_params == EXPECTED_PARAM_COUNT, (
        f"architecture has {n_params} trainable parameters, DESIGN.md locks "
        f"{EXPECTED_PARAM_COUNT} = (1*32*9+32) + (32*32*9+32) + (32*1*9+1)")
    for leaf in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)):
        assert leaf.dtype == dtype, (
            f"parameter dtype {leaf.dtype} is not the explicit CNN_DTYPE {dtype} -- "
            f"equinox resolves dtype=None against jax_enable_x64, so an implicit dtype "
            f"would depend on module import order")
    return model


def make_optimizer():
    """DESIGN.md's literal optimizer, every argument explicit.

    Returned as a bare `optax.adam`, not wrapped in `optax.chain`, so "no
    weight decay, no gradient clipping" is structural rather than a
    promise: there is no transformation slot for one to be added to."""
    return optax.adam(learning_rate=ADAM_LEARNING_RATE, b1=ADAM_B1, b2=ADAM_B2,
                      eps=ADAM_EPS, eps_root=ADAM_EPS_ROOT)


# ---- Losses: the two call sites of the one masking primitive ----

def training_loss(model, x, y, mask):
    """RAW (unclipped) masked active-support MSE -- what gradients are
    taken of. `clip_predictions=False`, per DESIGN.md's locked
    training/selection distinction."""
    pred = jax.vmap(model)(x)
    return masked_mse(pred, y, mask, clip_predictions=False)


def clipped_validation_per_image_mse(model, x, y, mask, batch_size=EVAL_BATCH_SIZE):
    """CLIPPED masked active-support MSE, per image -- the checkpoint-
    selection criterion. `clip_predictions=True`, through the same
    primitive `training_loss` uses.

    Chunked purely for memory: the 32-channel intermediates on a
    6,000-image validation partition are the only large array in this
    module. Per-image values are concatenated and the mean taken once, so
    no batch-boundary weighting enters the average.

    `batch_size` is not perfectly inert, and the claim is not made:
    `jax.vmap` vectorizes differently at different batch widths, which
    moves float32 results by ~1e-7 relative (measured on this
    architecture). Runs are reproducible because `EVAL_BATCH_SIZE` is
    fixed, not because the value is irrelevant."""
    n = x.shape[0]
    parts = []
    for start in range(0, n, int(batch_size)):
        xb, yb = x[start:start + int(batch_size)], y[start:start + int(batch_size)]
        pred = jax.vmap(model)(xb)
        parts.append(masked_per_image_mse(pred, yb, mask, clip_predictions=True))
    return jnp.concatenate(parts) if len(parts) > 1 else parts[0]


def clipped_validation_mse(model, x, y, mask, batch_size=EVAL_BATCH_SIZE):
    """The scalar checkpoint-selection criterion."""
    return jnp.mean(clipped_validation_per_image_mse(model, x, y, mask, batch_size))


# ---- Early stopping: the locked rule as a pure function ----

def early_stopping_init():
    """`best_val = inf`, so the first epoch always improves and a run that
    never improves again stops after `patience` further epochs -- at the
    locked patience of 10, a monotonically worsening run stops after 11
    epochs, with epoch 0 as the best checkpoint."""
    return {"best_val": float("inf"), "best_epoch": -1,
            "epochs_since_improvement": 0, "n_epochs_seen": 0}


def early_stopping_update(state, val_mse, patience=PATIENCE, min_delta=MIN_DELTA):
    """One epoch of DESIGN.md's locked rule: "early-stopping patience 10 on
    clipped active-support validation MSE, `min_delta=0.0` with strict `<`
    improvement, best checkpoint restored".

    Improvement is `val_mse < best_val - min_delta`. At the locked
    `min_delta = 0.0` that is exactly strict `<`: an epoch that ties the
    best is NOT an improvement, and the patience counter advances.

    The epoch index comes from the state's own counter rather than a
    caller-supplied argument, so a caller cannot pass one that disagrees
    with the number of updates actually applied.

    Returns `(new_state, improved, should_stop)`. `should_stop` is True
    once `patience` consecutive epochs have failed to improve. The
    training loop calls this function -- it does not reimplement the rule
    -- so the synthetic-sequence tests exercise the production rule."""
    val = float(val_mse)
    epoch_index = int(state["n_epochs_seen"])
    improved = val < state["best_val"] - float(min_delta)
    new_state = {
        "best_val": val if improved else state["best_val"],
        "best_epoch": epoch_index if improved else state["best_epoch"],
        "epochs_since_improvement": 0 if improved else state["epochs_since_improvement"] + 1,
        "n_epochs_seen": epoch_index + 1,
    }
    return new_state, improved, new_state["epochs_since_improvement"] >= int(patience)


def early_stopping_trace(val_mse_sequence, patience=PATIENCE, min_delta=MIN_DELTA):
    """`early_stopping_update` folded over a whole validation-MSE sequence.

    Analysis and test helper: it applies the same function the training
    loop applies, so a sequence tested here and a run that produced that
    sequence agree by construction. Returns the best epoch (0-based), the
    epoch training would have stopped after, whether it stopped early, and
    the per-epoch improvement flags."""
    state = early_stopping_init()
    improved_flags, stop_epoch, stopped_early = [], None, False
    for epoch, val in enumerate(val_mse_sequence):
        state, improved, should_stop = early_stopping_update(
            state, val, patience=patience, min_delta=min_delta)
        improved_flags.append(bool(improved))
        stop_epoch = epoch
        if should_stop:
            stopped_early = True
            break
    return {"best_epoch": state["best_epoch"], "best_val": state["best_val"],
            "stop_epoch": stop_epoch, "stopped_early": stopped_early,
            "improved": improved_flags, "n_epochs_run": len(improved_flags),
            "patience": int(patience), "min_delta": float(min_delta)}


# ---- Seeds and minibatch order ----

def seed_keys(seed):
    """One integer seed, split into the two keys it jointly governs:
    `(init_key, shuffle_key)`.

    DESIGN.md: "Three seeds (0,1,2), each jointly governing
    initialization, minibatch order, and framework randomness." There is
    no other randomness in this model -- no dropout, no batch norm, no
    stochastic augmentation -- so those two keys are the whole RNG story,
    and both descend from the one seed."""
    init_key, shuffle_key = jr.split(jr.PRNGKey(int(seed)), 2)
    return init_key, shuffle_key


def epoch_permutation(shuffle_key, n, epoch):
    """The minibatch order for one epoch: a permutation of `0..n-1` from
    `shuffle_key` folded with the epoch index, so each epoch reshuffles
    and the whole sequence is a deterministic function of the seed."""
    return jr.permutation(jr.fold_in(shuffle_key, int(epoch)), int(n))


# ---- Training ----

@eqx.filter_jit
def _train_step(model, opt_state, x_batch, y_batch, mask, optimizer):
    loss, grads = eqx.filter_value_and_grad(training_loss)(model, x_batch, y_batch, mask)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, updates), opt_state, loss


def as_image_batch(a, name="array"):
    """Cast to `CNN_DTYPE` and give the array the model's channel-first
    `(n, 1, side, side)` shape, accepting `(n, side, side)` or
    `(n, side*side)` as well.

    Public because the cast is not optional: a float64 array reaching the
    float32 model raises inside `lax.conv_general_dilated` rather than
    promoting, and under `jax_enable_x64` -- which `stage2b_ridge` turns
    on at import -- plain numpy arrays convert to float64 by default. Any
    caller reaching `training_loss` or `clipped_validation_mse` directly
    should pass its arrays through here first, the same way `train_cnn`
    does, rather than writing its own cast."""
    arr = jnp.asarray(np.asarray(a), dtype=CNN_DTYPE)
    if arr.ndim == 2:
        side = int(round(float(np.sqrt(arr.shape[1]))))
        if side * side != arr.shape[1]:
            raise ValueError(f"{name} has {arr.shape[1]} values per image, not a square grid")
        arr = arr.reshape(arr.shape[0], 1, side, side)
    elif arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim != 4:
        raise ValueError(f"{name} must be (n, p), (n, side, side) or (n, 1, side, side); "
                         f"got shape {arr.shape}")
    return arr


def train_cnn(x_fit, y_fit, x_val, y_val, mask, *, init_key, shuffle_key,
              batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
              min_delta=MIN_DELTA, eval_batch_size=EVAL_BATCH_SIZE):
    """Train one model to DESIGN.md's locked procedure, returning the
    best-checkpoint model and the full training-derived history.

    `x_fit` is the corrupted input `x_t_clip`; `y_fit` is the clean target
    `x_0`. Both are full 28x28 grids -- the restriction to the active
    support is `mask`'s job, applied identically in the training loss and
    the validation metric.

    ONE `mask` argument serves both. There is deliberately no separate
    validation-mask parameter: a train/validation scope divergence has no
    way to be expressed.

    `init_key` and `shuffle_key` are taken separately so a caller can hold
    one fixed while varying the other. The locked entry point is
    `train_cnn_for_seed`, which derives both from one seed via
    `seed_keys`; this signature is the lower level it is built on.

    The last batch of an epoch is partial when `n_fit` is not a multiple
    of `batch_size` and is included rather than dropped. Because the loss
    is a mean, images in a short batch carry proportionally more weight in
    that one update.

    Returns a dict of training-derived quantities only -- nothing in it is
    a function of any evaluation corpus."""
    x_fit, y_fit = as_image_batch(x_fit, "x_fit"), as_image_batch(y_fit, "y_fit")
    x_val, y_val = as_image_batch(x_val, "x_val"), as_image_batch(y_val, "y_val")
    mask = jnp.asarray(np.asarray(mask), dtype=CNN_DTYPE)
    if x_fit.shape != y_fit.shape:
        raise ValueError(f"x_fit {x_fit.shape} and y_fit {y_fit.shape} shapes differ")
    if x_val.shape != y_val.shape:
        raise ValueError(f"x_val {x_val.shape} and y_val {y_val.shape} shapes differ")

    model = make_model(init_key)
    optimizer = make_optimizer()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    n_fit = int(x_fit.shape[0])
    state = early_stopping_init()
    best_model = model
    val_history, train_history = [], []
    stopped_early = False

    for epoch in range(int(max_epochs)):
        order = epoch_permutation(shuffle_key, n_fit, epoch)
        epoch_loss, seen = 0.0, 0
        for start in range(0, n_fit, int(batch_size)):
            batch = order[start:start + int(batch_size)]
            xb, yb = x_fit[batch], y_fit[batch]
            model, opt_state, loss = _train_step(model, opt_state, xb, yb, mask, optimizer)
            epoch_loss += float(loss) * int(batch.shape[0])
            seen += int(batch.shape[0])
        train_history.append(epoch_loss / max(seen, 1))

        val = float(clipped_validation_mse(model, x_val, y_val, mask, eval_batch_size))
        val_history.append(val)
        state, improved, should_stop = early_stopping_update(
            state, val, patience=patience, min_delta=min_delta)
        if improved:
            best_model = model          # best checkpoint restored: this is the returned one
        if should_stop:
            stopped_early = True
            break

    return {
        "model": best_model,
        "best_clipped_val_mse": state["best_val"],
        "best_epoch": state["best_epoch"],
        "clipped_val_mse_history": np.asarray(val_history, dtype=np.float64),
        "raw_train_loss_history": np.asarray(train_history, dtype=np.float64),
        "n_epochs_run": len(val_history),
        "stopped_early": stopped_early,
        "n_params": count_trainable_parameters(best_model),
        "batch_size": int(batch_size), "max_epochs": int(max_epochs),
        "patience": int(patience), "min_delta": float(min_delta),
    }


def train_cnn_for_seed(x_fit, y_fit, x_val, y_val, mask, *, seed, **kwargs):
    """`train_cnn` with the locked joint seed derivation: one seed governs
    initialization and minibatch order together, via `seed_keys`."""
    init_key, shuffle_key = seed_keys(seed)
    result = train_cnn(x_fit, y_fit, x_val, y_val, mask,
                       init_key=init_key, shuffle_key=shuffle_key, **kwargs)
    result["seed"] = int(seed)
    return result


def select_best_seed(seeds, clipped_val_mses):
    """DESIGN.md's best-of-3 rule: the seed with the lowest CLIPPED
    validation MSE wins.

    Deliberately takes two plain sequences rather than the run dicts.
    "Training-derived only -- official-test performance never inspected
    during selection" is then structural: this function is not given
    anything else to select on.

    Ties (exact equality) go to the lower seed, so the choice is
    deterministic. DESIGN.md does not lock a tie-break; exact ties in a
    float64 validation MSE are not expected.

    Returns `(seed, index)`, mirroring `stage2b_ridge.select_alpha`."""
    seeds_arr = np.asarray(seeds)
    mse = np.asarray(clipped_val_mses, dtype=np.float64)
    if seeds_arr.shape != mse.shape:
        raise ValueError(f"seeds shape {seeds_arr.shape} does not match validation-MSE "
                         f"shape {mse.shape}")
    if mse.size == 0:
        raise ValueError("no runs to select from")
    if not np.all(np.isfinite(mse)):
        raise ValueError("non-finite clipped validation MSE -- seed selection is undefined")
    best = float(mse.min())
    tied = np.where(mse == best)[0]
    idx = int(tied[int(np.argmin(seeds_arr[tied]))])
    return seeds_arr[idx].item(), idx


def train_best_of_seeds(x_fit, y_fit, x_val, y_val, mask, *, seeds=SEEDS, **kwargs):
    """The locked three-seed protocol: train once per seed, select by
    clipped validation MSE.

    Returns a dict with the selected run under `best`, every run under
    `runs`, and the selection inputs recorded alongside."""
    runs = [train_cnn_for_seed(x_fit, y_fit, x_val, y_val, mask, seed=s, **kwargs)
            for s in seeds]
    best_seed, best_index = select_best_seed(
        [r["seed"] for r in runs], [r["best_clipped_val_mse"] for r in runs])
    return {
        "best": runs[best_index], "best_seed": best_seed, "best_index": best_index,
        "runs": runs,
        "seeds": [int(s) for s in seeds],
        "clipped_val_mse_per_seed": [r["best_clipped_val_mse"] for r in runs],
    }
