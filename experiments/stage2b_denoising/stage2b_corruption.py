"""
Stage 2B's forward corruption: the locked per-image RNG, the
clipped-Gaussian forward process, the analytical censoring profile, and
the corruption diagnostics -- DESIGN.md's "Corpus construction",
"Corruption level", and "Corruption RNG, exact values" sections.

    x_t^784      = sqrt(alpha_bar) * x_0^784 + sqrt(1 - alpha_bar) * epsilon
    x_t^784_clip = clip(x_t^784, 0, 1)

with `alpha_bar = 0.5`, frozen, and

    seed_bytes = SHA256(f"{split}:{index}:{MASTER_SEED}".encode()).digest()
    seed       = int.from_bytes(seed_bytes[:8], "little")
    rng        = numpy.random.Generator(numpy.random.PCG64(seed))
    epsilon    = rng.standard_normal(784, dtype=numpy.float64)

`split` is the LITERAL string "train" or "test" -- never Python's
process-salted `hash()`, which would make the corpus irreproducible
across processes. One realization per image, reused identically across
every condition.

## Two layers, deliberately separated

`corruption_seed` / `epsilon_for` are a pure seed-and-draw layer: they
take a split label and an integer index and touch no image data at all.
`corrupt_image` / `corrupt_corpus` are the corpus layer, and they refuse
`split="test"` unless a caller passes `allow_test_split=True`.

That refusal implements DESIGN.md's hard lock -- "no Stage 2B test-side
result is accessed during stages 1-3" -- structurally rather than by
convention. Feasibility stages 1-3 cannot reach the test corpus by
accident, typo, or a loop variable that ran one iteration too far. The
single confirmatory evaluation at stage 4 is the one place that flips
the flag, and it has to do so deliberately and visibly.

## Index semantics, load-bearing

`index` is the image's index within its OWN official split, not its
position within whatever subset is being processed. A 1,000-image
feasibility subset drawn from the 60,000 official training images must
pass those images' original indices; passing `0..999` instead would give
them a different corruption realization from the one they get at stage
3, silently breaking "one realization per image, reused identically".
`corrupt_corpus` therefore takes `indices` as a required argument rather
than defaulting to `arange(n)`.

## Test-use scope

DESIGN.md, verbatim in effect: the official KMNIST test images were used
extensively by Stage 2A and are not project-unseen. What is locked is
that the prespecified Stage 2B corrupted test corpus, test features,
model predictions, and denoising scores are generated and inspected in
one final confirmatory evaluation only. Nothing in this module should be
described as producing a "held-out test set".

dtype: float64 throughout. Not numerically required for the corruption
itself -- kept because the locked RNG spec produces it and spec
reproducibility outweighs a trivial saving.
"""
import hashlib

import numpy as np
from scipy.stats import norm

# ---- Locked constants (DESIGN.md) ----
MASTER_SEED = 42
ALPHA_BAR = 0.5          # frozen; adjusting it because clip rates "look awkward" is refused
N_PIXELS = 784
VALID_SPLITS = ("train", "test")

# DESIGN.md's analytical censoring profile, transcribed literally. Columns:
# x_0, P(clip below 0), P(clip above 1), total. Rounded to 3 decimals in
# the design document; `analytical_clip_rates` reproduces them from the
# closed form, and the test asserts agreement against this table.
ANALYTICAL_CLIP_TABLE = (
    (0.00, 0.500, 0.079, 0.579),
    (0.25, 0.401, 0.122, 0.523),
    (0.50, 0.309, 0.180, 0.489),
    (0.75, 0.227, 0.253, 0.480),
    (1.00, 0.159, 0.339, 0.498),
)


def signal_coefficient(alpha_bar=ALPHA_BAR):
    return float(np.sqrt(alpha_bar))


def noise_coefficient(alpha_bar=ALPHA_BAR):
    return float(np.sqrt(1.0 - alpha_bar))


# ---- Pure seed-and-draw layer: no image data, no corpus ----

def corruption_seed(split, index, master_seed=MASTER_SEED):
    """The locked seed derivation, exactly as specified: SHA256 of the
    literal string `"{split}:{index}:{master_seed}"`, first 8 bytes read
    little-endian.

    `split` must be the literal "train" or "test". This is deliberately
    not `hash()`: CPython salts `hash()` per process, so a `hash()`-based
    corpus would silently differ between runs."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS!r}, got {split!r}")
    if int(index) != index or index < 0:
        raise ValueError(f"index must be a non-negative integer, got {index!r}")
    digest = hashlib.sha256(f"{split}:{int(index)}:{master_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def epsilon_for(split, index, n_pixels=N_PIXELS, master_seed=MASTER_SEED):
    """The one noise realization for one image, as a flat (784,) float64
    vector. Takes no image data -- only the split label and the index."""
    rng = np.random.Generator(np.random.PCG64(corruption_seed(split, index, master_seed)))
    return rng.standard_normal(n_pixels, dtype=np.float64)


# ---- Forward process ----

def forward_corrupt(x0_flat, epsilon, alpha_bar=ALPHA_BAR):
    """The forward noise formula and its clipping, given an already-drawn
    epsilon. Returns (x_t, x_t_clip), both float64, both flat."""
    x0_flat = np.asarray(x0_flat, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    if x0_flat.shape != epsilon.shape:
        raise ValueError(f"x_0 shape {x0_flat.shape} does not match epsilon "
                         f"shape {epsilon.shape}")
    x_t = signal_coefficient(alpha_bar) * x0_flat + noise_coefficient(alpha_bar) * epsilon
    return x_t, np.clip(x_t, 0.0, 1.0)


def _check_split_allowed(split, allow_test_split):
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS!r}, got {split!r}")
    if split == "test" and not allow_test_split:
        raise PermissionError(
            "refusing to build the Stage 2B corrupted TEST corpus: DESIGN.md locks "
            "'no Stage 2B test-side result is accessed during stages 1-3'. Only the "
            "single confirmatory evaluation at feasibility stage 4 may pass "
            "allow_test_split=True, and it must do so deliberately.")


def corrupt_image(x0, split, index, alpha_bar=ALPHA_BAR, allow_test_split=False):
    """Corrupts one image. `x0` must be (28, 28) or (784,), in [0, 1];
    the returned arrays match its shape. Row-major flattening, matching
    the rest of this project's 28x28 <-> 784 convention.

    The size is asserted rather than adapted to. The spec locks
    `standard_normal(784)`; silently drawing a shorter vector for, say, a
    505-restricted array would produce a valid-looking result from a
    realization that is not the locked one -- the same class of silent
    adaptation the index-semantics note above is about. Corruption
    happens on the full grid, before any restriction."""
    _check_split_allowed(split, allow_test_split)
    x0 = np.asarray(x0, dtype=np.float64)
    shape = x0.shape
    flat = x0.reshape(-1)
    if flat.size != N_PIXELS:
        raise ValueError(
            f"corruption is defined on the full {N_PIXELS}-pixel grid (DESIGN.md: "
            f"full-image noise, then encode, then restrict), got {flat.size} values "
            f"with shape {shape}")
    x_t, x_t_clip = forward_corrupt(flat, epsilon_for(split, index), alpha_bar)
    return x_t.reshape(shape), x_t_clip.reshape(shape)


def corrupt_corpus(images, split, indices, alpha_bar=ALPHA_BAR, allow_test_split=False):
    """Corrupts a corpus. `indices` is REQUIRED and must be each image's
    index within its own official split -- see the module docstring's
    "Index semantics" note. Returns (x_t, x_t_clip), both matching
    `images`' shape."""
    _check_split_allowed(split, allow_test_split)
    images = np.asarray(images, dtype=np.float64)
    indices = np.asarray(indices)
    if indices.shape != (images.shape[0],):
        raise ValueError(f"indices must have one entry per image: expected shape "
                         f"{(images.shape[0],)}, got {indices.shape}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("indices contain duplicates -- each image needs its own "
                         "corruption realization")
    x_t = np.empty_like(images)
    x_t_clip = np.empty_like(images)
    for i, idx in enumerate(indices):
        x_t[i], x_t_clip[i] = corrupt_image(images[i], split, int(idx), alpha_bar,
                                             allow_test_split=allow_test_split)
    return x_t, x_t_clip


def rescaled_identity(x_t_clip, alpha_bar=ALPHA_BAR):
    """DESIGN.md's descriptive rescaled-identity baseline:

        x_hat_0_rescale = clip( x_t_clip / sqrt(alpha_bar), 0, 1 )

    The order is the spec's: the already-clipped value is divided, then
    clipped again. That inner clip turns out to be immaterial -- dividing
    by `sqrt(alpha_bar) <= 1` is monotone with gain >= 1, so
    `clip(clip(x)/c, 0, 1)` and `clip(x/c, 0, 1)` are the same function
    on all of R. The OUTER clip is not immaterial: without it the
    baseline emits values above 1.

    Uses only the known corruption coefficient and zero learned
    parameters -- it shows whether a model does more than undo the
    deterministic signal attenuation. Reported descriptively; it is not
    part of the hierarchical identity gate."""
    return np.clip(np.asarray(x_t_clip, dtype=np.float64) / np.sqrt(alpha_bar), 0.0, 1.0)


# ---- Censoring profile: analytical and empirical ----

def analytical_clip_rates(x0, alpha_bar=ALPHA_BAR):
    """Closed-form censoring probabilities at a given clean intensity:

        P(x_t < 0) = Phi( -sqrt(alpha_bar) * x_0 / sqrt(1 - alpha_bar) )
        P(x_t > 1) = 1 - Phi( (1 - sqrt(alpha_bar) * x_0) / sqrt(1 - alpha_bar) )

    Returns (p_below, p_above, p_total), broadcasting over `x0`. These are
    what DESIGN.md's censoring table states; the empirical rates measured
    on a real corpus are reported as confirmation against them, not as
    discovery."""
    x0 = np.asarray(x0, dtype=np.float64)
    mu = signal_coefficient(alpha_bar) * x0
    sigma = noise_coefficient(alpha_bar)
    p_below = norm.cdf((0.0 - mu) / sigma)
    p_above = 1.0 - norm.cdf((1.0 - mu) / sigma)
    return p_below, p_above, p_below + p_above


def empirical_clip_rates(x_t):
    """Measured censoring rates on a PRE-clip `x_t` array: below zero and
    above one reported separately (DESIGN.md is explicit that they are not
    collapsed into one number -- background-dominated MSE is shaped by the
    clip-at-zero mechanism specifically)."""
    x_t = np.asarray(x_t, dtype=np.float64)
    below = float(np.mean(x_t < 0.0))
    above = float(np.mean(x_t > 1.0))
    return {"below_zero": below, "above_one": above, "total": below + above,
            "n": int(x_t.size)}


def _mse_per_image(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)
    b = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)
    return np.mean((a - b) ** 2, axis=1)


def corruption_diagnostics(x0, x_t, x_t_clip, active_indices, labels=None):
    """DESIGN.md's required corruption diagnostics: pre-clip `MSE(x_t, x_0)`
    and post-clip `MSE(clip(x_t), x_0)`, each on both the 505 and 784
    scopes, plus empirical clip rates (below-zero and above-one
    separately) on both scopes and, if `labels` is given, per class.

    Averaging is per-image first, then across images.

    `mse_postclip_505` is the identity baseline the hierarchical gate
    uses."""
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.shape[0]
    flat0 = x0.reshape(n, -1)
    flat_t = np.asarray(x_t, dtype=np.float64).reshape(n, -1)
    flat_c = np.asarray(x_t_clip, dtype=np.float64).reshape(n, -1)
    act = np.asarray(active_indices)

    out = {
        "mse_preclip_784": float(np.mean(_mse_per_image(flat_t, flat0))),
        "mse_postclip_784": float(np.mean(_mse_per_image(flat_c, flat0))),
        "mse_preclip_505": float(np.mean(_mse_per_image(flat_t[:, act], flat0[:, act]))),
        "mse_postclip_505": float(np.mean(_mse_per_image(flat_c[:, act], flat0[:, act]))),
        "per_image_mse_postclip_505": _mse_per_image(flat_c[:, act], flat0[:, act]),
        "clip_rates_784": empirical_clip_rates(flat_t),
        "clip_rates_505": empirical_clip_rates(flat_t[:, act]),
        "n_images": n, "n_active": int(act.size),
    }
    if labels is not None:
        labels = np.asarray(labels)
        out["clip_rates_784_by_class"] = {
            int(c): empirical_clip_rates(flat_t[labels == c]) for c in np.unique(labels)}
        out["clip_rates_505_by_class"] = {
            int(c): empirical_clip_rates(flat_t[labels == c][:, act])
            for c in np.unique(labels)}
    return out


def format_analytical_table(alpha_bar=ALPHA_BAR):
    """The censoring profile as a printable table -- the reference the
    measured rates are compared against."""
    lines = [f"analytical censoring profile (alpha_bar = {alpha_bar}, "
             f"noise std {noise_coefficient(alpha_bar):.3f})",
             "  x_0   P(clip below 0)  P(clip above 1)  total"]
    for x0 in (0.00, 0.25, 0.50, 0.75, 1.00):
        below, above, total = analytical_clip_rates(x0, alpha_bar)
        lines.append(f"  {x0:.2f}  {below:15.3f}  {above:15.3f}  {total:5.3f}")
    return "\n".join(lines)
