"""
Stage 2B's encoder-on-noisy-inputs gate (DESIGN.md, feasibility stage 1).

`_local_converged_phases` was built and validated on clean, spatially-
smooth images. Whether it still converges on majority-censored noisy
inputs is a question, not an assertion -- so at feasibility stage 1, on
the same 1,000 images, each image's final-iteration maximum absolute
phase update (final-Delta) is recorded for the clean and the noisy
version separately, and:

    rho = median(Delta_noisy) / max(median(Delta_clean), 1e-15)
    PASS iff rho <= 10 OR (median_clean < ABS_CONV_EPS AND median_noisy < ABS_CONV_EPS)

The 1e-15 floor is numerical protection, not a scientific threshold. The
10x multiplier is arbitrary but pre-registered. **Automatic failures,
regardless of rho or the absolute-convergence escape**: any non-finite
encoded phase; any non-finite final-Delta. Both medians and both
95th-percentile final-Deltas are recorded regardless of outcome -- the
95th percentile gives visibility into a passing-median-but-exploding-tail
pattern and is explicitly NOT a second gate.

## The absolute-convergence escape (post-lock amendment)

At ENCODER_STEPS=150, the gate's first real run on noisy KMNIST FAILED at
rho=169.851 -- diagnosed in `diagnose_encoder_gate_failure.py` as genuine
slow convergence, not an irreducible floor: both clean and noisy
final-Delta decay geometrically until, by 1,200 steps, the median and the
p95 of each are exact float64 zero across those 1,000 images.
ENCODER_STEPS was raised to 1200 uniformly (clean and noisy identically,
every encoding site) as a result.

Note what that does and does not say. Two quantiles hitting zero is not
"every image is exactly zero" -- no per-image exact-zero count and no max
was reported at that rung, and the later 54,000-image measurement shows
the stronger reading is false at larger corpus sizes. The gate turns on
medians, so it is unaffected; the narrower statement is the true one.

That diagnostic surfaced a second, independent defect the ratio formula
has near either series' numerical floor: at steps=600, clean's median had
already hit exact 0.0 while noisy's sat at 1.776e-14 -- nine orders of
magnitude below the smallest meaningful final-Delta measured anywhere
(2.177e-07, clean at 150 steps) -- yet the gate reported FAIL at
rho=17.76, because `max(0.0, 1e-15)` silently turned a RATIO gate into an
ABSOLUTE test against the 1e-15 floor. The full rho trajectory across the
diagnostic's five step counts (14.98, 169.9, 1.915e4, 17.76, 0.0) is
non-monotone for exactly this reason: it tracks which of the two series
crossed its own float64 floor first, not whether the underlying mechanism
converged.

`ABS_CONV_EPS = 1e-12` is the fix: when BOTH medians are already below
it, the gate passes regardless of rho, because a ratio between two
quantities each indistinguishable from numerical dust measures which one
underflowed first, not the mechanism. 1e-12 is chosen with the same
margin logic as `RHO_THRESHOLD` and `MEDIAN_FLOOR` -- 5+ orders below the
smallest meaningful measured Delta (2.177e-07) and well above observed
float64 dust (1e-14 to 1e-16) -- so it cannot fire on a genuine, still-
converging signal, only on values already at the numerical floor.

## How final-Delta is measured without touching the encoder

`_local_converged_phases` returns only the converged field; it exposes no
per-iteration diagnostic, and reimplementing its update loop here is
exactly the failure mode CLAUDE.md principle 16 names. Instead the
encoder is called twice, unmodified, at `steps-1` and `steps`:

- The initial draw (`rng.normal(0, perturbation_std, target_phase.shape)`)
  depends on `seed` and the image shape, not on `steps`, and the update
  loop is deterministic -- so the `steps=149` field is a byte-identical
  prefix of the `steps=150` trajectory, not an approximation of it.
- The applied per-iteration update is `dt * dtheta` with
  `|dtheta| <= 4*k_coupling + k_bias`, so at the locked defaults
  `|dt * dtheta| <= 0.1 * 5 = 0.5 < pi`. The wrapped difference between
  the two fields therefore recovers the applied update exactly, with no
  2*pi aliasing. That bound is input-independent: it holds on
  majority-censored noisy inputs exactly as it does on clean ones, which
  is the property that matters for this gate.

**Convention for the logged numbers**: final-Delta here is the APPLIED
phase update `dt * dtheta`, i.e. the actual change in phase over the
final iteration. DESIGN.md's phrase "maximum absolute phase update" also
admits the rate `dtheta`, which is 10x larger at the locked `dt = 0.1`.
`rho` is invariant to that choice (a constant factor cancels in the
ratio, so the gate decision is identical either way), but the logged
medians and 95th percentiles are not -- they are `dt * dtheta` values.

final-Delta is taken over the encoder's own iteration domain, the full
28x28 grid, since it is a property of that iteration; the restriction to
the active support happens after encoding and does not enter it.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage2a_dynamics_classification"))

from stage2a_core import ENCODER_SEED  # noqa: E402
from bonsai.dynamics.learned_topology_construction import _local_converged_phases  # noqa: E402

ENCODER_STEPS = 1200     # raised 2026-08-06 from 150 (this module's own
                         # encoding sites only -- see module docstring;
                         # `_local_converged_phases`'s shared default and
                         # `stage2a_core.encode_and_restrict`, load-bearing
                         # for Stage 2A's own established pipeline, are
                         # deliberately untouched)
RHO_THRESHOLD = 10.0     # pre-registered, locked before any data existed
MEDIAN_FLOOR = 1e-15     # numerical protection, not a scientific threshold
ABS_CONV_EPS = 1e-12     # numerical-floor detection, not a scientific
                         # threshold -- see module docstring's "absolute-
                         # convergence escape"


def final_delta(image_01, seed=ENCODER_SEED, steps=ENCODER_STEPS):
    """Maximum absolute applied phase update over the encoder's final
    iteration, for one (28, 28) image in [0, 1].

    Measured by calling the unmodified encoder at `steps-1` and `steps`
    and wrapping the difference into (-pi, pi] -- see the module
    docstring for why that is exact rather than approximate. Non-finite
    phases propagate to a non-finite final-Delta rather than being
    silently absorbed."""
    theta_prev = _local_converged_phases(image_01, steps=steps - 1, seed=seed)
    theta_last = _local_converged_phases(image_01, steps=steps, seed=seed)
    with np.errstate(invalid="ignore"):
        wrapped = (theta_last - theta_prev + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.max(np.abs(wrapped)))


def _encode_one(args):
    """Module-level worker so `encode_with_final_delta_batch` can use a
    process Pool under either fork or spawn start methods.

    Computes `theta_last` (the converged field at the REQUESTED `steps`)
    and `theta_prev` (`steps - 1`) directly, then derives both the
    returned encoded theta and the final-Delta from them -- two encoder
    calls, not three.

    This deliberately does NOT go through `stage2a_core.encode_and_restrict`
    the way it used to. That function has no `steps` parameter of its own
    -- it is hardwired to `_local_converged_phases`'s bare default (150,
    Stage 2A's separate, unrelated convention) -- so the old version of
    this worker silently returned final-Delta measured at the CALLER's
    requested `steps` alongside a theta frozen at 150 regardless of it: an
    asymmetry invisible while `ENCODER_STEPS` also happened to be 150, and
    a real bug the moment it stopped being (2026-08-06, 150 -> 1200).
    Measured directly: at steps=1200 the two thetas differ by up to
    2.664e-07 (max wrapped abs difference, a single test image) -- small
    in absolute terms, but a downstream pipeline built from the STALE
    150-step theta while the gate's own verdict correctly reflects 1200
    would defeat the point of raising ENCODER_STEPS at all."""
    image_01, active_indices, seed, steps = args
    theta_prev = _local_converged_phases(image_01, steps=steps - 1, seed=seed)
    theta_last = _local_converged_phases(image_01, steps=steps, seed=seed)
    with np.errstate(invalid="ignore"):
        wrapped = (theta_last - theta_prev + np.pi) % (2.0 * np.pi) - np.pi
    delta = float(np.max(np.abs(wrapped)))
    theta = theta_last.flatten()[active_indices]
    return theta, delta


def encode_with_final_delta_batch(images_01, active_indices, seed=ENCODER_SEED,
                                   steps=ENCODER_STEPS, n_workers=1):
    """Encodes every image and records its final-Delta.

    The returned theta is `_local_converged_phases` run to exactly
    `steps` -- the same call `stage2a_core.encode_and_restrict` makes
    when `steps` happens to equal ITS hardwired default, and a genuinely
    different one otherwise (see `_encode_one`'s docstring for why that
    distinction is now load-bearing). Pass
    `active_indices = np.arange(784)` for the unrestricted full-grid
    field.

    Returns (thetas (n, len(active_indices)), deltas (n,)).

    Single-process by default. Two encoder passes per image (`steps - 1`
    and `steps`), cost scaling with `steps`; the caller measures and
    reports actual wall-clock rather than this docstring hand-maintaining
    a number that would go stale the next time `ENCODER_STEPS` moves.
    `n_workers > 1` uses a process Pool for callers that want it; the
    heavy per-image encode pipeline is a separate concern from this
    diagnostic and is not this module's job."""
    images_01 = np.asarray(images_01, dtype=np.float64)
    active_indices = np.asarray(active_indices)
    jobs = [(img, active_indices, seed, steps) for img in images_01]
    if n_workers and n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_encode_one, jobs)
    else:
        results = [_encode_one(job) for job in jobs]
    thetas = np.stack([r[0] for r in results])
    deltas = np.array([r[1] for r in results], dtype=np.float64)
    return thetas, deltas


def evaluate_rho_gate(delta_clean, delta_noisy, thetas_clean=None, thetas_noisy=None,
                       threshold=RHO_THRESHOLD, floor=MEDIAN_FLOOR,
                       abs_conv_eps=ABS_CONV_EPS):
    """The gate itself, as a pure function of measured final-Deltas (and,
    optionally, the encoded phases, for the non-finite-phase check).

    Separated from the encoding so it is testable on synthetic
    final-Delta sequences with known median and tail behaviour, without
    any image data.

    Returns a dict recording every quantity DESIGN.md requires logged
    regardless of outcome. `passed` is True if EITHER rho <= threshold OR
    both medians are already below `abs_conv_eps` (the absolute-
    convergence escape -- see module docstring), AND no automatic-failure
    condition fired. The escape exists because a ratio between two
    quantities that have each decayed to numerical dust measures which one
    underflowed first, not whether the mechanism converged -- see the
    module docstring's account of the S=600 in-data demonstration
    (FAIL at rho=17.76 while noisy's median sat at 1.776e-14).

    Three scoping notes worth knowing when reading the log:

    - The non-finite-PHASE check sees only the coordinates the caller
      passed in `thetas_*`, i.e. the active support. A non-finite phase
      OUTSIDE that support is still caught, but by the non-finite
      final-Delta condition (measured over the encoder's full 28x28
      iteration domain), not by the phase condition -- so the two counts
      are not interchangeable.
    - The medians and percentiles are computed over all values, so a
      single non-finite final-Delta makes them nan. That is faithful to
      the data and the gate has already failed automatically at that
      point, but it does mean the logged medians carry no information
      about the healthy images in that case.
    - The absolute-convergence escape requires BOTH medians below
      `abs_conv_eps`, not either. A lopsided case -- one series genuinely
      converged, the other still measurably moving -- does not qualify,
      and falls through to the ordinary rho test on its own merits."""
    delta_clean = np.asarray(delta_clean, dtype=np.float64)
    delta_noisy = np.asarray(delta_noisy, dtype=np.float64)
    if delta_clean.shape != delta_noisy.shape:
        raise ValueError(f"clean/noisy final-Delta shapes differ: "
                         f"{delta_clean.shape} vs {delta_noisy.shape}")

    median_clean = float(np.median(delta_clean))
    median_noisy = float(np.median(delta_noisy))
    p95_clean = float(np.percentile(delta_clean, 95))
    p95_noisy = float(np.percentile(delta_noisy, 95))

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = median_noisy / max(median_clean, floor)
    rho = float(rho)

    n_nonfinite_delta_clean = int(np.sum(~np.isfinite(delta_clean)))
    n_nonfinite_delta_noisy = int(np.sum(~np.isfinite(delta_noisy)))
    n_nonfinite_phase_clean = (0 if thetas_clean is None
                                else int(np.sum(~np.isfinite(np.asarray(thetas_clean)))))
    n_nonfinite_phase_noisy = (0 if thetas_noisy is None
                                else int(np.sum(~np.isfinite(np.asarray(thetas_noisy)))))

    reasons = []
    if n_nonfinite_phase_clean or n_nonfinite_phase_noisy:
        reasons.append(
            f"non-finite encoded phase (clean={n_nonfinite_phase_clean}, "
            f"noisy={n_nonfinite_phase_noisy})")
    if n_nonfinite_delta_clean or n_nonfinite_delta_noisy:
        reasons.append(
            f"non-finite final-Delta (clean={n_nonfinite_delta_clean}, "
            f"noisy={n_nonfinite_delta_noisy})")
    automatic_failure = bool(reasons)

    rho_ok = bool(np.isfinite(rho) and rho <= threshold)
    # Both medians already numerical dust: a ratio between them measures
    # which one underflowed first, not the mechanism. Computed regardless
    # of rho_ok, so it is always recorded -- not only when it was needed.
    absolute_convergence = bool(median_clean < abs_conv_eps and median_noisy < abs_conv_eps)
    gate_ok = rho_ok or absolute_convergence
    if not gate_ok:
        reasons.append(
            f"rho = {rho:.6g} exceeds the pre-registered threshold {threshold:g} "
            f"(median_clean={median_clean:.6g}, median_noisy={median_noisy:.6g}, "
            f"neither below the absolute-convergence floor {abs_conv_eps:.1e})")

    return {
        "rho": rho, "passed": bool(gate_ok and not automatic_failure),
        "absolute_convergence": absolute_convergence, "abs_conv_eps": float(abs_conv_eps),
        "automatic_failure": automatic_failure, "failure_reasons": reasons,
        "median_delta_clean": median_clean, "median_delta_noisy": median_noisy,
        "p95_delta_clean": p95_clean, "p95_delta_noisy": p95_noisy,
        "n_nonfinite_delta_clean": n_nonfinite_delta_clean,
        "n_nonfinite_delta_noisy": n_nonfinite_delta_noisy,
        "n_nonfinite_phase_clean": n_nonfinite_phase_clean,
        "n_nonfinite_phase_noisy": n_nonfinite_phase_noisy,
        "threshold": float(threshold), "floor": float(floor),
        "n_images": int(delta_clean.size),
    }


def run_encoder_gate(clean_images_01, noisy_images_01, active_indices,
                      seed=ENCODER_SEED, steps=ENCODER_STEPS, n_workers=1):
    """Encodes the clean and noisy versions of the same images separately,
    measures final-Delta for each, and applies the gate.

    Returns the `evaluate_rho_gate` dict with the encoded phases attached
    under `thetas_clean` / `thetas_noisy`, so a caller can inspect them
    without re-encoding. `steps` is also attached: a gate artifact carries
    no record of which step count produced it otherwise, and this module's
    own default has already moved once (150 -> 1200) -- provenance a
    caller would otherwise have to infer from a module constant that may
    have moved again by the time the artifact is read."""
    thetas_clean, delta_clean = encode_with_final_delta_batch(
        clean_images_01, active_indices, seed=seed, steps=steps, n_workers=n_workers)
    thetas_noisy, delta_noisy = encode_with_final_delta_batch(
        noisy_images_01, active_indices, seed=seed, steps=steps, n_workers=n_workers)
    result = evaluate_rho_gate(delta_clean, delta_noisy,
                                thetas_clean=thetas_clean, thetas_noisy=thetas_noisy)
    result["delta_clean"] = delta_clean
    result["delta_noisy"] = delta_noisy
    result["thetas_clean"] = thetas_clean
    result["thetas_noisy"] = thetas_noisy
    result["steps"] = steps
    result["seed"] = seed
    return result


def format_gate_log(result):
    """One-block human-readable summary of everything DESIGN.md requires
    logged whether the gate passes or fails."""
    lines = [
        f"encoder-on-noisy-inputs gate: {'PASS' if result['passed'] else 'FAIL'}",
        f"  n images                 : {result['n_images']}",
        f"  median final-Delta clean : {result['median_delta_clean']:.6e}",
        f"  median final-Delta noisy : {result['median_delta_noisy']:.6e}",
        f"  rho                      : {result['rho']:.6g} "
        f"(threshold {result['threshold']:g})",
        f"  absolute convergence     : {result['absolute_convergence']} "
        f"(both medians < {result['abs_conv_eps']:.1e})",
        f"  p95 final-Delta clean    : {result['p95_delta_clean']:.6e}  "
        f"(visibility only, not a gate)",
        f"  p95 final-Delta noisy    : {result['p95_delta_noisy']:.6e}  "
        f"(visibility only, not a gate)",
        f"  non-finite phases        : clean={result['n_nonfinite_phase_clean']}, "
        f"noisy={result['n_nonfinite_phase_noisy']}",
        f"  non-finite final-Deltas  : clean={result['n_nonfinite_delta_clean']}, "
        f"noisy={result['n_nonfinite_delta_noisy']}",
    ]
    for reason in result["failure_reasons"]:
        lines.append(f"  failure reason           : {reason}")
    return "\n".join(lines)
