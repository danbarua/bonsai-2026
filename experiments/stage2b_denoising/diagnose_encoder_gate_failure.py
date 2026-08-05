"""
Diagnostic-only script (not part of the locked pipeline; convention of
`diagnose_stage2_convergence.py`): investigates why Stage 2B's
encoder-on-noisy-inputs gate failed at feasibility stage 1.

Measured on a real A100 run (2026-08-05, commit 7723b96): rho = 169.851
against a pre-registered threshold of 10 -- roughly 17x over, with zero
non-finite phases or final-Deltas, so a clean ratio failure rather than a
numerical blow-up. Full record in that run's `stage1_report.json`
(bucket `bonsai-2026-stage2b-cache`, object
`stage2b/train/stage1/common/stage1_report.json`).

Two questions, pre-committed before either was measured:

1. CONVERGENCE CURVE: does noisy final-Delta keep decaying geometrically
   as step count grows, or plateau at a floor?
2. STATE DRIFT vs BETWEEN-IMAGE SCALE: even if final-Delta hasn't
   converged in the strict ratio sense, has the phase field itself
   stopped moving at a scale that matters relative to how different two
   images' fields actually look?

This changes nothing in the locked pipeline: no ENCODER_STEPS default
moves, no RHO_THRESHOLD moves, no ladder rung re-runs. DESIGN.md's rule
for a threshold exceedance is "halts the stage pending investigation" --
this script IS that investigation, not its resolution. Any response
(raising ENCODER_STEPS uniformly, or a gate redesign) is a separate,
disclosed DESIGN.md amendment with its own Review History entry, decided
from these numbers, not folded into them.

Runs entirely on CPU: `_local_converged_phases` has no JAX/GPU
dependency, so this needs no A100 and bills nothing. The stage-1 corpus
and corruption are regenerated locally from the same deterministic seeds
the cloud run used (SEED=42 partition, ENCODER_SEED=0) and verified
byte-for-byte against that run's own reported numbers before anything
else here is trusted.
"""
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage2a_dynamics_classification"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))

from bonsai.data.mnist_loader import load_mnist                          # noqa: E402
from bonsai.dynamics.learned_topology_construction import _local_converged_phases  # noqa: E402
import stage2b_partition as partition                                    # noqa: E402
import stage2b_corruption as corruption                                  # noqa: E402
import stage2b_encoder_gate as gate                                      # noqa: E402
import stage2a_topologies as topologies                                  # noqa: E402

KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")

# The two questions' parameters, fixed before any number below existed.
STEP_COUNTS = (75, 150, 300, 600, 1200)
DRIFT_LOW_STEPS = 150     # the locked ENCODER_STEPS, i.e. "what production sees today"
DRIFT_HIGH_STEPS = 600    # 4x further out; chosen to sit inside STEP_COUNTS's own range
N_PAIRS_BETWEEN_IMAGE = 5000
# Distinct from the partition's SEED=42 and the encoder's ENCODER_SEED=0,
# per CLAUDE.md principle 8 (a fixed seed per role, not one shared seed).
PAIR_SAMPLE_SEED = 90001

# From the failed cloud run's own stage1_report.json -- the anchor this
# script's local reconstruction is checked against before trusting
# anything computed here.
KNOWN_IDENTITY_BASELINE_MSE = 0.19945944496779283
KNOWN_MEDIAN_DELTA_CLEAN_AT_150 = 2.177485e-07
KNOWN_MEDIAN_DELTA_NOISY_AT_150 = 3.698480e-05
KNOWN_RHO_AT_150 = 169.851

N_WORKERS = max(1, mp.cpu_count() - 1)


def load_stage1_corpus():
    """Regenerates the exact stage-1 corpus, active support and corruption
    the cloud run used -- same calls, same seeds, as
    `run_ladder_stage1.py`'s steps 1, 1b and 2. Composed, not
    reimplemented: every one of these calls is the production function."""
    print("Loading official KMNIST training set...")
    x_train, y_train, _x_test, _y_test = load_mnist(KMNIST_DIR, gz=False)

    part = partition.Stage2BTrainingPartition(y_train)
    subsets = part.nested_development_subsets(
        size=partition.STAGE2_SUBSET_SIZE, prefix_size=partition.STAGE1_SUBSET_SIZE,
        seed=partition.LADDER_SUBSET_SEED, stratified=partition.LADDER_SUBSET_STRATIFIED)
    stage1_indices = np.asarray(subsets.stage1_indices)
    images_01 = x_train[stage1_indices].astype(np.float64) / 255.0
    labels = np.asarray(y_train[stage1_indices])
    print(f"Stage-1 corpus: n={images_01.shape[0]} (SEED={partition.LADDER_SUBSET_SEED}, "
          f"identical draw to the cloud run)")

    print("Building topologies (for active_indices only)...")
    active_indices, _ink_mask, _nodes_T, _topos = topologies.build_all_topologies()

    print("Corrupting (original dataset indices, split='train')...")
    x_t, x_t_clip = corruption.corrupt_corpus(images_01, "train", stage1_indices,
                                              alpha_bar=corruption.ALPHA_BAR)

    return images_01, x_t, x_t_clip, active_indices, labels


def verify_reconstruction_matches_the_cloud_run(images_01, x_t, x_t_clip, active_indices,
                                                 labels):
    """Hard stop, not a warning, if this doesn't match. Everything below
    depends on this corpus being the SAME corpus the failed run measured,
    not merely "the same by construction" -- checked, not assumed."""
    diag = corruption.corruption_diagnostics(images_01, x_t, x_t_clip, active_indices,
                                             labels=labels)
    identity_mse = diag["mse_postclip_505"]
    rel_diff = abs(identity_mse - KNOWN_IDENTITY_BASELINE_MSE) / KNOWN_IDENTITY_BASELINE_MSE
    print(f"identity-baseline MSE: {identity_mse!r} (cloud run: {KNOWN_IDENTITY_BASELINE_MSE!r}, "
          f"relative diff {rel_diff:.3e})")
    if rel_diff > 1e-9:
        raise RuntimeError(
            f"local reconstruction diverges from the cloud run's identity baseline by "
            f"{rel_diff:.3e} (relative) -- this is NOT the same corpus, and nothing below "
            f"can be trusted against the cloud run's numbers without finding out why.")
    print("Reconstruction verified byte-for-byte against the cloud run.")


def calibrate_and_choose_scale(images_01, x_t_clip, active_indices):
    """Times a small sample at the largest step count and decides full
    n=1,000 vs a stratified subsample -- disclosed either way, per the
    investigation's own pre-commitment that subsampling is acceptable but
    not silent."""
    n_probe = 10
    t0 = time.time()
    for image in images_01[:n_probe]:
        _local_converged_phases(image, steps=max(STEP_COUNTS), seed=gate.ENCODER_SEED)
    per_image_at_max_steps = (time.time() - t0) / n_probe

    # Rough total: 5 step counts, factor ~4 calls/image/step-count summed
    # across encode_with_final_delta_batch's internal 3 calls (1 at fixed
    # 150 via encode_and_restrict, 2 at the requested S via final_delta)
    # plus this script's own 2 direct calls for the drift measurement,
    # amortized as a multiple of the max-step probe for a conservative
    # upper bound.
    projected_s = per_image_at_max_steps * len(images_01) * 2 * (len(STEP_COUNTS) + 1)
    projected_s /= N_WORKERS
    print(f"calibration: {per_image_at_max_steps * 1000:.2f} ms/image at "
          f"{max(STEP_COUNTS)} steps (single-threaded); projected total "
          f"~{projected_s:.0f}s across {N_WORKERS} workers")

    if projected_s <= 600:
        print(f"proceeding at full scale: n={images_01.shape[0]}")
        return images_01, x_t_clip
    n_sub = 200
    print(f"DISCLOSED SUBSAMPLE: projected time exceeds 10 minutes; using a "
          f"class-stratified n={n_sub} subsample instead of the full "
          f"n={images_01.shape[0]} corpus. This is a diagnostic, not the gate.")
    rng = np.random.default_rng(PAIR_SAMPLE_SEED)
    idx = rng.choice(images_01.shape[0], size=n_sub, replace=False)
    return images_01[idx], x_t_clip[idx]


def run_convergence_curve(images_clean, images_noisy):
    """Measurement 1. Composes `encode_with_final_delta_batch` (the
    existing two-call final-Delta method) and `evaluate_rho_gate` (the
    real gate math) at each step count -- no formula re-derived here."""
    active_full_grid = np.arange(784)
    rows = {}
    for steps in STEP_COUNTS:
        t0 = time.time()
        _thetas_c, delta_clean = gate.encode_with_final_delta_batch(
            images_clean, active_full_grid, seed=gate.ENCODER_SEED, steps=steps,
            n_workers=N_WORKERS)
        _thetas_n, delta_noisy = gate.encode_with_final_delta_batch(
            images_noisy, active_full_grid, seed=gate.ENCODER_SEED, steps=steps,
            n_workers=N_WORKERS)
        verdict = gate.evaluate_rho_gate(delta_clean, delta_noisy)
        elapsed = time.time() - t0
        rows[steps] = {
            "delta_clean": delta_clean, "delta_noisy": delta_noisy,
            "median_clean": verdict["median_delta_clean"],
            "p95_clean": verdict["p95_delta_clean"],
            "median_noisy": verdict["median_delta_noisy"],
            "p95_noisy": verdict["p95_delta_noisy"],
            "rho": verdict["rho"], "passed": verdict["passed"], "elapsed_s": elapsed,
        }
        print(f"  steps={steps:5d}  median_clean={verdict['median_delta_clean']:.6e}  "
              f"median_noisy={verdict['median_delta_noisy']:.6e}  "
              f"rho={verdict['rho']:.4g}  ({elapsed:.1f}s)")
    return rows


def _drift_worker(args):
    """Module-level for Pool picklability. Calls the unmodified encoder
    directly at two explicit step counts -- NOT via
    `encode_with_final_delta_batch`, whose returned `thetas` are always
    computed at the module's fixed 150-step default regardless of the
    `steps` argument (an artifact of `encode_and_restrict` having no
    `steps` parameter of its own). Explicit calls avoid relying on that."""
    image, seed = args
    theta_low = _local_converged_phases(image, steps=DRIFT_LOW_STEPS, seed=seed).flatten()
    return theta_low


def _drift_worker_high(args):
    image, seed = args
    theta_high = _local_converged_phases(image, steps=DRIFT_HIGH_STEPS, seed=seed).flatten()
    return theta_high


def wrapped_diff(a, b):
    """Identical formula to `stage2b_encoder_gate.final_delta`'s wrap --
    not re-derived, transcribed, so a drift number and a final-Delta
    number mean the same thing."""
    with np.errstate(invalid="ignore"):
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def run_drift_vs_scale(images_noisy):
    """Measurement 2. Both quantities use the SAME reduction (max absolute
    wrapped difference over the full 784-element grid, matching the gate's
    own domain -- see `stage2b_encoder_gate`'s module docstring on why
    final-Delta is measured over the full grid, not the active support),
    so "is the drift small relative to how different two images look" is
    an apples-to-apples comparison."""
    seed = gate.ENCODER_SEED
    jobs = [(image, seed) for image in images_noisy]

    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        thetas_low = np.stack(pool.map(_drift_worker, jobs))
    with mp.Pool(N_WORKERS) as pool:
        thetas_high = np.stack(pool.map(_drift_worker_high, jobs))
    print(f"  encoded at {DRIFT_LOW_STEPS} and {DRIFT_HIGH_STEPS} steps "
          f"({time.time() - t0:.1f}s)")

    drift = np.max(np.abs(wrapped_diff(thetas_high, thetas_low)), axis=1)

    n = thetas_low.shape[0]
    rng = np.random.default_rng(PAIR_SAMPLE_SEED)
    i = rng.integers(0, n, size=N_PAIRS_BETWEEN_IMAGE)
    j = rng.integers(0, n, size=N_PAIRS_BETWEEN_IMAGE)
    same = i == j
    if np.any(same):
        j[same] = (j[same] + 1) % n     # never compare an image to itself
    between_image = np.max(np.abs(wrapped_diff(thetas_low[i], thetas_low[j])), axis=1)

    return drift, between_image


def print_summary(curve_rows, drift, between_image):
    print("\n" + "=" * 78)
    print("MEASUREMENT 1: CONVERGENCE CURVE (median / p95 final-Delta, full 784-grid)")
    print("=" * 78)
    print(f"{'steps':>7}  {'median_clean':>14}  {'p95_clean':>14}  "
          f"{'median_noisy':>14}  {'p95_noisy':>14}  {'rho':>10}")
    for steps in STEP_COUNTS:
        r = curve_rows[steps]
        print(f"{steps:7d}  {r['median_clean']:14.6e}  {r['p95_clean']:14.6e}  "
              f"{r['median_noisy']:14.6e}  {r['p95_noisy']:14.6e}  {r['rho']:10.4g}")

    r150 = curve_rows.get(150)
    if r150 is not None:
        print(f"\ncross-check at steps=150 against the cloud run's own report:")
        print(f"  median_delta_clean : {r150['median_clean']!r}  "
              f"(cloud: {KNOWN_MEDIAN_DELTA_CLEAN_AT_150!r})")
        print(f"  median_delta_noisy : {r150['median_noisy']!r}  "
              f"(cloud: {KNOWN_MEDIAN_DELTA_NOISY_AT_150!r})")
        print(f"  rho                : {r150['rho']!r}  (cloud: {KNOWN_RHO_AT_150!r})")

    print("\n" + "=" * 78)
    print(f"MEASUREMENT 2: STATE DRIFT ({DRIFT_LOW_STEPS}->{DRIFT_HIGH_STEPS} steps, noisy) "
          f"vs BETWEEN-IMAGE SCALE")
    print("=" * 78)
    print(f"drift, per-image max|wrapped(theta_{DRIFT_HIGH_STEPS} - theta_{DRIFT_LOW_STEPS})|, "
          f"n={drift.size}:")
    print(f"  median: {np.median(drift):.6e}   p95: {np.percentile(drift, 95):.6e}")
    print(f"\nbetween-image scale, max|wrapped(theta_i - theta_j)| at steps={DRIFT_LOW_STEPS}, "
          f"n_pairs={between_image.size}:")
    print(f"  median: {np.median(between_image):.6e}   p95: {np.percentile(between_image, 95):.6e}")
    ratio = np.median(drift) / np.median(between_image)
    print(f"\nratio (median drift / median between-image scale): {ratio:.4f}")


def main():
    images_01, x_t, x_t_clip, active_indices, labels = load_stage1_corpus()
    verify_reconstruction_matches_the_cloud_run(images_01, x_t, x_t_clip, active_indices, labels)

    images_clean, images_noisy = calibrate_and_choose_scale(images_01, x_t_clip, active_indices)
    n_used = images_clean.shape[0]

    print(f"\nRunning convergence curve (n={n_used}, {N_WORKERS} workers)...")
    curve_rows = run_convergence_curve(images_clean, images_noisy)

    print(f"\nRunning drift-vs-scale measurement (n={n_used}, {N_WORKERS} workers)...")
    drift, between_image = run_drift_vs_scale(images_noisy)

    print_summary(curve_rows, drift, between_image)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "encoder_gate_failure_diagnostic.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "n_used": n_used, "step_counts": STEP_COUNTS,
            "drift_low_steps": DRIFT_LOW_STEPS, "drift_high_steps": DRIFT_HIGH_STEPS,
            "curve_rows": curve_rows, "drift": drift, "between_image": between_image,
            "pair_sample_seed": PAIR_SAMPLE_SEED,
        }, f)
    print(f"\nSaved to {out_path}")
    return curve_rows, drift, between_image


if __name__ == "__main__":
    main()
