"""Stage 2B ladder stage 3, PHASE A: encode the 54,000-image fit side
locally, on CPU, and write the encoded array to GCS for the GPU phase to
read.

Runs on this machine, not on a Colab runtime, and provisions nothing.
`mighty-colab` is not involved in this phase at all.

## Why this phase is local, and why that does not contradict DESIGN.md

DESIGN.md's computational strategy says generation runs "entirely in the
cloud environment; artifacts pushed to Google Cloud Storage from within
it -- never round-tripped through local upload", and names its own
reason in the same sentence: "Stage 2A's 242MB-vs-~6-15MB Colab upload
limit, already hit once". That limit is a property of getting bytes INTO
a Colab session through the session's own upload mechanism. Writing from
here straight to GCS never touches it -- it goes through the same
`google-cloud-storage` client and the same chunked, resumable,
crc32c-verified transport in `stage2b_gcs.py` that a cloud-side write
uses, and `stage_kmnist_inputs.py` has already moved 47MB of KMNIST that
way for both prior ladder rungs.

The thing actually worth avoiding was different and was caught before it
was built: running a CPU-only encode INSIDE a provisioned A100 session,
so a metered GPU sits idle for the majority of the run's wall-clock
while numpy churns. Encoding is the one genuinely CPU-bound step in the
pipeline; everything downstream (evolution, ridge, CNN) actually uses
the GPU. Splitting them is what keeps the A100's billed time to the work
that needs an A100.

**Measured, which is why this is worth doing at all**: this machine runs
the encoder at ~10.6 ms/image across 9 workers at ENCODER_STEPS=1200, vs
the 218.28 ms/image single-worker rate stage 2 measured on the Colab CPU
-- roughly 3.4x faster per core before parallelising. Stage 3's fit side
lands around 10 minutes here. Numbers this script re-measures and
reports rather than assuming; see `report` at the end.

## What crosses the phase boundary, and what deliberately does not

Only the ENCODED array. Corruption and the clean targets are both cheap
and deterministic, so Phase B regenerates them in-session rather than
receiving them:

- encoded noisy phases, (54000, 505) float64  -- ~218 MB, THE handoff
- corrupted images, (54000, 784)              -- ~339 MB, NOT sent;
  `corrupt_corpus` reproduces them from the original dataset indices,
  bit-exact (already proven across rungs by the stage-2 driver's own
  cross-stage check), in ~2.5 min of cloud CPU
- clean targets, (54000, 505)                 -- ~218 MB, NOT sent; a
  pure slice of the KMNIST already staged in the bucket

That is 218 MB over the wire instead of 775 MB, for about two and a half
minutes of Colab CPU. The encoded array is the only thing here that is
expensive to recompute, so it is the only thing that travels.

At 218 MB the upload crosses `ensure_artifact`'s 64 MB auto-chunk
threshold, so the resumable chunked path engages on its own -- no call
site has to ask for it.

## Scope

Fit side only (54,000 images). The 6,000-image locked validation
partition is NOT encoded: the CNN consumes validation images as raw
corrupted 28x28 grids, and ridge selects alpha by cross-validation
internally on the fit side. Encoding it would produce an artifact
nothing reads.

No test-side data of any kind is touched.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage2a_dynamics_classification"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))

import stage2b_corruption as corruption          # noqa: E402
import stage2b_encoder_gate as encoder_gate      # noqa: E402
import stage2b_gcs as gcs                        # noqa: E402
import stage2b_partition as partition            # noqa: E402
import stage2a_topologies as topologies          # noqa: E402
from bonsai.data.mnist_loader import load_mnist  # noqa: E402

LADDER_STAGE = 3
SPLIT = "train"
KMNIST_DIR = os.path.join(_REPO_ROOT, "datasets", "kmnist")
EXPECTED_N_ACTIVE = 505

# 54,000 / 5,400 = 10 chunks exactly. Chunked for memory, not for speed:
# `encode_with_final_delta_batch` builds one job tuple per image and
# replicates `active_indices` into each, so a single 54,000-image call
# would materialise well over a gigabyte of job list before any work
# started. Ten passes over the same function cost nothing extra and keep
# peak memory to one chunk. Corruption is chunked alongside it for the
# same reason -- it returns both pre- and post-clip corpora, and only
# the clipped one is needed here.
CHUNK = 5_400


def encode_fit_side(kmnist_dir=KMNIST_DIR, n_workers=None, limit=None, chunk=CHUNK):
    """Corrupt and encode the fit side, chunk by chunk.

    Composes `corrupt_corpus` and `encode_with_final_delta_batch`
    unchanged -- the same functions both prior ladder rungs ran, so
    nothing about the numerics changes by running them here. Only the
    machine changes.

    Returns (thetas (n, 505), deltas (n,), fit_indices (n,), summary)."""
    n_workers = int(n_workers or max(1, mp.cpu_count() - 1))
    steps = encoder_gate.ENCODER_STEPS

    print(f"loading official KMNIST training split from {kmnist_dir}", flush=True)
    x_train, y_train, _x_test, _y_test = load_mnist(kmnist_dir, gz=False)

    part = partition.Stage2BTrainingPartition(y_train)
    fit_indices = np.asarray(part.fit_indices)
    if limit is not None:
        fit_indices = fit_indices[:int(limit)]
    n = int(fit_indices.size)

    print("building topologies (for active_indices only)", flush=True)
    active_indices, _ink, _nodes, _topos = topologies.build_all_topologies()
    active_indices = np.asarray(active_indices)
    if active_indices.size != EXPECTED_N_ACTIVE:
        raise RuntimeError(f"active support has {active_indices.size} nodes, "
                           f"expected {EXPECTED_N_ACTIVE}")

    print(f"encoding n={n} fit-side images, {steps} steps, {n_workers} workers, "
          f"chunk={chunk}", flush=True)
    thetas = np.empty((n, EXPECTED_N_ACTIVE), dtype=np.float64)
    deltas = np.empty(n, dtype=np.float64)
    t_start = time.time()

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        idx = fit_indices[lo:hi]
        images_01 = x_train[idx].astype(np.float64) / 255.0
        # split="train", ORIGINAL dataset indices -- never positional.
        _x_t, x_t_clip = corruption.corrupt_corpus(
            images_01, SPLIT, idx, alpha_bar=corruption.ALPHA_BAR)
        t0 = time.time()
        chunk_thetas, chunk_deltas = encoder_gate.encode_with_final_delta_batch(
            x_t_clip, active_indices, seed=encoder_gate.ENCODER_SEED,
            steps=steps, n_workers=n_workers)
        elapsed = time.time() - t0
        thetas[lo:hi] = chunk_thetas
        deltas[lo:hi] = chunk_deltas
        done = hi
        rate = elapsed / (hi - lo)
        eta = (n - done) * rate
        print(f"  chunk {lo:6d}:{hi:6d}  {elapsed:6.1f}s  "
              f"{rate*1000:6.2f} ms/image   done {done}/{n}   eta {eta/60:5.1f} min",
              flush=True)

    total = time.time() - t_start
    summary = {
        "n_images": n, "steps": steps, "n_workers": n_workers, "chunk": chunk,
        "encode_elapsed_s": total, "encode_per_image_ms": total / n * 1000,
        "max_delta": float(np.max(deltas)),
        "p95_delta": float(np.percentile(deltas, 95)),
        "median_delta": float(np.median(deltas)),
        "n_nonfinite_theta": int(np.sum(~np.isfinite(thetas))),
        "n_nonfinite_delta": int(np.sum(~np.isfinite(deltas))),
        "n_active": int(active_indices.size),
        "machine": f"{os.uname().sysname} {os.uname().machine}, "
                   f"{mp.cpu_count()} cores",
        "index_space": "official KMNIST training split (0-based), fit side only",
        "note": ("encoded from the CLIPPED corrupted images; clean targets and the "
                 "corrupted corpus are deterministic and regenerated by Phase B "
                 "rather than transported"),
    }
    return thetas, deltas, fit_indices, active_indices, summary


def upload(thetas, deltas, fit_indices, active_indices, summary,
           bucket_name=None, credentials=None, force=False):
    """Write the encoded array to GCS through the production transport.

    Above 64 MB `ensure_artifact` selects the chunked resumable route on
    its own, so a dropped connection mid-upload resumes from the last
    confirmed chunk rather than restarting ~218 MB."""
    steps = summary["steps"]
    bucket = gcs.get_bucket(name=bucket_name, credentials=credentials)
    name = gcs.object_path(stage=LADDER_STAGE, condition=None,
                           kind=f"encoded_fit_s{steps}", ext="npz", split=SPLIT)
    local_path = os.path.join(_THIS_DIR, "results",
                              f"stage3_encoded_fit_s{steps}.npz")

    def produce(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path, thetas_505=thetas, deltas=deltas, fit_indices=fit_indices,
            active_indices=active_indices,
            summary_json=np.array(json.dumps(summary, indent=2, sort_keys=True)))

    print(f"\nbucket        : {bucket.name}")
    print(f"object        : {name}")
    print(f"checksum      : {gcs.checksum_backend()}")
    t0 = time.time()
    result = gcs.ensure_artifact(name, local_path, produce=produce, bucket=bucket,
                                 force=force)
    print(f"transfer      : {time.time() - t0:.1f}s")
    print(f"step result   : {result.summary()}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kmnist-dir", default=KMNIST_DIR)
    parser.add_argument("--workers", type=int, default=None,
                        help="default cpu_count()-1")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--limit", type=int, default=None,
                        help="encode only the first N fit-side images (smoke runs)")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-upload", action="store_true",
                        help="encode and report, write nothing to GCS")
    args = parser.parse_args(argv)

    thetas, deltas, fit_indices, active_indices, summary = encode_fit_side(
        kmnist_dir=args.kmnist_dir, n_workers=args.workers, limit=args.limit,
        chunk=args.chunk)

    print("\n" + "=" * 70)
    print("PHASE A -- measured, not projected")
    print("=" * 70)
    print(f"  images            : {summary['n_images']}")
    print(f"  steps             : {summary['steps']}")
    print(f"  workers           : {summary['n_workers']}  ({summary['machine']})")
    print(f"  encode wall-clock : {summary['encode_elapsed_s']:.1f}s "
          f"({summary['encode_elapsed_s']/60:.1f} min)")
    print(f"  per image         : {summary['encode_per_image_ms']:.2f} ms")
    print(f"  final-Delta       : max {summary['max_delta']!r}, "
          f"p95 {summary['p95_delta']!r}, median {summary['median_delta']!r}")
    print(f"  non-finite        : theta={summary['n_nonfinite_theta']}, "
          f"delta={summary['n_nonfinite_delta']}")
    print(f"  encoded array     : {thetas.shape} float64 "
          f"({thetas.nbytes/1e6:.0f} MB uncompressed)")

    if summary["n_nonfinite_theta"] or summary["n_nonfinite_delta"]:
        print("\nREFUSING to upload: non-finite values in the encoded output.")
        return 1

    if args.no_upload:
        print("\n--no-upload set; nothing written to GCS.")
        return 0

    upload(thetas, deltas, fit_indices, active_indices, summary,
           bucket_name=args.bucket, credentials=args.credentials, force=args.force)
    print("\nPhase A complete. Phase B reads this object and regenerates "
          "corruption and targets in-session.")
    return 0


# The __main__ guard is load-bearing, not decoration: `mp.Pool` uses the
# spawn start method on macOS, so each worker re-imports this module, and
# without the guard every child would re-execute the whole encode --
# recursively. Hit for real in a throwaway timing script for this very
# measurement, which is why it is called out rather than assumed known.
if __name__ == "__main__":
    sys.exit(main())
