"""Stage 2B ladder stage 3, PHASE A: encode the 60,000-image official
KMNIST training side locally, on CPU, and write the encoded array to GCS
for the GPU phase to read.

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

- encoded noisy phases, (60000, 505) float64  -- ~242 MB, THE handoff
- corrupted images, (60000, 784)              -- ~376 MB, NOT sent;
  `corrupt_corpus` reproduces them from the original dataset indices,
  bit-exact (already proven across rungs by the stage-2 driver's own
  cross-stage check), in ~2.5 min of cloud CPU
- clean targets, (60000, 505)                 -- ~242 MB, NOT sent; a
  pure slice of the KMNIST already staged in the bucket

That is 242 MB over the wire instead of 860 MB, for about two and a half
minutes of Colab CPU. The encoded array is the only thing here that is
expensive to recompute, so it is the only thing that travels.

The upload crosses `ensure_artifact`'s 64 MB auto-chunk threshold, so the
resumable chunked path engages on its own -- no call site has to ask for
it.

## Row order: ascending official index, and why that is worth stating

Row `i` of every returned array IS official KMNIST training image `i`.
The artifact records `train_indices` explicitly anyway rather than
leaving that implied, and carries `fit_indices` and `validation_indices`
alongside so a consumer never reconstructs the partition itself.

`AUDIT_PROTOCOL.md` requires all cross-artifact comparison to happen by
official image index, never by positional prefix. Encoding in ascending
official order makes the two coincide here -- which removes a mapping
step rather than making one safe to skip. Anything comparing this
artifact against another still joins on indices; see
`compare_stage3_regeneration.py`, whose whole job is that the 54,000
overlapping images are matched by index and NOT by taking a prefix.

## Scope -- and a disclosed error in this script's first run

As first written and first run, this script encoded the fit side only
(54,000 images), on the stated reasoning that the 6,000-image locked
validation partition "would produce an artifact nothing reads": the CNN
consumes validation images as raw corrupted 28x28 grids, and ridge
selects alpha by cross-validation internally.

**That reasoning was wrong, and the population is 60,000.** The second
clause is where it fails. `DESIGN.md:479` defines the term outright --
at stage 3, "full training" = 54,000 fit + 6,000 locked validation --
and `DESIGN.md:492`'s compute table corroborates it arithmetically:
"~48-60k x 1008" is 0.8 x 60,000 for the fold-level SVDs and 60,000 for
the final refits. Under a 54,000 corpus that cell would read ~43-54k.
The 6,000 are held out from CNN gradient updates only; they are not held
out from the ridge path, which fits and cross-validates on all 60,000.
So the artifact IS read, by the ridge.

The error was mine, made in the direction that avoided re-work, which is
exactly why it is disclosed here rather than quietly edited away. The
regeneration closes it empirically: the 54,000 fit-side arrays
payload-compare bit-exact against the artifact this script already
uploaded, and the 6,000 validation arrays are new evidence with no prior
artifact to compare against -- fingerprinted at birth, with their
final-Delta tail computed and reported separately rather than folded
into one "regeneration passed".

No test-side data of any kind is touched, at either population.
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
import stage2b_fingerprint as fingerprint        # noqa: E402
import stage2b_gcs as gcs                        # noqa: E402
import stage2b_partition as partition            # noqa: E402
import stage2a_topologies as topologies          # noqa: E402
from bonsai.data.mnist_loader import load_mnist  # noqa: E402

LADDER_STAGE = 3
SPLIT = "train"
KMNIST_DIR = os.path.join(_REPO_ROOT, "datasets", "kmnist")
EXPECTED_N_ACTIVE = 505
N_OFFICIAL_TRAIN = 60_000

# 60,000 / 6,000 = 10 chunks exactly. Chunked for memory, not for speed:
# `encode_with_final_delta_batch` builds one job tuple per image and
# replicates `active_indices` into each, so a single 60,000-image call
# would materialise well over a gigabyte of job list before any work
# started. Ten passes over the same function cost nothing extra and keep
# peak memory to one chunk. Corruption is chunked alongside it for the
# same reason -- it returns both pre- and post-clip corpora, and only
# the clipped one is needed here.
#
# CLAUDE.md principle 19 says a chunked draw is not automatically the same
# stream as the unchunked one, so this is checked rather than reasoned
# about: `encode_with_final_delta_batch` passes a CONSTANT `seed` into
# every per-image job and `_local_converged_phases` builds a fresh
# `default_rng(seed)` per call, so an image's perturbation depends on
# nothing but the image and that seed. `test_stage2b_encode_stage3_local`
# pins it by sweeping chunk sizes, including a non-divisor and a chunk of
# one, and asserting bit-identical output -- so a later "just make the
# chunks bigger" cannot silently move the numbers.
CHUNK = 6_000


def encode_indices(x_train, indices, active_indices, steps, n_workers,
                   chunk=CHUNK, progress=True):
    """Corrupt and encode `indices` of `x_train`, chunk by chunk.

    Separated from `encode_training_side` so the chunking is testable
    without loading KMNIST: the chunk loop is the only place in this
    script where a numerical property could depend on batching, and a
    property that can only be exercised by a nine-minute full run is a
    property nobody re-checks.

    `indices` are OFFICIAL dataset indices, and they are what corruption
    is keyed on -- never a positional counter.

    Returns (thetas (n, len(active_indices)), deltas (n,))."""
    indices = np.asarray(indices)
    n = int(indices.size)
    thetas = np.empty((n, active_indices.size), dtype=np.float64)
    deltas = np.empty(n, dtype=np.float64)
    t_start = time.time()

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        idx = indices[lo:hi]
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
        if progress:
            rate = elapsed / (hi - lo)
            eta = (n - hi) * rate
            print(f"  chunk {lo:6d}:{hi:6d}  {elapsed:6.1f}s  "
                  f"{rate*1000:6.2f} ms/image   done {hi}/{n}   eta {eta/60:5.1f} min",
                  flush=True)

    return thetas, deltas, time.time() - t_start


def encode_training_side(kmnist_dir=KMNIST_DIR, n_workers=None, limit=None,
                         chunk=CHUNK):
    """Corrupt and encode the whole official training side, in ascending
    official index order.

    Composes `corrupt_corpus` and `encode_with_final_delta_batch`
    unchanged -- the same functions both prior ladder rungs ran, so
    nothing about the numerics changes by running them here. Only the
    machine changes.

    Returns (thetas, deltas, roles, active_indices, summary), where
    `roles` maps `train_indices` / `fit_indices` / `validation_indices`
    to official index arrays."""
    n_workers = int(n_workers or max(1, mp.cpu_count() - 1))
    steps = encoder_gate.ENCODER_STEPS

    print(f"loading official KMNIST training split from {kmnist_dir}", flush=True)
    x_train, y_train, _x_test, _y_test = load_mnist(kmnist_dir, gz=False)

    part = partition.Stage2BTrainingPartition(y_train)
    fit_indices = np.asarray(part.fit_indices)
    validation_indices = np.asarray(part.validation_indices)
    train_indices = np.arange(N_OFFICIAL_TRAIN, dtype=fit_indices.dtype)

    # AUDIT_PROTOCOL.md Freeze 2's population roles, asserted rather than
    # inherited from stage2b_partition. The claim "the 6,000 are held out
    # from CNN gradient updates only, and the ridge corpus is all 60,000"
    # is only meaningful if the three arrays really do partition the
    # official split -- and getting this wrong is exactly the error that
    # scoped Phase A to 54,000 in the first place.
    assert fit_indices.size == 54_000, fit_indices.size
    assert validation_indices.size == 6_000, validation_indices.size
    assert not (set(fit_indices.tolist()) & set(validation_indices.tolist())), \
        "fit and validation overlap"
    assert (set(fit_indices.tolist()) | set(validation_indices.tolist())
            == set(range(N_OFFICIAL_TRAIN))), \
        "fit u validation is not the official 60,000-image training split"
    print(f"population verified: {fit_indices.size} fit + "
          f"{validation_indices.size} validation = {N_OFFICIAL_TRAIN} official "
          f"training images, disjoint and exhaustive", flush=True)

    if limit is not None:
        train_indices = train_indices[:int(limit)]
    n = int(train_indices.size)

    print("building topologies (for active_indices only)", flush=True)
    active_indices, _ink, _nodes, _topos = topologies.build_all_topologies()
    active_indices = np.asarray(active_indices)
    if active_indices.size != EXPECTED_N_ACTIVE:
        raise RuntimeError(f"active support has {active_indices.size} nodes, "
                           f"expected {EXPECTED_N_ACTIVE}")

    print(f"encoding n={n} training-side images, {steps} steps, {n_workers} "
          f"workers, chunk={chunk}", flush=True)
    thetas, deltas, total = encode_indices(
        x_train, train_indices, active_indices, steps, n_workers, chunk=chunk)

    in_fit = np.isin(train_indices, fit_indices)
    nonzero = deltas > 0.0
    summary = {
        "n_images": n, "steps": steps, "n_workers": n_workers, "chunk": chunk,
        "encode_elapsed_s": total, "encode_per_image_ms": total / n * 1000,
        "max_delta": float(np.max(deltas)),
        "p95_delta": float(np.percentile(deltas, 95)),
        "median_delta": float(np.median(deltas)),
        "n_nonfinite_theta": int(np.sum(~np.isfinite(thetas))),
        "n_nonfinite_delta": int(np.sum(~np.isfinite(deltas))),
        "n_active": int(active_indices.size),
        "tail": tail_report(deltas, in_fit),
        "machine": f"{os.uname().sysname} {os.uname().machine}, "
                   f"{mp.cpu_count()} cores",
        "index_space": ("official KMNIST training split (0-based), all "
                        "60,000 images, rows in ascending index order"),
        "n_fit": int(in_fit.sum()),
        "n_validation": int(n - in_fit.sum()),
        "n_nonzero_delta": int(nonzero.sum()),
        "note": ("encoded from the CLIPPED corrupted images; clean targets and the "
                 "corrupted corpus are deterministic and regenerated by Phase B "
                 "rather than transported"),
    }
    roles = {"train_indices": train_indices, "fit_indices": fit_indices,
             "validation_indices": validation_indices}
    return thetas, deltas, roles, active_indices, summary


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial confidence interval for k successes in n.

    `AUDIT_PROTOCOL.md` requires the 6,000-image final-Delta tail rate be
    compared against the 54,000-image rate WITH UNCERTAINTY and with no
    expected-agreement criterion. A handful out of 6,000 and 79 out of
    54,000 are estimated with very different precision, and comparing the
    bare percentages is exactly the error that requirement guards against.
    Clopper-Pearson rather than a normal approximation because the counts
    are small and the rate is near zero, where the normal interval is
    badly wrong and can extend below zero."""
    from scipy.stats import beta
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def tail_report(deltas, in_fit):
    """The final-Delta convergence tail, split by population role.

    Reported per `AUDIT_PROTOCOL.md`'s companion requirement: numerator,
    denominator and split membership for each role, with a 95% interval.
    The 6,000 validation images have never been encoded before, so their
    rate is a NEW measurement, not a reproduction -- a different
    proportion from the fit side's 79/54,000 (0.146%) is a finding to
    report, not a reproducibility failure."""
    deltas = np.asarray(deltas, dtype=np.float64)
    in_fit = np.asarray(in_fit, dtype=bool)
    out = {}
    for role, mask in (("all", np.ones_like(in_fit)), ("fit", in_fit),
                       ("validation", ~in_fit)):
        d = deltas[mask]
        n = int(d.size)
        k = int(np.count_nonzero(d > 0.0))
        lo, hi = clopper_pearson(k, n)
        out[role] = {
            "n": n, "n_nonzero": k,
            "rate": (k / n) if n else float("nan"),
            "ci95_lower": lo, "ci95_upper": hi,
            "n_gt_1e_13": int(np.count_nonzero(d > 1e-13)),
            "n_gt_1e_12": int(np.count_nonzero(d > 1e-12)),
            "n_gt_1e_10": int(np.count_nonzero(d > 1e-10)),
            "max": float(np.max(d)) if n else float("nan"),
        }
    return out


def object_name_for(steps):
    """The artifact's object path.

    `encoded_train`, not `encoded_fit`: the array covers all 60,000
    official training images, and a name that still said "fit" would be
    claiming a population the object does not have. Keeping the old name
    would also have meant overwriting the 54,000-image artifact that the
    regeneration is COMPARED AGAINST -- destroying the baseline with the
    same `force=True` bypass this project spent a commit documenting."""
    return gcs.object_path(stage=LADDER_STAGE, condition=None,
                           kind=f"encoded_train_s{steps}", ext="npz", split=SPLIT)


def local_path_for(steps):
    return os.path.join(_THIS_DIR, "results", f"stage3_encoded_train_s{steps}.npz")


def build_fingerprint(summary, require_clean=True):
    """Provenance for this run, established BEFORE generation.

    `config` carries what is scientifically load-bearing about the run --
    not its timings, which change every time and would make two
    numerically identical runs look like different provenance."""
    return fingerprint.compute(
        entrypoint=os.path.abspath(__file__),
        repo_root=_REPO_ROOT,
        require_clean=require_clean,
        config={
            "encoder_steps": summary["steps"],
            "encoder_seed": encoder_gate.ENCODER_SEED,
            "alpha_bar": corruption.ALPHA_BAR,
            "corruption_scheme": "SHA256(split:index:42) -> PCG64, per image",
            "split": SPLIT,
            "population": "official KMNIST training split, all 60,000",
            "n_images": summary["n_images"],
            "n_active": summary["n_active"],
            "row_order": "ascending official training index",
            "dtype": "float64",
        })


def upload(thetas, deltas, roles, active_indices, summary, fp,
           bucket_name=None, credentials=None, force=False):
    """Write the encoded array to GCS through the production transport,
    then publish its manifest.

    Above 64 MB `ensure_artifact` selects the chunked resumable route on
    its own, so a dropped connection mid-upload resumes from the last
    confirmed chunk rather than restarting ~242 MB.

    The manifest is published AFTER the payload verifies, which is the
    ordering that makes it meaningful: a sidecar that exists is a sidecar
    describing a complete object. This is the first Stage 2B artifact
    written under the contract -- every earlier one carries no provenance
    at all, which is what the regeneration exists to fix."""
    steps = summary["steps"]
    bucket = gcs.get_bucket(name=bucket_name, credentials=credentials)
    name = object_name_for(steps)
    local_path = local_path_for(steps)

    def produce(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path, thetas_505=thetas, deltas=deltas,
            train_indices=roles["train_indices"],
            fit_indices=roles["fit_indices"],
            validation_indices=roles["validation_indices"],
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

    manifest = gcs.publish_manifest(result.local_path, name, bucket=bucket,
                                    fingerprint=fp)
    print(f"manifest      : {gcs.manifest_object_name(name)}")
    print(f"  payload sha256   : {manifest['payload_sha256']}")
    print(f"  arrays recorded  : {sorted(manifest.get('arrays', {}))}")
    print(f"  source files     : {len(fp['source_manifest'])}")
    print_git_state(fp["git"], indent="  ")
    return result, manifest


def print_git_state(git, indent=""):
    """Both cleanliness claims, closure first.

    Printing only `clean` would report this artifact as coming from a
    dirty tree, which is true of the REPOSITORY and false of the thing a
    reader cares about. The closure is what determines reproducibility, so
    it leads -- and the tree's state is shown underneath rather than
    hidden, since a reader who wants to check that the dirt is irrelevant
    needs to see what it was."""
    print(f"{indent}commit           : {git['commit']}")
    if "closure_clean" in git:
        print(f"{indent}source closure   : "
              f"{'CLEAN -- every file committed at HEAD' if git['closure_clean'] else 'DIRTY: ' + ', '.join(git['closure_dirty_paths'])}")
    print(f"{indent}working tree     : {'clean' if git['clean'] else 'dirty elsewhere'}")
    porcelain = git.get("tree_dirty_porcelain")
    if porcelain:
        for line in porcelain.splitlines():
            print(f"{indent}  | {line}")
        print(f"{indent}  (recorded, not blocking: none of these are in the "
              f"source closure)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kmnist-dir", default=KMNIST_DIR)
    parser.add_argument("--workers", type=int, default=None,
                        help="default cpu_count()-1")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--limit", type=int, default=None,
                        help="encode only the first N training images (smoke runs)")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-upload", action="store_true",
                        help="encode and report, write nothing to GCS")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="record a dirty tree instead of refusing (smoke runs "
                             "only -- a real artifact must be reproducible)")
    args = parser.parse_args(argv)

    # BEFORE generation, and before the nine minutes of encoding: a dirty
    # tree is refused here rather than discovered at upload time. The
    # closure is a claim about what will run, and it is revalidated below
    # against what did.
    pre = fingerprint.compute(
        entrypoint=os.path.abspath(__file__), repo_root=_REPO_ROOT,
        require_clean=not args.allow_dirty,
        config={"probe": "pre-generation source closure"})
    print(f"source closure    : {len(pre['source_manifest'])} files")
    print_git_state(pre["git"], indent="  ")

    thetas, deltas, roles, active_indices, summary = encode_training_side(
        kmnist_dir=args.kmnist_dir, n_workers=args.workers, limit=args.limit,
        chunk=args.chunk)

    print("\n" + "=" * 70)
    print("PHASE A -- measured, not projected")
    print("=" * 70)
    print(f"  images            : {summary['n_images']} "
          f"({summary['n_fit']} fit + {summary['n_validation']} validation)")
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

    print("\n  convergence tail (final-Delta > 0), by population role:")
    header = (f"  {'role':<12}{'n':>8}{'nonzero':>9}{'rate':>10}"
              f"{'95% CI (Clopper-Pearson)':>30}{'max':>14}")
    print(header)
    for role in ("all", "fit", "validation"):
        t = summary["tail"][role]
        ci = f"[{t['ci95_lower'] * 100:.4f}%, {t['ci95_upper'] * 100:.4f}%]"
        print(f"  {role:<12}{t['n']:>8}{t['n_nonzero']:>9}"
              f"{t['rate'] * 100:>9.3f}%{ci:>30}{t['max']:>14.3e}")
    print("  The validation rate is a NEW measurement on never-encoded images; "
          "AUDIT_PROTOCOL.md sets\n  no expected-agreement criterion against the "
          "fit-side rate. A different proportion is a\n  finding, not a "
          "reproducibility failure.")

    if summary["n_nonfinite_theta"] or summary["n_nonfinite_delta"]:
        print("\nREFUSING to upload: non-finite values in the encoded output.")
        return 1

    fp = build_fingerprint(summary, require_clean=not args.allow_dirty)
    new_paths = fingerprint.revalidate_after_execution(fp, _REPO_ROOT)
    print(f"\nrevalidated       : source closure intact, "
          f"{len(fp['source_manifest'])} files, {len(new_paths)} newly imported")

    if args.no_upload:
        print("\n--no-upload set; nothing written to GCS.")
        return 0

    upload(thetas, deltas, roles, active_indices, summary, fp,
           bucket_name=args.bucket, credentials=args.credentials, force=args.force)
    print("\nPhase A complete. Phase B reads this object and regenerates "
          "corruption and targets in-session.")
    print("Next: `make stage2b-compare-stage3` -- the 54,000 overlapping images "
          "must match the\nprevious artifact bit-exactly, joined by official "
          "index.")
    return 0


# The __main__ guard is load-bearing, not decoration: `mp.Pool` uses the
# spawn start method on macOS, so each worker re-imports this module, and
# without the guard every child would re-execute the whole encode --
# recursively. Hit for real in a throwaway timing script for this very
# measurement, which is why it is called out rather than assumed known.
if __name__ == "__main__":
    sys.exit(main())
