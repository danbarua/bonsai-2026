"""Backfill the trained CNN weights into stage 3's `cnn_production.npz`.

`stage2b_cnn.train_cnn` has always returned its best checkpoint under
`"model"`, and every caller dropped it. Stage 3 kept only loss histories
and a summary; stage 4 therefore retrains all three seeds from scratch to
get a model back, and FINDINGS records three separate real-GPU retrains in
its run alone. This network has been trained four times and discarded four
times. `run_ladder_stage3.step8_cnn` now persists the weights, so it never
recurs -- but only on a FRESH run, and `cnn_production.npz` already exists,
so `ensure_artifact` skips that step forever. This driver is the one-off
that fixes the object already in the bucket.

## Why a NEW object, and not weights added to `cnn_production.npz`

That was the intent, and the bucket refused it -- correctly. On the first
run that got as far as uploading, `ensure_artifact(force=True)` raised
`WriteOnceViolation`: `cnn_production.npz` is a LINEAGE artifact, lineage
artifacts are create-once, and object versioning is off on this bucket, so
replacing it destroys a reference another manifest may hold as a parent.
The error prescribes the remedy this driver now takes -- "To regenerate,
write a NEW name" -- the same move Phase A made when it produced
`encoded_train_s1200` rather than overwriting `encoded_fit_s1200`.

A second reason points the same way. Retraining does not reproduce stage
3's loss histories bit-exactly (FINDINGS measures 9.328e-07 and 2.385e-07
of drift across real-GPU retrains; this run measured 2.290e-06), so
rewriting that object would have replaced reported numbers with slightly
different ones. Keeping the histories where they are and the weights
beside them avoids the question entirely.

## What licenses these weights as stage 3's model

They come from a retrain verified to reproduce stage 3's SELECTION --
`best_seed` and `best_epoch`, exactly -- through the same
`cnn_reproduction_mismatch_reason` stage 4 uses. Nothing is written if
that check fails. `cnn_production.npz` is recorded as this object's PARENT
with its payload digest, so the pairing is pinned rather than implied, and
`weights_json` states plainly that the weights and those histories come
from different runs, because they do.

## What this does not carry

The manifest is written with no `fingerprint`: this driver cannot rebuild
stage 3's without duplicating its `build_fingerprint`, and a fabricated
one would be worse than none. Provenance lives in `weights_json` instead
-- the source commit, per-seed wallclock, the GPU it ran on, and the
reproduction outcome.

Run:  mighty-colab exec -s <session> -f backfill_cnn_weights.py \\
          --env BONSAI_COMMIT=<sha> --env JAX_ENABLE_X64=1 ...
Set `BONSAI_BACKFILL_DRYRUN=1` to do everything except the upload.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

DRIVER_FILENAME = "backfill_cnn_weights.py"
TRAIN_STAGE = 3
KMNIST_STAGING_STAGE = 1
SPLIT = "train"
OK_SENTINEL = "BACKFILL_OK"
FAIL_SENTINEL = "BACKFILL_FAIL"

REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
WORK_DIR = "/content/stage2b_backfill"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)

ENV_COMMIT = "BONSAI_COMMIT"
ENV_BUCKET = "BONSAI_GCS_BUCKET"
ENV_CREDENTIALS = "BONSAI_GCS_CREDENTIALS"
ENV_DRYRUN = "BONSAI_BACKFILL_DRYRUN"

EXPECTED_N_ACTIVE = 505

# Ladder stage 1 wrote `topologies.npz` before the fingerprint contract
# existed, so it carries no manifest and `consume_validated` refuses it
# under the default. The opt-out alone would be worse than the refusal:
# `require_manifest=False` disables EVERY provenance check, and these are
# exactly the names nothing else guards, stages 1 and 2 being closed and
# write-once having no purchase on objects written before it existed. So
# the opt-out is paired with a pinned digest, as
# `run_ladder_stage3.consume_pinned` does.
#
# The value is copied from `run_ladder_stage3.PINNED_SHA256` rather than
# imported, because importing that module executes `main()` whenever
# BONSAI_COMMIT is set -- which is every Colab run, including this one. If
# the two ever drift apart this run HALTS, which is the safe direction.
# Do not update this to make a failure pass; find out what changed.
PINNED_TOPOLOGIES_SHA256 = \
    "f671e63cc00b1612db0da5976c14b8880e4c4f90ae7fb192297721665f1907a4"

_T0 = time.time()


class BackfillHalt(Exception):
    """Stops the run with a stated reason rather than a traceback."""


def say(line):
    print(f"[backfill {time.time() - _T0:7.1f}s] {line}", flush=True)


def _run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def local_path_for(object_name):
    return os.path.join(WORK_DIR, object_name.replace("/", "__"))


def bootstrap_repo(commit, clone_dir=CLONE_DIR):
    """Clone at the pinned commit. Same shape as the ladder drivers': the
    file is transmitted as TEXT into a live kernel, so nothing of this
    repository exists on disk until this has run."""
    if not os.path.isdir(os.path.join(clone_dir, ".git")):
        os.makedirs(clone_dir, exist_ok=True)
        _run(["git", "init", "-q", clone_dir])
        _run(["git", "remote", "add", "origin", REPO_URL], cwd=clone_dir)
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=clone_dir)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=clone_dir)
    else:
        say(f"{clone_dir} already checked out; reusing it")
    head = _run(["git", "rev-parse", "HEAD"], cwd=clone_dir).stdout.strip()
    if head != commit:
        raise BackfillHalt(f"clone is at {head}, expected {commit}")
    try:
        _run([sys.executable, "-m", "pip", "install", "-e", clone_dir,
              "--no-deps", "--ignore-requires-python", "-q"])
    except subprocess.CalledProcessError as exc:
        say(f"editable install failed ({exc.returncode}); falling back to sys.path")
    return head


def add_repo_to_path(clone_dir):
    for directory in (*EXPERIMENT_DIRS, "src"):
        entry = os.path.join(clone_dir, directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def load_modules(clone_dir):
    """`stage2b_ridge` FIRST: it enables jax's x64 mode at import, and the
    ladder drivers depend on that ordering."""
    add_repo_to_path(clone_dir)
    import stage2b_ridge                                              # noqa: F401
    import stage2b_cnn as cnn
    import stage2b_gcs as gcs
    import stage2b_partition as partition
    import types
    return types.SimpleNamespace(cnn=cnn, gcs=gcs, partition=partition)


def fetch(mods, bucket, name, pinned_sha256=None):
    """Consume one artifact. `pinned_sha256` selects the pre-contract route
    -- the manifest opt-out plus a digest check, never the opt-out alone."""
    local = local_path_for(name)
    mods.gcs.consume_validated(name, local, bucket=bucket,
                               require_manifest=pinned_sha256 is None)
    if pinned_sha256 is not None:
        digest = sha256_of(local)
        if digest != pinned_sha256:
            raise BackfillHalt(
                f"{name!r} does not match its pinned digest: expected "
                f"{pinned_sha256}, got {digest}. This object carries no manifest "
                f"(pre-contract history), so the pin is the only thing standing "
                f"between this run and silently different input. Do not update the "
                f"pin to make this pass -- find out what changed.")
        say(f"consumed {name} (pre-contract, pinned sha256 {digest[:16]}...)")
    with np.load(local, allow_pickle=False) as handle:
        loaded = {key: handle[key] for key in handle.files}
    say(f"fetched {name} ({len(loaded)} arrays)")
    return loaded, local


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    commit = os.environ.get(ENV_COMMIT)
    if not commit:
        raise BackfillHalt(f"{ENV_COMMIT} is required -- it pins the code that runs")
    dry_run = bool(os.environ.get(ENV_DRYRUN))
    os.makedirs(WORK_DIR, exist_ok=True)

    head = bootstrap_repo(commit)
    say(f"repo at {head}")
    mods = load_modules(CLONE_DIR)
    bucket = mods.gcs.get_bucket(
        name=os.environ.get(ENV_BUCKET) or None,
        credentials=os.environ.get(ENV_CREDENTIALS))

    obj = lambda kind, stage=TRAIN_STAGE: mods.gcs.object_path(   # noqa: E731
        stage=stage, condition=None, kind=kind, ext="npz", split=SPLIT)

    corpus, _ = fetch(mods, bucket, obj("corpus"))
    corr, _ = fetch(mods, bucket, obj("corruption"))
    topo, _ = fetch(mods, bucket, obj("topologies", KMNIST_STAGING_STAGE),
                    pinned_sha256=PINNED_TOPOLOGIES_SHA256)
    cnn_name = obj("cnn_production")
    original, cnn_local = fetch(mods, bucket, cnn_name)

    before_sha = sha256_of(cnn_local)
    original_summary = json.loads(original["summary_json"].item())
    say(f"stage {TRAIN_STAGE} selected seed={original_summary['best_seed']} "
        f"epoch={original_summary['best_epoch']}, sha256={before_sha[:12]}")

    # The SAME derivation stage 3 uses -- not a second one written to agree
    # with it (CLAUDE.md principle 16).
    inputs = mods.cnn.training_inputs(
        images=corpus["images_01"], x_t_clip=corr["x_t_clip"],
        active_indices=topo["active_indices"],
        train_indices=corpus["train_indices"], fit_indices=corpus["fit_indices"],
        validation_indices=corpus["validation_indices"],
        index_join=mods.partition.index_join, expect_n_active=EXPECTED_N_ACTIVE)
    say(f"fit={inputs['fit_clean'].shape[0]}, "
        f"validation={inputs['val_clean'].shape[0]}")

    runs, wallclock = [], {}
    for seed in mods.cnn.SEEDS:
        t0 = time.time()
        run = mods.cnn.train_cnn_for_seed(
            inputs["fit_noisy"], inputs["fit_clean"],
            inputs["val_noisy"], inputs["val_clean"], inputs["mask"], seed=seed)
        wallclock[str(seed)] = time.time() - t0
        runs.append(run)
        say(f"seed={seed}: best_epoch={run['best_epoch']}, "
            f"best_clipped_val_mse={run['best_clipped_val_mse']!r}, "
            f"{wallclock[str(seed)]:.1f}s")

    best_seed, best_index = mods.cnn.select_best_seed(
        [r["seed"] for r in runs], [r["best_clipped_val_mse"] for r in runs])
    reproduced = {
        "best_seed": best_seed, "best_index": best_index,
        "best_epoch": int(runs[best_index]["best_epoch"]),
        "best_clipped_val_mse": float(runs[best_index]["best_clipped_val_mse"]),
    }
    mismatch = mods.cnn.cnn_reproduction_mismatch_reason(
        reproduced, original_summary, train_stage=TRAIN_STAGE)
    if mismatch:
        raise BackfillHalt(
            f"reproduction FAILED, so these weights are not stage {TRAIN_STAGE}'s "
            f"model and must not be written into its artifact: {mismatch}")
    mse_diff = abs(reproduced["best_clipped_val_mse"]
                   - float(original_summary["best_clipped_val_mse"]))
    say(f"reproduction VERIFIED: seed={best_seed} epoch={reproduced['best_epoch']}; "
        f"best_clipped_val_mse differs by {mse_diff:.3e} (reported, not gated)")

    arrays = {}
    probe = mods.cnn.as_image_batch(inputs["val_noisy"][:4], "round-trip probe")
    for run in runs:
        blob = mods.cnn.serialise_model(run["model"])
        lossy = mods.cnn.model_round_trip_mismatch(run["model"], blob, probe)
        if lossy:
            raise BackfillHalt(f"seed={run['seed']} weights do not round-trip: {lossy}")
        arrays[f"weights_seed{run['seed']}"] = blob
    say(f"serialised {len(mods.cnn.SEEDS)} models, each round-trip verified")

    arrays["weights_json"] = np.array(json.dumps({
        "what": "the trained CNN weights stage 3 produced and did not persist",
        "companion_to": cnn_name,
        "weights_from": "a retrain on this run, NOT the run that produced the loss "
                        "histories in the companion -- iterative float32 training "
                        "does not reproduce bit-exactly, which is why those "
                        "histories were left where they are rather than rewritten",
        "reproduction": {"verified": True, "matched_on": ["best_seed", "best_epoch"],
                         "best_clipped_val_mse_difference": mse_diff,
                         "reproduced": reproduced, "original": original_summary},
        "weights_format": "equinox.tree_serialise_leaves -> uint8, load via "
                          "stage2b_cnn.deserialise_model",
        "weights_keys": [f"weights_seed{r['seed']}" for r in runs],
        "best_weights_key": f"weights_seed{best_seed}",
        "companion_sha256": before_sha,
        "commit": head, "driver": DRIVER_FILENAME,
        "wallclock_s_per_seed": wallclock,
        "platform": {"python": sys.version, "numpy": np.__version__,
                     "gpu": os.environ.get("BONSAI_GPU", "unrecorded")},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, sort_keys=True))

    def produce(path):
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as handle:
            reloaded = {key: handle[key] for key in handle.files}
        for key, value in arrays.items():
            if not np.array_equal(reloaded[key], value):
                raise BackfillHalt(f"written array {key!r} does not read back equal")
        say(f"wrote {len(reloaded)} arrays, all verified on readback")

    weights_name = obj("cnn_weights")
    if dry_run:
        check = local_path_for(weights_name) + ".dryrun.npz"
        produce(check)
        say(f"{ENV_DRYRUN} set -- not uploading. File at {check}")
        print(OK_SENTINEL, flush=True)
        return 0

    # `cnn_production.npz` as a PINNED parent: this object is only meaningful
    # against the selection recorded there, so the pairing is recorded rather
    # than left for a reader to infer. No force anywhere -- this is a create.
    mods.gcs.ensure_artifact(weights_name, local_path_for(weights_name),
                             produce=produce, bucket=bucket,
                             parents={cnn_name: before_sha})
    say(f"uploaded {weights_name}, sha256 "
        f"{sha256_of(local_path_for(weights_name))[:12]}; parent {cnn_name} "
        f"pinned at {before_sha[:12]} and NOT modified")
    print(OK_SENTINEL, flush=True)
    return 0


if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    try:
        _status = main()
    except BackfillHalt as _exc:
        print(f"{FAIL_SENTINEL} {_exc}", flush=True)
        _status = 1
    if _status:
        sys.exit(_status)
