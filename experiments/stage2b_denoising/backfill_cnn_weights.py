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

## Why this MERGES rather than regenerates

Retraining does not reproduce those histories bit-exactly. FINDINGS
measures the drift across real-GPU retrains at 9.328e-07 and 2.385e-07 in
`best_clipped_val_mse`. Regenerating the artifact would therefore replace
numbers this project has already reported with slightly different ones --
silently, and for no gain.

So the seven arrays stage 3 wrote are carried through BYTE-IDENTICAL, and
only new keys are added. The weights come from a retrain that is verified
to reproduce stage 3's SELECTION (`best_seed`, `best_epoch`) before
anything is written, via the same `cnn_reproduction_mismatch_reason` stage
4 uses. `backfill_json` records that the weights and the histories come
from different runs, because they do, and a reader must not have to infer
it.

## What this does not carry

The manifest is written with no `fingerprint`: this driver cannot rebuild
stage 3's without duplicating its `build_fingerprint`, and a fabricated
one would be worse than none. Provenance lives in `backfill_json` instead
-- the source commit, the digest of the object as it stood BEFORE this
run, per-seed wallclock, and the reproduction outcome -- so the byte change
is auditable rather than mysterious.

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


def fetch(mods, bucket, name):
    local = local_path_for(name)
    mods.gcs.consume_validated(name, local, bucket=bucket)
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
    topo, _ = fetch(mods, bucket, obj("topologies", KMNIST_STAGING_STAGE))
    cnn_name = obj("cnn_production")
    original, cnn_local = fetch(mods, bucket, cnn_name)

    before_sha = sha256_of(cnn_local)
    original_summary = json.loads(original["summary_json"].item())
    say(f"stage {TRAIN_STAGE} selected seed={original_summary['best_seed']} "
        f"epoch={original_summary['best_epoch']}, sha256={before_sha[:12]}")
    if any(key.startswith("weights_seed") for key in original):
        raise BackfillHalt(f"{cnn_name} already carries weights; nothing to backfill")

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

    # Every original array carried through untouched; only new keys added.
    merged = dict(original)
    probe = mods.cnn.as_image_batch(inputs["val_noisy"][:4], "round-trip probe")
    for run in runs:
        blob = mods.cnn.serialise_model(run["model"])
        lossy = mods.cnn.model_round_trip_mismatch(run["model"], blob, probe)
        if lossy:
            raise BackfillHalt(f"seed={run['seed']} weights do not round-trip: {lossy}")
        merged[f"weights_seed{run['seed']}"] = blob
    say(f"serialised {len(mods.cnn.SEEDS)} models, each round-trip verified")

    merged["backfill_json"] = np.array(json.dumps({
        "what": "CNN weights added to an artifact stage 3 wrote without them",
        "weights_from": "a retrain on this run, NOT the run that produced the "
                        "histories in this file -- iterative float32 training does "
                        "not reproduce bit-exactly, so the histories were carried "
                        "through unchanged rather than regenerated",
        "reproduction": {"verified": True, "matched_on": ["best_seed", "best_epoch"],
                         "best_clipped_val_mse_difference": mse_diff,
                         "reproduced": reproduced, "original": original_summary},
        "weights_format": "equinox.tree_serialise_leaves -> uint8, load via "
                          "stage2b_cnn.deserialise_model",
        "weights_keys": [f"weights_seed{r['seed']}" for r in runs],
        "object_sha256_before_backfill": before_sha,
        "commit": head, "driver": DRIVER_FILENAME,
        "wallclock_s_per_seed": wallclock,
        "platform": {"python": sys.version, "numpy": np.__version__},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, sort_keys=True))

    # `.npz` already, so savez_compressed does not append a second one.
    def produce(path):
        np.savez_compressed(path, **merged)
        with np.load(path, allow_pickle=False) as handle:
            reloaded = {key: handle[key] for key in handle.files}
        for key, value in original.items():
            if not np.array_equal(reloaded[key], value):
                raise BackfillHalt(f"merge altered the original array {key!r}")
        say(f"wrote {len(reloaded)} arrays; all {len(original)} originals unchanged")

    if dry_run:
        check = local_path_for(cnn_name) + ".merged.npz"
        produce(check)
        say(f"{ENV_DRYRUN} set -- not uploading. Merged file at {check}")
        print(OK_SENTINEL, flush=True)
        return 0

    mods.gcs.ensure_artifact(cnn_name, local_path_for(cnn_name),
                             produce=produce, bucket=bucket, force=True)
    say(f"uploaded {cnn_name}, sha256 {sha256_of(local_path_for(cnn_name))[:12]} "
        f"(was {before_sha[:12]})")
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
