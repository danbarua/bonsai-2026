"""Stage 2B feasibility-ladder stage 1 (n=1,000): the driver that joins the
modules together for the first time.

Run by `make stage2b-ladder-stage1`, which executes it ON a Colab runtime
(`mighty-colab exec --file`) -- the file is transmitted as code, not as a
path that exists on the VM.

## What this file is allowed to do

It COMPOSES. Every computation belongs to a module that already has its own
tests; this file chooses the order, moves values between steps, and decides
pass/fail. Any formula written inline here is a defect, not a shortcut --
three of this project's worst bugs (Stage 1D's replica directions, the
`stage2a-verify` no-op gate, the single-trial-function near-miss) lived in
exactly this layer, wrapped around kernels that were themselves correct.

## How the code and the data get to the runtime

Code arrives by `git fetch` of one pinned commit from the public repo:
`datasets/` and the cached construction pickles are gitignored, so a clone
carries the pipeline but no inputs, and `git archive` of the tracked tree is
14MB -- past the Colab upload ceiling this project has already hit once.

KMNIST arrives from GCS, staged there once by `make stage2b-stage-inputs`.
All four IDX files are needed, not two: `load_mnist` opens the t10k pair
unconditionally, and topology construction goes through it. Those bytes are
never bound to a variable here. Reading them to rebuild a graph from class-0
TRAINING images is not a Stage 2B test-side result, and the `stage2b/testsplit`
guards are untouched by it.

Credentials arrive as a file whose path is passed explicitly, never read
from a default location and never printed.

## Which code ran

`BONSAI_DRIVER_SHA256` is computed by the make target from the local file it
transmits; this script hashes the CLONE's copy of itself and compares. The
executed code has no `__file__` to hash, so proving the transmitted text
matches the pinned commit has to come at it from the other side.

## Output contract

`STAGE1_OK` or `STAGE1_FAIL <reason>` on stdout, and a non-zero exit on
failure. The make target requires BOTH, because an exit code cannot
distinguish "ran and passed" from "exited cleanly without reaching its
verdict". A heartbeat line prints every 30s: `exec --timeout` bounds the gap
between OUTPUTS, not the run, so a healthy silent step is indistinguishable
from a hang.
"""
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import types

import numpy as np

# ---- identity and sentinels ----
DRIVER_FILENAME = "run_ladder_stage1.py"
LADDER_STAGE = 1
SPLIT = "train"                     # no test-side object is touched at this rung
OK_SENTINEL = "STAGE1_OK"
FAIL_SENTINEL = "STAGE1_FAIL"

# ---- bootstrap ----
REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
WORK_DIR = "/content/stage2b_stage1"
KMNIST_SUBDIR = "datasets/kmnist"   # must match build_stage1d_constructions.KMNIST_DIR
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)

# The four IDX files, mapped to their GCS `kind` tokens.
#
# The `split="train"` token in these object paths is the PIPELINE split --
# these are inputs to a training-side rung. It is not a claim about byte
# provenance. The `kind` carries provenance literally, matching the IDX
# filename, which is why the t10k pair is named `t10k` and deliberately NOT
# `test`: a `kmnist_test_images` object would read as a Stage 2B test-side
# artifact, the exact misreading `stage2b_gcs`'s split guards exist to
# prevent.
KMNIST_FILES = {
    "train-images-idx3-ubyte": "kmnist_train_images",
    "train-labels-idx1-ubyte": "kmnist_train_labels",
    "t10k-images-idx3-ubyte": "kmnist_t10k_images",
    "t10k-labels-idx1-ubyte": "kmnist_t10k_labels",
}
KMNIST_EXT = "idx"

# ---- environment variable names (set by `mighty-colab exec --env K=V`) ----
ENV_COMMIT = "BONSAI_COMMIT"
ENV_BUCKET = "BONSAI_GCS_BUCKET"
ENV_CREDENTIALS = "BONSAI_GCS_CREDENTIALS"
ENV_DRIVER_SHA = "BONSAI_DRIVER_SHA256"

# ---- run parameters ----
#
# Everything the DESIGN locks (subset sizes, seeds, alpha grid, fold count,
# encoder steps, rho threshold) is read from the modules that own it. Only
# choices this driver genuinely makes appear here.
ENCODER_WORKERS = 1     # see step 4 for why not a Pool
EVOLVE_CHUNK = 250      # 1000 / 250 = 4 equal chunks: one compiled shape, no ragged tail
FULL_GRID = 784
EXPECTED_N_ACTIVE = 505
EXPECTED_REF_IDX = 363          # nodes_T["median"], a position in 505-space
EXPECTED_FEATURE_DIM = 2 * EXPECTED_N_ACTIVE - 2    # 1008
IDENTITY_KEY = "identity"
RAW_CONDITIONS = ("raw_505", "raw_784")
HEARTBEAT_SECONDS = 30.0

SMOKE_BANNER = ("SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE, "
                "NON-INFERENTIAL, NOT A RESULT")

_RUN_T0 = time.time()
_STEP = {"name": "startup", "t0": _RUN_T0}


class Stage1Halt(Exception):
    """A gate returned a negative verdict. Distinct from a bug: the run did
    what it was asked and the answer was no."""


# ---------------------------------------------------------------- plumbing

def say(line):
    print(f"[stage1] {line}", flush=True)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


def _dumps(obj):
    return json.dumps(obj, indent=2, default=_json_default, sort_keys=True)


def _run(argv, cwd=None):
    say(f"$ {' '.join(argv)}")
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)


def _heartbeat_loop(stop):
    while not stop.wait(HEARTBEAT_SECONDS):
        now = time.time()
        print(f"[heartbeat] t={now - _RUN_T0:.0f}s step={_STEP['name']} "
              f"step_elapsed={now - _STEP['t0']:.0f}s", flush=True)


def start_heartbeat():
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(stop,), daemon=True)
    thread.start()
    return stop, thread


@contextlib.contextmanager
def timed_step(name, timings):
    """Marks the current step for the heartbeat and records its wall clock.

    Records on the way out even when the step raises, so a halted step's
    duration still reaches the report (principle 18: every stage gets its
    own measurement; one behaving linearly is not evidence another does)."""
    previous, t0 = dict(_STEP), time.time()
    _STEP.update(name=name, t0=t0)
    say(f"begin {name}")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        timings[name] = elapsed
        _STEP.update(previous)
        say(f"end {name} ({elapsed:.1f}s)")


def local_path_for(object_name):
    """One local path per object, derived mechanically.

    `ensure_artifact` trusts an existing local file when the object is
    already in the bucket, so two objects sharing a local path would give
    the second one the first one's bytes. Deriving the path from the name
    makes that collision impossible rather than merely unlikely."""
    return os.path.join(WORK_DIR, object_name.replace("/", "__"))


# ------------------------------------------------------------- bootstrap

def bootstrap_repo(commit, clone_dir=CLONE_DIR):
    """Fetch exactly `commit` from the public repo and put it on the path.

    `git clone --depth 1` fetches only the default branch's tip and cannot
    check out an arbitrary SHA; `fetch --depth 1 origin <sha>` can, and is
    what pins the run to the commit the make target verified was pushed."""
    info = {"clone_dir": clone_dir, "requested_commit": commit,
            "python": sys.version, "pip_editable": False}

    if not os.path.isdir(os.path.join(clone_dir, ".git")):
        os.makedirs(clone_dir, exist_ok=True)
        _run(["git", "init", "-q", clone_dir])
        _run(["git", "remote", "add", "origin", REPO_URL], cwd=clone_dir)
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=clone_dir)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=clone_dir)
    else:
        say(f"{clone_dir} already checked out; reusing it")

    head = _run(["git", "rev-parse", "HEAD"], cwd=clone_dir).stdout.strip()
    info["head_sha"] = head
    if head != commit:
        raise Stage1Halt(f"clone is at {head}, expected {commit}")

    # `--ignore-requires-python`: pyproject declares >=3.14 and Colab runs
    # older. Not a leap -- `stage2b_ridge` and `stage2b_cnn` have already
    # executed under this interpreter in the two GPU verify targets, so the
    # interpreter risk was retired by measurement, not waved through.
    # `--no-deps` is load-bearing: `dependencies` pins jax, and letting pip
    # resolve it would replace the runtime's CUDA build with a CPU wheel,
    # after which `device_preflight()` fails for a reason unrelated to the
    # science.
    try:
        _run([sys.executable, "-m", "pip", "install", "-e", clone_dir,
              "--no-deps", "--ignore-requires-python", "-q"])
        info["pip_editable"] = True
    except subprocess.CalledProcessError as exc:
        say(f"editable install failed ({exc.returncode}); falling back to sys.path")
        info["pip_install_stderr"] = (exc.stderr or "")[-2000:]

    return info


def add_repo_to_path(clone_dir):
    """Put a checkout's source directories on `sys.path`, idempotently.

    Owned by the import side rather than the fetch side so that
    `load_modules` is self-contained: a local test can point both at this
    repo and check the driver's whole dependency closure resolves and its
    call sites match the real signatures, without a runtime, a clone or a
    network. That check is worth a great deal here -- caller-side glue
    around correct kernels is this project's most expensive recurring bug,
    and signature drift is the cheap half of it to catch."""
    for directory in (*EXPERIMENT_DIRS, "src"):
        entry = os.path.join(clone_dir, directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def verify_driver_identity(clone_dir, expected_sha256):
    """Compare the clone's copy of this file against the hash of the file
    the make target actually transmitted.

    The executed code has no `__file__`, so this is the only non-circular
    way to show that what ran is what the pinned commit contains."""
    path = os.path.join(clone_dir, "experiments", "stage2b_denoising", DRIVER_FILENAME)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    result = {"path": path, "sha256": digest, "expected": expected_sha256,
              "matches": bool(expected_sha256) and digest == expected_sha256}
    if expected_sha256 and not result["matches"]:
        raise Stage1Halt(
            f"driver identity mismatch: the transmitted file hashes to "
            f"{expected_sha256}, the commit's copy to {digest}. The code that ran "
            f"is not the code at {os.path.basename(clone_dir)}'s pinned commit.")
    if not expected_sha256:
        say(f"{ENV_DRIVER_SHA} unset; recorded {digest} without comparison")
    return result


def load_modules(clone_dir):
    """Every repo import, in one place and in an order that matters.

    `stage2b_ridge` FIRST: it enables jax's x64 mode at import, before
    `jax.numpy` binds. `evolve_on_graph_jax` and `stage2b_cnn` do not, so a
    different order runs the graph evolution in float32 -- silently,
    plausibly, with nothing raised anywhere."""
    add_repo_to_path(clone_dir)
    import stage2b_ridge as ridge                                       # noqa: E402
    from evolve_on_graph_jax import batched_evolve_on_graph_jax         # noqa: E402
    import jax                                                          # noqa: E402
    import jax.numpy as jnp                                             # noqa: E402

    import stage2b_conditions as conditions                             # noqa: E402
    import stage2b_corruption as corruption                             # noqa: E402
    import stage2b_encoder_gate as encoder_gate                         # noqa: E402
    import stage2b_gcs as gcs                                           # noqa: E402
    import stage2b_partition as partition                               # noqa: E402
    import stage2b_stats as stats                                       # noqa: E402
    import stage2b_verify_gpu as verify_gpu                             # noqa: E402
    import stage2a_core as core                                         # noqa: E402
    import stage2a_topologies as topologies                             # noqa: E402
    import build_stage1d_constructions as s1d                           # noqa: E402
    from bonsai.data.mnist_loader import load_mnist                     # noqa: E402
    from bonsai.dynamics.learned_topology_construction import (         # noqa: E402
        _local_converged_phases)

    mods = types.SimpleNamespace(
        ridge=ridge, batched_evolve_on_graph_jax=batched_evolve_on_graph_jax,
        jax=jax, jnp=jnp, conditions=conditions, corruption=corruption,
        encoder_gate=encoder_gate, gcs=gcs, partition=partition, stats=stats,
        verify_gpu=verify_gpu, core=core, topologies=topologies, s1d=s1d,
        load_mnist=load_mnist, local_converged_phases=_local_converged_phases)

    # `stage2b_verify_gpu` puts /content at the front of sys.path at import.
    # Nothing importable is uploaded there by this target, but assert the
    # provenance rather than reasoning about it: a stale module shadowing the
    # clone is precisely the class of bug this run exists to avoid.
    for name in ("ridge", "corruption", "encoder_gate", "gcs", "partition",
                 "stats", "core", "topologies"):
        origin = getattr(mods, name).__file__
        if not os.path.abspath(origin).startswith(os.path.abspath(clone_dir)):
            raise Stage1Halt(f"module {name} resolved to {origin}, outside {clone_dir}")
    return mods


def stage_kmnist(mods, bucket, clone_dir):
    """Download the four staged IDX files into the clone's datasets dir.

    The destination is not free: `build_stage1d_constructions` computes
    KMNIST_DIR relative to its own location, so the files must land inside
    the clone, not in /content."""
    dest_dir = os.path.join(clone_dir, KMNIST_SUBDIR)
    os.makedirs(dest_dir, exist_ok=True)
    staged = {}
    for filename, kind in sorted(KMNIST_FILES.items()):
        name = mods.gcs.object_path(stage=LADDER_STAGE, condition=None, kind=kind,
                                    ext=KMNIST_EXT, split=SPLIT)
        dest = os.path.join(dest_dir, filename)
        if os.path.isfile(dest):
            say(f"{filename} already present ({os.path.getsize(dest)} bytes)")
        else:
            mods.gcs.download_file(name, dest, bucket=bucket)
            say(f"downloaded {name} -> {filename} ({os.path.getsize(dest)} bytes)")
        staged[filename] = dest
    return dest_dir, staged


# ------------------------------------------------------- artifact wrappers
#
# Two rules hold the resumability story together, and both are easy to
# break silently:
#
#   1. All expensive work happens INSIDE the produce closure. The classic
#      failure is `produce=lambda p: save(p, expensive())` where
#      `expensive()` already ran on the line above -- the skip branch then
#      skips only the write.
#   2. Callers use the value LOADED FROM DISK, never the closure's return.
#      That is the only path identical across produce / download /
#      trusted-local-file, and it is why the asserts in each step run on the
#      loaded arrays: a resumed run re-checks the bytes it downloaded.

def ensure_npz(mods, bucket, object_name, compute, force=False):
    def produce(path):
        arrays = compute()
        np.savez_compressed(path, **arrays)

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket, force=force)
    with np.load(result.local_path, allow_pickle=False) as handle:
        loaded = {key: handle[key] for key in handle.files}
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_json(mods, bucket, object_name, compute, force=False):
    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_dumps(compute()))

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket, force=force)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_text(mods, bucket, object_name, compute, force=False):
    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(compute())

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket, force=force)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = handle.read()
    say(f"artifact {result.summary()}")
    return loaded, result


def _obj(mods, kind, ext, condition=None):
    return mods.gcs.object_path(stage=LADDER_STAGE, condition=condition, kind=kind,
                                ext=ext, split=SPLIT)


# ------------------------------------------------------------------ steps

def step0_preflight(mods, record):
    """GPU present, and float64 realised ON the device rather than merely
    requested in config. `stage2b_verify_gpu` already implements exactly
    this check -- calling it beats writing a second copy that could drift."""
    try:
        mods.verify_gpu.device_preflight()
    except SystemExit as exc:
        raise Stage1Halt(f"device preflight: {exc}") from exc
    probe = mods.jnp.zeros(1, dtype=mods.jnp.float64)
    record["run"]["devices"] = [str(d) for d in mods.jax.devices()]
    record["run"]["realised_float64_dtype"] = str(probe.dtype)
    record["run"]["jax_enable_x64"] = bool(mods.jax.config.jax_enable_x64)


def step1_corpus(mods, bucket, kmnist_dir):
    def compute():
        x_train, y_train, _x_test, _y_test = mods.load_mnist(kmnist_dir, gz=False)
        part = mods.partition.Stage2BTrainingPartition(y_train)
        subsets = part.nested_development_subsets(
            size=mods.partition.STAGE2_SUBSET_SIZE,
            prefix_size=mods.partition.STAGE1_SUBSET_SIZE,
            seed=mods.partition.LADDER_SUBSET_SEED,
            stratified=mods.partition.LADDER_SUBSET_STRATIFIED)
        stage1_indices = np.asarray(subsets.stage1_indices)
        # Scaled to [0, 1] float64 in the same expression that selects the
        # rung. `corrupt_corpus`'s unit-interval guard is the backstop for
        # this, not the mechanism -- the driver satisfies it rather than
        # discovering it.
        images_01 = x_train[stage1_indices].astype(np.float64) / 255.0
        return {
            "stage1_indices": stage1_indices,
            "stage2_indices": np.asarray(subsets.stage2_indices),
            "images_01": images_01,
            "labels": np.asarray(y_train[stage1_indices]),
            "partition_summary": np.array(_dumps(part.summary())),
            "subset_summary": np.array(_dumps(subsets.summary())),
        }

    corpus, _ = ensure_npz(mods, bucket, _obj(mods, "corpus", "npz"), compute)
    n = corpus["images_01"].shape[0]
    if n != mods.partition.STAGE1_SUBSET_SIZE:
        raise Stage1Halt(f"corpus has {n} images, expected "
                         f"{mods.partition.STAGE1_SUBSET_SIZE}")
    lo, hi = float(corpus["images_01"].min()), float(corpus["images_01"].max())
    say(f"corpus n={n}, range [{lo:g}, {hi:g}], dtype {corpus['images_01'].dtype}")
    return corpus


def step1b_topologies(mods, bucket):
    def compute():
        active_indices, ink_mask_active, nodes_T, topos = \
            mods.topologies.build_all_topologies()
        meta = {
            "nodes_T": {key: int(value) for key, value in nodes_T.items()},
            "n_active": int(np.asarray(active_indices).size),
            # On a fresh clone the gitignored class0_constructions.pkl is
            # absent, so build_and_verify_T reconstructs identically but
            # SKIPS its byte-exact check against the historical artifact.
            # Which branch fired is a real caveat on this rung's numbers.
            "historical_cache_present": bool(
                os.path.exists(mods.s1d.CLASS0_CONSTRUCTIONS_PATH)),
            "historical_cache_path": str(mods.s1d.CLASS0_CONSTRUCTIONS_PATH),
        }
        arrays = {"active_indices": np.asarray(active_indices),
                  "ink_mask_active": np.asarray(ink_mask_active),
                  "summary_json": np.array(_dumps(meta))}
        for name, W in topos.items():
            arrays[f"W_{name}"] = np.asarray(W)
        return arrays

    topo, _ = ensure_npz(mods, bucket, _obj(mods, "topologies", "npz"), compute)
    meta = json.loads(topo["summary_json"].item())
    n_active = int(np.asarray(topo["active_indices"]).size)
    if n_active != EXPECTED_N_ACTIVE:
        raise Stage1Halt(f"active support has {n_active} nodes, expected "
                         f"{EXPECTED_N_ACTIVE}")
    ref_idx = int(meta["nodes_T"]["median"])
    if ref_idx != EXPECTED_REF_IDX:
        raise Stage1Halt(f"T's median-degree node is {ref_idx}, expected "
                         f"{EXPECTED_REF_IDX}")
    say(f"topologies built; n_active={n_active}, ref_idx={ref_idx} (505-space), "
        f"historical cache present: {meta['historical_cache_present']}")
    return topo, meta, ref_idx


def step2_corruption(mods, bucket, corpus):
    def compute():
        x_t, x_t_clip = mods.corruption.corrupt_corpus(
            corpus["images_01"], SPLIT, corpus["stage1_indices"],
            alpha_bar=mods.corruption.ALPHA_BAR)
        return {"x_t": x_t, "x_t_clip": x_t_clip}

    corr, _ = ensure_npz(mods, bucket, _obj(mods, "corruption", "npz"), compute)

    # Assert (a): the nesting property itself. `epsilon_for` takes no subset
    # context, so "the same epsilon through the stage-2 draw's context" is
    # true by construction and cannot fail -- what IS checkable, and what
    # the ladder actually depends on, is that stage 1's draw is a prefix of
    # stage 2's, so a rung reuses realisations rather than redrawing them.
    prefix = np.asarray(corpus["stage2_indices"])[:corpus["stage1_indices"].size]
    if not np.array_equal(np.asarray(corpus["stage1_indices"]), prefix):
        raise Stage1Halt("stage-1 indices are not a prefix of the stage-2 draw; "
                         "corruption realisations would not nest across rungs")

    # Assert (b): the one with teeth. Recompute epsilon from the ORIGINAL
    # dataset index and confirm it reproduces that row of the produced
    # corpus. Had positional indices (0..999) been passed to corrupt_corpus,
    # this fails at k=0 for every image whose drawn index is not 0.
    # array_equal, not allclose: identical arithmetic on identical bytes is
    # bit-exact, and a tolerance would blunt the only assert here that can
    # catch the bug it exists for.
    for k in (0, corpus["images_01"].shape[0] // 2, corpus["images_01"].shape[0] - 1):
        eps = mods.corruption.epsilon_for(SPLIT, int(corpus["stage1_indices"][k]))
        x_t_k, x_t_clip_k = mods.corruption.forward_corrupt(
            corpus["images_01"][k].reshape(-1), eps, mods.corruption.ALPHA_BAR)
        if not (np.array_equal(x_t_k, np.asarray(corr["x_t"])[k].reshape(-1))
                and np.array_equal(x_t_clip_k,
                                   np.asarray(corr["x_t_clip"])[k].reshape(-1))):
            raise Stage1Halt(
                f"row {k} does not reproduce from its ORIGINAL dataset index "
                f"{int(corpus['stage1_indices'][k])}; corrupt_corpus was very "
                f"likely given positional indices")
    say(f"corruption verified against original indices at rows "
        f"0/{corpus['images_01'].shape[0] // 2}/{corpus['images_01'].shape[0] - 1}")
    return corr


def step3_diagnostics(mods, bucket, corpus, corr, topo, record):
    def compute():
        diag = mods.corruption.corruption_diagnostics(
            corpus["images_01"], corr["x_t"], corr["x_t_clip"],
            topo["active_indices"], labels=corpus["labels"])
        agreement = mods.corruption.clip_rate_agreement(
            corpus["images_01"], corr["x_t"], alpha_bar=mods.corruption.ALPHA_BAR)
        identity_vector = np.asarray(diag["per_image_mse_postclip_505"])
        summary = {key: value for key, value in diag.items()
                   if not isinstance(value, np.ndarray)}
        # The active-support post-clip MSE is the identity baseline: it is
        # what "return the noisy input unchanged" scores on the task. Named
        # here so the report cannot present it as anything else.
        summary["identity_baseline_mse_505"] = summary.get("mse_postclip_505")
        summary["clip_rate_agreement"] = agreement
        summary["analytical_table"] = mods.corruption.format_analytical_table()
        return {"per_image_mse_postclip_505": identity_vector,
                "summary_json": np.array(_dumps(summary))}

    diag_npz, _ = ensure_npz(mods, bucket, _obj(mods, "corruption_diagnostics", "npz"),
                             compute)
    summary = json.loads(diag_npz["summary_json"].item())
    record["diagnostics"] = summary
    agreement = summary.get("clip_rate_agreement", {})
    outside = [key for key in ("below_zero", "above_one", "total")
               if isinstance(agreement.get(key), dict)
               and not agreement[key].get("within_tolerance", True)]
    say(f"clip rates vs analytical table: "
        f"{'all within tolerance' if not outside else 'OUTSIDE on ' + ', '.join(outside)}")
    say(f"identity baseline (post-clip active-support MSE): "
        f"{summary.get('identity_baseline_mse_505')}")
    # Diagnostic, not a gate -- DESIGN.md gives the censoring table no
    # pass/fail role. Recorded loudly so a disagreement is visible.
    if outside:
        record["diagnostics"]["clip_rate_outside_tolerance"] = outside
    return diag_npz, summary


def step4_encoder_gate(mods, bucket, corpus, corr, record):
    def compute():
        result = mods.encoder_gate.run_encoder_gate(
            corpus["images_01"],            # clean
            corr["x_t_clip"],               # noisy: the CLIPPED corpus, the model input
            np.arange(FULL_GRID),           # full grid back; step 5 restricts
            seed=mods.encoder_gate.ENCODER_SEED,
            steps=mods.encoder_gate.ENCODER_STEPS,
            n_workers=ENCODER_WORKERS)
        print(mods.encoder_gate.format_gate_log(result), flush=True)
        summary = {key: value for key, value in result.items()
                   if not isinstance(value, np.ndarray)}
        return {"thetas_clean_784": np.asarray(result["thetas_clean"]),
                "thetas_noisy_784": np.asarray(result["thetas_noisy"]),
                "delta_clean": np.asarray(result["delta_clean"]),
                "delta_noisy": np.asarray(result["delta_noisy"]),
                "gate_log": np.array(mods.encoder_gate.format_gate_log(result)),
                "summary_json": np.array(_dumps(summary))}

    # Step count in the object name, self-invalidating: a step-count change
    # (e.g. the 2026-08-06 150->1200 amendment) mints a NEW object rather
    # than silently reusing a cached artifact computed at the old count --
    # `ensure_artifact` treats an object's existence as proof its step is
    # done, so reusing one name across a semantics change would serve a
    # stale FAIL verdict forever. The old steps=150 FAIL artifact stays in
    # the bucket, untouched, as the historical record of the first real run.
    gate_kind = f"encoder_gate_s{mods.encoder_gate.ENCODER_STEPS}"
    gate, _ = ensure_npz(mods, bucket, _obj(mods, gate_kind, "npz"), compute)
    print(gate["gate_log"].item(), flush=True)
    summary = json.loads(gate["summary_json"].item())
    # Into the report on BOTH branches, and before the halt: the gate's
    # verdict is stage 1's first genuinely novel scientific output, and it
    # is wanted whichever way it lands.
    record["gates"]["encoder"] = summary
    say(f"ENCODER GATE rho = {summary['rho']!r} (threshold {summary['threshold']}) "
        f"-> {'PASS' if summary['passed'] else 'FAIL'}")
    say(f"  final-Delta clean : median {summary['median_delta_clean']!r}, "
        f"p95 {summary['p95_delta_clean']!r}")
    say(f"  final-Delta noisy : median {summary['median_delta_noisy']!r}, "
        f"p95 {summary['p95_delta_noisy']!r}")
    if not summary["passed"]:
        raise Stage1Halt("encoder gate: " + "; ".join(summary["failure_reasons"]))
    return gate


def step5_restrict(mods, gate, corr, topo):
    active_indices = np.asarray(topo["active_indices"])
    # The NOISY encoding is the feature source; the clean encoding exists
    # only for the gate's ratio. Getting this backwards would leak the
    # target into the input.
    theta0_505 = np.asarray(gate["thetas_noisy_784"])[:, active_indices]
    expected = (np.asarray(corr["x_t_clip"]).shape[0], EXPECTED_N_ACTIVE)
    if theta0_505.shape != expected:
        raise Stage1Halt(f"restricted phases are {theta0_505.shape}, expected {expected}")
    # One-image proof that slicing the gate's full-grid field equals a
    # fresh call to the same underlying encoder AT THE GATE'S OWN STEP
    # COUNT -- not `stage2a_core.encode_and_restrict`, which has no `steps`
    # parameter of its own and is hardwired to `_local_converged_phases`'s
    # bare default. That default is Stage 2A's separate, unrelated
    # convention (still 150, load-bearing for ~14 of its own already-
    # verified pipeline files) and must not move for Stage 2B's sake.
    # Comparing against it would have silently re-anchored this check to
    # the WRONG step count the moment ENCODER_STEPS diverged from
    # `_local_converged_phases`'s default -- which it now has (the ladder
    # stage-1 amendment raising ENCODER_STEPS to 1200 after the encoder
    # gate's first real FAIL). Caught by tracing the amendment's own
    # "every encoding site" requirement against this driver's actual call
    # graph, not by assumption.
    reference = mods.local_converged_phases(
        np.asarray(corr["x_t_clip"])[0], steps=mods.encoder_gate.ENCODER_STEPS,
        seed=mods.encoder_gate.ENCODER_SEED).flatten()[active_indices]
    if not np.array_equal(reference, theta0_505[0]):
        raise Stage1Halt("restricting the gate's full-grid phases does not reproduce "
                         "a fresh encode at the gate's own step count on image 0")
    say(f"restricted to {theta0_505.shape}; matches a fresh encode at "
        f"{mods.encoder_gate.ENCODER_STEPS} steps on image 0")
    return theta0_505


def step6_evolution(mods, bucket, theta0_505, topo, record):
    n = theta0_505.shape[0]
    if n % EVOLVE_CHUNK:
        raise Stage1Halt(f"{n} images does not divide into {EVOLVE_CHUNK}-row chunks; "
                         f"a ragged tail would compile a second shape")
    n_chunks = n // EVOLVE_CHUNK
    warmed = {"done": False}

    def warm_up(W):
        """One tiny call so the JIT compile is bounded by a print on each
        side. It is the only part of this step that cannot print from the
        inside, and it is lazy so a fully-cached rerun compiles nothing."""
        if warmed["done"]:
            return
        t0 = time.time()
        theta_warm, _success = mods.batched_evolve_on_graph_jax(
            mods.jnp.asarray(theta0_505[:EVOLVE_CHUNK]), W)
        mods.jax.block_until_ready(theta_warm)
        say(f"evolve warm-up compile {time.time() - t0:.1f}s (excluded from step timings)")
        warmed["done"] = True

    evolved, failures = {}, []
    for graph in mods.conditions.EVOLVED_GRAPHS:
        W_np = np.asarray(topo[f"W_{graph}"])

        def compute(graph=graph, W_np=W_np):
            W = mods.jnp.asarray(W_np)
            warm_up(W)
            thetas, flags = [], []
            for chunk in range(n_chunks):
                lo, hi = chunk * EVOLVE_CHUNK, (chunk + 1) * EVOLVE_CHUNK
                t0 = time.time()
                # Exactly two positional arguments: this binding is
                # jax.jit(jax.vmap(..., in_axes=(0, None))), so it has no
                # k_coupling parameter and returns a bare 2-tuple.
                theta_T, success = mods.batched_evolve_on_graph_jax(
                    mods.jnp.asarray(theta0_505[lo:hi]), W)
                mods.jax.block_until_ready(theta_T)
                success_np = np.asarray(success)
                say(f"evolve/{graph} chunk {chunk + 1}/{n_chunks} rows {lo}:{hi} "
                    f"ok={int(success_np.sum())}/{hi - lo} ({time.time() - t0:.2f}s)")
                thetas.append(np.asarray(theta_T))
                flags.append(success_np)
            return {"theta_T": np.concatenate(thetas),
                    "success": np.concatenate(flags)}

        loaded, _ = ensure_npz(mods, bucket,
                               _obj(mods, "theta_T", "npz",
                                    condition=mods.conditions.path_segment(graph)),
                               compute)
        evolved[graph] = loaded

        # Gate on the success FLAGS -- the batched path cannot raise
        # per-trial, so an ungated caller silently keeps garbage rows. Stage
        # 1 is a correctness rung: the expectation is zero failures.
        success = np.asarray(loaded["success"])
        n_failed = int(np.count_nonzero(~success))
        entry = {"n_images": int(success.size), "n_failed": n_failed}

        # The reference path signals failure differently: theta_T is None
        # and diag["failed"] is True, but on SUCCESS the "failed" key is
        # ABSENT rather than False -- so diag["failed"] would raise KeyError.
        # Note the `not`: the success condition is "not None AND not failed".
        theta_ref, diag = mods.core.evolve_on_graph(theta0_505[0], W_np)
        reference_ok = (theta_ref is not None) and not diag.get("failed", False)
        entry["cpu_reference"] = {
            "succeeded": bool(reference_ok),
            "recovery_step": diag.get("recovery_step"),
            "method": diag.get("method"),
            "solver_message": diag.get("solver_message"),
        }
        record["evolution"][graph] = entry
        say(f"evolve/{graph}: {n_failed} failed of {success.size}; "
            f"cpu reference ok={reference_ok}")
        if n_failed:
            failures.append(f"{graph}: {n_failed}/{success.size} solves failed")
        if not reference_ok:
            failures.append(f"{graph}: the numpy reference path failed on image 0")

    # Every graph's counts are recorded before any halt, so the report shows
    # the whole picture rather than stopping at the first bad one.
    if failures:
        raise Stage1Halt("graph evolution: " + "; ".join(failures))
    return evolved


def step7_features(mods, bucket, theta0_505, evolved, topo, corr, corpus, ref_idx):
    n = theta0_505.shape[0]
    active_indices = np.asarray(topo["active_indices"])
    features = {}

    for condition in mods.conditions.ALL_CONDITIONS:
        theta = (theta0_505 if condition == mods.conditions.PRE_EVOLUTION
                 else np.asarray(evolved[condition]["theta_T"]))

        def compute(theta=theta):
            # reference_node_features asserts the dropped pair is exactly
            # (1.0, 0.0) on every call -- the constant-column drop is
            # checked at runtime n times per condition, by the module that
            # owns it.
            return {"X": np.stack([mods.core.reference_node_features(theta[i], ref_idx)
                                   for i in range(n)])}

        loaded, _ = ensure_npz(mods, bucket,
                               _obj(mods, "features", "npz",
                                    condition=mods.conditions.path_segment(condition)),
                               compute)
        X = np.asarray(loaded["X"])
        if X.shape != (n, EXPECTED_FEATURE_DIM):
            raise Stage1Halt(f"{condition} features are {X.shape}, expected "
                             f"{(n, EXPECTED_FEATURE_DIM)}")
        features[condition] = X

    # Raw conditions are slices of already-cached arrays: no computation, so
    # no artifact. This is also why they need no path-segment vocabulary --
    # path_segment() rejects them by design.
    raw_784 = np.asarray(corr["x_t_clip"]).reshape(n, FULL_GRID)
    features["raw_784"] = raw_784
    features["raw_505"] = raw_784[:, active_indices]

    Y = np.asarray(corpus["images_01"]).reshape(n, FULL_GRID)[:, active_indices]
    say(f"features: " + ", ".join(f"{k}{v.shape}" for k, v in sorted(features.items())))
    say(f"target Y {Y.shape} (clean, active support)")
    return features, Y


def step8_ridge(mods, bucket, features, Y, y_strat, record):
    conditions = (*RAW_CONDITIONS, *mods.conditions.ALL_CONDITIONS)

    def compute_cv():
        out = {"conditions": {}, "halt_reasons": [], "order": list(conditions)}
        for condition in conditions:
            entry = {}
            X = features[condition]
            try:
                # X goes in UNSCALED: the per-fold StandardScaler lives
                # inside cross_validate_alpha, fitted on the training fold
                # only.
                entry["cv"] = mods.ridge.cross_validate_alpha(X, Y, y_strat)
            except AssertionError as exc:
                # assert_scaler_centered raises from inside svd_ridge_fit, so
                # cross_validate_alpha returns nothing at all. Catching per
                # condition is what stops one firing guard from costing the
                # other six conditions' fold margins -- which are the
                # measurement the tolerance amendment needs.
                entry["cv_error"] = f"{type(exc).__name__}: {exc}"
                out["halt_reasons"].append(f"{condition}: centering guard: {exc}")
                out["conditions"][condition] = entry
                say(f"ridge/{condition}: CENTERING GUARD FIRED -- {exc}")
                continue
            say(f"ridge/{condition}: alpha={entry['cv']['alpha']}, "
                f"fold cond={np.asarray(entry['cv']['fold_cond']).tolist()}")
            try:
                equivalence = mods.ridge.ridge_equivalence_check(X, Y, y_strat)
                entry["equivalence"] = equivalence
                say(f"ridge/{condition}: equivalence pred_diff="
                    f"{float(equivalence['max_abs_clipped_pred_diff']):.3e} "
                    f"(tol {float(equivalence['tol']):.0e}), "
                    f"alpha agrees={bool(equivalence['alpha_agrees'])}")
                if not equivalence["passed"]:
                    out["halt_reasons"].append(
                        f"{condition}: equivalence pred_diff="
                        f"{float(equivalence['max_abs_clipped_pred_diff']):.3e} "
                        f"(tol {float(equivalence['tol']):.0e}), "
                        f"alpha_jax={equivalence['alpha_jax']}, "
                        f"alpha_sklearn={equivalence['alpha_sklearn']}")
            except AssertionError as exc:
                entry["equivalence_error"] = f"{type(exc).__name__}: {exc}"
                out["halt_reasons"].append(
                    f"{condition}: equivalence centering guard: {exc}")
            out["conditions"][condition] = entry
        return out

    cv_json, _ = ensure_json(mods, bucket, _obj(mods, "ridge_cv", "json"), compute_cv)
    record["gates"]["ridge"] = cv_json
    if cv_json["halt_reasons"]:
        raise Stage1Halt("ridge: " + "; ".join(cv_json["halt_reasons"]))

    def compute_final():
        arrays, meta = {}, {"alphas": {}, "centering_margins": {}}
        for condition in conditions:
            alpha = cv_json["conditions"][condition]["cv"]["alpha"]
            fit, scaler = mods.ridge.fit_final(features[condition], Y, alpha)
            # alpha_index=0, not cv["alpha_index"]: fit_final decomposes for
            # the single selected alpha, so fit["W"] has length 1 and a grid
            # index would read the wrong slab (or raise).
            prediction = mods.ridge.ridge_predict(
                fit, scaler.transform(features[condition]), 0)
            arrays[f"mse_{condition}"] = mods.ridge.clipped_per_image_mse(prediction, Y)
            meta["alphas"][condition] = alpha
            meta["centering_margins"][condition] = fit.get("centering_margin")
        arrays["summary_json"] = np.array(_dumps(meta))
        return arrays

    final, _ = ensure_npz(mods, bucket, _obj(mods, "ridge_final", "npz"), compute_final)
    record["ridge_final"] = json.loads(final["summary_json"].item())
    return cv_json, final


def step9_stats_smoke(mods, bucket, final, diag_npz, corpus, record):
    # Six keys, not seven. validate_conditions requires pre_evolution, the
    # four evolved graphs and the identity baseline; raw_505/raw_784 are
    # ridge conditions that belong to no comparison family in DESIGN.md, so
    # they are reported descriptively and passed to nothing.
    mse_by_condition = {mods.conditions.PRE_EVOLUTION:
                        np.asarray(final[f"mse_{mods.conditions.PRE_EVOLUTION}"])}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        mse_by_condition[graph] = np.asarray(final[f"mse_{graph}"])
    mse_by_condition[IDENTITY_KEY] = np.asarray(diag_npz["per_image_mse_postclip_505"])

    def compute():
        return mods.stats.run_stage2b_inference(
            mse_by_condition, np.asarray(corpus["labels"]),
            identity_key=IDENTITY_KEY)

    inference, _ = ensure_json(mods, bucket, _obj(mods, "stats_smoke", "json"), compute)

    def compute_text():
        descriptive = {condition: float(np.mean(features_mse))
                       for condition, features_mse in sorted(mse_by_condition.items())}
        for condition in RAW_CONDITIONS:
            key = f"mse_{condition}"
            if key in final:
                descriptive[condition] = float(np.mean(np.asarray(final[key])))
        return (f"{SMOKE_BANNER}\n\n"
                f"Mean per-image clipped MSE, in-sample, all ridge conditions:\n"
                f"{_dumps(descriptive)}\n\n"
                f"Full inference output:\n{_dumps(inference)}\n")

    ensure_text(mods, bucket, _obj(mods, "stats_smoke", "txt"), compute_text)
    record["stats_smoke"] = inference
    say("stats machinery exercised end to end (in-sample; not a result)")
    return inference


def step10_report(mods, bucket, record):
    """Written from main()'s finally, so a halted run still leaves one.

    force=True here and ONLY here: every other artifact is deterministic
    given the pinned commit, so skipping an existing one is right. The
    report describes THIS run, and skip-if-exists would silently preserve a
    previous run's verdict."""
    def compute_json():
        return record

    def compute_text():
        lines = [f"Stage 2B feasibility-ladder stage 1 report",
                 f"commit: {record['run'].get('head_sha')}",
                 f"verdict: {record.get('verdict')}", ""]
        if record.get("halt_reason"):
            lines += [f"halt reason: {record['halt_reason']}", ""]
        lines += ["timings (s):", _dumps(record.get("timings", {})), "",
                  "full record:", _dumps(record), "",
                  str(record.get("verdict", FAIL_SENTINEL))]
        return "\n".join(lines) + "\n"

    ensure_json(mods, bucket, _obj(mods, "stage1_report", "json"), compute_json,
                force=True)
    ensure_text(mods, bucket, _obj(mods, "stage1_report", "txt"), compute_text,
                force=True)


# ------------------------------------------------------------------- main

def new_record():
    return {"run": {}, "timings": {}, "gates": {}, "evolution": {}, "artifacts": {},
            "verdict": None, "halt_reason": None}


def main():
    stop, heartbeat = start_heartbeat()
    record, status, mods, bucket = new_record(), 0, None, None
    try:
        commit = os.environ[ENV_COMMIT]
        credentials = os.environ[ENV_CREDENTIALS]
        bucket_name = os.environ.get(ENV_BUCKET) or None

        with timed_step("bootstrap", record["timings"]):
            record["run"].update(bootstrap_repo(commit))
            record["run"]["driver_identity"] = verify_driver_identity(
                CLONE_DIR, os.environ.get(ENV_DRIVER_SHA))
            mods = load_modules(CLONE_DIR)
            record["run"]["package_versions"] = package_versions()
            os.makedirs(WORK_DIR, exist_ok=True)
            bucket = mods.gcs.get_bucket(name=bucket_name, credentials=credentials)
            record["run"]["bucket"] = str(bucket.name)
            record["run"]["credentials_path"] = credentials
            record["run"]["checksum_backend"] = mods.gcs.checksum_backend()

        with timed_step("stage_kmnist", record["timings"]):
            kmnist_dir, staged = stage_kmnist(mods, bucket, CLONE_DIR)
            record["run"]["kmnist"] = {name: os.path.getsize(path)
                                       for name, path in sorted(staged.items())}

        with timed_step("0_preflight", record["timings"]):
            step0_preflight(mods, record)
        with timed_step("1_corpus", record["timings"]):
            corpus = step1_corpus(mods, bucket, kmnist_dir)
        with timed_step("1b_topologies", record["timings"]):
            topo, topo_meta, ref_idx = step1b_topologies(mods, bucket)
            record["run"]["topologies"] = topo_meta
        with timed_step("2_corruption", record["timings"]):
            corr = step2_corruption(mods, bucket, corpus)
        with timed_step("3_diagnostics", record["timings"]):
            diag_npz, _diag_summary = step3_diagnostics(mods, bucket, corpus, corr,
                                                        topo, record)
        with timed_step("4_encoder_gate", record["timings"]):
            gate = step4_encoder_gate(mods, bucket, corpus, corr, record)
        with timed_step("5_restrict", record["timings"]):
            theta0_505 = step5_restrict(mods, gate, corr, topo)
        with timed_step("6_evolution", record["timings"]):
            evolved = step6_evolution(mods, bucket, theta0_505, topo, record)
        with timed_step("7_features", record["timings"]):
            features, Y = step7_features(mods, bucket, theta0_505, evolved, topo,
                                         corr, corpus, ref_idx)
        with timed_step("8_ridge", record["timings"]):
            _cv_json, final = step8_ridge(mods, bucket, features, Y,
                                          np.asarray(corpus["labels"]), record)
        with timed_step("9_stats_smoke", record["timings"]):
            step9_stats_smoke(mods, bucket, final, diag_npz, corpus, record)

        record["verdict"] = OK_SENTINEL
    except Stage1Halt as halt:
        record.update(verdict=FAIL_SENTINEL, halt_reason=str(halt))
        status = 1
    except SystemExit as exc:
        record.update(verdict=FAIL_SENTINEL, halt_reason=f"SystemExit: {exc}")
        status = 1
    except BaseException as exc:                    # noqa: BLE001 - reported, not swallowed
        traceback.print_exc()
        record.update(verdict=FAIL_SENTINEL,
                      halt_reason=f"{type(exc).__name__}: {exc}")
        status = 1
    finally:
        stop.set()
        heartbeat.join(timeout=2)
        record["timings"]["total"] = time.time() - _RUN_T0
        if mods is not None and bucket is not None:
            try:
                step10_report(mods, bucket, record)
            except Exception as exc:                # noqa: BLE001
                print(f"[stage1] report FAILED to write: {type(exc).__name__}: {exc}",
                      flush=True)

    verdict = record["verdict"] or FAIL_SENTINEL
    print(verdict if status == 0 else f"{verdict} {record['halt_reason']}", flush=True)
    return status


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version}
    for package in ("numpy", "scipy", "scikit-learn", "jax", "jaxlib", "diffrax",
                    "google-cloud-storage", "google-crc32c"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


# Runs as a script, and also when a remote kernel gives this code some other
# __name__ -- in which case the presence of BONSAI_COMMIT is what says "this
# is the runtime, go". Importing this module locally satisfies neither, so a
# test can read its constants without side effects.
#
# Success returns normally rather than through sys.exit(0): under a kernel,
# SystemExit surfaces as a raised exception, and a success that looks like
# one to the caller is worse than a missing exit code. Failure does exit
# non-zero, on top of the sentinel line.
if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    _status = main()
    if _status:
        sys.exit(_status)
