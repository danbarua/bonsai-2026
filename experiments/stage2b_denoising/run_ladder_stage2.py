"""Stage 2B feasibility-ladder stage 2 (n=5,000 development subset): the
second rung, adding runtime/feature-validity measurement at scale, the
production SVD's own condition-number diagnostic, ridge-grid behaviour,
the ladder's second real-data ridge equivalence gate, and the first CNN
training against real data.

Same architecture and discipline as `run_ladder_stage1.py`: composes
existing, verified modules and implements nothing itself; runs on a
Colab runtime that fetches one pinned commit of this repo rather than
being uploaded with its dependencies; every artifact through
`ensure_artifact` under `stage2b/train/stage2/`, so a dead session
resumes having lost at most one step. `STAGE2_OK` / `STAGE2_FAIL` on
stdout, non-zero exit on failure -- the make target requires both.

## What stage 2 adds over stage 1

- The full 5,000-image development subset (stage 1's 1,000 is its
  prefix, checked explicitly, not merely trusted from construction).
- Encoding at ladder scale, recorded as a DIAGNOSTIC (max/p95 final-Delta
  and measured per-image cost) -- no gate. Stage 1 already established
  the encoder converges at ENCODER_STEPS=1200; stage 2 is not the place
  to invent a second gate on the same question.
- The production SVD's own condition number, per fold per condition --
  free from the decomposition `cross_validate_alpha` already computes,
  reported rather than recomputed via a second `cond()` call.
- Ridge-grid behaviour: selected alpha and the full validation-MSE curve
  per condition, with edge-of-grid selections flagged.
- The ladder's SECOND real-data ridge equivalence gate (stage 1 was the
  first) -- HALTS on failure, same as stage 1.
- CNN development: the corrupted 5,000-image subset as training input,
  early-stopped on the LOCKED 6,000-image validation partition (not a
  held-out slice of the dev subset), best of three seeds.

## What stage 2 explicitly does not halt on

Per the amendment's own halt/record distinction: condition numbers,
final-Delta's tail, grid-edge alpha selections, and the CNN's standing
against the identity baseline are all RECORDED FACT, not gates. Only
four things halt this stage: the ridge equivalence gate, the
scaler-centering guard, a graph-evolution success-flag failure, and any
non-finite feature. No new gate is invented mid-ladder.

## Explicitly out of scope

The official KMNIST test set (untouchable until feasibility stage 4).
Any confirmatory statistic. Any FINDINGS conclusion beyond mechanical or
development reporting -- this stage produces measurements, not a result.
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import types
import contextlib

import numpy as np

# ---- identity and sentinels ----
DRIVER_FILENAME = "run_ladder_stage2.py"
LADDER_STAGE = 2
KMNIST_STAGING_STAGE = 1   # staged once by stage_kmnist_inputs.py under
                           # stage=1; every later ladder stage downloads
                           # from THAT object path, never re-stages. Must
                           # equal run_ladder_stage1.LADDER_STAGE -- checked
                           # by value in tests/test_stage2b_ladder_stage2.py,
                           # not imported (see KMNIST_FILES below for why).
SPLIT = "train"
OK_SENTINEL = "STAGE2_OK"
FAIL_SENTINEL = "STAGE2_FAIL"

# ---- bootstrap ----
REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
WORK_DIR = "/content/stage2b_stage2"
KMNIST_SUBDIR = "datasets/kmnist"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)
# Duplicated from run_ladder_stage1.py rather than imported -- deliberately
# reversed from an earlier version of this file that imported these from
# run_ladder_stage1 at module scope via a `__file__`-relative sys.path
# insert. That failed on the FIRST real run: `mighty-colab exec -f script`
# transmits this file's TEXT directly into an existing IPython kernel cell
# rather than running it as a script or importing it as a module, so
# `__file__` is undefined there (confirmed: `NameError: name '__file__' is
# not defined`, before main() ever started) -- and even with that fixed,
# run_ladder_stage1.py does not exist anywhere on the exec'd kernel's
# filesystem until bootstrap_repo() has cloned the repo, which happens
# INSIDE main(), after every module-scope statement has already run.
# Nothing but stdlib+numpy is available at module-import time under this
# execution model, not even another file from this same repository -- the
# same constraint stage 1's driver was self-contained under from the start.
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
ENCODER_WORKERS = 1     # single-process: see load_modules' note on why a
                         # Pool is not used under this execution model
EVOLVE_CHUNK = 250       # 5000 / 250 = 20 exact chunks, same size stage 1 used
FULL_GRID = 784
EXPECTED_N_ACTIVE = 505
EXPECTED_REF_IDX = 363
EXPECTED_FEATURE_DIM = 2 * EXPECTED_N_ACTIVE - 2    # 1008
IDENTITY_KEY = "identity"
RAW_CONDITIONS = ("raw_505", "raw_784")
HEARTBEAT_SECONDS = 30.0
STAGE3_FIT_N = 54_000    # DESIGN.md's stage-3 fit-side size (54k fit / 6k validation)

# Stats smoke is optional at this rung -- run only if projected cost from
# stage 1's own MEASURED number stays under budget. Not re-derived from
# first principles: a linear projection off one real measurement, exactly
# the kind of number principle 18 says to keep separate per stage rather
# than assume.
STATS_SMOKE_STAGE1_REFERENCE_S = 10.86
STATS_SMOKE_STAGE1_REFERENCE_N = 1000
STATS_SMOKE_BUDGET_S = 60.0

SMOKE_BANNER = ("SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE, "
                "NON-INFERENTIAL, NOT A RESULT")

_RUN_T0 = time.time()
_STEP = {"name": "startup", "t0": _RUN_T0}


class Stage2Halt(Exception):
    """A gate returned a negative verdict. Distinct from a bug: the run did
    what it was asked and the answer was no."""


# ---------------------------------------------------------------- plumbing

def say(line):
    print(f"[stage2] {line}", flush=True)


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
    return os.path.join(WORK_DIR, object_name.replace("/", "__"))


# ------------------------------------------------------------- bootstrap

def bootstrap_repo(commit, clone_dir=CLONE_DIR):
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
        raise Stage2Halt(f"clone is at {head}, expected {commit}")

    try:
        _run([sys.executable, "-m", "pip", "install", "-e", clone_dir,
              "--no-deps", "--ignore-requires-python", "-q"])
        info["pip_editable"] = True
    except subprocess.CalledProcessError as exc:
        say(f"editable install failed ({exc.returncode}); falling back to sys.path")
        info["pip_install_stderr"] = (exc.stderr or "")[-2000:]

    return info


def add_repo_to_path(clone_dir):
    for directory in (*EXPERIMENT_DIRS, "src"):
        entry = os.path.join(clone_dir, directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def verify_driver_identity(clone_dir, expected_sha256):
    path = os.path.join(clone_dir, "experiments", "stage2b_denoising", DRIVER_FILENAME)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    result = {"path": path, "sha256": digest, "expected": expected_sha256,
              "matches": bool(expected_sha256) and digest == expected_sha256}
    if expected_sha256 and not result["matches"]:
        raise Stage2Halt(
            f"driver identity mismatch: the transmitted file hashes to "
            f"{expected_sha256}, the commit's copy to {digest}. The code that ran "
            f"is not the code at {os.path.basename(clone_dir)}'s pinned commit.")
    if not expected_sha256:
        say(f"{ENV_DRIVER_SHA} unset; recorded {digest} without comparison")
    return result


def load_modules(clone_dir):
    """Every repo import, in one place and in an order that matters.

    `stage2b_ridge` FIRST: it enables jax's x64 mode at import. Adds
    `stage2b_cnn` over stage 1's closure -- the only new dependency this
    stage introduces (equinox, optax; installed by the make target
    alongside jax/diffrax, not assumed already present).

    `n_workers` for the encoder stays at 1, matching stage 1's own
    decision, not revisited casually here: a `multiprocessing.Pool` under
    `spawn` needs a picklable `__main__` module to re-import in each
    child, and code delivered via `mighty-colab exec --file` is exec'd
    into an existing kernel's namespace, not run as a standalone script
    with a real `__main__` guard -- which is precisely why this driver's
    OWN entry guard checks `BONSAI_COMMIT` as well as `__name__`. Whether
    a worker Pool is actually safe under that execution model is
    unverified; getting a correct single-threaded measurement of encode
    cost at this scale is worth more here than a speed guess that risks
    a silent multiprocessing failure."""
    add_repo_to_path(clone_dir)
    import stage2b_ridge as ridge                                       # noqa: E402
    from evolve_on_graph_jax import batched_evolve_on_graph_jax         # noqa: E402
    import jax                                                          # noqa: E402
    import jax.numpy as jnp                                             # noqa: E402

    import stage2b_cnn as cnn                                           # noqa: E402
    import stage2b_conditions as conditions                             # noqa: E402
    import stage2b_corruption as corruption                             # noqa: E402
    import stage2b_encoder_gate as encoder_gate                         # noqa: E402
    import stage2b_gcs as gcs                                           # noqa: E402
    import stage2b_partition as partition                               # noqa: E402
    import stage2b_stats as stats                                       # noqa: E402
    import stage2b_verify_gpu as verify_gpu                             # noqa: E402
    import stage2a_core as core                                         # noqa: E402
    import stage2a_topologies as topologies                             # noqa: E402
    from bonsai.data.mnist_loader import load_mnist                     # noqa: E402
    from bonsai.dynamics.learned_topology_construction import (         # noqa: E402
        _local_converged_phases)

    mods = types.SimpleNamespace(
        ridge=ridge, batched_evolve_on_graph_jax=batched_evolve_on_graph_jax,
        jax=jax, jnp=jnp, cnn=cnn, conditions=conditions, corruption=corruption,
        encoder_gate=encoder_gate, gcs=gcs, partition=partition, stats=stats,
        verify_gpu=verify_gpu, core=core, topologies=topologies,
        load_mnist=load_mnist, local_converged_phases=_local_converged_phases)

    for name in ("ridge", "cnn", "corruption", "encoder_gate", "gcs", "partition",
                 "stats", "core", "topologies"):
        origin = getattr(mods, name).__file__
        if not os.path.abspath(origin).startswith(os.path.abspath(clone_dir)):
            raise Stage2Halt(f"module {name} resolved to {origin}, outside {clone_dir}")
    return mods


def stage_kmnist(mods, bucket, clone_dir):
    """Downloads the four IDX files staged once by `stage_kmnist_inputs.py`
    under `stage=1` -- reused verbatim by every later ladder stage, never
    re-staged, per the same object path stage 1 itself reads."""
    dest_dir = os.path.join(clone_dir, KMNIST_SUBDIR)
    os.makedirs(dest_dir, exist_ok=True)
    staged = {}
    for filename, kind in sorted(KMNIST_FILES.items()):
        name = mods.gcs.object_path(stage=KMNIST_STAGING_STAGE, condition=None, kind=kind,
                                    ext=KMNIST_EXT, split=SPLIT)
        dest = os.path.join(dest_dir, filename)
        if os.path.isfile(dest):
            say(f"{filename} already present ({os.path.getsize(dest)} bytes)")
        else:
            # Through the central validated consume path, not a raw
            # download -- there is exactly one way bytes get from GCS into
            # a consumer here, and it validates. `require_manifest=False`
            # by name: the IDX objects were staged under the stage-1 prefix
            # before the fingerprint contract existed. See stage2b_gcs's
            # legacy policy.
            manifest, _ = mods.gcs.consume_validated(name, dest, bucket=bucket,
                                                     require_manifest=False)
            say(f"downloaded {name} -> {filename} ({os.path.getsize(dest)} bytes)"
                f"{'' if manifest is None else ', manifest validated'}")
        staged[filename] = dest
    return dest_dir, staged


# ------------------------------------------------------- artifact wrappers

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


def _obj(mods, kind, ext, condition=None, stage=LADDER_STAGE):
    return mods.gcs.object_path(stage=stage, condition=condition, kind=kind,
                                ext=ext, split=SPLIT)


# ------------------------------------------------------------------ steps

def step0_preflight(mods, record):
    try:
        mods.verify_gpu.device_preflight()
    except SystemExit as exc:
        raise Stage2Halt(f"device preflight: {exc}") from exc
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
        stage2_indices = np.asarray(subsets.stage2_indices)
        stage1_indices = np.asarray(subsets.stage1_indices)
        images_01 = x_train[stage2_indices].astype(np.float64) / 255.0
        labels = np.asarray(y_train[stage2_indices])

        validation_indices = np.asarray(part.validation_indices)
        validation_images_01 = x_train[validation_indices].astype(np.float64) / 255.0
        validation_labels = np.asarray(part.validation_labels())

        return {
            "stage1_indices": stage1_indices, "stage2_indices": stage2_indices,
            "images_01": images_01, "labels": labels,
            "validation_indices": validation_indices,
            "validation_images_01": validation_images_01,
            "validation_labels": validation_labels,
            "partition_summary": np.array(_dumps(part.summary())),
            "subset_summary": np.array(_dumps(subsets.summary())),
        }

    corpus, _ = ensure_npz(mods, bucket, _obj(mods, "corpus", "npz"), compute)
    n = corpus["images_01"].shape[0]
    if n != mods.partition.STAGE2_SUBSET_SIZE:
        raise Stage2Halt(f"corpus has {n} images, expected "
                         f"{mods.partition.STAGE2_SUBSET_SIZE}")
    n_val = np.asarray(corpus["validation_images_01"]).shape[0]
    if n_val != mods.partition.N_VALIDATION:
        raise Stage2Halt(f"validation partition has {n_val} images, expected "
                         f"{mods.partition.N_VALIDATION}")
    # Explicit prefix-property assert. `nested_development_subsets` makes
    # this true by construction (stage1_indices is a VIEW of
    # stage2_indices[:n]), but the spec asks for it checked here too --
    # defense in depth against a future change to how the artifact is
    # cached (a view's aliasing does not survive an npz round-trip).
    prefix_len = mods.partition.STAGE1_SUBSET_SIZE
    if not np.array_equal(np.asarray(corpus["stage1_indices"]),
                          np.asarray(corpus["stage2_indices"])[:prefix_len]):
        raise Stage2Halt("stage-1 indices are not a prefix of stage-2's draw")
    say(f"corpus n={n}, validation n={n_val}, prefix property holds")
    return corpus


def step1b_topologies(mods, bucket):
    """Reused directly from stage 1's own cached artifact, not rebuilt.
    Topologies depend on nothing about which images are processed -- the
    same construction at every ladder stage -- so reading stage 1's
    already-verified object guarantees byte-identical active_indices and
    graphs, not merely "the same by construction." `ensure_npz` still
    falls back to building it if the object is somehow absent (e.g. this
    stage run before stage 1 ever has)."""
    def compute():
        active_indices, ink_mask_active, nodes_T, topos = \
            mods.topologies.build_all_topologies()
        meta = {
            "nodes_T": {key: int(value) for key, value in nodes_T.items()},
            "n_active": int(np.asarray(active_indices).size),
        }
        arrays = {"active_indices": np.asarray(active_indices),
                  "ink_mask_active": np.asarray(ink_mask_active),
                  "summary_json": np.array(_dumps(meta))}
        for name, W in topos.items():
            arrays[f"W_{name}"] = np.asarray(W)
        return arrays

    name = _obj(mods, "topologies", "npz", stage=KMNIST_STAGING_STAGE)
    topo, _ = ensure_npz(mods, bucket, name, compute)
    meta = json.loads(topo["summary_json"].item())
    n_active = int(np.asarray(topo["active_indices"]).size)
    if n_active != EXPECTED_N_ACTIVE:
        raise Stage2Halt(f"active support has {n_active} nodes, expected "
                         f"{EXPECTED_N_ACTIVE}")
    ref_idx = int(meta["nodes_T"]["median"])
    if ref_idx != EXPECTED_REF_IDX:
        raise Stage2Halt(f"T's median-degree node is {ref_idx}, expected "
                         f"{EXPECTED_REF_IDX}")
    say(f"topologies reused from stage {KMNIST_STAGING_STAGE}'s artifact; "
        f"n_active={n_active}, ref_idx={ref_idx} (505-space)")
    return topo, meta, ref_idx


def step2_corruption(mods, bucket, corpus):
    def compute():
        x_t, x_t_clip = mods.corruption.corrupt_corpus(
            corpus["images_01"], SPLIT, corpus["stage2_indices"],
            alpha_bar=mods.corruption.ALPHA_BAR)
        return {"x_t": x_t, "x_t_clip": x_t_clip}

    corr, _ = ensure_npz(mods, bucket, _obj(mods, "corruption", "npz"), compute)

    n = corpus["images_01"].shape[0]
    for k in (0, n // 2, n - 1):
        eps = mods.corruption.epsilon_for(SPLIT, int(corpus["stage2_indices"][k]))
        x_t_k, x_t_clip_k = mods.corruption.forward_corrupt(
            corpus["images_01"][k].reshape(-1), eps, mods.corruption.ALPHA_BAR)
        if not (np.array_equal(x_t_k, np.asarray(corr["x_t"])[k].reshape(-1))
                and np.array_equal(x_t_clip_k,
                                   np.asarray(corr["x_t_clip"])[k].reshape(-1))):
            raise Stage2Halt(
                f"row {k} does not reproduce from its ORIGINAL dataset index "
                f"{int(corpus['stage2_indices'][k])}; corrupt_corpus was very "
                f"likely given positional indices")

    # Cross-stage spot-check: stage 1's own cached corruption artifact,
    # compared bit-exact at the prefix rows. Proves the same corruption
    # REALIZATION reproduces across ladder rungs, not merely that this
    # rung's own formula is internally consistent (that is what the loop
    # above already checks).
    stage1_corr_name = _obj(mods, "corruption", "npz", stage=1)
    stage1_local = local_path_for(stage1_corr_name)
    # Central validated consume, with the pre-contract opt-out named:
    # stage 1's corruption artifact is completed-rung history, written
    # before the fingerprint contract. See stage2b_gcs's legacy policy --
    # a refusal here would mean new code reached for old history, and this
    # reach is deliberate and documented.
    mods.gcs.consume_validated(stage1_corr_name, stage1_local, bucket=bucket,
                               require_manifest=False)
    with np.load(stage1_local) as handle:
        stage1_x_t_clip = handle["x_t_clip"]
    n_prefix = corpus["stage1_indices"].size
    for k in (0, n_prefix // 2, n_prefix - 1):
        if not np.array_equal(np.asarray(corr["x_t_clip"])[k], stage1_x_t_clip[k]):
            raise Stage2Halt(
                f"row {k} disagrees with stage 1's own cached corruption -- the "
                f"same corruption realization did not reproduce across ladder rungs")
    say(f"corruption verified: own-formula spot-check at rows 0/{n // 2}/{n - 1}; "
        f"cross-stage bit-exact match with stage 1 at rows 0/{n_prefix // 2}/{n_prefix - 1}")
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
        summary["identity_baseline_mse_505"] = summary.get("mse_postclip_505")
        summary["clip_rate_agreement"] = agreement
        summary["clip_rate_agreement_note"] = (
            "n=5,000 tightens the Monte Carlo SE by sqrt(5) relative to stage 1's "
            "n=1,000 -- a genuinely stronger check, same tolerance rule.")
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
    say(f"identity baseline (post-clip active-support MSE, dev subset): "
        f"{summary.get('identity_baseline_mse_505')}")
    if outside:
        record["diagnostics"]["clip_rate_outside_tolerance"] = outside
    return diag_npz, summary


def step4_encode(mods, bucket, corr, record):
    """Noisy-only encoding, at ladder scale. Clean images are ridge/CNN
    targets and are never encoded anywhere in this pipeline. No gate --
    stage 1 already established convergence at ENCODER_STEPS=1200; this
    step exists to MEASURE cost and tail behaviour at 5x scale, recorded
    as a diagnostic fact for stage-3 planning, not re-litigated as a
    pass/fail question."""
    steps_now = mods.encoder_gate.ENCODER_STEPS

    def compute():
        t0 = time.time()
        thetas, deltas = mods.encoder_gate.encode_with_final_delta_batch(
            corr["x_t_clip"], np.arange(FULL_GRID), seed=mods.encoder_gate.ENCODER_SEED,
            steps=steps_now, n_workers=ENCODER_WORKERS)
        elapsed = time.time() - t0
        n = corr["x_t_clip"].shape[0]
        summary = {
            "steps": steps_now, "n_images": n, "elapsed_s": elapsed,
            "per_image_s": elapsed / n, "n_workers": ENCODER_WORKERS,
            "max_delta": float(np.max(deltas)),
            "p95_delta": float(np.percentile(deltas, 95)),
            "median_delta": float(np.median(deltas)),
            "n_nonfinite_theta": int(np.sum(~np.isfinite(thetas))),
            "n_nonfinite_delta": int(np.sum(~np.isfinite(deltas))),
        }
        return {"thetas_noisy_784": np.asarray(thetas), "deltas": np.asarray(deltas),
                "summary_json": np.array(_dumps(summary))}

    # Step-count in the object name, same self-invalidation discipline as
    # stage 1's encoder-gate artifact: a future ENCODER_STEPS change mints
    # a new object rather than silently resuming a stale one.
    kind = f"encode_diagnostic_s{steps_now}"
    gate, _ = ensure_npz(mods, bucket, _obj(mods, kind, "npz"), compute)
    summary = json.loads(gate["summary_json"].item())
    record["encode_diagnostic"] = summary
    say(f"ENCODE (noisy only, {summary['steps']} steps, {summary['n_workers']} worker(s)): "
        f"{summary['per_image_s'] * 1000:.3f} ms/image, {summary['elapsed_s']:.1f}s total "
        f"for n={summary['n_images']}")
    say(f"  final-Delta: max {summary['max_delta']!r}, p95 {summary['p95_delta']!r}, "
        f"median {summary['median_delta']!r}")
    if summary["n_nonfinite_theta"] or summary["n_nonfinite_delta"]:
        say(f"  WARNING: non-finite encode output (theta={summary['n_nonfinite_theta']}, "
            f"delta={summary['n_nonfinite_delta']}) -- will be caught by the features "
            f"finiteness check downstream")
    return gate


def step5_restrict(mods, gate, corr, topo):
    active_indices = np.asarray(topo["active_indices"])
    theta0_505 = np.asarray(gate["thetas_noisy_784"])[:, active_indices]
    expected = (np.asarray(corr["x_t_clip"]).shape[0], EXPECTED_N_ACTIVE)
    if theta0_505.shape != expected:
        raise Stage2Halt(f"restricted phases are {theta0_505.shape}, expected {expected}")
    reference = mods.local_converged_phases(
        np.asarray(corr["x_t_clip"])[0], steps=mods.encoder_gate.ENCODER_STEPS,
        seed=mods.encoder_gate.ENCODER_SEED).flatten()[active_indices]
    if not np.array_equal(reference, theta0_505[0]):
        raise Stage2Halt("restricting the cached full-grid phases does not reproduce "
                         "a fresh encode at the same step count on image 0")
    say(f"restricted to {theta0_505.shape}; matches a fresh encode at "
        f"{mods.encoder_gate.ENCODER_STEPS} steps on image 0")
    return theta0_505


def step6_evolution(mods, bucket, theta0_505, topo, record):
    n = theta0_505.shape[0]
    if n % EVOLVE_CHUNK:
        raise Stage2Halt(f"{n} images does not divide into {EVOLVE_CHUNK}-row chunks")
    n_chunks = n // EVOLVE_CHUNK
    warmed = {"done": False}

    def warm_up(W):
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
                theta_T, success = mods.batched_evolve_on_graph_jax(
                    mods.jnp.asarray(theta0_505[lo:hi]), W)
                mods.jax.block_until_ready(theta_T)
                success_np = np.asarray(success)
                if chunk % 5 == 0 or chunk == n_chunks - 1:
                    say(f"evolve/{graph} chunk {chunk + 1}/{n_chunks} rows {lo}:{hi} "
                        f"ok={int(success_np.sum())}/{hi - lo} ({time.time() - t0:.2f}s)")
                thetas.append(np.asarray(theta_T))
                flags.append(success_np)
            return {"theta_T": np.concatenate(thetas), "success": np.concatenate(flags)}

        loaded, _ = ensure_npz(mods, bucket,
                               _obj(mods, "theta_T", "npz",
                                    condition=mods.conditions.path_segment(graph)),
                               compute)
        evolved[graph] = loaded

        success = np.asarray(loaded["success"])
        n_failed = int(np.count_nonzero(~success))
        entry = {"n_images": int(success.size), "n_failed": n_failed}

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

    if failures:
        raise Stage2Halt("graph evolution: " + "; ".join(failures))
    return evolved


def step7_features(mods, bucket, theta0_505, evolved, topo, corr, corpus, ref_idx):
    n = theta0_505.shape[0]
    active_indices = np.asarray(topo["active_indices"])
    features = {}

    for condition in mods.conditions.ALL_CONDITIONS:
        theta = (theta0_505 if condition == mods.conditions.PRE_EVOLUTION
                 else np.asarray(evolved[condition]["theta_T"]))

        def compute(theta=theta):
            return {"X": np.stack([mods.core.reference_node_features(theta[i], ref_idx)
                                   for i in range(n)])}

        loaded, _ = ensure_npz(mods, bucket,
                               _obj(mods, "features", "npz",
                                    condition=mods.conditions.path_segment(condition)),
                               compute)
        X = np.asarray(loaded["X"])
        if X.shape != (n, EXPECTED_FEATURE_DIM):
            raise Stage2Halt(f"{condition} features are {X.shape}, expected "
                             f"{(n, EXPECTED_FEATURE_DIM)}")
        features[condition] = X

    raw_784 = np.asarray(corr["x_t_clip"]).reshape(n, FULL_GRID)
    features["raw_784"] = raw_784
    features["raw_505"] = raw_784[:, active_indices]

    # Explicit halt: stage 1 relied on the encoder gate's own automatic-
    # failure check for this; stage 2 has no gate, so this is the one
    # check that actually enforces "non-finite features (halt)".
    non_finite = {cond: int(np.sum(~np.isfinite(X))) for cond, X in features.items()}
    non_finite = {cond: count for cond, count in non_finite.items() if count}
    if non_finite:
        raise Stage2Halt(f"non-finite features: {non_finite}")

    Y = np.asarray(corpus["images_01"]).reshape(n, FULL_GRID)[:, active_indices]
    say("features: " + ", ".join(f"{k}{v.shape}" for k, v in sorted(features.items())))
    say(f"target Y {Y.shape} (clean, active support); zero non-finite across all conditions")
    return features, Y


def step8_ridge(mods, bucket, features, Y, y_strat, record):
    conditions = (*RAW_CONDITIONS, *mods.conditions.ALL_CONDITIONS)

    def compute_cv():
        out = {"conditions": {}, "halt_reasons": [], "order": list(conditions)}
        for condition in conditions:
            entry = {}
            X = features[condition]
            try:
                cv = mods.ridge.cross_validate_alpha(X, Y, y_strat)
                entry["cv"] = cv
                alphas = np.asarray(cv["alphas"])
                entry["alpha_at_grid_edge"] = bool(
                    cv["alpha"] == float(alphas.min()) or cv["alpha"] == float(alphas.max()))
            except AssertionError as exc:
                entry["cv_error"] = f"{type(exc).__name__}: {exc}"
                out["halt_reasons"].append(f"{condition}: centering guard: {exc}")
                out["conditions"][condition] = entry
                say(f"ridge/{condition}: CENTERING GUARD FIRED -- {exc}")
                continue
            say(f"ridge/{condition}: alpha={entry['cv']['alpha']}"
                f"{' [GRID EDGE]' if entry['alpha_at_grid_edge'] else ''}, "
                f"cond(X) per fold={np.asarray(entry['cv']['fold_cond']).tolist()}")
            try:
                equivalence = mods.ridge.ridge_equivalence_check(X, Y, y_strat)
                entry["equivalence"] = equivalence
                say(f"ridge/{condition}: equivalence PASS 2: pred_diff="
                    f"{float(equivalence['max_abs_clipped_pred_diff']):.3e} "
                    f"(tol {float(equivalence['tol']):.0e}), "
                    f"alpha agrees={bool(equivalence['alpha_agrees'])}")
                if not equivalence["passed"]:
                    out["halt_reasons"].append(
                        f"{condition}: equivalence pass 2 pred_diff="
                        f"{float(equivalence['max_abs_clipped_pred_diff']):.3e} "
                        f"(tol {float(equivalence['tol']):.0e}), "
                        f"alpha_jax={equivalence['alpha_jax']}, "
                        f"alpha_sklearn={equivalence['alpha_sklearn']}")
            except AssertionError as exc:
                entry["equivalence_error"] = f"{type(exc).__name__}: {exc}"
                out["halt_reasons"].append(f"{condition}: equivalence centering guard: {exc}")
            out["conditions"][condition] = entry
        return out

    cv_json, _ = ensure_json(mods, bucket, _obj(mods, "ridge_cv", "json"), compute_cv)
    record["gates"]["ridge"] = cv_json
    if cv_json["halt_reasons"]:
        raise Stage2Halt("ridge: " + "; ".join(cv_json["halt_reasons"]))

    def compute_final():
        arrays, meta = {}, {"alphas": {}, "centering_margins": {}}
        for condition in conditions:
            alpha = cv_json["conditions"][condition]["cv"]["alpha"]
            fit, scaler = mods.ridge.fit_final(features[condition], Y, alpha)
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


def step9_cnn(mods, bucket, corpus, topo, corr, record):
    active_indices = np.asarray(topo["active_indices"])
    mask = mods.cnn.build_active_support_mask(active_indices, expect_n_active=EXPECTED_N_ACTIVE)

    def compute_val_corruption():
        x_t, x_t_clip = mods.corruption.corrupt_corpus(
            corpus["validation_images_01"], SPLIT, corpus["validation_indices"],
            alpha_bar=mods.corruption.ALPHA_BAR)
        return {"x_t": x_t, "x_t_clip": x_t_clip}

    val_corr, _ = ensure_npz(mods, bucket, _obj(mods, "validation_corruption", "npz"),
                             compute_val_corruption)

    # Identity baseline on the LOCKED validation partition, active support
    # only -- composed from the exact function stage 1 used for its own
    # identity baseline (corruption_diagnostics), called on the validation
    # corpus instead of the dev corpus. Not reimplemented.
    val_diag = mods.corruption.corruption_diagnostics(
        corpus["validation_images_01"], val_corr["x_t"], val_corr["x_t_clip"],
        active_indices, labels=corpus["validation_labels"])
    identity_val_mse = float(val_diag["mse_postclip_505"])
    say(f"CNN identity baseline (validation partition, active support, "
        f"n={corpus['validation_images_01'].shape[0]}): {identity_val_mse!r}")

    def compute_cnn():
        runs, per_seed_wallclock = [], {}
        for seed in mods.cnn.SEEDS:
            t0 = time.time()
            run = mods.cnn.train_cnn_for_seed(
                corr["x_t_clip"], corpus["images_01"],
                val_corr["x_t_clip"], corpus["validation_images_01"],
                mask, seed=seed)
            elapsed = time.time() - t0
            per_seed_wallclock[str(seed)] = elapsed
            runs.append(run)
            say(f"CNN seed={seed}: best_epoch={run['best_epoch']}, "
                f"best_clipped_val_mse={run['best_clipped_val_mse']!r}, "
                f"n_epochs_run={run['n_epochs_run']}, "
                f"stopped_early={run['stopped_early']}, {elapsed:.1f}s")

        best_seed, best_index = mods.cnn.select_best_seed(
            [r["seed"] for r in runs], [r["best_clipped_val_mse"] for r in runs])

        arrays = {}
        for run in runs:
            s = run["seed"]
            arrays[f"train_history_seed{s}"] = run["raw_train_loss_history"]
            arrays[f"val_history_seed{s}"] = run["clipped_val_mse_history"]
        meta = {
            "best_seed": best_seed, "best_index": best_index,
            "best_epoch": int(runs[best_index]["best_epoch"]),
            "best_clipped_val_mse": float(runs[best_index]["best_clipped_val_mse"]),
            "seeds": list(mods.cnn.SEEDS),
            "clipped_val_mse_per_seed": [float(r["best_clipped_val_mse"]) for r in runs],
            "best_epoch_per_seed": [int(r["best_epoch"]) for r in runs],
            "n_epochs_run_per_seed": [int(r["n_epochs_run"]) for r in runs],
            "stopped_early_per_seed": [bool(r["stopped_early"]) for r in runs],
            "wallclock_s_per_seed": per_seed_wallclock,
            "total_wallclock_s": sum(per_seed_wallclock.values()),
            "identity_val_mse": identity_val_mse,
            "n_params": int(runs[0]["n_params"]),
        }
        arrays["summary_json"] = np.array(_dumps(meta))
        return arrays

    cnn_npz, _ = ensure_npz(mods, bucket, _obj(mods, "cnn_development", "npz"), compute_cnn)
    summary = json.loads(cnn_npz["summary_json"].item())
    record["cnn"] = summary
    say(f"CNN best: seed={summary['best_seed']}, best_epoch={summary['best_epoch']}, "
        f"clipped val MSE={summary['best_clipped_val_mse']!r} vs identity "
        f"{summary['identity_val_mse']!r} (mechanical sanity, NON-INFERENTIAL, "
        f"not a locked comparison)")
    return cnn_npz, summary


def step10_stats_smoke(mods, bucket, final, diag_npz, corpus, record):
    n = corpus["images_01"].shape[0]
    projected_s = STATS_SMOKE_STAGE1_REFERENCE_S * (n / STATS_SMOKE_STAGE1_REFERENCE_N)
    if projected_s > STATS_SMOKE_BUDGET_S:
        record["stats_smoke"] = {
            "skipped": True,
            "reason": (f"projected {projected_s:.1f}s (linear from stage 1's measured "
                      f"{STATS_SMOKE_STAGE1_REFERENCE_S}s at n={STATS_SMOKE_STAGE1_REFERENCE_N}) "
                      f"exceeds the {STATS_SMOKE_BUDGET_S:.0f}s budget; the ridge-to-stats "
                      f"glue is already proven by stage 1's run"),
        }
        say(f"stats smoke SKIPPED: {record['stats_smoke']['reason']}")
        return None

    mse_by_condition = {mods.conditions.PRE_EVOLUTION:
                        np.asarray(final[f"mse_{mods.conditions.PRE_EVOLUTION}"])}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        mse_by_condition[graph] = np.asarray(final[f"mse_{graph}"])
    mse_by_condition[IDENTITY_KEY] = np.asarray(diag_npz["per_image_mse_postclip_505"])

    def compute():
        return mods.stats.run_stage2b_inference(
            mse_by_condition, np.asarray(corpus["labels"]), identity_key=IDENTITY_KEY)

    inference, _ = ensure_json(mods, bucket, _obj(mods, "stats_smoke", "json"), compute)

    def compute_text():
        descriptive = {condition: float(np.mean(arr))
                       for condition, arr in sorted(mse_by_condition.items())}
        for condition in RAW_CONDITIONS:
            key = f"mse_{condition}"
            if key in final:
                descriptive[condition] = float(np.mean(np.asarray(final[key])))
        return (f"{SMOKE_BANNER}\n\n"
                f"Mean per-image clipped MSE, in-sample, all ridge conditions "
                f"(n={n}):\n{_dumps(descriptive)}\n\n"
                f"Full inference output:\n{_dumps(inference)}\n")

    ensure_text(mods, bucket, _obj(mods, "stats_smoke", "txt"), compute_text)
    record["stats_smoke"] = dict(inference)
    record["stats_smoke"]["skipped"] = False
    say(f"stats machinery exercised end to end at n={n} (in-sample; not a result)")
    return inference


def step11_report(mods, bucket, record):
    def compute_json():
        return record

    def compute_text():
        lines = [f"Stage 2B feasibility-ladder stage 2 report",
                 f"commit: {record['run'].get('head_sha')}",
                 f"verdict: {record.get('verdict')}", ""]
        if record.get("halt_reason"):
            lines += [f"halt reason: {record['halt_reason']}", ""]
        lines += ["timings (s):", _dumps(record.get("timings", {})), "",
                  "full record:", _dumps(record), "",
                  str(record.get("verdict", FAIL_SENTINEL))]
        return "\n".join(lines) + "\n"

    # Run-scoped, and therefore create-once like every other artifact.
    # `force=True` used to overwrite one fixed report name, which meant a
    # resumed run DESTROYED the record of what the attempt that died had
    # seen -- the same "an unwritten result does not survive" failure this
    # project already has a lesson about. Both reports now survive,
    # distinguishable by run id, and the write-once policy needs no
    # exception carved out for reports.
    kind = f"stage2_report_{record['run']['run_id']}"
    ensure_json(mods, bucket, _obj(mods, kind, "json"), compute_json)
    ensure_text(mods, bucket, _obj(mods, kind, "txt"), compute_text)


def build_stage3_projections(record, n_measured=5000):
    """Per-stage linear projections to stage 3's fit-side size (54,000
    images) from what was actually measured here -- never one blended
    rate. Each is explicitly labeled a projection, not a measurement."""
    scale = STAGE3_FIT_N / n_measured
    projections = {"n_measured": n_measured, "n_projected": STAGE3_FIT_N,
                   "scale_factor": scale, "basis": "linear in n, single measurement"}

    encode = record.get("encode_diagnostic", {})
    if encode:
        projections["encode"] = {
            "measured_s": encode.get("elapsed_s"),
            "measured_per_image_ms": encode.get("per_image_s", 0) * 1000,
            "projected_s": encode.get("elapsed_s", 0) * scale,
        }

    evo_total = record.get("timings", {}).get("6_evolution")
    if evo_total is not None:
        projections["evolution"] = {"measured_s": evo_total, "projected_s": evo_total * scale}

    ridge_total = record.get("timings", {}).get("8_ridge")
    if ridge_total is not None:
        projections["ridge"] = {"measured_s": ridge_total, "projected_s": ridge_total * scale}

    cnn = record.get("cnn", {})
    if cnn:
        cnn_total = cnn.get("total_wallclock_s")
        projections["cnn"] = {
            "measured_s_at_n5000_fit": cnn_total,
            "note": ("CNN cost scales with EPOCHS x BATCHES, not simply n -- more fit "
                     "images means more batches per epoch but early stopping means the "
                     "epoch count itself is not fixed. This projection scales wall-clock "
                     "linearly in n as a first approximation only; it is not validated."),
            "projected_s": (cnn_total * scale) if cnn_total is not None else None,
        }
    return projections


def new_record():
    # A run id, minted once per process. It scopes the report object name,
    # so a resumed run writes a new report rather than replacing the one
    # its predecessor left.
    return {"run": {"run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
            "timings": {}, "gates": {}, "evolution": {}, "cnn": {},
            "encode_diagnostic": {}, "stats_smoke": {},
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
        with timed_step("4_encode", record["timings"]):
            gate = step4_encode(mods, bucket, corr, record)
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
        with timed_step("9_cnn", record["timings"]):
            step9_cnn(mods, bucket, corpus, topo, corr, record)
        with timed_step("10_stats_smoke", record["timings"]):
            step10_stats_smoke(mods, bucket, final, diag_npz, corpus, record)

        record["stage3_projections"] = build_stage3_projections(
            record, n_measured=corpus["images_01"].shape[0])
        record["verdict"] = OK_SENTINEL
    except Stage2Halt as halt:
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
                step11_report(mods, bucket, record)
            except Exception as exc:                # noqa: BLE001
                print(f"[stage2] report FAILED to write: {type(exc).__name__}: {exc}",
                      flush=True)

    verdict = record["verdict"] or FAIL_SENTINEL
    print(verdict if status == 0 else f"{verdict} {record['halt_reason']}", flush=True)
    return status


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version}
    for package in ("numpy", "scipy", "scikit-learn", "jax", "jaxlib", "diffrax",
                    "equinox", "optax", "google-cloud-storage", "google-crc32c"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    _status = main()
    if _status:
        sys.exit(_status)
