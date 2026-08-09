"""Stage 2B feasibility-ladder stage 4: THE ONE LOCKED EVALUATION,
DESIGN.md's "ONE locked evaluation on the official 10,000-image test
corpus." This is the first and only driver in this project permitted to
touch the official KMNIST test split, and the first that produces an
actual, inferential Stage 2B result -- everything upstream of this file
is either closed feasibility work or, at stage 3, explicitly banered
"SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE, NON-INFERENTIAL,
NOT A RESULT".

**WRITE-ONLY AT THE TIME THIS FILE WAS ADDED.** Execution requires the
pre-Stage-4 package review (`STAGE3_PLAN.md`'s item 9 -- this file is that
deliverable) and Dan's explicit release, per `AUDIT_PROTOCOL.md`: "Stage 4
stays blocked behind the package review and explicit release." Nothing in
this repository runs this file automatically.

## Two things stage 3's artifacts do NOT carry, discovered while designing
   this driver, and what that implies for its shape

`stage3`'s `ridge_final_*.npz` stores per-image IN-SAMPLE MSE and a
`summary_json` of selected alphas -- not the fitted coefficients or the
`StandardScaler`. `stage3`'s `cnn_production.npz` stores training/
validation loss HISTORIES and a `summary_json` -- not the trained model
weights (`stage2b_cnn.train_cnn`'s `"model"` key is discarded before the
array dict is built). Neither omission is a defect to fix here: per
principle 24, a number that anchors a decision is either reproducible
from committed code or it is not a decision anchor, and both the ridge
solve and the CNN training procedure ARE committed, deterministic (given
fixed seeds and fixed data) code. So:

- **Ridge**: refit fresh in this driver, from stage 3's persisted TRAIN
  features and the FROZEN production alpha stage 3 already selected, then
  applied once to TEST features. `svd_ridge_fit` is a closed-form SVD
  solve with no randomness anywhere in it, so this refit is bit-exact by
  construction given the same `(X_train, Y_train, alpha)` -- no
  reproduction check is needed or written, unlike the CNN below.
- **CNN**: retrained from the same three fixed seeds on the same locked
  54,000/6,000 fit/validation split, then VERIFIED against stage 3's
  persisted `(best_seed, best_epoch)` before its weights are trusted for
  test-corpus inference (see `cnn_reproduction_mismatch_reason`). Unlike
  the ridge solve, iterative SGD-style training over many float32
  matmuls is not guaranteed bit-exact across sessions or hardware even
  under a fixed seed, so this is checked rather than assumed. The
  structural check (same seed, same stopping epoch) halts on mismatch;
  the numeric `best_clipped_val_mse` difference is reported but does NOT
  halt on its own -- inventing a numeric tolerance with no measured basis
  is exactly what `AUDIT_PROTOCOL.md`'s Freeze 1 rejected once already
  (empirical tolerance, chosen after seeing a number). If retraining
  turns out not to reproduce reliably, that is itself a finding for the
  package review, not a threshold for this file to guess at.
  **Execution cost this implies, flagged for Dan rather than decided
  here**: running this driver pays the CNN's full three-seed training
  cost a second time, because stage 3 never persisted the weights.
  Whether to instead patch stage 3 to persist them (so stage 4 only
  verifies-and-loads) is a design change to an already-executed, already-
  fingerprinted artifact -- it would need its own release, not a default
  taken while writing this file.

## Why this is ONE file, not a Phase-A/Phase-B split like stage 3

Stage 3 splits into a local CPU encode (`encode_stage3_local.py`) and a
cloud GPU driver specifically because encoding 60,000 images at 1200
steps is the pipeline's one genuinely CPU-bound step, and running it
inside a metered A100 session would leave that GPU idle for most of the
run. At stage 3's own measured local rate (~10.6 ms/image, 9 workers),
that argument is a real ~10-minute cost; at 10,000 images -- one sixth
the population -- the SAME arithmetic gives roughly two minutes, against
a run whose other steps (evolution, ridge, CNN training, the 20,000-
resample bootstrap) already occupy the GPU for longer than that. The
cost the phase split exists to avoid is present here in degree, not in
kind, and does not justify a second local script, a second artifact
crossing the wire, and a second `bootstrap_repo`/`load_modules` copy for
a two-minute saving. This driver therefore encodes IN-SESSION, and this
paragraph is the disclosed decision, not a silent departure from stage
3's pattern.

## The single opt-in site

`stage2b_corruption._check_split_allowed` and `stage2b_gcs._check_object_path_allowed`
each refuse `split="test"` without `allow_test_split=True`, independently,
in their own modules. This file is the only DRIVER in
`experiments/stage2b_denoising/` where the literal `allow_test_split=True`
appears -- both call sites, corruption's and GCS's, live here and nowhere
else among the files that read or write scientific artifacts. One other
file in the directory, `smoke_stage2b_gcs.py`, also carries the literal:
it is a pre-existing, hand-run infrastructure probe for the GCS transport
layer's opt-in MECHANISM itself (round-trips a throwaway object through
every stage/split combination and deletes what it wrote), produces no
scientific artifact, and is not a driver -- a named, justified exemption,
not a second driver-side opt-in.
`tests/test_stage2b_ladder_stage4.py`'s
`test_allow_test_split_true_appears_in_exactly_this_driver_plus_the_exemption`
derives the full set from every `.py` file's AST rather than trusting
this paragraph, per principle 21 -- a claim like this one, stated in
prose and never checked, is exactly the failure mode a hand-maintained
list produces.

## One-shot semantics

DESIGN.md's primary test is described as "evaluated once". `ensure_*`'s
ordinary skip-on-match behaviour is right for every OTHER artifact this
driver writes -- a dead session should resume having lost at most one
step, the same property stage 3 has. It is wrong for exactly one object:
the official result. `step11_report`'s official-result write REFUSES
outright if that object already exists, rather than silently reusing it
or requiring a `force=True` this driver never offers -- see
`refuse_if_official_result_exists`. Every earlier artifact (corpus,
corruption, encoded phases, evolved thetas, features, the ridge test
MSEs, the CNN reproduction) stays ordinarily resumable.

## What this driver does not do

Retune anything against the test result. Compute a second test-corpus
evaluation under different settings "to compare". Widen a tolerance
because a first run failed it. Persist CNN or ridge model objects for
reuse elsewhere -- their outputs (per-image test MSE) are what downstream
statistics need, and nothing here claims those weights as a reusable
artifact. Touch `stage2b/testsplit/` under any stage other than 4.
"""
import hashlib
import json
import multiprocessing as mp
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
DRIVER_FILENAME = "run_ladder_stage4.py"
LADDER_STAGE = 4
TRAIN_STAGE = 3            # the rung whose artifacts this one refits from
                            # and whose CNN selection it must reproduce
KMNIST_STAGING_STAGE = 1   # staged once by stage_kmnist_inputs.py; every
                            # ladder stage downloads from THAT object path
SPLIT = "test"
TRAIN_SPLIT = "train"
OK_SENTINEL = "STAGE4_OK"
FAIL_SENTINEL = "STAGE4_FAIL"

# ---- bootstrap ----
#
# Duplicated from run_ladder_stage3.py rather than imported -- that file's
# own comment records why: `mighty-colab exec -f script` transmits this
# file's TEXT into an existing IPython kernel, so `__file__` is undefined
# and no other file of this repository exists on disk until bootstrap_repo()
# has cloned it, which happens INSIDE main(), after every module-scope
# statement has already run. Nothing but stdlib+numpy is importable here.
REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
WORK_DIR = "/content/stage2b_stage4"
KMNIST_SUBDIR = "datasets/kmnist"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)
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
EXPECTED_N_TEST = 10_000
EXPECTED_N_TRAIN = 60_000
EXPECTED_N_ACTIVE = 505
EXPECTED_REF_IDX = 363
FULL_GRID = 784
EXPECTED_FEATURE_DIM = 2 * EXPECTED_N_ACTIVE - 2    # 1008
ENCODER_STEPS = 1200      # must match the production value stage 3 used;
                          # asserted against stage 3's own recorded config
                          # rather than assumed equal by construction
EVOLVE_CHUNK = 250        # 10,000 / 250 = 40 exact chunks; 60,000 / 250 =
                          # 240 exact chunks (train-side CNN regeneration
                          # needs no evolution, so this only bounds the
                          # test-side loop, but the same constant stage 3
                          # used is reused rather than re-chosen)
TEST_ENCODE_CHUNK = 2_500  # 10,000 / 2,500 = 4 exact chunks. Chunking here
                          # is PURE memory management, not a numerics
                          # concern: unlike a shared-RNG batched draw
                          # (CLAUDE.md principle 19), each image's encode
                          # is independent -- `_encode_one` opens no state
                          # that spans images -- so no chunk size can
                          # change the result, and no bit-identical-across-
                          # chunk-sizes test is needed the way stage 3's
                          # sign-flip test needed one.

IDENTITY_KEY = "identity"
RAW_CONDITIONS = ("raw_505", "raw_784")
HEARTBEAT_SECONDS = 30.0

OFFICIAL_BANNER = ("OFFICIAL STAGE 4 RESULT -- ONE-SHOT, OFFICIAL 10,000-IMAGE "
                   "KMNIST TEST CORPUS, INFERENTIAL. DESIGN.md's primary "
                   "comparison, evaluated once.")

_RUN_T0 = time.time()
_STEP = {"name": "startup", "t0": _RUN_T0}


class Stage4Halt(Exception):
    """A prespecified halt condition fired."""


# ---------------------------------------------------------------- plumbing

def say(line):
    print(f"[stage4 {time.time() - _RUN_T0:7.1f}s] {line}", flush=True)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _dumps(obj):
    return json.dumps(obj, indent=2, sort_keys=True, default=_json_default)


def _run(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)


def _heartbeat_loop(stop):
    while not stop.wait(HEARTBEAT_SECONDS):
        step = _STEP
        print(f"[stage4 heartbeat] {step['name']} {time.time() - step['t0']:.0f}s",
              flush=True)


def start_heartbeat():
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(stop,), daemon=True)
    thread.start()
    return stop, thread


@contextlib.contextmanager
def timed_step(name, timings):
    previous = dict(_STEP)
    _STEP.update(name=name, t0=time.time())
    t0 = time.time()
    try:
        yield
    finally:
        timings[name] = time.time() - t0
        say(f"step {name} took {timings[name]:.2f}s")
        _STEP.update(previous)


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
        raise Stage4Halt(f"clone is at {head}, expected {commit}")

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
        raise Stage4Halt(
            f"driver identity mismatch: the transmitted file hashes to "
            f"{expected_sha256}, the commit's copy to {digest}. The code that ran "
            f"is not the code at {os.path.basename(clone_dir)}'s pinned commit.")
    if not expected_sha256:
        say(f"{ENV_DRIVER_SHA} unset; recorded {digest} without comparison")
    return result


def load_modules(clone_dir):
    """Every repo import, in one place and in an order that matters.

    `stage2b_ridge` FIRST: it enables jax's x64 mode at import. Mirrors
    `run_ladder_stage3.py::load_modules` exactly -- the same module set,
    because this driver needs everything stage 3's Phase B needed
    (evolution, ridge, CNN, stats) plus corruption's split gate used in
    anger for the first time."""
    add_repo_to_path(clone_dir)
    import stage2b_ridge as ridge                                       # noqa: E402
    from evolve_on_graph_jax import batched_evolve_on_graph_jax         # noqa: E402
    import jax                                                          # noqa: E402
    import jax.numpy as jnp                                             # noqa: E402

    import stage2b_cnn as cnn                                           # noqa: E402
    import stage2b_conditions as conditions                             # noqa: E402
    import stage2b_corruption as corruption                             # noqa: E402
    import stage2b_encoder_gate as encoder_gate                         # noqa: E402
    import stage2b_fingerprint as fingerprint                           # noqa: E402
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
        encoder_gate=encoder_gate, fingerprint=fingerprint, gcs=gcs,
        partition=partition, stats=stats, verify_gpu=verify_gpu, core=core,
        topologies=topologies, load_mnist=load_mnist,
        local_converged_phases=_local_converged_phases)

    for name in ("ridge", "cnn", "corruption", "encoder_gate", "fingerprint", "gcs",
                 "partition", "stats", "core", "topologies"):
        origin = getattr(mods, name).__file__
        if not os.path.abspath(origin).startswith(os.path.abspath(clone_dir)):
            raise Stage4Halt(f"module {name} resolved to {origin}, outside {clone_dir}")
    return mods


def stage_kmnist(mods, bucket, clone_dir):
    """The four IDX files staged once under `stage=1`, reused verbatim --
    including the official `t10k-*` files this driver reads. Staging them
    is NOT itself a test-side act: `stage_kmnist_inputs.py`'s own docstring
    is explicit that "none of `stage2b_gcs`'s test-split machinery is
    involved... every object written here lives under the train root".
    `allow_test_split` is not passed here and is not needed here."""
    dest_dir = os.path.join(clone_dir, KMNIST_SUBDIR)
    os.makedirs(dest_dir, exist_ok=True)
    staged = {}
    for filename, kind in sorted(KMNIST_FILES.items()):
        name = mods.gcs.object_path(stage=KMNIST_STAGING_STAGE, condition=None, kind=kind,
                                    ext=KMNIST_EXT, split=TRAIN_SPLIT)
        dest = os.path.join(dest_dir, filename)
        if os.path.isfile(dest):
            say(f"{filename} already present ({os.path.getsize(dest)} bytes)")
        else:
            manifest, _ = mods.gcs.consume_validated(name, dest, bucket=bucket,
                                                     require_manifest=False)
            say(f"downloaded {name} -> {filename} ({os.path.getsize(dest)} bytes)"
                f"{'' if manifest is None else ', manifest validated'}")
        staged[filename] = dest
    return dest_dir, staged


# ------------------------------------------------------- artifact wrappers

def build_fingerprint(mods, clone_dir, config):
    """This run's provenance, established BEFORE anything is generated."""
    return mods.fingerprint.compute(
        entrypoint=os.path.join(clone_dir, "experiments", "stage2b_denoising",
                                DRIVER_FILENAME),
        repo_root=clone_dir,
        require_clean=True,
        config=config)


def ensure_npz(mods, bucket, object_name, compute, *, fingerprint=None, parents=None,
               allow_test_split=False, force=False):
    def produce(path):
        arrays = compute()
        np.savez_compressed(path, **arrays)

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket, force=force,
                                      fingerprint=fingerprint, parents=parents,
                                      allow_test_split=allow_test_split)
    with np.load(result.local_path, allow_pickle=False) as handle:
        loaded = {key: handle[key] for key in handle.files}
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_json(mods, bucket, object_name, compute, *, fingerprint=None, parents=None,
                allow_test_split=False):
    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_dumps(compute()))

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket,
                                      fingerprint=fingerprint, parents=parents,
                                      allow_test_split=allow_test_split)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_text(mods, bucket, object_name, compute, *, fingerprint=None, parents=None,
                allow_test_split=False):
    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(compute())

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket,
                                      fingerprint=fingerprint, parents=parents,
                                      allow_test_split=allow_test_split)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = handle.read()
    say(f"artifact {result.summary()}")
    return loaded, result


def _obj_test(mods, kind, ext, condition=None):
    """The ONLY function in this file that passes `allow_test_split=True`
    to `stage2b_gcs.object_path`. Every test-side artifact this driver
    writes or names goes through here."""
    return mods.gcs.object_path(stage=LADDER_STAGE, condition=condition, kind=kind,
                                ext=ext, split=SPLIT, allow_test_split=True)


def _obj_train(mods, kind, ext, condition=None, stage=TRAIN_STAGE):
    """A stage-3 (or stage-1) TRAIN-side object name -- no test-split
    opt-in, because none is needed: reading what stage 3 already computed
    is ordinary train-side access."""
    return mods.gcs.object_path(stage=stage, condition=condition, kind=kind, ext=ext,
                                split=TRAIN_SPLIT)


def consume_pinned(mods, bucket, object_name, pin_key, pinned_sha256, local_path=None):
    """A pre-contract artifact (stage 1's topologies), consumed with the
    named opt-out and checked against its pinned digest. Mirrors
    `run_ladder_stage3.py::consume_pinned` exactly."""
    local_path = local_path or local_path_for(object_name)
    mods.gcs.consume_validated(object_name, local_path, bucket=bucket,
                               require_manifest=False)
    with open(local_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    expected = pinned_sha256[pin_key]
    if digest != expected:
        raise Stage4Halt(
            f"{object_name!r} does not match its pinned digest: expected {expected}, "
            f"got {digest}. This object carries no manifest (pre-contract history), "
            f"so the pin is the only thing standing between this run and silently "
            f"different input. Do not update the pin to make this pass -- find out "
            f"what changed.")
    say(f"consumed {object_name} (pre-contract, pinned sha256 {digest[:16]}...)")
    return local_path


PINNED_SHA256 = {
    "stage1/topologies":
        "f671e63cc00b1612db0da5976c14b8880e4c4f90ae7fb192297721665f1907a4",
}


def parent_map(mods, bucket, names, *, allow_test_split=False):
    """`{object_name: payload_sha256}` for a set of parents.

    Unlike `run_ladder_stage3.py::parent_map` -- which this is otherwise a
    straight copy of -- every parent here IS a test-side object, so this
    takes the same opt-in every other test-side accessor in this file
    takes, rather than defaulting to it silently. `read_manifest` gates on
    the object path regardless of who constructed it or when; a name built
    through `_obj_test` still needs the opt-in threaded to THIS call."""
    out = {}
    for name in names:
        manifest = mods.gcs.read_manifest(name, bucket=bucket,
                                          allow_test_split=allow_test_split)
        digest = (manifest or {}).get("payload_sha256")
        if digest is None:
            local = local_path_for(name)
            if not os.path.isfile(local):
                raise Stage4Halt(
                    f"cannot record {name!r} as a parent: it carries no manifest and "
                    f"no local copy exists at {local}. A parent digest that cannot be "
                    f"resolved would pin nothing.")
            with open(local, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        out[name] = digest
    return out


# ---------------------------------------------------------------- helpers

def refuse_if_official_result_exists(mods, bucket, name):
    """DESIGN.md's primary test is "evaluated once". Every OTHER artifact
    this driver writes is ordinarily resumable through `ensure_artifact`'s
    skip-on-match behaviour; this one object is not, because a second
    silent pass over it would mean either quietly reusing a result that
    already stands or requiring a `force=True` this driver deliberately
    never offers. Both are unavailable; this halts instead."""
    if mods.gcs.object_exists(name, bucket=bucket, allow_test_split=True):
        raise Stage4Halt(
            f"the official Stage 4 result already exists at {name!r}. DESIGN.md's "
            f"primary comparison is 'evaluated once', so this driver refuses to "
            f"compute or overwrite it a second time. A genuine, deliberate "
            f"re-evaluation is a decision for the package review to make explicitly, "
            f"not a default this driver takes on its own.")


def cnn_reproduction_mismatch_reason(reproduced, original):
    """Whether a freshly-retrained best-of-3 CNN selection reproduces
    stage 3's persisted one, structurally.

    Compares `best_seed` and `best_epoch` for EXACT equality -- both are
    integer selections (an argmin over three seeds; an early-stopping
    epoch count), not floating-point measurements, so exact equality is
    the right comparison and not a fragile one. Deliberately does NOT
    gate on `best_clipped_val_mse` with an invented numeric tolerance:
    `AUDIT_PROTOCOL.md`'s Freeze 1 already rejected choosing a tolerance
    after seeing a number, and this driver has no measured basis for one.
    The MSE difference is reported by the caller for human review, never
    used here to pass or fail anything.

    Returns a halt-reason string, or None if seed and epoch both match."""
    r_seed, o_seed = int(reproduced["best_seed"]), int(original["best_seed"])
    if r_seed != o_seed:
        return (f"CNN retraining selected seed={r_seed}, stage {TRAIN_STAGE} selected "
                f"seed={o_seed}. Same three fixed seeds, same fit/validation data -- a "
                f"different argmin means the training run did not reproduce, not that "
                f"a coin landed differently.")
    r_epoch, o_epoch = int(reproduced["best_epoch"]), int(original["best_epoch"])
    if r_epoch != o_epoch:
        return (f"CNN retraining's selected seed ({r_seed}) stopped at "
                f"best_epoch={r_epoch}, stage {TRAIN_STAGE}'s stopped at "
                f"best_epoch={o_epoch}. Early stopping is deterministic given the "
                f"validation trajectory; a different stopping point means the "
                f"trajectory itself differed.")
    return None


# ------------------------------------------------------------------ steps

def step0_preflight(mods, record):
    try:
        mods.verify_gpu.device_preflight()
    except SystemExit as exc:
        raise Stage4Halt(f"device preflight: {exc}") from exc
    probe = mods.jnp.zeros(1, dtype=mods.jnp.float64)
    record["run"]["devices"] = [str(d) for d in mods.jax.devices()]
    record["run"]["realised_float64_dtype"] = str(probe.dtype)
    record["run"]["jax_enable_x64"] = bool(mods.jax.config.jax_enable_x64)


def step1_test_corpus(mods, bucket, kmnist_dir, fp):
    """All 10,000 official KMNIST test images, ascending official index.

    This is where `allow_test_split=True` is first exercised: the object
    this step writes lives under `stage2b/testsplit/stage4/...`, and
    without the opt-in `_obj_test` would raise before any bytes moved."""
    def compute():
        _x_train, _y_train, x_test, y_test = mods.load_mnist(kmnist_dir, gz=False)
        n = int(x_test.shape[0])
        if n != EXPECTED_N_TEST:
            raise Stage4Halt(f"KMNIST test split has {n} images, expected "
                             f"{EXPECTED_N_TEST}")
        return {
            "test_indices": np.arange(n, dtype=np.int64),
            "images_01": x_test.astype(np.float64) / 255.0,
            "labels": np.asarray(y_test),
        }

    corpus, _ = ensure_npz(mods, bucket, _obj_test(mods, "corpus_test", "npz"), compute,
                           fingerprint=fp, allow_test_split=True)
    n = corpus["images_01"].shape[0]
    if n != EXPECTED_N_TEST:
        raise Stage4Halt(f"test corpus has {n} images, expected {EXPECTED_N_TEST}")
    if not np.array_equal(np.asarray(corpus["test_indices"]),
                          np.arange(EXPECTED_N_TEST, dtype=np.int64)):
        raise Stage4Halt("test corpus row order is not ascending official index")
    say(f"test corpus n={n}, ascending official index 0..{n - 1}")
    return corpus


def step1b_topologies(mods, bucket):
    """Stage 1's own cached artifact, reused unchanged -- byte-identical
    `active_indices` and graphs across every rung, this one included.
    Population-independent, so no test-split concern arises here at all."""
    name = mods.gcs.object_path(stage=KMNIST_STAGING_STAGE, condition=None,
                                kind="topologies", ext="npz", split=TRAIN_SPLIT)
    local = consume_pinned(mods, bucket, name, "stage1/topologies", PINNED_SHA256)
    with np.load(local, allow_pickle=False) as handle:
        topo = {key: handle[key] for key in handle.files}
    meta = json.loads(topo["summary_json"].item())
    n_active = int(np.asarray(topo["active_indices"]).size)
    if n_active != EXPECTED_N_ACTIVE:
        raise Stage4Halt(f"active support has {n_active} nodes, expected "
                         f"{EXPECTED_N_ACTIVE}")
    ref_idx = int(meta["nodes_T"]["median"])
    if ref_idx != EXPECTED_REF_IDX:
        raise Stage4Halt(f"T's median-degree node is {ref_idx}, expected "
                         f"{EXPECTED_REF_IDX}")
    say(f"topologies reused from stage {KMNIST_STAGING_STAGE}; n_active={n_active}, "
        f"ref_idx={ref_idx} (505-space)")
    return topo, ref_idx


def step2_test_corruption(mods, bucket, corpus, fp, corpus_name):
    """The SECOND `allow_test_split=True` call site -- `stage2b_corruption`'s
    own gate, independent of `stage2b_gcs`'s. Regenerated deterministically
    from official test indices; spot-checked against a fresh single-image
    recomputation at three rows, mirroring stage 3's own corruption
    cross-check."""
    def compute():
        x_t, x_t_clip = mods.corruption.corrupt_corpus(
            corpus["images_01"], SPLIT, corpus["test_indices"],
            alpha_bar=mods.corruption.ALPHA_BAR, allow_test_split=True)
        return {"x_t": x_t, "x_t_clip": x_t_clip}

    corr, _ = ensure_npz(mods, bucket, _obj_test(mods, "corruption_test", "npz"),
                         compute, fingerprint=fp, allow_test_split=True,
                         parents=parent_map(mods, bucket, (corpus_name,), allow_test_split=True))

    n = corpus["images_01"].shape[0]
    for k in (0, n // 2, n - 1):
        eps = mods.corruption.epsilon_for(SPLIT, int(corpus["test_indices"][k]))
        x_t_k, x_t_clip_k = mods.corruption.forward_corrupt(
            corpus["images_01"][k].reshape(-1), eps, mods.corruption.ALPHA_BAR)
        if not (np.array_equal(x_t_k, np.asarray(corr["x_t"])[k].reshape(-1))
                and np.array_equal(x_t_clip_k,
                                   np.asarray(corr["x_t_clip"])[k].reshape(-1))):
            raise Stage4Halt(
                f"row {k} does not reproduce from its ORIGINAL test index "
                f"{int(corpus['test_indices'][k])}")
    say(f"test corruption regenerated and spot-checked at rows 0, {n // 2}, {n - 1}")
    return corr


def _encode_test_indices(mods, x_test, active_indices, chunk=TEST_ENCODE_CHUNK):
    """Encodes the whole test corpus in-session, chunked purely for peak
    memory (see the module docstring's `TEST_ENCODE_CHUNK` note on why
    chunk size cannot change the result here)."""
    n = int(x_test.shape[0])
    thetas = np.empty((n, active_indices.size), dtype=np.float64)
    n_workers = max(1, mp.cpu_count() - 1)
    t_start = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        t0 = time.time()
        chunk_thetas, _deltas = mods.encoder_gate.encode_with_final_delta_batch(
            x_test[lo:hi], active_indices, seed=mods.encoder_gate.ENCODER_SEED,
            steps=ENCODER_STEPS, n_workers=n_workers)
        thetas[lo:hi] = chunk_thetas
        elapsed = time.time() - t0
        say(f"encode chunk {lo}:{hi} {elapsed:.1f}s ({elapsed / (hi - lo) * 1000:.2f} "
            f"ms/image)")
    return thetas, time.time() - t_start


def step3_encode_test(mods, bucket, corr, topo, fp, parents):
    active_indices = np.asarray(topo["active_indices"])

    def compute():
        thetas, elapsed = _encode_test_indices(mods, np.asarray(corr["x_t_clip"]),
                                               active_indices)
        return {"thetas_505": thetas,
                "summary_json": np.array(_dumps({
                    "n_images": int(thetas.shape[0]), "steps": ENCODER_STEPS,
                    "encode_elapsed_s": elapsed,
                    "encode_per_image_ms": elapsed / thetas.shape[0] * 1000,
                    "n_nonfinite": int(np.sum(~np.isfinite(thetas))),
                }))}

    encoded, _ = ensure_npz(mods, bucket,
                            _obj_test(mods, f"encoded_test_s{ENCODER_STEPS}", "npz"),
                            compute, fingerprint=fp, allow_test_split=True,
                            parents=parents)
    theta0_505 = np.asarray(encoded["thetas_505"])
    expected = (np.asarray(corr["x_t_clip"]).shape[0], EXPECTED_N_ACTIVE)
    if theta0_505.shape != expected:
        raise Stage4Halt(f"encoded test phases are {theta0_505.shape}, expected "
                         f"{expected}")
    summary = json.loads(encoded["summary_json"].item())
    if summary["n_nonfinite"]:
        raise Stage4Halt(f"{summary['n_nonfinite']} non-finite encoded values")
    say(f"test corpus encoded: {theta0_505.shape}, "
        f"{summary['encode_per_image_ms']:.2f} ms/image")
    return theta0_505


def step4_test_evolution(mods, bucket, theta0_505, topo, record, fp, parents):
    """Four graphs on the test corpus. Structurally identical to stage 3's
    `step5_evolution` (batched JAX solve, success-flag gated), scoped to
    10,000 rows and written under the test-split root."""
    n = theta0_505.shape[0]
    if n % EVOLVE_CHUNK:
        raise Stage4Halt(f"{n} images does not divide into {EVOLVE_CHUNK}-row chunks")
    n_chunks = n // EVOLVE_CHUNK
    warmed = {"done": False}

    def warm_up(W):
        if warmed["done"]:
            return
        t0 = time.time()
        theta_warm, _success = mods.batched_evolve_on_graph_jax(
            mods.jnp.asarray(theta0_505[:EVOLVE_CHUNK]), W)
        mods.jax.block_until_ready(theta_warm)
        say(f"evolve warm-up compile {time.time() - t0:.1f}s (excluded from timings)")
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
                theta_T, success = mods.batched_evolve_on_graph_jax(
                    mods.jnp.asarray(theta0_505[lo:hi]), W)
                mods.jax.block_until_ready(theta_T)
                thetas.append(np.asarray(theta_T))
                flags.append(np.asarray(success))
            return {"theta_T": np.concatenate(thetas), "success": np.concatenate(flags)}

        name = _obj_test(mods, "theta_T", "npz",
                         condition=mods.conditions.path_segment(graph))
        loaded, _ = ensure_npz(mods, bucket, name, compute, fingerprint=fp,
                               allow_test_split=True, parents=parents)
        evolved[graph] = loaded

        success = np.asarray(loaded["success"])
        n_failed = int(np.count_nonzero(~success))
        record["evolution"][graph] = {"n_images": int(success.size),
                                      "n_failed": n_failed, "object": name}
        say(f"evolve/{graph}: {n_failed} failed of {success.size}")
        if n_failed:
            failures.append(f"{graph}: {n_failed}/{success.size} solves failed")

    if failures:
        raise Stage4Halt("test-corpus graph evolution: " + "; ".join(failures))
    return evolved


def step5_test_features(mods, bucket, theta0_505, evolved, topo, corr, corpus, ref_idx,
                        fp, parents_by_condition):
    """The same seven conditions stage 3 fit ridge on, computed here for
    the TEST corpus: `raw_505`, `raw_784`, `pre_evolution`, and the four
    evolved graphs."""
    n = theta0_505.shape[0]
    active_indices = np.asarray(topo["active_indices"])
    features = {}

    for condition in mods.conditions.ALL_CONDITIONS:
        theta = (theta0_505 if condition == mods.conditions.PRE_EVOLUTION
                 else np.asarray(evolved[condition]["theta_T"]))

        def compute(theta=theta):
            return {"X": np.stack([mods.core.reference_node_features(theta[i], ref_idx)
                                   for i in range(n)])}

        name = _obj_test(mods, "features", "npz",
                         condition=mods.conditions.path_segment(condition))
        loaded, _ = ensure_npz(mods, bucket, name, compute, fingerprint=fp,
                               allow_test_split=True,
                               parents=parents_by_condition.get(condition))
        X = np.asarray(loaded["X"])
        if X.shape != (n, EXPECTED_FEATURE_DIM):
            raise Stage4Halt(f"{condition} test features are {X.shape}, expected "
                             f"{(n, EXPECTED_FEATURE_DIM)}")
        features[condition] = X

    raw_784 = np.asarray(corr["x_t_clip"]).reshape(n, FULL_GRID)
    features["raw_784"] = raw_784
    features["raw_505"] = raw_784[:, active_indices]

    non_finite = {cond: int(np.sum(~np.isfinite(X))) for cond, X in features.items()}
    non_finite = {cond: count for cond, count in non_finite.items() if count}
    if non_finite:
        raise Stage4Halt(f"non-finite test features: {non_finite}")

    Y = np.asarray(corpus["images_01"]).reshape(n, FULL_GRID)[:, active_indices]
    say("test features: " + ", ".join(f"{k}{v.shape}" for k, v in sorted(features.items())))
    return features, Y


# ------------------------------------------------------------- ridge (real)

def _load_train_ridge_inputs(mods, bucket):
    """Everything the ridge step needs from stage 3's TRAIN-side artifacts:
    the full 60,000-image corpus (for `Y_train` and raw features), the
    seven conditions' feature arrays, and the frozen production alpha per
    condition. No test-split object is touched here."""
    corpus_name = _obj_train(mods, "corpus", "npz")
    corpus_local = local_path_for(corpus_name)
    mods.gcs.consume_validated(corpus_name, corpus_local, bucket=bucket)
    with np.load(corpus_local, allow_pickle=False) as handle:
        train_corpus = {key: handle[key] for key in handle.files}
    n_train = train_corpus["images_01"].shape[0]
    if n_train != EXPECTED_N_TRAIN:
        raise Stage4Halt(f"stage {TRAIN_STAGE} train corpus has {n_train} images, "
                         f"expected {EXPECTED_N_TRAIN}")

    train_features = {}
    for condition in mods.conditions.ALL_CONDITIONS:
        name = _obj_train(mods, "features", "npz",
                          condition=mods.conditions.path_segment(condition))
        local = local_path_for(name)
        mods.gcs.consume_validated(name, local, bucket=bucket)
        with np.load(local, allow_pickle=False) as handle:
            train_features[condition] = np.asarray(handle["X"])

    tag = grid_tag(mods.ridge.ALPHA_GRID)
    ridge_final_name = _obj_train(mods, f"ridge_final_{tag}", "npz")
    ridge_final_local = local_path_for(ridge_final_name)
    mods.gcs.consume_validated(ridge_final_name, ridge_final_local, bucket=bucket)
    with np.load(ridge_final_local, allow_pickle=False) as handle:
        ridge_final_summary = json.loads(handle["summary_json"].item())
    alphas = ridge_final_summary["alphas"]

    return train_corpus, train_features, alphas, {corpus_name: corpus_local,
                                                    ridge_final_name: ridge_final_local}


def grid_tag(alphas):
    """Identical to `run_ladder_stage3.py::grid_tag` -- reproduced here
    rather than imported so this driver's naming is self-contained, and
    because the two drivers cannot share code across the bootstrap
    boundary (see the module docstring)."""
    canonical = ",".join(repr(float(a)) for a in alphas)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"g{len(tuple(alphas))}_{digest}"


def train_raw_pixel_conditions(mods, images_train, train_indices, active_indices):
    """`raw_505`/`raw_784`'s TRAIN-side X, and `Y_train`, built the way the
    corrupted-to-clean reconstruction task actually requires: X from the
    CORRUPTED train images (`x_t_clip`), Y from the clean ones.

    A caller that reaches `images_train` directly for X fits Y-on-Y --
    ridge learns an identity map on the training population, then meets
    genuinely corrupted pixels for the first time at test time, where its
    near-identity fit reproduces something close to the input verbatim --
    numerically close to the identity/do-nothing baseline, which is
    exactly how this bug presented (PR review, not local testing: the
    corrected and buggy versions still disagree once real GCS artifacts
    are involved, which this pure function cannot exercise, so the
    regression test for it asserts the two OUTPUTS differ on synthetic
    data rather than asserting a specific downstream MSE).

    Both stage 3's own `step6_features` and this driver's own
    `step5_test_features` build the raw conditions from `x_t_clip`; this
    is the same construction, applied to the train population."""
    _x_t_train, x_t_clip_train = mods.corruption.corrupt_corpus(
        images_train, TRAIN_SPLIT, train_indices, alpha_bar=mods.corruption.ALPHA_BAR)
    n = images_train.shape[0]
    raw_784_train = np.asarray(x_t_clip_train).reshape(n, FULL_GRID)
    raw_505_train = raw_784_train[:, active_indices]
    clean_784_train = np.asarray(images_train).reshape(n, FULL_GRID)
    Y_train = clean_784_train[:, active_indices]
    return raw_784_train, raw_505_train, Y_train


def step6_test_ridge(mods, bucket, features_test, Y_test, topo, fp, parents):
    """Refit each condition's ridge on the full 60,000-image TRAIN corpus
    at stage 3's FROZEN production alpha, then predict once on the TEST
    corpus. Deterministic SVD solve -- no reproduction check needed,
    unlike the CNN below (see the module docstring)."""
    active_indices = np.asarray(topo["active_indices"])
    train_corpus, train_features, alphas, _train_parents = _load_train_ridge_inputs(
        mods, bucket)
    raw_784_train, raw_505_train, Y_train = train_raw_pixel_conditions(
        mods, np.asarray(train_corpus["images_01"]),
        np.asarray(train_corpus["train_indices"]), active_indices)
    train_features["raw_784"] = raw_784_train
    train_features["raw_505"] = raw_505_train

    conditions = (*RAW_CONDITIONS, *mods.conditions.ALL_CONDITIONS)

    def compute():
        arrays, meta = {}, {"alphas": {}}
        for condition in conditions:
            alpha = alphas[condition]
            fit, scaler = mods.ridge.fit_final(train_features[condition], Y_train, alpha)
            X_test_scaled = scaler.transform(features_test[condition])
            prediction = mods.ridge.ridge_predict(fit, X_test_scaled, 0)
            arrays[f"mse_{condition}"] = mods.ridge.clipped_per_image_mse(
                prediction, Y_test)
            meta["alphas"][condition] = alpha
            say(f"ridge/{condition}: refit on {EXPECTED_N_TRAIN} train images at "
                f"frozen alpha={alpha}, evaluated on {Y_test.shape[0]} test images, "
                f"mean clipped test MSE={float(np.mean(arrays[f'mse_{condition}'])):.6e}")
        arrays["summary_json"] = np.array(_dumps(meta))
        return arrays

    tag = grid_tag(mods.ridge.ALPHA_GRID)
    final, _ = ensure_npz(mods, bucket, _obj_test(mods, f"ridge_final_test_{tag}", "npz"),
                          compute, fingerprint=fp, allow_test_split=True, parents=parents)
    return final


# --------------------------------------------------------------- CNN (real)

def step7_test_cnn(mods, bucket, topo, fp, parents):
    """Retrain the CNN from the same three fixed seeds on the same locked
    54,000/6,000 fit/validation split stage 3 used, VERIFY the selection
    reproduces stage 3's, then evaluate the reproduced model on the TEST
    corpus. See the module docstring for why retraining rather than
    loading is the right design given what stage 3 persisted."""
    active_indices = np.asarray(topo["active_indices"])
    mask = mods.cnn.build_active_support_mask(active_indices,
                                              expect_n_active=EXPECTED_N_ACTIVE)

    train_corpus_name = _obj_train(mods, "corpus", "npz")
    train_corpus_local = local_path_for(train_corpus_name)
    mods.gcs.consume_validated(train_corpus_name, train_corpus_local, bucket=bucket)
    with np.load(train_corpus_local, allow_pickle=False) as handle:
        train_corpus = {key: handle[key] for key in handle.files}

    train_indices = np.asarray(train_corpus["train_indices"])
    fit_rows, _ = mods.partition.index_join(
        np.asarray(train_corpus["fit_indices"]), train_indices,
        source_name="the fit role", target_name="the corpus")
    val_rows, _ = mods.partition.index_join(
        np.asarray(train_corpus["validation_indices"]), train_indices,
        source_name="the validation role", target_name="the corpus")

    images = np.asarray(train_corpus["images_01"])
    _x_t, x_t_clip = mods.corruption.corrupt_corpus(
        images, TRAIN_SPLIT, train_indices, alpha_bar=mods.corruption.ALPHA_BAR)
    fit_clean, val_clean = images[fit_rows], images[val_rows]
    fit_noisy, val_noisy = x_t_clip[fit_rows], x_t_clip[val_rows]

    cnn_train_name = _obj_train(mods, "cnn_production", "npz")
    cnn_train_local = local_path_for(cnn_train_name)
    mods.gcs.consume_validated(cnn_train_name, cnn_train_local, bucket=bucket)
    with np.load(cnn_train_local, allow_pickle=False) as handle:
        original_summary = json.loads(handle["summary_json"].item())

    runs, per_seed_wallclock = [], {}
    for seed in mods.cnn.SEEDS:
        t0 = time.time()
        run = mods.cnn.train_cnn_for_seed(fit_noisy, fit_clean, val_noisy, val_clean,
                                          mask, seed=seed)
        per_seed_wallclock[str(seed)] = time.time() - t0
        runs.append(run)
        say(f"CNN retrain seed={seed}: best_epoch={run['best_epoch']}, "
            f"best_clipped_val_mse={run['best_clipped_val_mse']!r}, "
            f"{per_seed_wallclock[str(seed)]:.1f}s")

    best_seed, best_index = mods.cnn.select_best_seed(
        [r["seed"] for r in runs], [r["best_clipped_val_mse"] for r in runs])
    reproduced_summary = {
        "best_seed": best_seed, "best_index": best_index,
        "best_epoch": int(runs[best_index]["best_epoch"]),
        "best_clipped_val_mse": float(runs[best_index]["best_clipped_val_mse"]),
    }

    mismatch = cnn_reproduction_mismatch_reason(reproduced_summary, original_summary)
    if mismatch:
        raise Stage4Halt(f"CNN reproduction: {mismatch}")
    mse_diff = abs(reproduced_summary["best_clipped_val_mse"]
                   - float(original_summary["best_clipped_val_mse"]))
    say(f"CNN reproduction VERIFIED: seed={best_seed} epoch={reproduced_summary['best_epoch']} "
        f"match stage {TRAIN_STAGE}; best_clipped_val_mse differs by {mse_diff:.3e} "
        f"(reported, not gated -- see module docstring)")

    best_model = runs[best_index]["model"]

    # Now the real thing: the reproduced model, evaluated ONCE on the
    # official test corpus. These are reads of TEST-SPLIT objects this
    # driver already WROTE in steps 1-2 (each write went through
    # `_obj_test`'s opt-in at construction time), so the read itself
    # still needs `allow_test_split=True` passed through to
    # `consume_validated` -- the gate checks the object PATH, not who
    # constructed it or when.
    test_corr_name = _obj_test(mods, "corruption_test", "npz")
    test_corr_local = local_path_for(test_corr_name)
    mods.gcs.consume_validated(test_corr_name, test_corr_local, bucket=bucket,
                               allow_test_split=True)
    test_corpus_name = _obj_test(mods, "corpus_test", "npz")
    test_corpus_local = local_path_for(test_corpus_name)
    mods.gcs.consume_validated(test_corpus_name, test_corpus_local, bucket=bucket,
                               allow_test_split=True)
    with np.load(test_corr_local, allow_pickle=False) as handle:
        test_x_t_clip = np.asarray(handle["x_t_clip"])
    with np.load(test_corpus_local, allow_pickle=False) as handle:
        test_images = np.asarray(handle["images_01"])

    # `as_image_batch`'s own docstring: "Any caller reaching ... clipped_validation_mse
    # directly should pass its arrays through here first, the same way train_cnn does,
    # rather than writing its own cast." train_cnn_for_seed does this internally for
    # fit/val; this driver's own direct call on the TEST arrays is new and missed it on
    # the first real run -- caught there, fixed here, not by static review.
    test_mse = np.asarray(mods.cnn.clipped_validation_per_image_mse(
        best_model, mods.cnn.as_image_batch(test_x_t_clip, "test_x_t_clip"),
        mods.cnn.as_image_batch(test_images, "test_images"), mask))

    def compute():
        return {"mse_cnn": test_mse,
                "summary_json": np.array(_dumps({
                    "reproduced": reproduced_summary, "original": original_summary,
                    "best_clipped_val_mse_diff": mse_diff,
                    "wallclock_s_per_seed": per_seed_wallclock,
                    "test_mean_clipped_mse": float(np.mean(test_mse)),
                }))}

    final, _ = ensure_npz(mods, bucket, _obj_test(mods, "cnn_test", "npz"), compute,
                          fingerprint=fp, allow_test_split=True, parents=parents)
    say(f"CNN test evaluation: mean clipped MSE={float(np.mean(test_mse)):.6e} "
        f"(descriptive -- not part of the ridge-based inference families)")
    return final, reproduced_summary


# ------------------------------------------------------------- the result

def step8_identity_baselines(mods, corpus, corr, Y, topo, labels):
    """The hierarchical gate's identity (`clip(x_t)`, post-clip, active
    support) plus the descriptive rescaled-identity baseline, both on the
    test corpus."""
    active_indices = np.asarray(topo["active_indices"])
    diag = mods.corruption.corruption_diagnostics(
        np.asarray(corpus["images_01"]), np.asarray(corr["x_t"]),
        np.asarray(corr["x_t_clip"]), active_indices, labels=labels)
    identity_mse = np.asarray(diag["per_image_mse_postclip_505"])

    x_t_clip_505 = np.asarray(corr["x_t_clip"]).reshape(
        corpus["images_01"].shape[0], FULL_GRID)[:, active_indices]
    rescaled = mods.corruption.rescaled_identity(x_t_clip_505, mods.corruption.ALPHA_BAR)
    rescaled_mse = mods.ridge.clipped_per_image_mse(rescaled, Y)
    return identity_mse, rescaled_mse, diag


def step9_inference(mods, final_ridge, identity_mse, labels):
    """DESIGN.md's locked inference, run for the first and only time on
    the official test corpus. `mse_by_condition` carries exactly what
    `run_stage2b_inference` requires -- `pre_evolution` plus the four
    evolved graphs, plus identity -- mirroring stage 3's own
    `step9_stats_smoke` construction, minus the smoke banner because this
    IS the result."""
    mse_by_condition = {mods.conditions.PRE_EVOLUTION:
                        np.asarray(final_ridge[f"mse_{mods.conditions.PRE_EVOLUTION}"])}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        mse_by_condition[graph] = np.asarray(final_ridge[f"mse_{graph}"])
    mse_by_condition[IDENTITY_KEY] = identity_mse

    return mods.stats.run_stage2b_inference(mse_by_condition, labels,
                                            identity_key=IDENTITY_KEY)


def step10_descriptive(final_ridge, cnn_final, rescaled_mse, diag):
    """Everything DESIGN.md wants reported that is NOT part of the
    inference families: raw-pixel ridge, the rescaled-identity baseline,
    the CNN's test MSE, and the corruption diagnostics."""
    descriptive = {}
    for condition in RAW_CONDITIONS:
        key = f"mse_{condition}"
        if key in final_ridge:
            descriptive[condition] = float(np.mean(np.asarray(final_ridge[key])))
    descriptive["rescaled_identity"] = float(np.mean(rescaled_mse))
    descriptive["cnn"] = float(np.mean(np.asarray(cnn_final["mse_cnn"])))
    descriptive["corruption_diagnostics"] = {
        k: v for k, v in diag.items()
        if k in ("mse_postclip_505", "mse_postclip_784", "mse_preclip_505",
                 "mse_preclip_784")}
    return descriptive


def step11_report(mods, bucket, record):
    """The per-run report is written every time, success or failure --
    that history is what a debugging pass reads. The OFFICIAL result is
    not: DESIGN.md's primary test is "evaluated once", and a run that
    halted before reaching `step9_inference` evaluated nothing. Writing
    `official_result` unconditionally would let a halted, pre-inference
    attempt occupy the one-shot slot and permanently block every
    subsequent, possibly-correct attempt with `refuse_if_official_result_exists`
    -- which is exactly what happened on this driver's own first real
    run, caught within the same session it was found in."""
    def compute_json():
        return record

    def compute_text():
        lines = [OFFICIAL_BANNER, "", f"commit: {record['run'].get('head_sha')}",
                 f"verdict: {record.get('verdict')}", ""]
        if record.get("halt_reason"):
            lines += [f"halt reason: {record['halt_reason']}", ""]
        lines += ["timings (s):", _dumps(record.get("timings", {})), "",
                  "full record:", _dumps(record), "",
                  str(record.get("verdict", FAIL_SENTINEL))]
        return "\n".join(lines) + "\n"

    kind = f"stage4_report_{record['run']['run_id']}"
    ensure_json(mods, bucket, _obj_test(mods, kind, "json"), compute_json,
               allow_test_split=True)
    ensure_text(mods, bucket, _obj_test(mods, kind, "txt"), compute_text,
               allow_test_split=True)

    if record.get("verdict") != OK_SENTINEL:
        say(f"official result NOT written: verdict is {record.get('verdict')!r}, "
            f"not {OK_SENTINEL!r}. Per-run report above carries the failure.")
        return

    official_name = _obj_test(mods, "official_result", "json")
    refuse_if_official_result_exists(mods, bucket, official_name)
    ensure_json(mods, bucket, official_name, compute_json, allow_test_split=True)
    ensure_text(mods, bucket, _obj_test(mods, "official_result", "txt"), compute_text,
               allow_test_split=True)


def new_record():
    return {"run": {"run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
            "timings": {}, "evolution": {}, "cnn": {}, "verdict": None,
            "halt_reason": None}


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

        # Refuse EARLY, before doing any work, not only at the report step
        # -- a run that pays every step's cost only to halt at the very
        # end on a pre-existing result would be its own kind of footgun.
        official_name = mods.gcs.object_path(
            stage=LADDER_STAGE, condition=None, kind="official_result", ext="json",
            split=SPLIT, allow_test_split=True)
        refuse_if_official_result_exists(mods, bucket, official_name)

        fp = build_fingerprint(mods, CLONE_DIR, {
            "ladder_stage": LADDER_STAGE,
            "split": SPLIT,
            "population": "official KMNIST test split, all 10,000",
            "n_images": EXPECTED_N_TEST,
            "row_order": "ascending official test index",
            "encoder_steps": ENCODER_STEPS,
            "alpha_bar": mods.corruption.ALPHA_BAR,
            "corruption_scheme": "SHA256(split:index:42) -> PCG64, per image",
            "evolve_chunk": EVOLVE_CHUNK,
            "n_active": EXPECTED_N_ACTIVE,
            "reference_node": EXPECTED_REF_IDX,
            "feature_dim": EXPECTED_FEATURE_DIM,
            "ridge_alpha_grid": list(mods.ridge.ALPHA_GRID),
            "cnn_seeds": list(mods.cnn.SEEDS),
            "dtype": "float64",
        })
        record["run"]["fingerprint"] = {
            "source_manifest_digest": fp["source_manifest_digest"],
            "config_digest": fp["config_digest"], "git": fp["git"],
            "n_source_files": len(fp["source_manifest"])}
        say(f"fingerprint established: config {fp['config_digest'][:16]}...")

        with timed_step("stage_kmnist", record["timings"]):
            kmnist_dir, staged = stage_kmnist(mods, bucket, CLONE_DIR)
            record["run"]["kmnist"] = {name: os.path.getsize(path)
                                       for name, path in sorted(staged.items())}

        with timed_step("0_preflight", record["timings"]):
            step0_preflight(mods, record)
        with timed_step("1_test_corpus", record["timings"]):
            corpus = step1_test_corpus(mods, bucket, kmnist_dir, fp)
            corpus_name = _obj_test(mods, "corpus_test", "npz")
        with timed_step("1b_topologies", record["timings"]):
            topo, ref_idx = step1b_topologies(mods, bucket)
        with timed_step("2_test_corruption", record["timings"]):
            corr = step2_test_corruption(mods, bucket, corpus, fp, corpus_name)
            corr_name = _obj_test(mods, "corruption_test", "npz")

        encode_parents = parent_map(mods, bucket, (corr_name,), allow_test_split=True)
        with timed_step("3_encode_test", record["timings"]):
            theta0_505 = step3_encode_test(mods, bucket, corr, topo, fp, encode_parents)
            encoded_name = _obj_test(mods, f"encoded_test_s{ENCODER_STEPS}", "npz")

        evo_parents = parent_map(mods, bucket, (encoded_name, corpus_name),
                                 allow_test_split=True)
        with timed_step("4_test_evolution", record["timings"]):
            evolved = step4_test_evolution(mods, bucket, theta0_505, topo, record, fp,
                                           evo_parents)

        feature_parents = {mods.conditions.PRE_EVOLUTION: dict(evo_parents)}
        for graph in mods.conditions.EVOLVED_GRAPHS:
            theta_name = _obj_test(mods, "theta_T", "npz",
                                   condition=mods.conditions.path_segment(graph))
            feature_parents[graph] = parent_map(mods, bucket, (theta_name,),
                                                allow_test_split=True)
        with timed_step("5_test_features", record["timings"]):
            features_test, Y_test = step5_test_features(
                mods, bucket, theta0_505, evolved, topo, corr, corpus, ref_idx, fp,
                feature_parents)

        ridge_parents = parent_map(mods, bucket, (corpus_name, corr_name),
                                   allow_test_split=True)
        with timed_step("6_test_ridge", record["timings"]):
            final_ridge = step6_test_ridge(mods, bucket, features_test, Y_test, topo,
                                           fp, ridge_parents)

        with timed_step("7_test_cnn", record["timings"]):
            cnn_final, cnn_reproduction = step7_test_cnn(
                mods, bucket, topo, fp, parent_map(mods, bucket, (corpus_name, corr_name),
                                                allow_test_split=True))
            record["cnn"] = cnn_reproduction

        with timed_step("8_identity_baselines", record["timings"]):
            identity_mse, rescaled_mse, diag = step8_identity_baselines(
                mods, corpus, corr, Y_test, topo, np.asarray(corpus["labels"]))

        with timed_step("9_inference", record["timings"]):
            inference = step9_inference(mods, final_ridge, identity_mse,
                                        np.asarray(corpus["labels"]))
            record["inference"] = inference

        with timed_step("10_descriptive", record["timings"]):
            record["descriptive"] = step10_descriptive(final_ridge, cnn_final,
                                                        rescaled_mse, diag)

        record["verdict"] = OK_SENTINEL
    except Stage4Halt as halt:
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
                with timed_step("11_report", record["timings"]):
                    step11_report(mods, bucket, record)
            except Exception as exc:                # noqa: BLE001
                print(f"[stage4] report FAILED to write: {type(exc).__name__}: {exc}",
                      flush=True)

    verdict = record["verdict"] or FAIL_SENTINEL
    print(verdict if status == 0 else f"{verdict} {record['halt_reason']}", flush=True)
    return status


if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    _status = main()
    if _status:
        sys.exit(_status)
