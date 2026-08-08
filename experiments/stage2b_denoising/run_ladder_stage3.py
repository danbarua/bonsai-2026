"""Stage 2B feasibility-ladder stage 3, Phase B: the full 60,000-image
training corpus on a Colab A100 -- evolution, features, ridge and CNN at
production scale, under the fingerprint/manifest contract that did not
exist when stages 1 and 2 ran.

`PHASE_B_PLAN.md` is the plan of record and this file implements its step
table. Same architecture and discipline as `run_ladder_stage2.py`:
composes existing, verified modules and implements nothing itself; runs on
a runtime that fetches one pinned commit rather than being uploaded with
its dependencies; every artifact through `ensure_artifact`, so a dead
session resumes having lost at most one step. `STAGE3_OK` / `STAGE3_FAIL`
on stdout, non-zero exit on failure -- the make target requires both.

## What Phase B changes from stage 2

- **No encode step.** Phase A encoded all 60,000 images locally on CPU and
  published `encoded_train_s1200.npz` with a manifest; stage 2's
  `step4_encode` becomes a validated consume, which is why the step
  numbering below differs from stage 2's.
- **The corpus is the whole official training split**, all 60,000 indices
  in ascending order, carrying Freeze 2's fit/validation role flags.
- **A sizing probe (step 2b) gates the run**, measuring both cost legs
  separately and halting against budgets fixed before it ran.
- **Thetas and features are persisted per graph and per condition with
  fingerprints** -- resumability needs it anyway, and Decision 4 makes it
  load-bearing: the amendment audit consumes these exact objects.

## What halts this stage

The ridge equivalence extension (Decision 2), the scaler-centering guard,
a graph-evolution success-flag failure, any non-finite feature, the
corruption cross-check, the encoded-input spot-check, and the sizing
probe. Condition numbers, grid-edge alpha selections and the CNN's
standing against identity remain RECORDED FACT, exactly as at stage 2. No
new gate is invented mid-ladder in either direction.

## Explicitly out of scope

The official KMNIST test set, untouchable until feasibility stage 4: no
test-split object is read, written, or named anywhere in this file. Any
confirmatory statistic. The amendment audit and the ARM/x86 stress run,
which are separate sessions after this one.
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
DRIVER_FILENAME = "run_ladder_stage3.py"
LADDER_STAGE = 3
KMNIST_STAGING_STAGE = 1   # staged once by stage_kmnist_inputs.py under
                           # stage=1; every later ladder stage downloads
                           # from THAT object path, never re-stages.
STAGE2_STAGE = 2           # the rung whose corruption rows this one
                           # cross-checks against, by official index
SPLIT = "train"
OK_SENTINEL = "STAGE3_OK"
FAIL_SENTINEL = "STAGE3_FAIL"

# ---- bootstrap ----
REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
WORK_DIR = "/content/stage2b_stage3"
KMNIST_SUBDIR = "datasets/kmnist"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)
# Duplicated from run_ladder_stage2.py rather than imported, for the reason
# that file records at length: `mighty-colab exec -f script` transmits this
# file's TEXT into an existing IPython kernel, so `__file__` is undefined
# and no other file of this repository exists on disk until bootstrap_repo()
# has cloned it -- which happens INSIDE main(), after every module-scope
# statement has already run. Nothing but stdlib+numpy is importable here.
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
EVOLVE_CHUNK = 250       # 60,000 / 250 = 240 exact chunks; the same chunk
                         # size stages 1 and 2 used, and the divisibility
                         # halt below stays valid unchanged
FULL_GRID = 784
EXPECTED_N = 60_000
EXPECTED_N_ACTIVE = 505
EXPECTED_REF_IDX = 363
EXPECTED_FEATURE_DIM = 2 * EXPECTED_N_ACTIVE - 2    # 1008
ENCODER_STEPS = 1200     # the step count Phase A encoded at; asserted
                         # against the artifact's own recorded config

# The re-encode spot-check's tolerance. Phase A measured max 3 ULP between
# ARM and x86 encodings of the same images, with within-architecture
# results bit-exact in both directions. 16 leaves room for a longer
# accumulation than the images Phase A sampled without admitting the
# order-of-magnitude drift that would mean the encode is not reproducing.
# Exact equality is NOT the right comparison here and would fail on a
# healthy run: this driver runs on x86, the artifact was produced on ARM.
ENCODED_REENCODE_MAX_ULP = 16.0
IDENTITY_KEY = "identity"
RAW_CONDITIONS = ("raw_505", "raw_784")
HEARTBEAT_SECONDS = 30.0

SMOKE_BANNER = ("SMOKE OF THE MACHINERY ONLY -- IN-SAMPLE, TRAINING-SIDE, "
                "NON-INFERENTIAL, NOT A RESULT")

# ---- pre-contract inputs, pinned by content ----
#
# Stages 1 and 2 wrote every artifact before the fingerprint contract
# existed, so these carry no manifest and are consumed with the named
# `require_manifest=False` opt-out. A pinned digest is what stands in.
#
# It is a BYTE-PIN, not provenance, and the difference is not cosmetic: it
# says "these are the exact bytes stage 2 read", which is true and
# checkable, and says nothing about what produced them. Retrofitting
# manifests onto completed rungs is refused because it would fabricate
# provenance rather than record it -- see stage2b_gcs's legacy policy.
PINNED_SHA256 = {
    "stage1/topologies":
        "f671e63cc00b1612db0da5976c14b8880e4c4f90ae7fb192297721665f1907a4",
    "stage2/corruption":
        "cd9ac32357f6aeefbaca006fb97cd53c52273f4acca4960115a72fc2983438cf",
}

# ---- the sizing probe's budgets ----
#
# Fixed here and in PHASE_B_PLAN.md BEFORE the probe ever ran, so the
# verdict cannot be chosen after a number exists -- the same discipline
# Decision 2 applies to the equivalence extension.
#
# Each leg gets its OWN multiplier. A single blended rate is what principle
# 18 forbids, and this pipeline is its own example: stage 2's `8_ridge` is
# 305.53s at n=5,000, and it is dominated not by the JAX SVD but by
# `sklearn_ridge_predict`, which fits `Ridge(solver="svd")` once per alpha
# on the CPU -- 315 oracle SVDs against 35 production ones.
PROBE_JAX_SVD_COUNT = 42       # DESIGN.md's accounting: 35 fold-level + 7 refits
PROBE_SKLEARN_FIT_COUNT = 315  # 7 conditions x 5 folds x 9 alphas
PROBE_RIDGE_BUDGET_S = 7_200.0
PROBE_RUN_BUDGET_S = 9_000.0
PROBE_DEVICE_PEAK_BUDGET_BYTES = 12 * 1024**3

# The device budget's derivation, so it can be argued with rather than
# obeyed. At n_train=48,000, `svd_ridge_fit` holds X (387 MB), U (387 MB,
# full_matrices=False), Y_tilde (194 MB), Vt (8 MB), W at (9, 1008, 505)
# (37 MB) and Z (4 MB) -- ~1.02 GB of named arrays, plus a cuSOLVER
# workspace this model does not pretend to know. The budget is 8x that: it
# catches an order-of-magnitude modelling error while leaving a healthy run
# room for a larger workspace. It sits far below the 40 GB device on
# purpose, because a threshold at 30 GB would pass almost any wrong model.
PROBE_MODELLED_ARRAY_BYTES = 1_020_000_000

_RUN_T0 = time.time()
_STEP = {"name": "startup", "t0": _RUN_T0}


class Stage3Halt(Exception):
    """A prespecified halt condition fired."""


# ---------------------------------------------------------------- plumbing

def say(line):
    print(f"[stage3 {time.time() - _RUN_T0:7.1f}s] {line}", flush=True)


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
        print(f"[stage3 heartbeat] {step['name']} "
              f"{time.time() - step['t0']:.0f}s", flush=True)


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
        raise Stage3Halt(f"clone is at {head}, expected {commit}")

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
        raise Stage3Halt(
            f"driver identity mismatch: the transmitted file hashes to "
            f"{expected_sha256}, the commit's copy to {digest}. The code that ran "
            f"is not the code at {os.path.basename(clone_dir)}'s pinned commit.")
    if not expected_sha256:
        say(f"{ENV_DRIVER_SHA} unset; recorded {digest} without comparison")
    return result


def load_modules(clone_dir):
    """Every repo import, in one place and in an order that matters.

    `stage2b_ridge` FIRST: it enables jax's x64 mode at import. Adds
    `stage2b_fingerprint` over stage 2's closure -- the contract this rung
    is the first driver to write under."""
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
            raise Stage3Halt(f"module {name} resolved to {origin}, outside {clone_dir}")
    return mods


def stage_kmnist(mods, bucket, clone_dir):
    """The four IDX files staged once under `stage=1`, reused verbatim."""
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
            # Through the central validated consume path. These four DO
            # carry manifests (staged by stage_kmnist_inputs.py), but the
            # opt-out stays: they were staged under the stage-1 prefix and
            # a future re-stage must not silently start failing here.
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
               force=False):
    def produce(path):
        arrays = compute()
        np.savez_compressed(path, **arrays)

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket, force=force,
                                      fingerprint=fingerprint, parents=parents)
    with np.load(result.local_path, allow_pickle=False) as handle:
        loaded = {key: handle[key] for key in handle.files}
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_json(mods, bucket, object_name, compute, *, fingerprint=None, parents=None):
    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_dumps(compute()))

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket,
                                      fingerprint=fingerprint, parents=parents)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_text(mods, bucket, object_name, compute, *, fingerprint=None, parents=None):
    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(compute())

    result = mods.gcs.ensure_artifact(object_name, local_path_for(object_name),
                                      produce=produce, bucket=bucket,
                                      fingerprint=fingerprint, parents=parents)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = handle.read()
    say(f"artifact {result.summary()}")
    return loaded, result


def _obj(mods, kind, ext, condition=None, stage=LADDER_STAGE):
    return mods.gcs.object_path(stage=stage, condition=condition, kind=kind,
                                ext=ext, split=SPLIT)


def consume_pinned(mods, bucket, object_name, pin_key, local_path=None):
    """A pre-contract artifact, consumed with the named opt-out and checked
    against its pinned digest.

    The digest check is not decoration. `require_manifest=False` disables
    every provenance check there is, so without this the driver would
    accept whatever bytes currently sit at that name -- and these are
    exactly the names nothing else guards, since stages 1 and 2 are closed
    and write-once has no purchase on objects written before it existed."""
    local_path = local_path or local_path_for(object_name)
    mods.gcs.consume_validated(object_name, local_path, bucket=bucket,
                               require_manifest=False)
    with open(local_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    expected = PINNED_SHA256[pin_key]
    if digest != expected:
        raise Stage3Halt(
            f"{object_name!r} does not match its pinned digest: expected {expected}, "
            f"got {digest}. This object carries no manifest (pre-contract history), "
            f"so the pin is the only thing standing between this run and silently "
            f"different input. Do not update the pin to make this pass -- find out "
            f"what changed.")
    say(f"consumed {object_name} (pre-contract, pinned sha256 {digest[:16]}...)")
    return local_path


def parent_map(mods, bucket, names):
    """`{object_name: payload_sha256}` for a set of parents.

    A parent entry exists to PIN the bytes it names, so recording `None`
    would be a lineage link that constrains nothing -- the manifest would
    claim provenance it cannot check. The digest comes from the parent's
    own manifest where it has one, and is computed from the local copy
    otherwise (every parent here has already been downloaded by the step
    that consumed it, so this reads a file rather than the network).

    A parent that can be resolved neither way is an error rather than a
    `None`: it means this run is claiming descent from something it never
    actually read."""
    out = {}
    for name in names:
        manifest = mods.gcs.read_manifest(name, bucket=bucket)
        digest = (manifest or {}).get("payload_sha256")
        if digest is None:
            local = local_path_for(name)
            if not os.path.isfile(local):
                raise Stage3Halt(
                    f"cannot record {name!r} as a parent: it carries no manifest and "
                    f"no local copy exists at {local}. A parent digest that cannot be "
                    f"resolved would pin nothing.")
            with open(local, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        out[name] = digest
    return out


# ------------------------------------------------------------ sizing probe

def probe_projections(measured):
    """Project each leg on its own multiplier. Pure, so the arithmetic is
    testable without a GPU."""
    jax_total = measured["jax_svd_s"] * PROBE_JAX_SVD_COUNT
    sklearn_total = measured["sklearn_fit_s"] * PROBE_SKLEARN_FIT_COUNT
    return {
        "jax_svd_s_measured": measured["jax_svd_s"],
        "jax_svd_count": PROBE_JAX_SVD_COUNT,
        "jax_projected_s": jax_total,
        "sklearn_fit_s_measured": measured["sklearn_fit_s"],
        "sklearn_fit_count": PROBE_SKLEARN_FIT_COUNT,
        "sklearn_projected_s": sklearn_total,
        "ridge_projected_s": jax_total + sklearn_total,
        "device_peak_bytes": measured.get("device_peak_bytes"),
    }


def evaluate_probe(measured, elapsed_so_far_s):
    """The probe's verdict against budgets fixed before it ran.

    Returns `(projections, reasons)`; a non-empty `reasons` is a halt. Kept
    pure and separate from the measuring so the halt path can be tested on
    synthetic over-budget input -- a guard that has never been seen to fail
    is not yet a guard."""
    proj = probe_projections(measured)
    proj["elapsed_before_probe_s"] = elapsed_so_far_s
    # NOT a forecast of total runtime, and named so it cannot be read as
    # one. The probe runs before the 229 MB encoded-input download, the
    # evolution and feature steps and the CNN, so none of those are in
    # here: it is elapsed-so-far plus the ridge projection, which is the
    # quantity the budget below actually gates. A run that overshoots does
    # so on the steps this deliberately excludes.
    proj["elapsed_plus_ridge_projected_s"] = elapsed_so_far_s + proj["ridge_projected_s"]
    proj["excluded_from_projection"] = [
        "encoded-input download (~229 MB)", "evolution", "features", "CNN",
        "stats smoke", "artifact uploads (~3.4 GB)"]
    proj["budgets"] = {
        "ridge_s": PROBE_RIDGE_BUDGET_S,
        "run_s": PROBE_RUN_BUDGET_S,
        "device_peak_bytes": PROBE_DEVICE_PEAK_BUDGET_BYTES,
    }

    reasons = []
    if proj["ridge_projected_s"] > PROBE_RIDGE_BUDGET_S:
        reasons.append(
            f"projected ridge {proj['ridge_projected_s']:.0f}s exceeds the "
            f"{PROBE_RIDGE_BUDGET_S:.0f}s budget (JAX {proj['jax_projected_s']:.0f}s "
            f"+ sklearn {proj['sklearn_projected_s']:.0f}s)")
    if proj["elapsed_plus_ridge_projected_s"] > PROBE_RUN_BUDGET_S:
        reasons.append(
            f"elapsed-plus-ridge {proj['elapsed_plus_ridge_projected_s']:.0f}s "
            f"exceeds the {PROBE_RUN_BUDGET_S:.0f}s budget")
    peak = proj.get("device_peak_bytes")
    if peak is not None and peak > PROBE_DEVICE_PEAK_BUDGET_BYTES:
        reasons.append(
            f"device peak {peak / 1024**3:.2f} GiB exceeds the "
            f"{PROBE_DEVICE_PEAK_BUDGET_BYTES / 1024**3:.0f} GiB budget "
            f"(model predicted ~{PROBE_MODELLED_ARRAY_BYTES / 1024**3:.2f} GiB of "
            f"named arrays plus solver workspace)")
    return proj, reasons


def _device_peak_bytes(mods):
    """JAX's live-buffer high-water mark, or None on a backend that does
    not report it (CPU). Absent is recorded as absent, never as zero."""
    try:
        stats = mods.jax.devices()[0].memory_stats()
    except Exception:                               # noqa: BLE001 - optional
        return None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


def step2b_sizing_probe(mods, bucket, record, fp, n_total=EXPECTED_N):
    """One condition, one fold, BOTH legs -- before anything expensive runs.

    ## The measurement is published BEFORE the halt is raised

    Order is load-bearing, not stylistic. The probe's numbers used to live
    only in the run record, which is written at teardown -- so a session
    that died after measuring and before finishing lost the measurement
    entirely, and the next run had to re-measure on a metered GPU. That is
    the "an unwritten result does not survive" failure this stage already
    has a lesson about, and it came within minutes of costing us the Phase
    B numbers when the launcher was killed mid-run.

    So the artifact lands first and the halt is raised second. A probe that
    HALTS is exactly the case where its evidence matters most -- it is the
    measurement someone will want to argue with -- and it would otherwise
    be the case most likely to lose it.

    The artifact is RUN_SCOPED by construction: its kind starts with
    `probe`, nothing consumes it, and it is never a parent. It carries the
    run id so a resumed attempt records its own measurement rather than
    displacing its predecessor's.

    ## Why the matrix is synthetic

    This runs before features exist, which is the point: a probe that can
    only run after evolution and feature extraction cannot stop the run
    from paying for them. The cost of a thin SVD and of a `Ridge` fit
    depends on the matrix SHAPE, not its contents, so a seeded random
    matrix at the exact production shape measures the right thing. It is a
    sizing measurement and is labelled one; no scientific quantity is
    computed here or reported anywhere as a result.
    """
    n_train = n_total - n_total // mods.ridge.N_SPLITS
    shape = (n_train, EXPECTED_FEATURE_DIM)
    rng = np.random.default_rng(0)
    X = rng.standard_normal(shape)
    Y = rng.standard_normal((n_train, EXPECTED_N_ACTIVE))
    say(f"sizing probe: one fold at {shape}, targets {(n_train, EXPECTED_N_ACTIVE)}")

    t0 = time.time()
    fit = mods.ridge.svd_ridge_fit(X, Y, check_centered=False)
    mods.jax.block_until_ready(mods.jnp.asarray(fit["singular_values"]))
    jax_svd_s = time.time() - t0
    peak = _device_peak_bytes(mods)
    say(f"sizing probe: JAX svd_ridge_fit {jax_svd_s:.1f}s"
        + (f", device peak {peak / 1024**3:.2f} GiB" if peak else ", device peak unreported"))

    t0 = time.time()
    mods.ridge.sklearn_ridge_predict(X, Y, X[:1], (float(mods.ridge.ALPHA_GRID[0]),))
    sklearn_fit_s = time.time() - t0
    say(f"sizing probe: one sklearn Ridge(solver='svd') fit {sklearn_fit_s:.1f}s")

    measured = {"jax_svd_s": jax_svd_s, "sklearn_fit_s": sklearn_fit_s,
                "device_peak_bytes": peak, "n_train": n_train,
                "feature_dim": EXPECTED_FEATURE_DIM,
                "matrix": "seeded standard normal at production shape; SIZING ONLY"}
    proj, reasons = evaluate_probe(measured, time.time() - _RUN_T0)
    record["sizing_probe"] = {"measured": measured, **proj, "halted": bool(reasons),
                              "reasons": reasons}

    # Durable BEFORE the halt -- see this function's docstring for why the
    # ordering is the point rather than an implementation detail.
    kind = f"probe_sizing_{record['run']['run_id']}"
    try:
        ensure_json(mods, bucket, _obj(mods, kind, "json"),
                    lambda: record["sizing_probe"], fingerprint=fp)
    except Exception as exc:                    # noqa: BLE001 - never mask the halt
        say(f"sizing probe: FAILED to publish its measurement "
            f"({type(exc).__name__}: {exc}); continuing to the verdict")

    say("sizing probe projections: JAX "
        f"{proj['jax_projected_s']:.0f}s (x{PROBE_JAX_SVD_COUNT}) + sklearn "
        f"{proj['sklearn_projected_s']:.0f}s (x{PROBE_SKLEARN_FIT_COUNT}) = ridge "
        f"{proj['ridge_projected_s']:.0f}s; elapsed+ridge "
        f"{proj['elapsed_plus_ridge_projected_s']:.0f}s "
        f"(EXCLUDES download, evolution, features, CNN -- not a runtime forecast)")
    if reasons:
        raise Stage3Halt("sizing probe: " + "; ".join(reasons))
    say("sizing probe: within every budget; proceeding")
    return proj


# ------------------------------------------------------------------ steps

def step0_preflight(mods, record):
    try:
        mods.verify_gpu.device_preflight()
    except SystemExit as exc:
        raise Stage3Halt(f"device preflight: {exc}") from exc
    probe = mods.jnp.zeros(1, dtype=mods.jnp.float64)
    record["run"]["devices"] = [str(d) for d in mods.jax.devices()]
    record["run"]["realised_float64_dtype"] = str(probe.dtype)
    record["run"]["jax_enable_x64"] = bool(mods.jax.config.jax_enable_x64)


def step1_corpus(mods, bucket, kmnist_dir, fp):
    """All 60,000 official training images, ascending, with Freeze 2's role
    flags carried alongside rather than left implicit in row order."""
    def compute():
        x_train, y_train, _x_test, _y_test = mods.load_mnist(kmnist_dir, gz=False)
        n = int(x_train.shape[0])
        if n != EXPECTED_N:
            raise Stage3Halt(f"KMNIST training split has {n} images, expected "
                             f"{EXPECTED_N}")
        train_indices = np.arange(n, dtype=np.int64)
        part = mods.partition.Stage2BTrainingPartition(y_train)
        fit_indices = np.asarray(part.fit_indices)
        validation_indices = np.asarray(part.validation_indices)
        return {
            "train_indices": train_indices,
            "fit_indices": fit_indices,
            "validation_indices": validation_indices,
            "images_01": x_train.astype(np.float64) / 255.0,
            "labels": np.asarray(y_train),
            "partition_summary": np.array(_dumps(part.summary())),
        }

    corpus, _ = ensure_npz(mods, bucket, _obj(mods, "corpus", "npz"), compute,
                           fingerprint=fp)
    n = corpus["images_01"].shape[0]
    if n != EXPECTED_N:
        raise Stage3Halt(f"corpus has {n} images, expected {EXPECTED_N}")

    # Freeze 2's roles, asserted rather than assumed: disjoint, exhaustive,
    # and the sizes DESIGN.md locks. A role array that silently overlapped
    # would put validation images in the fit side with nothing failing.
    fit = np.asarray(corpus["fit_indices"])
    val = np.asarray(corpus["validation_indices"])
    if fit.size != mods.partition.N_FIT or val.size != mods.partition.N_VALIDATION:
        raise Stage3Halt(f"roles are {fit.size} fit / {val.size} validation, expected "
                         f"{mods.partition.N_FIT} / {mods.partition.N_VALIDATION}")
    if np.intersect1d(fit, val).size:
        raise Stage3Halt("fit and validation roles overlap")
    if not np.array_equal(np.sort(np.concatenate([fit, val])),
                          np.arange(EXPECTED_N, dtype=np.int64)):
        raise Stage3Halt("fit and validation roles are not exhaustive over 0..59,999")
    if not np.array_equal(np.asarray(corpus["train_indices"]),
                          np.arange(EXPECTED_N, dtype=np.int64)):
        raise Stage3Halt("corpus row order is not ascending official index")
    say(f"corpus n={n}, roles {fit.size} fit / {val.size} validation, disjoint and "
        f"exhaustive, ascending official order")
    return corpus


def step1b_topologies(mods, bucket):
    """Stage 1's own cached artifact, reused rather than rebuilt.

    Topologies depend on nothing about which images are processed, so
    reading stage 1's object guarantees byte-identical active_indices and
    graphs across every rung -- not merely "the same by construction".

    PHASE_B_PLAN.md originally said this is consumed under `CONTENT_ONLY`.
    That was wrong on its premise: `CONTENT_ONLY` is a fingerprint policy,
    and this object carries no manifest, so no fingerprint check can run
    under any policy. What runs instead is the pre-contract opt-out plus a
    pinned digest. The plan is amended at that line."""
    name = _obj(mods, "topologies", "npz", stage=KMNIST_STAGING_STAGE)
    local = consume_pinned(mods, bucket, name, "stage1/topologies")
    with np.load(local, allow_pickle=False) as handle:
        topo = {key: handle[key] for key in handle.files}
    meta = json.loads(topo["summary_json"].item())
    n_active = int(np.asarray(topo["active_indices"]).size)
    if n_active != EXPECTED_N_ACTIVE:
        raise Stage3Halt(f"active support has {n_active} nodes, expected "
                         f"{EXPECTED_N_ACTIVE}")
    ref_idx = int(meta["nodes_T"]["median"])
    if ref_idx != EXPECTED_REF_IDX:
        raise Stage3Halt(f"T's median-degree node is {ref_idx}, expected "
                         f"{EXPECTED_REF_IDX}")
    say(f"topologies reused from stage {KMNIST_STAGING_STAGE}; n_active={n_active}, "
        f"ref_idx={ref_idx} (505-space)")
    return topo, meta, ref_idx


def step2_corruption(mods, bucket, corpus, fp, corpus_name):
    """Regenerated in-session for all 60,000, then cross-checked BIT-EXACT
    against stage 2's stored rows, joined by official index.

    Bit-exact is the right tolerance and anything looser would be a defect:
    the scheme is pure numpy PCG64 keyed on `SHA256(split:index:42)`, which
    is architecture-independent. This is the check that the same corruption
    REALIZATION reproduces across rungs, not merely that this rung's
    formula is internally consistent."""
    def compute():
        x_t, x_t_clip = mods.corruption.corrupt_corpus(
            corpus["images_01"], SPLIT, corpus["train_indices"],
            alpha_bar=mods.corruption.ALPHA_BAR)
        return {"x_t": x_t, "x_t_clip": x_t_clip}

    corr, _ = ensure_npz(mods, bucket, _obj(mods, "corruption", "npz"), compute,
                         fingerprint=fp,
                         parents=parent_map(mods, bucket, (corpus_name,)))

    n = corpus["images_01"].shape[0]
    for k in (0, n // 2, n - 1):
        eps = mods.corruption.epsilon_for(SPLIT, int(corpus["train_indices"][k]))
        x_t_k, x_t_clip_k = mods.corruption.forward_corrupt(
            corpus["images_01"][k].reshape(-1), eps, mods.corruption.ALPHA_BAR)
        if not (np.array_equal(x_t_k, np.asarray(corr["x_t"])[k].reshape(-1))
                and np.array_equal(x_t_clip_k,
                                   np.asarray(corr["x_t_clip"])[k].reshape(-1))):
            raise Stage3Halt(
                f"row {k} does not reproduce from its ORIGINAL dataset index "
                f"{int(corpus['train_indices'][k])}")

    # The cross-rung check, joined by official index through the SHARED
    # helper rather than a fresh positional slice. Two artifacts built from
    # differently-ordered index lists would align row-for-row, agree on
    # shape, and compare entirely wrong numbers with nothing raised.
    stage2_corr_name = _obj(mods, "corruption", "npz", stage=STAGE2_STAGE)
    stage2_local = consume_pinned(mods, bucket, stage2_corr_name, "stage2/corruption")
    stage2_corpus_name = _obj(mods, "corpus", "npz", stage=STAGE2_STAGE)
    stage2_corpus_local = local_path_for(stage2_corpus_name)
    mods.gcs.consume_validated(stage2_corpus_name, stage2_corpus_local, bucket=bucket,
                               require_manifest=False)
    with np.load(stage2_corpus_local, allow_pickle=False) as handle:
        stage2_indices = np.asarray(handle["stage2_indices"])
    rows, join_report = mods.partition.index_join(
        stage2_indices, np.asarray(corpus["train_indices"]),
        source_name="stage 2's corpus", target_name="this rung's corpus")

    with np.load(stage2_local, allow_pickle=False) as handle:
        stage2_x_t_clip = handle["x_t_clip"]
    mine = np.asarray(corr["x_t_clip"])[rows]
    if not np.array_equal(mine, stage2_x_t_clip):
        n_diff = int(np.count_nonzero(np.any(mine != stage2_x_t_clip, axis=1)))
        raise Stage3Halt(
            f"{n_diff} of {rows.size} stage-2 images disagree with this rung's "
            f"corruption. The same realization did not reproduce across rungs.")
    say(f"corruption verified: own-formula spot-check at rows 0/{n // 2}/{n - 1}; "
        f"cross-rung BIT-EXACT over all {rows.size} stage-2 images, joined by "
        f"official index ({join_report['n_rows_moved']} rows moved by the join)")
    return corr, join_report


def step3_encoded_input(mods, bucket, corpus, corr, record):
    """Phase A's 60,000-image encode, consumed under the full contract.

    This is the one input written under the fingerprint contract, so unlike
    every other consume here it gets the real thing: manifest read first,
    payload fetched at the COMMITTED generation, digest recomputed.

    The re-encode spot-check runs at ULP tolerance, never exact equality.
    Phase A measured max 3 ULP between ARM and x86 encodings of the same
    images, with within-architecture results bit-exact in both directions;
    this driver runs on x86 and the artifact was produced on ARM, so an
    exact comparison would fail on a healthy run for a reason already
    understood and recorded."""
    name = _obj(mods, f"encoded_train_s{ENCODER_STEPS}", "npz")
    local = local_path_for(name)
    manifest, _ = mods.gcs.consume_validated(name, local, bucket=bucket)
    if manifest is None:
        raise Stage3Halt(f"{name} carries no manifest; Phase A's artifact must")

    config = (manifest.get("fingerprint") or {}).get("config") or {}
    record["encoded_input"] = {
        "object": name,
        "payload_sha256": manifest.get("payload_sha256"),
        "payload_generation": manifest.get("payload_generation"),
        "fingerprint_commit": ((manifest.get("fingerprint") or {}).get("git") or {}
                               ).get("commit"),
        "config": config,
    }
    say(f"encoded input {name}: sha256 {manifest.get('payload_sha256')}, "
        f"generation {manifest.get('payload_generation')}, produced at commit "
        f"{record['encoded_input']['fingerprint_commit']}")

    if int(config.get("encoder_steps", -1)) != ENCODER_STEPS:
        raise Stage3Halt(f"encoded input was produced at encoder_steps="
                         f"{config.get('encoder_steps')}, this driver expects "
                         f"{ENCODER_STEPS}")
    if int(config.get("n_images", -1)) != EXPECTED_N:
        raise Stage3Halt(f"encoded input covers {config.get('n_images')} images, "
                         f"expected {EXPECTED_N}")

    with np.load(local, allow_pickle=False) as handle:
        encoded = {key: handle[key] for key in handle.files}
    if not np.array_equal(np.asarray(encoded["train_indices"]),
                          np.asarray(corpus["train_indices"])):
        raise Stage3Halt("the encoded artifact's row order does not match the corpus")

    # One-image re-encode, ULP tolerance.
    active = np.asarray(encoded["active_indices"])
    fresh = mods.local_converged_phases(
        np.asarray(corr["x_t_clip"])[0], steps=ENCODER_STEPS,
        seed=mods.encoder_gate.ENCODER_SEED).flatten()[active]
    stored = np.asarray(encoded["thetas_505"])[0]
    ulps = np.abs(fresh - stored) / np.spacing(np.maximum(np.abs(stored), 1e-300))
    max_ulp = float(np.max(ulps))
    max_abs = float(np.max(np.abs(fresh - stored)))
    record["encoded_input"]["reencode_max_ulp"] = max_ulp
    record["encoded_input"]["reencode_max_abs"] = max_abs
    if max_ulp > ENCODED_REENCODE_MAX_ULP:
        raise Stage3Halt(
            f"re-encoding image 0 differs from the stored array by {max_ulp:.1f} ULP "
            f"(max abs {max_abs:.3e}), beyond the {ENCODED_REENCODE_MAX_ULP} ULP "
            f"tolerance. Cross-architecture drift of a few ULP is expected; this is "
            f"larger and means the encode is not reproducing.")
    say(f"encoded input spot-check: image 0 re-encodes to within {max_ulp:.1f} ULP "
        f"(max abs {max_abs:.3e}), tolerance {ENCODED_REENCODE_MAX_ULP}")
    return encoded, name


def step4_restrict(mods, encoded, corr):
    theta0_505 = np.asarray(encoded["thetas_505"])
    expected = (np.asarray(corr["x_t_clip"]).shape[0], EXPECTED_N_ACTIVE)
    if theta0_505.shape != expected:
        raise Stage3Halt(f"encoded phases are {theta0_505.shape}, expected {expected}")
    say(f"encoded phases {theta0_505.shape} already restricted to the active support")
    return theta0_505


def step5_evolution(mods, bucket, theta0_505, topo, record, fp, parents):
    """Four graphs, batched, success-flag gated, thetas persisted per graph
    WITH fingerprints -- Decision 4 makes that load-bearing: the amendment
    audit consumes these exact objects rather than a recomputation that
    merely ought to match them."""
    n = theta0_505.shape[0]
    if n % EVOLVE_CHUNK:
        raise Stage3Halt(f"{n} images does not divide into {EVOLVE_CHUNK}-row chunks")
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

    evolved, failures, names = {}, [], {}
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
                if chunk % 20 == 0 or chunk == n_chunks - 1:
                    say(f"evolve/{graph} chunk {chunk + 1}/{n_chunks} rows {lo}:{hi} "
                        f"ok={int(success_np.sum())}/{hi - lo} ({time.time() - t0:.2f}s)")
                thetas.append(np.asarray(theta_T))
                flags.append(success_np)
            return {"theta_T": np.concatenate(thetas), "success": np.concatenate(flags)}

        name = _obj(mods, "theta_T", "npz",
                    condition=mods.conditions.path_segment(graph))
        loaded, _ = ensure_npz(mods, bucket, name, compute, fingerprint=fp,
                               parents=parents)
        evolved[graph] = loaded
        names[graph] = name

        success = np.asarray(loaded["success"])
        n_failed = int(np.count_nonzero(~success))
        entry = {"n_images": int(success.size), "n_failed": n_failed, "object": name}

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
        raise Stage3Halt("graph evolution: " + "; ".join(failures))
    return evolved, names


def step6_features(mods, bucket, theta0_505, evolved, topo, corr, corpus, ref_idx,
                   fp, parents_by_condition):
    n = theta0_505.shape[0]
    active_indices = np.asarray(topo["active_indices"])
    features, names = {}, {}

    for condition in mods.conditions.ALL_CONDITIONS:
        theta = (theta0_505 if condition == mods.conditions.PRE_EVOLUTION
                 else np.asarray(evolved[condition]["theta_T"]))

        def compute(theta=theta):
            return {"X": np.stack([mods.core.reference_node_features(theta[i], ref_idx)
                                   for i in range(n)])}

        name = _obj(mods, "features", "npz",
                    condition=mods.conditions.path_segment(condition))
        loaded, _ = ensure_npz(mods, bucket, name, compute, fingerprint=fp,
                               parents=parents_by_condition.get(condition))
        X = np.asarray(loaded["X"])
        if X.shape != (n, EXPECTED_FEATURE_DIM):
            raise Stage3Halt(f"{condition} features are {X.shape}, expected "
                             f"{(n, EXPECTED_FEATURE_DIM)}")
        features[condition] = X
        names[condition] = name

    raw_784 = np.asarray(corr["x_t_clip"]).reshape(n, FULL_GRID)
    features["raw_784"] = raw_784
    features["raw_505"] = raw_784[:, active_indices]

    non_finite = {cond: int(np.sum(~np.isfinite(X))) for cond, X in features.items()}
    non_finite = {cond: count for cond, count in non_finite.items() if count}
    if non_finite:
        raise Stage3Halt(f"non-finite features: {non_finite}")

    Y = np.asarray(corpus["images_01"]).reshape(n, FULL_GRID)[:, active_indices]
    say("features: " + ", ".join(f"{k}{v.shape}" for k, v in sorted(features.items())))
    say(f"target Y {Y.shape} (clean, active support); zero non-finite across all "
        f"{len(features)} conditions")
    return features, Y, names


def step7_ridge(mods, bucket, features, Y, y_strat, record, fp, parents_by_condition):
    """5-fold CV over the nine-alpha grid, seven conditions, plus Decision
    2's equivalence extension.

    ## What the equivalence check is at this scale, stated before it runs

    `DESIGN.md:330` is literal and scoped -- "at both the 1,000- and
    5,000-image stages" -- it named those two, it passed at both, and it is
    DISCHARGED. What runs here is a NEW, PRUDENTIAL EXTENSION at 12x the
    largest verified scale, fold-level only: it preserves the gate's
    literal quantities (clipped validation predictions, identical alpha
    selection) without seven full-corpus oracle fits.

    That framing describes what the check IS. It does not soften what a
    failure MEANS. Disagreement beyond the frozen tolerance on any fold or
    condition -- predictions or alpha selection -- HALTS EVERYTHING before
    Stage 4, full diagnosis required. "The locked gate still passed" is not
    an available reading of an extension failure at production scale.

    All seven conditions and all five folds run. Narrowing to the primary
    condition to save oracle time would make Decision 2's "any fold or
    condition" halt rule mean less than it says.
    """
    conditions = (*RAW_CONDITIONS, *mods.conditions.ALL_CONDITIONS)

    def compute_cv():
        out = {"conditions": {}, "halt_reasons": [], "order": list(conditions),
               "equivalence_framing": (
                   "new prudential extension at 12x the largest verified scale, "
                   "fold-level only; DESIGN.md:330's locked gate named 1,000 and "
                   "5,000 and is discharged. A failure here halts regardless."),
               "alpha_column_convention": (
                   "fold_kappa_alpha / fold_coef_norm are indexed at the SELECTED "
                   "alpha_index under the fixed-alpha regime; the *_by_alpha arrays "
                   "carry all nine columns for the reselected regime.")}
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
                t0 = time.time()
                equivalence = mods.ridge.ridge_equivalence_check(X, Y, y_strat)
                entry["equivalence"] = equivalence
                entry["equivalence_wallclock_s"] = time.time() - t0
                say(f"ridge/{condition}: equivalence EXTENSION: pred_diff="
                    f"{float(equivalence['max_abs_clipped_pred_diff']):.3e} "
                    f"(tol {float(equivalence['tol']):.0e}), "
                    f"alpha agrees={bool(equivalence['alpha_agrees'])}, "
                    f"{entry['equivalence_wallclock_s']:.0f}s")
                if not equivalence["passed"]:
                    out["halt_reasons"].append(
                        f"{condition}: equivalence extension pred_diff="
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

    cv_json, _ = ensure_json(mods, bucket, _obj(mods, "ridge_cv", "json"), compute_cv,
                             fingerprint=fp, parents=parents_by_condition.get("all"))
    record["gates"]["ridge"] = cv_json
    if cv_json["halt_reasons"]:
        raise Stage3Halt("ridge: " + "; ".join(cv_json["halt_reasons"]))

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

    final, _ = ensure_npz(mods, bucket, _obj(mods, "ridge_final", "npz"), compute_final,
                          fingerprint=fp, parents=parents_by_condition.get("all"))
    record["ridge_final"] = json.loads(final["summary_json"].item())
    return cv_json, final


def step8_cnn(mods, bucket, corpus, topo, corr, record, fp, parents):
    """The locked CNN config, fit on the 54,000 fit side, early-stopped on
    the LOCKED 6,000 validation partition, best of three seeds.

    The fit/validation split is Freeze 2's, taken by official index through
    the shared join rather than by slicing rows -- the corpus is in
    ascending official order, so a positional slice would silently be a
    different set the moment that order ever changed."""
    active_indices = np.asarray(topo["active_indices"])
    mask = mods.cnn.build_active_support_mask(active_indices,
                                              expect_n_active=EXPECTED_N_ACTIVE)
    train_indices = np.asarray(corpus["train_indices"])
    fit_rows, _ = mods.partition.index_join(
        np.asarray(corpus["fit_indices"]), train_indices,
        source_name="the fit role", target_name="the corpus")
    val_rows, _ = mods.partition.index_join(
        np.asarray(corpus["validation_indices"]), train_indices,
        source_name="the validation role", target_name="the corpus")

    images = np.asarray(corpus["images_01"])
    x_t_clip = np.asarray(corr["x_t_clip"])
    fit_clean, val_clean = images[fit_rows], images[val_rows]
    fit_noisy, val_noisy = x_t_clip[fit_rows], x_t_clip[val_rows]
    val_x_t = np.asarray(corr["x_t"])[val_rows]

    # Identity baseline on the locked validation partition, active support
    # only -- the exact function stage 1 used, called on this corpus.
    val_diag = mods.corruption.corruption_diagnostics(
        val_clean, val_x_t, val_noisy, active_indices,
        labels=np.asarray(corpus["labels"])[val_rows])
    identity_val_mse = float(val_diag["mse_postclip_505"])
    say(f"CNN identity baseline (locked validation partition, active support, "
        f"n={val_clean.shape[0]}): {identity_val_mse!r}")

    def compute_cnn():
        runs, per_seed_wallclock = [], {}
        for seed in mods.cnn.SEEDS:
            t0 = time.time()
            run = mods.cnn.train_cnn_for_seed(fit_noisy, fit_clean, val_noisy,
                                              val_clean, mask, seed=seed)
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
            "n_fit": int(fit_clean.shape[0]), "n_validation": int(val_clean.shape[0]),
        }
        arrays["summary_json"] = np.array(_dumps(meta))
        return arrays

    cnn_npz, _ = ensure_npz(mods, bucket, _obj(mods, "cnn_production", "npz"),
                            compute_cnn, fingerprint=fp, parents=parents)
    summary = json.loads(cnn_npz["summary_json"].item())
    record["cnn"] = summary
    say(f"CNN best: seed={summary['best_seed']}, best_epoch={summary['best_epoch']}, "
        f"clipped val MSE={summary['best_clipped_val_mse']!r} vs identity "
        f"{summary['identity_val_mse']!r} (mechanical sanity, NON-INFERENTIAL, "
        f"not a locked comparison)")
    return cnn_npz, summary


def step9_stats_smoke(mods, bucket, final, corr, corpus, topo, record, fp, parents):
    """The stats machinery exercised end to end, in-sample and
    non-inferential, exactly as at stages 1 and 2. Not a result, and the
    banner says so in the artifact itself."""
    n = corpus["images_01"].shape[0]
    active_indices = np.asarray(topo["active_indices"])
    diag = mods.corruption.corruption_diagnostics(
        np.asarray(corpus["images_01"]), np.asarray(corr["x_t"]),
        np.asarray(corr["x_t_clip"]), active_indices,
        labels=np.asarray(corpus["labels"]))

    mse_by_condition = {mods.conditions.PRE_EVOLUTION:
                        np.asarray(final[f"mse_{mods.conditions.PRE_EVOLUTION}"])}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        mse_by_condition[graph] = np.asarray(final[f"mse_{graph}"])
    mse_by_condition[IDENTITY_KEY] = np.asarray(diag["per_image_mse_postclip_505"])

    def compute():
        return mods.stats.run_stage2b_inference(
            mse_by_condition, np.asarray(corpus["labels"]), identity_key=IDENTITY_KEY)

    inference, _ = ensure_json(mods, bucket, _obj(mods, "stats_smoke", "json"), compute,
                               fingerprint=fp, parents=parents)

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

    ensure_text(mods, bucket, _obj(mods, "stats_smoke", "txt"), compute_text,
                fingerprint=fp, parents=parents)
    record["stats_smoke"] = dict(inference)
    record["stats_smoke"]["skipped"] = False
    say(f"stats machinery exercised end to end at n={n} (in-sample; not a result)")
    return inference


def step10_report(mods, bucket, record):
    def compute_json():
        return record

    def compute_text():
        lines = ["Stage 2B feasibility-ladder stage 3 (Phase B) report",
                 f"commit: {record['run'].get('head_sha')}",
                 f"verdict: {record.get('verdict')}", ""]
        if record.get("halt_reason"):
            lines += [f"halt reason: {record['halt_reason']}", ""]
        lines += ["timings (s):", _dumps(record.get("timings", {})), "",
                  "full record:", _dumps(record), "",
                  str(record.get("verdict", FAIL_SENTINEL))]
        return "\n".join(lines) + "\n"

    # Run-scoped, and therefore carrying a run id: a resumed run writes a
    # NEW report rather than destroying the record of what the attempt that
    # died had seen. No kind retains an overwrite path anywhere.
    kind = f"stage3_report_{record['run']['run_id']}"
    ensure_json(mods, bucket, _obj(mods, kind, "json"), compute_json)
    ensure_text(mods, bucket, _obj(mods, kind, "txt"), compute_text)


def new_record():
    return {"run": {"run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
            "timings": {}, "gates": {}, "evolution": {}, "cnn": {},
            "sizing_probe": {}, "encoded_input": {}, "stats_smoke": {},
            "verdict": None, "halt_reason": None}


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

        # Provenance established BEFORE anything is generated, so every
        # artifact this run writes commits to the same recorded producer.
        fp = build_fingerprint(mods, CLONE_DIR, {
            "ladder_stage": LADDER_STAGE,
            "split": SPLIT,
            "population": "official KMNIST training split, all 60,000",
            "n_images": EXPECTED_N,
            "row_order": "ascending official training index",
            "encoder_steps": ENCODER_STEPS,
            "alpha_bar": mods.corruption.ALPHA_BAR,
            "corruption_scheme": "SHA256(split:index:42) -> PCG64, per image",
            "evolve_chunk": EVOLVE_CHUNK,
            "n_active": EXPECTED_N_ACTIVE,
            "reference_node": EXPECTED_REF_IDX,
            "feature_dim": EXPECTED_FEATURE_DIM,
            "ridge_alpha_grid": list(mods.ridge.ALPHA_GRID),
            "ridge_n_splits": mods.ridge.N_SPLITS,
            "ridge_fold_seed": mods.ridge.FOLD_SEED,
            "dtype": "float64",
        })
        record["run"]["fingerprint"] = {
            "source_manifest_digest": fp["source_manifest_digest"],
            "config_digest": fp["config_digest"],
            "git": fp["git"],
            "n_source_files": len(fp["source_manifest"]),
        }
        say(f"fingerprint established: config {fp['config_digest'][:16]}..., "
            f"sources {fp['source_manifest_digest'][:16]}... over "
            f"{len(fp['source_manifest'])} files")

        with timed_step("stage_kmnist", record["timings"]):
            kmnist_dir, staged = stage_kmnist(mods, bucket, CLONE_DIR)
            record["run"]["kmnist"] = {name: os.path.getsize(path)
                                       for name, path in sorted(staged.items())}

        with timed_step("0_preflight", record["timings"]):
            step0_preflight(mods, record)
        with timed_step("1_corpus", record["timings"]):
            corpus = step1_corpus(mods, bucket, kmnist_dir, fp)
            corpus_name = _obj(mods, "corpus", "npz")
        with timed_step("1b_topologies", record["timings"]):
            topo, topo_meta, ref_idx = step1b_topologies(mods, bucket)
            record["run"]["topologies"] = topo_meta
        with timed_step("2_corruption", record["timings"]):
            corr, join_report = step2_corruption(mods, bucket, corpus, fp, corpus_name)
            record["run"]["corruption_join"] = join_report
            corr_name = _obj(mods, "corruption", "npz")

        # The probe runs BEFORE anything expensive. A probe that could only
        # run after evolution and feature extraction could not stop the run
        # from paying for them.
        with timed_step("2b_sizing_probe", record["timings"]):
            step2b_sizing_probe(mods, bucket, record, fp)

        with timed_step("3_encoded_input", record["timings"]):
            encoded, encoded_name = step3_encoded_input(mods, bucket, corpus, corr,
                                                        record)
        with timed_step("4_restrict", record["timings"]):
            theta0_505 = step4_restrict(mods, encoded, corr)

        evo_parents = parent_map(mods, bucket, (encoded_name, corpus_name))
        with timed_step("5_evolution", record["timings"]):
            evolved, theta_names = step5_evolution(mods, bucket, theta0_505, topo,
                                                   record, fp, evo_parents)

        feature_parents = {mods.conditions.PRE_EVOLUTION: dict(evo_parents)}
        for graph in mods.conditions.EVOLVED_GRAPHS:
            feature_parents[graph] = parent_map(mods, bucket, (theta_names[graph],))
        with timed_step("6_features", record["timings"]):
            features, Y, feature_names = step6_features(
                mods, bucket, theta0_505, evolved, topo, corr, corpus, ref_idx,
                fp, feature_parents)

        all_parents = {"all": parent_map(
            mods, bucket, (*feature_names.values(), corpus_name, corr_name))}
        with timed_step("7_ridge", record["timings"]):
            _cv_json, final = step7_ridge(mods, bucket, features, Y,
                                          np.asarray(corpus["labels"]), record, fp,
                                          all_parents)
        with timed_step("8_cnn", record["timings"]):
            step8_cnn(mods, bucket, corpus, topo, corr, record, fp,
                      parent_map(mods, bucket, (corpus_name, corr_name)))
        with timed_step("9_stats_smoke", record["timings"]):
            step9_stats_smoke(mods, bucket, final, corr, corpus, topo, record, fp,
                              parent_map(mods, bucket,
                                         (_obj(mods, "ridge_final", "npz"),)))

        record["verdict"] = OK_SENTINEL
    except Stage3Halt as halt:
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
                print(f"[stage3] report FAILED to write: {type(exc).__name__}: {exc}",
                      flush=True)

    verdict = record["verdict"] or FAIL_SENTINEL
    print(verdict if status == 0 else f"{verdict} {record['halt_reason']}", flush=True)
    return status


if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    _status = main()
    if _status:
        sys.exit(_status)
