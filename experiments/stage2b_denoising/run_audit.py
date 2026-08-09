"""Stage 2B amendment-impact audit: quantifies the representational impact
of raising the encoder budget from 150 to 1,200 steps -- the amendment made
after ladder stage 1's gate failure, per the frozen `AUDIT_PROTOCOL.md`.

## Architecture: replaces the earlier local-CLI stub entirely

The file this replaces was an `argparse` CLI (`--encoded-150`,
`--production-ridge-rerun`, `--preflight`) checking two LOCAL paths via
`stage2b_audit.require_audit_prerequisites`. That shape does not fit what
this audit actually needs: real GPU evolution and OOF ridge at 60,000
images, which only runs the way stage 3/4 do -- `mighty-colab exec -f
script` transmitting this file's TEXT into a Colab kernel, config read from
environment variables because `argparse`/`__file__`/local paths do not
exist in that context until `bootstrap_repo()` has cloned the repo, inside
`main()`. This is a disclosed, deliberate replacement, not a silent one:
`stage2b_audit.require_audit_prerequisites` stays in the pure module,
tested and available, but this driver does not call it -- `step1_*`'s own
`consume_validated` calls are strictly stronger (manifest + digest, not
merely `os.path.exists`) and supersede it.

## What this audits, and what it does not recompute

Per `PHASE_B_PLAN.md`'s Decision 4 (frozen): the 1,200-step side is Phase
B's (stage 3's) own persisted evolved-feature artifacts, consumed through
the validated path -- never re-evolved. Only the 150-step budget is new
compute here. `gates.toml` names this driver as the consumer of five
`binding_gate` rows and six `binding_value` rows still `pending_consumer`;
each step below says which it discharges.

## No one-shot lock

Unlike stage 4 (the single locked confirmatory evaluation), this audit is
diagnostic -- AUDIT_PROTOCOL.md: "It is not a model-selection knob." Every
artifact goes through `ensure_artifact`'s ordinary skip-if-exists path;
there is no `official_result`-style write-once object.

`AUDIT_OK` / `AUDIT_FAIL` on stdout, non-zero exit on failure.
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
from itertools import combinations

import numpy as np

# ---- identity and sentinels ----
DRIVER_FILENAME = "run_audit.py"
LADDER_STAGE = 5          # this driver's own artifacts (150-step thetas
                           # and features, OOF results, distances, report)
TRAIN_STAGE = 3            # stage 3 (Phase B): source of the reused
                           # 1,200-step thetas/features and production alphas
STAGE1_STAGE = 1
STAGE2_STAGE = 2
SPLIT = "train"
OK_SENTINEL = "AUDIT_OK"
FAIL_SENTINEL = "AUDIT_FAIL"

# ---- bootstrap ----
REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
WORK_DIR = "/content/stage2b_audit"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)

# ---- environment variable names (set by `mighty-colab exec --env K=V`) ----
ENV_COMMIT = "BONSAI_COMMIT"
ENV_BUCKET = "BONSAI_GCS_BUCKET"
ENV_CREDENTIALS = "BONSAI_GCS_CREDENTIALS"
ENV_DRIVER_SHA = "BONSAI_DRIVER_SHA256"

# ---- run parameters ----
EVOLVE_CHUNK = 250        # 60,000 / 250 = 240 exact chunks; same size stage
                           # 3 uses, so the divisibility halt stays valid
FULL_GRID = 784
EXPECTED_N = 60_000
EXPECTED_N_ACTIVE = 505
EXPECTED_REF_IDX = 363    # position WITHIN the 505-active-support array,
                           # not a raw 784-pixel index -- matches
                           # stage2b_audit.GAUGE_NODE, asserted equal below
EXPECTED_FEATURE_DIM = 2 * EXPECTED_N_ACTIVE - 2   # 1008
AUDIT_NEW_STEPS = 150      # the budget this driver evolves
PRODUCTION_STEPS = 1200    # consumed from stage 3, never re-evolved
HEARTBEAT_SECONDS = 30.0

# ---- pre-contract inputs, pinned by content ----
#
# Stages 1 and 2 wrote every artifact before the fingerprint contract
# existed (same situation `run_ladder_stage3.py`'s PINNED_SHA256
# documents), so these carry no manifest and are consumed with the named
# `require_manifest=False` opt-out plus a byte pin -- the only thing
# standing between a resumed run and silently different input. Computed by
# downloading each object live and hashing it (2026-08-09), read-only, no
# GPU/compute cost. `stage1/topologies` is reused VERBATIM from
# `run_ladder_stage3.py`'s own pin -- same immutable object, unchanged.
PINNED_SHA256 = {
    "stage1/topologies":
        "f671e63cc00b1612db0da5976c14b8880e4c4f90ae7fb192297721665f1907a4",
    "stage1/corpus":
        "9e61f1353fa0da9179f0257202639ab20e90ad070b3b0bd55e8851b1dabafadb",
    "stage1/ridge_cv":
        "99c07b0f852c99bdb5e4f11b4a6d59cfb7fd96e8c47bd4e3c5e259da12ab0dde",
    "stage1/features/pre_evolution":
        "1b0097bb9c024f17cf1301a4519ccacc0484b7b04de978142fa56f7ead010dbb",
    "stage1/features/T":
        "2f866e8cbd30546bbdc7124c07f3ea63c981d003194b73a7e967bfcc51ff581d",
    "stage1/features/lattice":
        "79296580ab605af29e493ad2978a0db42276809074f2af9d499cc33d24b15530",
    "stage1/features/rewired":
        "129503d1f0d124f3f79f1a7de2f339723ad6d61386b1993ecd076ece7fbe27ac",
    "stage1/features/curr_random":
        "8cc336c86168c70bf809873c1fd483475609a5d96d7f68bc139375a800c94390",
    "stage2/corpus":
        "0b2ca2d0e62dac2f669eb70109ee0b094648c727bce8f6ef484f2be1b49f51fd",
    "stage2/ridge_cv":
        "7f29dd36c4a278d5e1d0361cd21f224c1b4d47b783aa60e2a6a294a02e5851d8",
    "stage2/features/pre_evolution":
        "3f5ac4e3d56a37ffbe16b968c955201e5c576ed9bf2664514772865916574899",
    "stage2/features/T":
        "d0a584e072a4e58d02b36ae695d1497251b3b3cbc228331c866b9220320dd0f9",
    "stage2/features/lattice":
        "e16fc0b263a9265a573b40fa0d850596ea22c516621178de2341a357e61faea9",
    "stage2/features/rewired":
        "5ec45f3de7371e371071a801dfddf82e8ba0abfdd25b0de6323a033974c8c002",
    "stage2/features/curr_random":
        "145fd31f4d1988dff75bc4772102f377df24ac1716c473d0785a242ced1e1f23",
}

# ---- the sizing probe's budgets ----
#
# This driver's ridge step (`step6_oof_both_regimes`) has a DIFFERENT cost
# shape from stage 3's: no sklearn oracle fits at all (this audit never
# calls `ridge_equivalence_check`), and TWO JAX SVDs per (steps, condition)
# pair instead of one -- `compute_oof_alpha_regimes` calls both
# `cross_validate_alpha` (5 fold-level SVDs) and `oof_per_image_mse`
# (5 more, same fold shape) for each pair. 2 budgets x 5 conditions x
# (5 + 5) = 100 JAX SVDs, ZERO sklearn fits. Copying stage 3's mixed
# JAX+sklearn budget constants would misdescribe this step's actual cost
# (principle 18) -- these are derived from this driver's own call count.
PROBE_JAX_SVD_COUNT = 100
# No measured per-SVD timing exists yet (nothing has run) -- these budgets
# are safety ceilings fixed in advance, not projections, mirroring stage
# 3's own PROBE_RIDGE_BUDGET_S/PROBE_RUN_BUDGET_S as round, conservative,
# stated-in-advance numbers. Set lower than stage 3's because stage 3's
# ridge step was dominated by its (removed here) sklearn leg: "315 oracle
# SVDs against 35 production ones" -- pure-JAX work at ~2.86x stage 3's
# fold-level SVD count is expected to cost markedly less wall-clock than a
# step whose dominant cost was 315 CPU sklearn fits.
PROBE_RIDGE_BUDGET_S = 3_600.0
PROBE_RUN_BUDGET_S = 5_400.0
PROBE_DEVICE_PEAK_BUDGET_BYTES = 12 * 1024**3
# Per-fold array footprint is IDENTICAL to stage 3's: same (n_train, 1008)
# X, same (n_train, 505) Y at the same n_train~48,000 -- so stage 3's own
# modelled-array-bytes derivation applies unchanged, not re-derived.
PROBE_MODELLED_ARRAY_BYTES = 1_020_000_000

_RUN_T0 = time.time()
_STEP = {"name": "startup", "t0": _RUN_T0}


class AuditHalt(Exception):
    """A prespecified halt condition fired."""


# ---------------------------------------------------------------- plumbing

def say(line):
    print(f"[audit {time.time() - _RUN_T0:7.1f}s] {line}", flush=True)


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
        print(f"[audit heartbeat] {step['name']} {time.time() - step['t0']:.0f}s",
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
        raise AuditHalt(f"clone is at {head}, expected {commit}")

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
        raise AuditHalt(
            f"driver identity mismatch: the transmitted file hashes to "
            f"{expected_sha256}, the commit's copy to {digest}. The code that ran "
            f"is not the code at {os.path.basename(clone_dir)}'s pinned commit.")
    if not expected_sha256:
        say(f"{ENV_DRIVER_SHA} unset; recorded {digest} without comparison")
    return result


def load_modules(clone_dir):
    """Every repo import, in one place and in an order that matters.

    `stage2b_ridge` FIRST: it enables jax's x64 mode at import. Adds
    `stage2b_audit` -- the pure calculation layer this driver composes and
    implements none of."""
    add_repo_to_path(clone_dir)
    import stage2b_ridge as ridge                                       # noqa: E402
    from evolve_on_graph_jax import batched_evolve_on_graph_jax         # noqa: E402
    import jax                                                          # noqa: E402
    import jax.numpy as jnp                                             # noqa: E402

    import stage2b_audit as audit                                       # noqa: E402
    import stage2b_conditions as conditions                             # noqa: E402
    import stage2b_fingerprint as fingerprint                           # noqa: E402
    import stage2b_gcs as gcs                                           # noqa: E402
    import stage2b_verify_gpu as verify_gpu                             # noqa: E402
    import stage2a_core as core                                        # noqa: E402

    mods = types.SimpleNamespace(
        ridge=ridge, batched_evolve_on_graph_jax=batched_evolve_on_graph_jax,
        jax=jax, jnp=jnp, audit=audit, conditions=conditions,
        fingerprint=fingerprint, gcs=gcs, verify_gpu=verify_gpu, core=core)

    for name in ("ridge", "audit", "conditions", "fingerprint", "gcs", "verify_gpu", "core"):
        origin = getattr(mods, name).__file__
        if not os.path.abspath(origin).startswith(os.path.abspath(clone_dir)):
            raise AuditHalt(f"module {name} resolved to {origin}, outside {clone_dir}")
    return mods


def build_fingerprint(mods, clone_dir, config):
    return mods.fingerprint.compute(
        entrypoint=os.path.join(clone_dir, "experiments", "stage2b_denoising",
                                DRIVER_FILENAME),
        repo_root=clone_dir, require_clean=True, config=config)


# ------------------------------------------------------- artifact wrappers

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
    """This driver never passes `allow_test_split` anywhere -- it reads
    stage 1/2/3's TRAIN-side artifacts and writes its own under
    `LADDER_STAGE`; no test-split object is touched by this file."""
    return mods.gcs.object_path(stage=stage, condition=condition, kind=kind,
                                ext=ext, split=SPLIT)


def consume_pinned(mods, bucket, object_name, pin_key, local_path=None):
    """A pre-contract artifact, consumed with the named opt-out and checked
    against its pinned digest -- mirrors `run_ladder_stage3.py::consume_pinned`
    exactly."""
    local_path = local_path or local_path_for(object_name)
    mods.gcs.consume_validated(object_name, local_path, bucket=bucket,
                               require_manifest=False)
    with open(local_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    expected = PINNED_SHA256[pin_key]
    if digest != expected:
        raise AuditHalt(
            f"{object_name!r} does not match its pinned digest: expected {expected}, "
            f"got {digest}. This object carries no manifest (pre-contract history), "
            f"so the pin is the only thing standing between this run and silently "
            f"different input. Do not update the pin to make this pass -- find out "
            f"what changed.")
    say(f"consumed {object_name} (pre-contract, pinned sha256 {digest[:16]}...)")
    return local_path


def parent_map(mods, bucket, names):
    out = {}
    for name in names:
        manifest = mods.gcs.read_manifest(name, bucket=bucket)
        digest = (manifest or {}).get("payload_sha256")
        if digest is None:
            local = local_path_for(name)
            if not os.path.isfile(local):
                raise AuditHalt(
                    f"cannot record {name!r} as a parent: it carries no manifest and "
                    f"no local copy exists at {local}. A parent digest that cannot be "
                    f"resolved would pin nothing.")
            with open(local, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        out[name] = digest
    return out


def grid_tag(alphas):
    canonical = ",".join(repr(float(a)) for a in alphas)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"g{len(tuple(alphas))}_{digest}"


# ------------------------------------------------------------ sizing probe

def probe_projections(measured):
    total = measured["jax_svd_s"] * PROBE_JAX_SVD_COUNT
    return {"jax_svd_s_measured": measured["jax_svd_s"],
            "jax_svd_count": PROBE_JAX_SVD_COUNT,
            "ridge_projected_s": total,
            "device_peak_bytes": measured.get("device_peak_bytes")}


def evaluate_probe(measured, elapsed_so_far_s):
    """Pure and separate from the measuring so the halt path is testable on
    synthetic over-budget input (a guard never seen to fail is not yet a
    guard). Mirrors `run_ladder_stage3.py::evaluate_probe`'s structure at
    this driver's own (JAX-only) cost shape."""
    proj = probe_projections(measured)
    proj["elapsed_before_probe_s"] = elapsed_so_far_s
    proj["elapsed_plus_ridge_projected_s"] = elapsed_so_far_s + proj["ridge_projected_s"]
    proj["excluded_from_projection"] = [
        "encoded-input downloads", "150-step evolution", "150-step features",
        "stage-1/2 cross-check", "feature distances", "artifact uploads"]
    proj["budgets"] = {"ridge_s": PROBE_RIDGE_BUDGET_S, "run_s": PROBE_RUN_BUDGET_S,
                       "device_peak_bytes": PROBE_DEVICE_PEAK_BUDGET_BYTES}

    reasons = []
    if proj["ridge_projected_s"] > PROBE_RIDGE_BUDGET_S:
        reasons.append(f"projected ridge {proj['ridge_projected_s']:.0f}s exceeds the "
                       f"{PROBE_RIDGE_BUDGET_S:.0f}s budget")
    if proj["elapsed_plus_ridge_projected_s"] > PROBE_RUN_BUDGET_S:
        reasons.append(f"elapsed-plus-ridge {proj['elapsed_plus_ridge_projected_s']:.0f}s "
                       f"exceeds the {PROBE_RUN_BUDGET_S:.0f}s budget")
    peak = proj.get("device_peak_bytes")
    if peak is not None and peak > PROBE_DEVICE_PEAK_BUDGET_BYTES:
        reasons.append(f"device peak {peak / 1024**3:.2f} GiB exceeds the "
                       f"{PROBE_DEVICE_PEAK_BUDGET_BYTES / 1024**3:.0f} GiB budget")
    return proj, reasons


def _device_peak_bytes(mods):
    try:
        stats = mods.jax.devices()[0].memory_stats()
    except Exception:                               # noqa: BLE001 - optional
        return None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


def step2_sizing_probe(mods, bucket, record, fp, n_total=EXPECTED_N):
    """One fold's JAX SVD at true production shape, measured BEFORE the
    expensive OOF ridge step runs -- principle 18 / `PHASE_B_PLAN.md:351-358`.
    Published BEFORE the halt is raised, mirroring
    `run_ladder_stage3.py::step2b_sizing_probe`'s own reasoning: a probe
    that halts is exactly the case where its evidence matters most."""
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

    measured = {"jax_svd_s": jax_svd_s, "device_peak_bytes": peak, "n_train": n_train,
                "feature_dim": EXPECTED_FEATURE_DIM,
                "matrix": "seeded standard normal at production shape; SIZING ONLY"}
    proj, reasons = evaluate_probe(measured, time.time() - _RUN_T0)
    record["sizing_probe"] = {"measured": measured, **proj, "halted": bool(reasons),
                              "reasons": reasons}

    kind = f"probe_sizing_{record['run']['run_id']}"
    try:
        ensure_json(mods, bucket, _obj(mods, kind, "json"),
                    lambda: record["sizing_probe"], fingerprint=fp)
    except Exception as exc:                    # noqa: BLE001 - never mask the halt
        say(f"sizing probe: FAILED to publish its measurement "
            f"({type(exc).__name__}: {exc}); continuing to the verdict")

    say(f"sizing probe projection: ridge {proj['ridge_projected_s']:.0f}s "
        f"(x{PROBE_JAX_SVD_COUNT} JAX SVDs, no sklearn leg); elapsed+ridge "
        f"{proj['elapsed_plus_ridge_projected_s']:.0f}s")
    if reasons:
        raise AuditHalt("sizing probe: " + "; ".join(reasons))
    say("sizing probe: within every budget; proceeding")
    return proj


# ------------------------------------------------------------------ steps

def step0_preflight(mods, record):
    try:
        mods.verify_gpu.device_preflight()
    except SystemExit as exc:
        raise AuditHalt(f"device preflight: {exc}") from exc
    probe = mods.jnp.zeros(1, dtype=mods.jnp.float64)
    record["run"]["devices"] = [str(d) for d in mods.jax.devices()]
    record["run"]["realised_float64_dtype"] = str(probe.dtype)
    record["run"]["jax_enable_x64"] = bool(mods.jax.config.jax_enable_x64)


def step1_load_train_side(mods, bucket, record):
    """Stage 3's 60,000-image corpus and both encoded budgets, validated.

    Discharges nothing on its own; establishes the official-index ordering
    every later step relies on (`assert_official_indices`), and confirms
    the 150/1200 encoded artifacts differ only in `encoder_steps`
    (`assert_same_audit_inputs`, over recorded fingerprint config, per
    `PHASE_B_PLAN.md:327-333`)."""
    corpus_name = _obj(mods, "corpus", "npz", stage=TRAIN_STAGE)
    corpus_local = local_path_for(corpus_name)
    mods.gcs.consume_validated(corpus_name, corpus_local, bucket=bucket)
    with np.load(corpus_local, allow_pickle=False) as handle:
        corpus = {key: handle[key] for key in handle.files}
    if corpus["images_01"].shape[0] != EXPECTED_N:
        raise AuditHalt(f"stage {TRAIN_STAGE} corpus has "
                        f"{corpus['images_01'].shape[0]} images, expected {EXPECTED_N}")
    train_indices = mods.audit.assert_official_indices(
        np.asarray(corpus["train_indices"]), name="stage3 corpus train_indices")

    encoded, configs = {}, {}
    for steps in (AUDIT_NEW_STEPS, PRODUCTION_STEPS):
        name = _obj(mods, f"encoded_train_s{steps}", "npz", stage=TRAIN_STAGE)
        local = local_path_for(name)
        manifest, _ = mods.gcs.consume_validated(name, local, bucket=bucket)
        if manifest is None:
            raise AuditHalt(f"{name} carries no manifest")
        configs[steps] = (manifest.get("fingerprint") or {}).get("config") or {}
        with np.load(local, allow_pickle=False) as handle:
            enc = {key: handle[key] for key in handle.files}
        idx = mods.audit.assert_official_indices(
            np.asarray(enc["train_indices"]), name=f"encoded_s{steps} train_indices")
        if not np.array_equal(idx, train_indices):
            raise AuditHalt(f"encoded_s{steps}'s official indices do not match the "
                            f"corpus's order")
        encoded[steps] = enc
        say(f"encoded_s{steps} consumed: sha256 {manifest.get('payload_sha256')}")

    mods.audit.assert_same_audit_inputs(configs[AUDIT_NEW_STEPS], configs[PRODUCTION_STEPS])
    say("150/1200 encoded artifacts confirmed to differ only in encoder_steps")

    record["train_side"] = {"corpus_object": corpus_name,
                            "encoded_objects": {str(s): _obj(mods, f"encoded_train_s{s}",
                                                             "npz", stage=TRAIN_STAGE)
                                                for s in (AUDIT_NEW_STEPS, PRODUCTION_STEPS)}}
    return corpus, encoded, train_indices, corpus_name


def step1b_topologies(mods, bucket):
    """Stage 1's pinned topologies, reused verbatim -- population-independent,
    byte-identical `active_indices`/`W_{graph}` across every rung."""
    name = mods.gcs.object_path(stage=STAGE1_STAGE, condition=None, kind="topologies",
                                ext="npz", split=SPLIT)
    local = consume_pinned(mods, bucket, name, "stage1/topologies")
    with np.load(local, allow_pickle=False) as handle:
        topo = {key: handle[key] for key in handle.files}
    meta = json.loads(topo["summary_json"].item())
    n_active = int(np.asarray(topo["active_indices"]).size)
    if n_active != EXPECTED_N_ACTIVE:
        raise AuditHalt(f"active support has {n_active} nodes, expected {EXPECTED_N_ACTIVE}")
    ref_idx = int(meta["nodes_T"]["median"])
    if ref_idx != EXPECTED_REF_IDX:
        raise AuditHalt(f"T's median-degree node is {ref_idx}, expected {EXPECTED_REF_IDX}")
    if ref_idx != mods.audit.GAUGE_NODE:
        raise AuditHalt(f"topology ref_idx {ref_idx} disagrees with "
                        f"stage2b_audit.GAUGE_NODE {mods.audit.GAUGE_NODE}")
    say(f"topologies reused from stage {STAGE1_STAGE}; n_active={n_active}, "
        f"ref_idx={ref_idx} (505-space, matches GAUGE_NODE)")
    return topo, ref_idx


def step1c_stage3_evolved(mods, bucket):
    """Phase B's OWN persisted 1,200-step thetas and features, consumed --
    never re-evolved. Discharges the reuse half of
    `PHASE_B_PLAN.md:313-325` (Decision 4)."""
    theta_1200, features_1200 = {}, {}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        seg = mods.conditions.path_segment(graph)
        name = mods.gcs.object_path(stage=TRAIN_STAGE, condition=seg, kind="theta_T",
                                    ext="npz", split=SPLIT)
        local = local_path_for(name)
        mods.gcs.consume_validated(name, local, bucket=bucket)
        with np.load(local, allow_pickle=False) as handle:
            theta_1200[graph] = np.asarray(handle["theta_T"])
    for condition in mods.conditions.ALL_CONDITIONS:
        seg = mods.conditions.path_segment(condition)
        name = mods.gcs.object_path(stage=TRAIN_STAGE, condition=seg, kind="features",
                                    ext="npz", split=SPLIT)
        local = local_path_for(name)
        mods.gcs.consume_validated(name, local, bucket=bucket)
        with np.load(local, allow_pickle=False) as handle:
            features_1200[condition] = np.asarray(handle["X"])
    say(f"stage {TRAIN_STAGE}'s 1,200-step thetas/features consumed for "
        f"{len(mods.conditions.ALL_CONDITIONS)} conditions")
    return theta_1200, features_1200


def step1d_production_alphas(mods, bucket):
    """The 1,200-step production alpha per condition, from stage 3's own
    ridge_final -- the "fixed" alpha regime's input. Restricted to the 5
    audit conditions; stage 3's raw-pixel conditions are not part of the
    audit's contrast."""
    tag = grid_tag(mods.ridge.ALPHA_GRID)
    name = _obj(mods, f"ridge_final_{tag}", "npz", stage=TRAIN_STAGE)
    local = local_path_for(name)
    mods.gcs.consume_validated(name, local, bucket=bucket)
    with np.load(local, allow_pickle=False) as handle:
        summary = json.loads(handle["summary_json"].item())
    alphas = {condition: float(summary["alphas"][condition])
             for condition in mods.conditions.ALL_CONDITIONS}
    say(f"production (1200-step) alphas: {alphas}")
    return alphas


def step3_evolve_150(mods, bucket, encoded, theta_1200, topo, record, fp):
    """Evolve ONLY the 150-step budget. The 1,200-step spot-check below is a
    REPORTED DIAGNOSTIC, not a gate: AUDIT_PROTOCOL.md's analytic
    resolution bound is derived for MSE/clipped-prediction perturbations
    and has no defined scale for a raw phase difference in radians -- using
    it as a halt threshold here would be principle 22's mis-scaled-tolerance
    failure. The actual provenance guarantee for the reused 1,200-step
    artifacts is validated consumption + fingerprint lineage
    (`PHASE_B_PLAN.md:313-325`), not this spot-check."""
    theta0_150 = np.asarray(encoded[AUDIT_NEW_STEPS]["thetas_505"])
    theta0_1200 = np.asarray(encoded[PRODUCTION_STEPS]["thetas_505"])
    n = theta0_150.shape[0]
    if n % EVOLVE_CHUNK:
        raise AuditHalt(f"{n} images does not divide into {EVOLVE_CHUNK}-row chunks")
    n_chunks = n // EVOLVE_CHUNK

    theta_150, spot_check = {}, {}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        W = mods.jnp.asarray(np.asarray(topo[f"W_{graph}"]))
        thetas, flags = [], []
        for chunk in range(n_chunks):
            lo, hi = chunk * EVOLVE_CHUNK, (chunk + 1) * EVOLVE_CHUNK
            theta_T, success = mods.batched_evolve_on_graph_jax(
                mods.jnp.asarray(theta0_150[lo:hi]), W)
            mods.jax.block_until_ready(theta_T)
            success_np = np.asarray(success)
            if chunk % 40 == 0 or chunk == n_chunks - 1:
                say(f"evolve150/{graph} chunk {chunk + 1}/{n_chunks} rows {lo}:{hi} "
                    f"ok={int(success_np.sum())}/{hi - lo}")
            thetas.append(np.asarray(theta_T))
            flags.append(success_np)
        theta_arr = np.concatenate(thetas)
        success_arr = np.concatenate(flags)
        n_failed = int(np.count_nonzero(~success_arr))
        if n_failed:
            raise AuditHalt(f"150-step evolution/{graph}: {n_failed}/{success_arr.size} "
                            f"solves failed")

        name = _obj(mods, "theta_T_s150", "npz", condition=mods.conditions.path_segment(graph))
        loaded, _ = ensure_npz(mods, bucket, name,
                               lambda theta_arr=theta_arr, success_arr=success_arr:
                               {"theta_T": theta_arr, "success": success_arr},
                               fingerprint=fp)
        theta_150[graph] = np.asarray(loaded["theta_T"])

        # Reported diagnostic only (see docstring): image 0 re-evolved at
        # 1,200 steps, diffed against stage 3's stored value.
        theta_T_1200_image0, success_image0 = mods.batched_evolve_on_graph_jax(
            mods.jnp.asarray(theta0_1200[0:1]), W)
        mods.jax.block_until_ready(theta_T_1200_image0)
        fresh = np.asarray(theta_T_1200_image0)[0]
        stored = theta_1200[graph][0]
        diff = mods.audit.wrapped_phase_difference(fresh, stored)
        max_abs = float(np.max(np.abs(diff)))
        spot_check[graph] = {"image0_max_abs_wrapped_phase_diff": max_abs,
                             "success": bool(np.asarray(success_image0)[0])}
        if not np.isfinite(max_abs):
            raise AuditHalt(f"1200-step spot-check for {graph}: non-finite phase diff")
        say(f"1200-step spot-check/{graph}: image0 max abs wrapped phase diff "
            f"{max_abs:.3e} (diagnostic only, not a gate)")

    record["evolution_150"] = {"n_images": n, "spot_check_vs_stage3_1200": spot_check}
    return theta_150


def step4_features_150(mods, bucket, encoded, theta_150, ref_idx, record, fp):
    theta0_150 = np.asarray(encoded[AUDIT_NEW_STEPS]["thetas_505"])
    n = theta0_150.shape[0]
    features_150 = {}
    for condition in mods.conditions.ALL_CONDITIONS:
        theta = (theta0_150 if condition == mods.conditions.PRE_EVOLUTION
                 else theta_150[condition])

        def compute(theta=theta):
            return {"X": np.stack([mods.core.reference_node_features(theta[i], ref_idx)
                                   for i in range(n)])}

        name = _obj(mods, "features_s150", "npz", condition=mods.conditions.path_segment(condition))
        loaded, _ = ensure_npz(mods, bucket, name, compute, fingerprint=fp)
        X = np.asarray(loaded["X"])
        if X.shape != (n, EXPECTED_FEATURE_DIM):
            raise AuditHalt(f"150-step {condition} features are {X.shape}, expected "
                            f"{(n, EXPECTED_FEATURE_DIM)}")
        features_150[condition] = X
    non_finite = {c: int(np.sum(~np.isfinite(X))) for c, X in features_150.items()}
    non_finite = {c: v for c, v in non_finite.items() if v}
    if non_finite:
        raise AuditHalt(f"non-finite 150-step features: {non_finite}")
    record["features_150"] = {"n_images": n, "conditions": list(features_150)}
    say(f"150-step features: {', '.join(f'{k}{v.shape}' for k, v in sorted(features_150.items()))}")
    return features_150


def step5_stage12_cross_check(mods, bucket, record):
    """Reproduce stage 1's and stage 2's OWN stored fold-aggregate values
    with the new OOF machinery, on THEIR OWN grid and THEIR OWN row order --
    discharges `gates.toml`'s `binding_gate.9bc6f9e3808a`.

    Each stage's `ridge_cv.json` carries its own `alphas` (the pre-amendment
    NINE-decade grid -- `FINDINGS.md:935-1002` -- not today's 13-decade
    `ALPHA_GRID`). Using the historical grid here is required, not
    optional: `oof_per_image_mse`'s default grid has 13 columns, the stored
    `fold_clipped_val_mse` has 9, and passing the wrong one fails on SHAPE
    before comparing a single value. This never calls `assert_frozen_grid`
    -- that assertion is for this driver's own 60k runs (step 6), which do
    stay frozen on the 13-decade grid. If this ever hits a shape error, the
    fix is never "rerun history on the new grid" -- that would silently
    convert a pin against real historical numbers into a self-comparison.

    Each stage's own row order is used throughout (its own `stageN_indices`,
    its own `images_01`/`labels`) -- no cross-stage or cross-artifact join
    is needed, since `StratifiedKFold` on the SAME rows in the SAME order
    with the SAME `random_state` reproduces the SAME fold assignment the
    stored aggregates were computed under."""
    results = {}
    for stage, stage_key, index_key in ((STAGE1_STAGE, "stage1", "stage1_indices"),
                                        (STAGE2_STAGE, "stage2", "stage2_indices")):
        corpus_name = mods.gcs.object_path(stage=stage, condition=None, kind="corpus",
                                           ext="npz", split=SPLIT)
        corpus_local = consume_pinned(mods, bucket, corpus_name, f"{stage_key}/corpus")
        with np.load(corpus_local, allow_pickle=False) as handle:
            corpus = {key: handle[key] for key in handle.files}
        official_indices = np.asarray(corpus[index_key])
        images = np.asarray(corpus["images_01"])
        n = images.shape[0]
        labels = np.asarray(corpus["labels"])

        cv_name = mods.gcs.object_path(stage=stage, condition=None, kind="ridge_cv",
                                       ext="json", split=SPLIT)
        cv_local = consume_pinned(mods, bucket, cv_name, f"{stage_key}/ridge_cv")
        with open(cv_local, "r", encoding="utf-8") as handle:
            ridge_cv = json.load(handle)

        topo_name = mods.gcs.object_path(stage=STAGE1_STAGE, condition=None,
                                         kind="topologies", ext="npz", split=SPLIT)
        topo_local = consume_pinned(mods, bucket, topo_name, "stage1/topologies")
        with np.load(topo_local, allow_pickle=False) as handle:
            active_indices = np.asarray(handle["active_indices"])
        Y = images.reshape(n, FULL_GRID)[:, active_indices]

        stage_results = {}
        for condition in mods.conditions.ALL_CONDITIONS:
            seg = mods.conditions.path_segment(condition)
            feat_name = mods.gcs.object_path(stage=stage, condition=seg, kind="features",
                                             ext="npz", split=SPLIT)
            feat_local = consume_pinned(mods, bucket, feat_name,
                                        f"{stage_key}/features/{condition}")
            with np.load(feat_local, allow_pickle=False) as handle:
                X = np.asarray(handle["X"])
            if X.shape[0] != n:
                raise AuditHalt(f"{stage_key}/{condition} features have {X.shape[0]} rows, "
                                f"corpus has {n}")

            stored_cv = ridge_cv["conditions"][condition]["cv"]
            historical_alphas = np.asarray(stored_cv["alphas"], dtype=np.float64)
            oof = mods.ridge.oof_per_image_mse(X, Y, labels, alphas=historical_alphas)
            check = mods.audit.assert_oof_matches_fold_aggregates(oof, stored_cv)
            stage_results[condition] = check
            say(f"{stage_key}/{condition}: OOF reproduces stored fold aggregates "
                f"(max abs diff {check['max_abs_diff']:.3e}, grid={len(historical_alphas)} "
                f"decades)")
        results[stage_key] = {"n_images": n, "n_official_indices": int(official_indices.size),
                              "conditions": stage_results}
    record["stage12_cross_check"] = results
    return results


def step6_oof_both_regimes(mods, bucket, corpus, topo, features_150, features_1200,
                           production_alphas, record, fp):
    """The audit's own 60,000-image OOF ridge, both alpha regimes, both
    budgets -- frozen on the 13-decade grid throughout. Discharges
    `binding_value`s `b682b6af449a`, `7ff6b43436c6`, `7946f9f0ccc3`,
    `1148ede15761`, `259d3b7f30fb`, `1837113c184a`."""
    active_indices = np.asarray(topo["active_indices"])
    n = corpus["images_01"].shape[0]
    Y = np.asarray(corpus["images_01"]).reshape(n, FULL_GRID)[:, active_indices]
    labels = np.asarray(corpus["labels"])
    features_by_budget = {AUDIT_NEW_STEPS: features_150, PRODUCTION_STEPS: features_1200}

    def compute():
        regimes = mods.audit.compute_oof_alpha_regimes(
            features_by_budget, Y, labels, production_alphas, ridge_module=mods.ridge)
        return {
            "fixed": {str(s): {c: v["mse"] for c, v in regimes["fixed"][str(s)].items()}
                     for s in (AUDIT_NEW_STEPS, PRODUCTION_STEPS)},
            "reselected": {str(s): {c: v["mse"]
                                    for c, v in regimes["reselected"][str(s)].items()}
                          for s in (AUDIT_NEW_STEPS, PRODUCTION_STEPS)},
            "alphas": {
                "fixed": production_alphas,
                "reselected": {str(s): {c: regimes["reselected"][str(s)][c]["alpha"]
                                       for c in regimes["reselected"][str(s)]}
                              for s in (AUDIT_NEW_STEPS, PRODUCTION_STEPS)},
            },
            "cross_checks": {str(s): {c: regimes["cross_checks"][str(s)][c]
                                      for c in regimes["cross_checks"][str(s)]}
                            for s in (AUDIT_NEW_STEPS, PRODUCTION_STEPS)},
        }

    name = _obj(mods, "oof_regimes", "json")
    result, _ = ensure_json(mods, bucket, name, compute, fingerprint=fp)
    record["oof_regimes"] = {"object": name}
    say("60k OOF ridge complete: both alpha regimes, both budgets, 5 conditions")
    return result


def step7_feature_distances(mods, bucket, encoded, theta_150, theta_1200,
                            features_150, features_1200, record, fp):
    """150-vs-1200 distance per condition (NOT pre-vs-evolved within a
    budget -- that answers a different question than the amendment's
    impact). Gauge-fixed at `stage2b_audit.GAUGE_NODE`, `active_indices`
    omitted: every theta array here is already restricted to the 505-active
    support with `GAUGE_NODE` already a valid position within it (asserted
    in `step1b_topologies`), so no full-to-active translation applies."""
    theta0_150 = np.asarray(encoded[AUDIT_NEW_STEPS]["thetas_505"])
    theta0_1200 = np.asarray(encoded[PRODUCTION_STEPS]["thetas_505"])

    def compute():
        distances = {}
        for condition in mods.conditions.ALL_CONDITIONS:
            left = (theta0_150 if condition == mods.conditions.PRE_EVOLUTION
                   else theta_150[condition])
            right = (theta0_1200 if condition == mods.conditions.PRE_EVOLUTION
                    else theta_1200[condition])
            distances[condition] = mods.audit.feature_distances(
                left, right, reference_column=mods.audit.GAUGE_NODE,
                features_left=features_150[condition],
                features_right=features_1200[condition])
        return distances

    name = _obj(mods, "feature_distances", "json")
    result, _ = ensure_json(mods, bucket, name, compute, fingerprint=fp)
    record["feature_distances"] = {"object": name}
    say(f"150-vs-1200 feature distances computed for {len(mods.conditions.ALL_CONDITIONS)} "
        f"conditions")
    return result


def combine_trigger_regimes(fixed, reselected):
    """The pure decision `step8_trigger_verdict` applies -- pulled out so it
    is testable on synthetic verdict dicts without GCS/GPU, the same reason
    `floor_halt_reason` exists as its own function in `run_ladder_stage3.py`.

    Discharges `binding_gate.0570557b8fa8` (all six pairwise comparisons --
    asserted, not assumed) and `binding_gate.b6e8be29a571` (the
    OR-over-regimes clause, deliberately separate from `12d7b0d5472d`: "a
    reversal seen under fixed-alpha alone, or under reselected-alpha alone,
    is sufficient" -- never "check once on whichever regime runs first")."""
    for regime_name, verdict in (("fixed", fixed), ("reselected", reselected)):
        if len(verdict["pairwise"]) != 6:
            raise AuditHalt(f"{regime_name} regime: expected 6 pairwise comparisons, "
                            f"got {len(verdict['pairwise'])}")
    combined_triggered = bool(fixed["triggered"] or reselected["triggered"])
    return {"fixed": fixed, "reselected": reselected,
           "combined_triggered": combined_triggered,
           "combination_rule": "fixed.triggered OR reselected.triggered"}


def step8_trigger_verdict(mods, bucket, oof_regimes, record, fp):
    """Discharges `binding_gate.12d7b0d5472d` (the three trigger
    conditions) via `mods.audit.trigger_verdict`; `combine_trigger_regimes`
    discharges the other two rows named in its own docstring."""
    def mse_by_budget(regime):
        return {s: oof_regimes[regime][str(s)] for s in (AUDIT_NEW_STEPS, PRODUCTION_STEPS)}

    def compute():
        fixed = mods.audit.trigger_verdict(mse_by_budget("fixed"))
        reselected = mods.audit.trigger_verdict(mse_by_budget("reselected"))
        return combine_trigger_regimes(fixed, reselected)

    name = _obj(mods, "trigger_verdict", "json")
    result, _ = ensure_json(mods, bucket, name, compute, fingerprint=fp)
    record["trigger_verdict"] = result
    say(f"trigger verdict: combined_triggered={result['combined_triggered']} "
        f"(fixed={result['fixed']['triggered']}, reselected={result['reselected']['triggered']})")
    return result


def step9_report(mods, bucket, record):
    """Always written, ordinarily resumable -- no one-shot lock (see module
    docstring). Embeds AUDIT_PROTOCOL.md's required scope statement
    (`:99-104`) verbatim so the eventual write-up cannot lose it."""
    scope_statement = (
        "Per-budget fold-fitted StandardScalers are retained (production "
        "preprocessing). Fixed-alpha therefore isolates the effect of alpha "
        "reselection -- it does NOT completely isolate raw representation "
        "change. A shared-scaler comparison is optional secondary work, not "
        "required, and must not be presented as the primary probe.")

    def compute_json():
        return {**record, "scope_statement": scope_statement}

    def compute_text():
        lines = ["Stage 2B amendment-impact audit report",
                 f"commit: {record['run'].get('head_sha')}",
                 f"verdict: {record.get('verdict')}", ""]
        if record.get("halt_reason"):
            lines += [f"halt reason: {record['halt_reason']}", ""]
        lines += ["SCOPE STATEMENT (AUDIT_PROTOCOL.md:99-104):", scope_statement, "",
                  "timings (s):", _dumps(record.get("timings", {})), "",
                  "full record:", _dumps(record), "",
                  str(record.get("verdict", FAIL_SENTINEL))]
        return "\n".join(lines) + "\n"

    kind = f"audit_report_{record['run']['run_id']}"
    ensure_json(mods, bucket, _obj(mods, kind, "json"), compute_json)
    ensure_text(mods, bucket, _obj(mods, kind, "txt"), compute_text)


def new_record():
    return {"run": {"run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
            "timings": {}, "train_side": {}, "evolution_150": {}, "features_150": {},
            "sizing_probe": {}, "stage12_cross_check": {}, "oof_regimes": {},
            "feature_distances": {}, "trigger_verdict": {},
            "verdict": None, "halt_reason": None}


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version}
    for package in ("numpy", "scipy", "scikit-learn", "jax", "jaxlib"):
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

        fp = build_fingerprint(mods, CLONE_DIR, {
            "ladder_stage": LADDER_STAGE, "split": SPLIT,
            "population": "official KMNIST training split, all 60,000",
            "n_images": EXPECTED_N, "row_order": "ascending official training index",
            "audit_new_steps": AUDIT_NEW_STEPS, "production_steps": PRODUCTION_STEPS,
            "evolve_chunk": EVOLVE_CHUNK, "n_active": EXPECTED_N_ACTIVE,
            "reference_node": EXPECTED_REF_IDX, "feature_dim": EXPECTED_FEATURE_DIM,
            "ridge_alpha_grid": list(mods.ridge.ALPHA_GRID),
            "ridge_n_splits": mods.ridge.N_SPLITS, "ridge_fold_seed": mods.ridge.FOLD_SEED,
            "dtype": "float64",
        })
        record["run"]["fingerprint"] = {
            "source_manifest_digest": fp["source_manifest_digest"],
            "config_digest": fp["config_digest"], "git": fp["git"],
            "n_source_files": len(fp["source_manifest"])}
        say(f"fingerprint established: config {fp['config_digest'][:16]}...")

        with timed_step("0_preflight", record["timings"]):
            step0_preflight(mods, record)
        with timed_step("1_load_train_side", record["timings"]):
            corpus, encoded, train_indices, corpus_name = step1_load_train_side(mods, bucket, record)
        with timed_step("1b_topologies", record["timings"]):
            topo, ref_idx = step1b_topologies(mods, bucket)
        with timed_step("1c_stage3_evolved", record["timings"]):
            theta_1200, features_1200 = step1c_stage3_evolved(mods, bucket)
        with timed_step("1d_production_alphas", record["timings"]):
            production_alphas = step1d_production_alphas(mods, bucket)
        with timed_step("2_sizing_probe", record["timings"]):
            step2_sizing_probe(mods, bucket, record, fp)
        with timed_step("3_evolve_150", record["timings"]):
            theta_150 = step3_evolve_150(mods, bucket, encoded, theta_1200, topo, record, fp)
        with timed_step("4_features_150", record["timings"]):
            features_150 = step4_features_150(mods, bucket, encoded, theta_150, ref_idx,
                                              record, fp)
        with timed_step("5_stage12_cross_check", record["timings"]):
            step5_stage12_cross_check(mods, bucket, record)
        with timed_step("6_oof_both_regimes", record["timings"]):
            oof_regimes = step6_oof_both_regimes(mods, bucket, corpus, topo, features_150,
                                                 features_1200, production_alphas, record, fp)
        with timed_step("7_feature_distances", record["timings"]):
            step7_feature_distances(mods, bucket, encoded, theta_150, theta_1200,
                                    features_150, features_1200, record, fp)
        with timed_step("8_trigger_verdict", record["timings"]):
            step8_trigger_verdict(mods, bucket, oof_regimes, record, fp)

        record["verdict"] = OK_SENTINEL
    except AuditHalt as halt:
        record.update(verdict=FAIL_SENTINEL, halt_reason=str(halt))
        status = 1
    except SystemExit as exc:
        record.update(verdict=FAIL_SENTINEL, halt_reason=f"SystemExit: {exc}")
        status = 1
    except BaseException as exc:                    # noqa: BLE001 - reported, not swallowed
        traceback.print_exc()
        record.update(verdict=FAIL_SENTINEL, halt_reason=f"{type(exc).__name__}: {exc}")
        status = 1
    finally:
        stop.set()
        heartbeat.join(timeout=2)
        record["timings"]["total"] = time.time() - _RUN_T0
        if mods is not None and bucket is not None:
            try:
                step9_report(mods, bucket, record)
            except Exception as exc:                # noqa: BLE001
                print(f"[audit] report FAILED to write: {type(exc).__name__}: {exc}",
                      flush=True)

    verdict = record["verdict"] or FAIL_SENTINEL
    print(verdict if status == 0 else f"{verdict} {record['halt_reason']}", flush=True)
    return status


if __name__ == "__main__" or os.environ.get(ENV_COMMIT):
    _status = main()
    if _status:
        sys.exit(_status)
