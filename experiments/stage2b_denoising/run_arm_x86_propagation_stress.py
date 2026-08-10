"""Companion Protocol 1 (`COMPANION_PROTOCOLS.md`): ARM/x86 propagation
stress set.

Three resumable phases, one driver:

  --phase arm-construct   Mac ARM: build B∪C∪D stress indices, slice the
                          authoritative Phase-A ARM encode onto them.
  --phase x86-encode      Colab x86: re-encode the same official indices
                          via unmodified encode_stage3_local.encode_indices.
  --phase propagate       Mac (default): rank regenerated A, evolve both
                          arches, fit_final once per condition, five-stage
                          report, stage-5 halt.

Component A is always the regenerate+disclose branch: no committed
per-image ARM−x86 array exists. Construction records carry
`component_a_source = "regenerated"`.

Synthesis kinds embed a UTC run_id (`protocol1_propagation_report_{run_id}`,
`protocol1_ridge_frozen_{run_id}`) so a re-run cannot hit ensure_artifact's
skip branch on stale content. Intermediate kinds stay fixed-name LINEAGE.

Sentinels: PROTOCOL1_OK / PROTOCOL1_HALT (exit 0 either way — halt is a
scientific consequence) / PROTOCOL1_FAIL (exit non-zero, infrastructure).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types

import numpy as np

# ---- identity and sentinels ----
DRIVER_FILENAME = "run_arm_x86_propagation_stress.py"
LADDER_STAGE = 3
KMNIST_STAGING_STAGE = 1
SPLIT = "train"
ENCODER_STEPS = 1200
EXPECTED_N_ACTIVE = 505
EXPECTED_REF_IDX = 363
EXPECTED_FEATURE_DIM = 1008
FULL_GRID = 784
N_OFFICIAL_TRAIN = 60_000
EVOLVE_CHUNK = 250
COMPONENT_A_SIZE = 100
COMPONENT_D_SEED = 42
COMPONENT_D_PER_CLASS = 20
TAIL_CAP = 500
ENCODING_SANITY_MAX_ABS = 1e-12
EXPECTED_RIDGE_GRID_TAG = "g13_88edf9ac"
FRAMING = (
    "adversarial upper bound on cross-architecture propagation; "
    "not a corpus sample"
)

OK_SENTINEL = "PROTOCOL1_OK"
HALT_SENTINEL = "PROTOCOL1_HALT"
FAIL_SENTINEL = "PROTOCOL1_FAIL"
X86_OK_SENTINEL = "PROTOCOL1_X86_ENCODE_OK"

# ---- bootstrap (Colab exec model) ----
REPO_URL = "https://github.com/danbarua/bonsai-2026.git"
CLONE_DIR = "/content/bonsai-2026"
REMOTE_WORK_DIR = "/content/stage2b_protocol1"
EXPERIMENT_DIRS = (
    "experiments/stage2b_denoising",
    "experiments/stage2a_dynamics_classification",
    "experiments/stage1d_topology_specificity",
)
KMNIST_SUBDIR = "datasets/kmnist"
KMNIST_FILES = {
    "train-images-idx3-ubyte": "kmnist_train_images",
    "train-labels-idx1-ubyte": "kmnist_train_labels",
    "t10k-images-idx3-ubyte": "kmnist_t10k_images",
    "t10k-labels-idx1-ubyte": "kmnist_t10k_labels",
}
KMNIST_EXT = "idx"

ENV_COMMIT = "BONSAI_COMMIT"
ENV_BUCKET = "BONSAI_GCS_BUCKET"
ENV_CREDENTIALS = "BONSAI_GCS_CREDENTIALS"
ENV_DRIVER_SHA = "BONSAI_DRIVER_SHA256"
ENV_ALLOW_NON_X86 = "PROTOCOL1_ALLOW_NON_X86"

# Fixed-name intermediate kinds (LINEAGE).
KIND_STRESS_INDICES = "protocol1_stress_indices"
KIND_ENCODED_ARM = "protocol1_encoded_stress_arm_s1200"
KIND_ENCODED_X86 = "protocol1_encoded_stress_x86_s1200"
KIND_THETA_ARM = "protocol1_theta_T_arm"
KIND_THETA_X86 = "protocol1_theta_T_x86"
KIND_FEATURES_ARM = "protocol1_features_arm"
KIND_FEATURES_X86 = "protocol1_features_x86"


class Protocol1Fail(Exception):
    """Infrastructure / input failure — prints PROTOCOL1_FAIL, exit non-zero."""


# ------------------------------------------------------------------ helpers


def say(msg):
    print(f"[protocol1] {msg}", flush=True)


def _dumps(obj):
    return json.dumps(obj, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


def new_record():
    return {
        "run": {"run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())},
        "timings": {},
        "construction": {},
        "platforms": {},
        "stages": {},
        "halt": {},
        "artifacts": {},
        "verdict": None,
        "halt_reason": None,
    }


def synthesis_kind_report(run_id):
    """Run-id-embedded report kind — never the bare fixed name."""
    return f"protocol1_propagation_report_{run_id}"


def synthesis_kind_ridge(run_id):
    """Run-id-embedded frozen-ridge kind — never the bare fixed name."""
    return f"protocol1_ridge_frozen_{run_id}"


def indices_sha256(indices):
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_construction_record(*, b_meta, n_stress, indices, indices_refined=False):
    """Construction record embedded in stress_indices + the report."""
    b_meta = dict(b_meta)
    b_meta["n_used"] = int(b_meta.get("n_used", min(
        int(b_meta["true_count"]), int(b_meta["cap"]))))
    return {
        "component_a_source": "regenerated",
        "component_a_size": int(COMPONENT_A_SIZE),
        "component_b": {
            "true_count": int(b_meta["true_count"]),
            "cap": int(b_meta["cap"]),
            "cap_applied": bool(b_meta["cap_applied"]),
            "n_used": int(b_meta["n_used"]),
        },
        "component_d_seed": int(COMPONENT_D_SEED),
        "component_d_per_class": int(COMPONENT_D_PER_CLASS),
        "n_stress": int(n_stress),
        "indices_sha256": indices_sha256(indices),
        "indices_refined": bool(indices_refined),
        "framing": FRAMING,
    }


def apply_frozen_ridge(fits, features_by_arch, ridge_module):
    """Predict both arches from already-fitted (fit, scaler) per condition.

    Does not call fit_final. Used so tests can assert one fit per condition
    is shared across arches.
    """
    pred = {arch: {} for arch in features_by_arch}
    mse = {arch: {} for arch in features_by_arch}
    # Y is not known here; caller supplies targets via a side channel when
    # computing MSE. This helper only produces predictions.
    for condition, bundle in fits.items():
        fit = bundle["fit"]
        scaler = bundle["scaler"]
        for arch, feats in features_by_arch.items():
            X = np.asarray(feats[condition], dtype=np.float64)
            pred[arch][condition] = ridge_module.ridge_predict(
                fit, scaler.transform(X), 0)
    return pred


def grid_tag(alphas):
    canonical = ",".join(repr(float(a)) for a in alphas)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    return f"g{len(tuple(alphas))}_{digest}"


def _is_x86_machine():
    machine = platform.machine().lower()
    return machine in ("x86_64", "amd64")


def _run(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)


# ------------------------------------------------------------- path / mods


def discover_local_repo_root():
    """Best-effort local repo root when not under Colab bootstrap."""
    # Function-local __file__ is fine; module-scope must stay Colab-safe.
    try:
        here = os.path.abspath(__file__)
        return os.path.abspath(os.path.join(os.path.dirname(here), "..", ".."))
    except NameError:
        pass
    cwd = os.path.abspath(os.getcwd())
    probe = cwd
    for _ in range(8):
        if os.path.isdir(os.path.join(probe, "experiments", "stage2b_denoising")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    raise Protocol1Fail(
        f"cannot locate repo root from cwd={cwd!r}; pass --kmnist-dir and run "
        f"from the bonsai-2026 tree")


def resolve_runtime():
    """Return (mode, repo_root, work_dir) for local CLI vs Colab exec."""
    if os.environ.get(ENV_COMMIT):
        return "remote", CLONE_DIR, REMOTE_WORK_DIR
    root = discover_local_repo_root()
    work = os.path.join(root, "experiments", "stage2b_denoising", "results")
    return "local", root, work


def add_repo_to_path(clone_dir):
    for directory in (*EXPERIMENT_DIRS, "src"):
        entry = os.path.join(clone_dir, directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)


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
        raise Protocol1Fail(f"clone is at {head}, expected {commit}")
    try:
        _run([sys.executable, "-m", "pip", "install", "-e", clone_dir,
              "--no-deps", "--ignore-requires-python", "-q"])
        info["pip_editable"] = True
    except subprocess.CalledProcessError as exc:
        say(f"editable install failed ({exc.returncode}); falling back to sys.path")
        info["pip_install_stderr"] = (exc.stderr or "")[-2000:]
    return info


def verify_driver_identity(clone_dir, expected_sha256):
    path = os.path.join(clone_dir, "experiments", "stage2b_denoising",
                        DRIVER_FILENAME)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    result = {"path": path, "sha256": digest, "expected": expected_sha256,
              "matches": bool(expected_sha256) and digest == expected_sha256}
    if expected_sha256 and not result["matches"]:
        raise Protocol1Fail(
            f"driver identity mismatch: transmitted {expected_sha256}, "
            f"commit copy {digest}")
    return result


def load_modules(repo_root):
    add_repo_to_path(repo_root)
    import stage2b_ridge as ridge                                       # noqa: E402
    import stage2b_audit as audit                                       # noqa: E402
    import stage2b_conditions as conditions                             # noqa: E402
    import stage2b_corruption as corruption                             # noqa: E402
    import stage2b_encoder_gate as encoder_gate                         # noqa: E402
    import stage2b_fingerprint as fingerprint                           # noqa: E402
    import stage2b_gcs as gcs                                           # noqa: E402
    import encode_stage3_local as encode_local                          # noqa: E402
    import stage2a_core as core                                         # noqa: E402
    import stage2a_topologies as topologies                             # noqa: E402
    from bonsai.data.mnist_loader import load_mnist                     # noqa: E402

    # JAX stack is optional at import for arm-construct / pure helpers;
    # propagate and any evolve path require it.
    try:
        from evolve_on_graph_jax import batched_evolve_on_graph_jax     # noqa: E402
        import jax                                                      # noqa: E402
        import jax.numpy as jnp                                         # noqa: E402
    except Exception as exc:  # noqa: BLE001
        batched_evolve_on_graph_jax = None
        jax = None
        jnp = None
        say(f"JAX stack unavailable at import ({type(exc).__name__}: {exc}); "
            f"evolve phases will refuse")

    mods = types.SimpleNamespace(
        ridge=ridge, audit=audit, conditions=conditions, corruption=corruption,
        encoder_gate=encoder_gate, fingerprint=fingerprint, gcs=gcs,
        encode_local=encode_local, core=core, topologies=topologies,
        load_mnist=load_mnist,
        batched_evolve_on_graph_jax=batched_evolve_on_graph_jax,
        jax=jax, jnp=jnp)
    return mods


def local_path_for(work_dir, object_name):
    base = os.path.basename(object_name)
    return os.path.join(work_dir, base)


def results_mirror(work_dir, basename):
    return os.path.join(work_dir, basename)


def _obj(mods, kind, ext, condition=None, stage=LADDER_STAGE):
    return mods.gcs.object_path(stage=stage, condition=condition, kind=kind,
                                ext=ext, split=SPLIT)


def ensure_npz(mods, bucket, work_dir, object_name, compute, *,
               fingerprint=None, parents=None, no_upload=False):
    local = local_path_for(work_dir, object_name)

    def produce(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = compute()
        np.savez_compressed(path, **arrays)

    if no_upload or bucket is None:
        if not os.path.isfile(local):
            produce(local)
            say(f"wrote local-only {local}")
        else:
            say(f"local-only reuse {local}")
        with np.load(local, allow_pickle=False) as handle:
            loaded = {key: handle[key] for key in handle.files}
        return loaded, types.SimpleNamespace(
            local_path=local, skipped=os.path.isfile(local),
            summary=lambda: f"local {local}")

    result = mods.gcs.ensure_artifact(
        object_name, local, produce=produce, bucket=bucket,
        fingerprint=fingerprint, parents=parents)
    with np.load(result.local_path, allow_pickle=False) as handle:
        loaded = {key: handle[key] for key in handle.files}
    say(f"artifact {result.summary()}")
    return loaded, result


def ensure_json(mods, bucket, work_dir, object_name, compute, *,
                fingerprint=None, parents=None, no_upload=False):
    local = local_path_for(work_dir, object_name)

    def produce(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_dumps(compute()))

    if no_upload or bucket is None:
        produce(local)
        say(f"wrote local-only {local}")
        with open(local, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded, types.SimpleNamespace(local_path=local)

    result = mods.gcs.ensure_artifact(
        object_name, local, produce=produce, bucket=bucket,
        fingerprint=fingerprint, parents=parents)
    with open(result.local_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    say(f"artifact {result.summary()}")
    return loaded, result


def consume_npz(mods, bucket, work_dir, object_name, *, require_manifest=True,
                no_upload=False, fallback_local=None):
    local = local_path_for(work_dir, object_name)
    if fallback_local and os.path.isfile(fallback_local):
        if no_upload or bucket is None or not _object_exists(
                mods, bucket, object_name):
            say(f"using local fallback {fallback_local}")
            with np.load(fallback_local, allow_pickle=False) as handle:
                return {key: handle[key] for key in handle.files}, fallback_local
    if bucket is None:
        if os.path.isfile(local):
            with np.load(local, allow_pickle=False) as handle:
                return {key: handle[key] for key in handle.files}, local
        if fallback_local and os.path.isfile(fallback_local):
            with np.load(fallback_local, allow_pickle=False) as handle:
                return {key: handle[key] for key in handle.files}, fallback_local
        raise Protocol1Fail(f"missing local artifact for {object_name}")
    mods.gcs.consume_validated(
        object_name, local, bucket=bucket, require_manifest=require_manifest)
    with np.load(local, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}, local


def _object_exists(mods, bucket, object_name):
    try:
        return bool(mods.gcs.object_exists(object_name, bucket=bucket))
    except Exception:  # noqa: BLE001
        return False



def get_bucket(mods, args):
    if args.no_upload:
        return None
    credentials = args.credentials or os.environ.get(ENV_CREDENTIALS)
    bucket_name = args.bucket or os.environ.get(ENV_BUCKET) or None
    return mods.gcs.get_bucket(name=bucket_name, credentials=credentials)


def build_fingerprint(mods, repo_root, config, require_clean=True):
    entry = os.path.join(repo_root, "experiments", "stage2b_denoising",
                         DRIVER_FILENAME)
    return mods.fingerprint.compute(
        entrypoint=entry, repo_root=repo_root, require_clean=require_clean,
        config=config)


def platform_summary():
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "system": platform.system(),
    }


# -------------------------------------------------------------- alignment


def assert_stress_row_contract(thetas, deltas, indices, *, label):
    thetas = np.asarray(thetas, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    n = int(indices.size)
    if thetas.shape != (n, EXPECTED_N_ACTIVE):
        raise Protocol1Fail(
            f"{label}: thetas_505 shape {thetas.shape} != {(n, EXPECTED_N_ACTIVE)}")
    if deltas.shape != (n,):
        raise Protocol1Fail(f"{label}: deltas shape {deltas.shape} != {(n,)}")
    if indices.ndim != 1 or indices.shape != (n,):
        raise Protocol1Fail(f"{label}: indices shape {indices.shape}")
    if not np.all(np.isfinite(thetas)):
        raise Protocol1Fail(f"{label}: non-finite thetas")
    if n and not np.all(indices[1:] > indices[:-1]):
        raise Protocol1Fail(f"{label}: indices are not strictly increasing")
    return thetas, deltas, indices


def join_rows(values, source_indices, target_indices, audit):
    try:
        return audit.align_by_official_index(values, source_indices, target_indices)
    except audit.AuditInputError as exc:
        raise Protocol1Fail(f"index join failed: {exc}") from exc


# ---------------------------------------------------------------- phases


def phase_preflight(mods, args, repo_root, work_dir, bucket):
    say(f"mode={resolve_runtime()[0]} machine={platform.machine()}")
    say(f"repo_root={repo_root}")
    say(f"work_dir={work_dir}")
    encoded_local = os.path.join(
        repo_root, "experiments", "stage2b_denoising", "results",
        f"stage3_encoded_train_s{ENCODER_STEPS}.npz")
    say(f"local encoded_train present: {os.path.isfile(encoded_local)}")
    kmnist = args.kmnist_dir or os.path.join(repo_root, KMNIST_SUBDIR)
    say(f"kmnist_dir={kmnist} exists={os.path.isdir(kmnist)}")
    if bucket is not None:
        say(f"bucket={bucket.name}")
        for kind in (KIND_STRESS_INDICES, KIND_ENCODED_ARM, KIND_ENCODED_X86):
            name = _obj(mods, kind, "npz")
            say(f"  exists {name}: {_object_exists(mods, bucket, name)}")
    else:
        say("bucket=None (--no-upload or missing creds)")
    tag = grid_tag(mods.ridge.ALPHA_GRID)
    say(f"ALPHA_GRID tag={tag} (expected {EXPECTED_RIDGE_GRID_TAG})")
    if tag != EXPECTED_RIDGE_GRID_TAG:
        raise Protocol1Fail(
            f"ALPHA_GRID tag {tag} != expected production {EXPECTED_RIDGE_GRID_TAG}")
    say("preflight OK")
    return 0


def phase_arm_construct(mods, args, repo_root, work_dir, bucket, fp):
    audit = mods.audit
    encoded_name = _obj(mods, f"encoded_train_s{ENCODER_STEPS}", "npz")
    encoded_local = os.path.join(
        repo_root, "experiments", "stage2b_denoising", "results",
        f"stage3_encoded_train_s{ENCODER_STEPS}.npz")
    encoded, _src = consume_npz(
        mods, bucket, work_dir, encoded_name, require_manifest=True,
        no_upload=args.no_upload, fallback_local=encoded_local)

    train_indices = np.asarray(encoded["train_indices"], dtype=np.int64)
    thetas_full = np.asarray(encoded["thetas_505"], dtype=np.float64)
    deltas_full = np.asarray(encoded["deltas"], dtype=np.float64)
    active_indices = np.asarray(encoded["active_indices"], dtype=np.int64)
    if active_indices.size != EXPECTED_N_ACTIVE:
        raise Protocol1Fail(
            f"active_indices size {active_indices.size} != {EXPECTED_N_ACTIVE}")
    if thetas_full.shape != (N_OFFICIAL_TRAIN, EXPECTED_N_ACTIVE):
        raise Protocol1Fail(f"unexpected full thetas shape {thetas_full.shape}")
    if not np.array_equal(train_indices, np.arange(N_OFFICIAL_TRAIN)):
        # still OK if it's a permutation — join by official index below
        say("train_indices is not 0..N-1 ascending; joining by official index")

    kmnist_dir = args.kmnist_dir or os.path.join(repo_root, KMNIST_SUBDIR)
    _x_train, y_train, _x_test, _y_test = mods.load_mnist(kmnist_dir, gz=False)
    labels = np.asarray(y_train, dtype=np.int64)
    if labels.shape != (N_OFFICIAL_TRAIN,):
        raise Protocol1Fail(f"KMNIST labels shape {labels.shape}")
    official_indices = np.arange(N_OFFICIAL_TRAIN, dtype=np.int64)

    positive, b_meta = audit.capped_positive_delta_indices(
        train_indices, deltas_full, tail_cap=TAIL_CAP)
    b_meta = dict(b_meta)
    b_meta["n_used"] = int(positive.size)
    say(f"component B: true_count={b_meta['true_count']} "
        f"cap_applied={b_meta['cap_applied']} n_used={b_meta['n_used']}")

    empty_a = np.asarray([], dtype=np.int64)
    provisional = audit.build_stress_indices(
        official_indices, labels, empty_a, positive,
        seed=COMPONENT_D_SEED, tail_cap=TAIL_CAP)
    say(f"provisional B∪C∪D stress set: n={provisional.size}")

    # Every stress index must be present in the production encode.
    missing = sorted(set(map(int, provisional)) - set(map(int, train_indices)))
    if missing:
        raise Protocol1Fail(
            f"{len(missing)} stress indices absent from train_indices "
            f"(first {missing[:5]})")

    thetas = join_rows(thetas_full, train_indices, provisional, audit)
    deltas = join_rows(deltas_full, train_indices, provisional, audit)
    thetas, deltas, provisional = assert_stress_row_contract(
        thetas, deltas, provisional, label="arm-construct")

    construction = build_construction_record(
        b_meta=b_meta, n_stress=int(provisional.size), indices=provisional,
        indices_refined=False)
    summary = {
        "phase": "arm-construct",
        "construction": construction,
        "platform": platform_summary(),
        "encoder_steps": ENCODER_STEPS,
        "n_active": int(active_indices.size),
        "source_encoded": encoded_name,
    }

    parents = None
    if bucket is not None:
        try:
            man = mods.gcs.read_manifest(encoded_name, bucket=bucket)
            if man and man.get("payload_sha256"):
                parents = {encoded_name: man["payload_sha256"]}
        except Exception as exc:  # noqa: BLE001
            say(f"parent digest for encoded_train unavailable: {exc}")

    def compute_indices():
        return {
            "indices": provisional.astype(np.int64),
            "labels_at_indices": labels[provisional].astype(np.int64),
            "summary_json": np.array(_dumps(summary)),
        }

    idx_name = _obj(mods, KIND_STRESS_INDICES, "npz")
    ensure_npz(mods, bucket, work_dir, idx_name, compute_indices,
               fingerprint=fp, parents=parents, no_upload=args.no_upload)

    def compute_arm():
        return {
            "thetas_505": thetas.astype(np.float64),
            "deltas": deltas.astype(np.float64),
            "indices": provisional.astype(np.int64),
            "active_indices": active_indices.astype(np.int64),
            "summary_json": np.array(_dumps({
                **summary,
                "arch": "arm",
                "slice": "index-join of production encoded_train_s1200",
            })),
        }

    arm_name = _obj(mods, KIND_ENCODED_ARM, "npz")
    ensure_npz(mods, bucket, work_dir, arm_name, compute_arm,
               fingerprint=fp, parents=parents, no_upload=args.no_upload)

    say(f"arm-construct done: n_stress={provisional.size} "
        f"indices_sha256={construction['indices_sha256'][:16]}...")
    say(f"objects: {idx_name}")
    say(f"objects: {arm_name}")
    return 0


def phase_x86_encode(mods, args, repo_root, work_dir, bucket, fp):
    if not _is_x86_machine() and os.environ.get(ENV_ALLOW_NON_X86) != "1":
        raise Protocol1Fail(
            f"x86-encode requires x86_64/AMD64 (got {platform.machine()!r}). "
            f"Set {ENV_ALLOW_NON_X86}=1 only for local dry-run / unit tests.")
    if os.environ.get(ENV_ALLOW_NON_X86) == "1":
        say(f"WARNING: {ENV_ALLOW_NON_X86}=1 — dry-run escape, not production")

    idx_name = _obj(mods, KIND_STRESS_INDICES, "npz")
    stress, _ = consume_npz(mods, bucket, work_dir, idx_name,
                            no_upload=args.no_upload)
    indices = np.asarray(stress["indices"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise Protocol1Fail("stress indices empty or malformed")
    if not np.all(indices[1:] > indices[:-1]):
        raise Protocol1Fail("stress indices not strictly increasing")

    # active_indices from ARM stress encode (or topologies fallback).
    arm_name = _obj(mods, KIND_ENCODED_ARM, "npz")
    try:
        arm, _ = consume_npz(mods, bucket, work_dir, arm_name,
                             no_upload=args.no_upload)
        active_indices = np.asarray(arm["active_indices"], dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        say(f"ARM stress encode unavailable ({exc}); falling back to topologies")
        topo_name = mods.gcs.object_path(
            stage=KMNIST_STAGING_STAGE, condition=None, kind="topologies",
            ext="npz", split=SPLIT)
        topo_local = local_path_for(work_dir, topo_name)
        mods.gcs.consume_validated(topo_name, topo_local, bucket=bucket,
                                   require_manifest=False)
        with np.load(topo_local, allow_pickle=False) as handle:
            active_indices = np.asarray(handle["active_indices"], dtype=np.int64)

    if active_indices.size != EXPECTED_N_ACTIVE:
        raise Protocol1Fail(
            f"active_indices size {active_indices.size} != {EXPECTED_N_ACTIVE}")

    kmnist_dir = args.kmnist_dir or os.path.join(repo_root, KMNIST_SUBDIR)
    if bucket is not None and not os.path.isdir(kmnist_dir):
        kmnist_dir = _stage_kmnist(mods, bucket, repo_root)

    x_train, _y_train, _x_test, _y_test = mods.load_mnist(kmnist_dir, gz=False)
    x_train = np.asarray(x_train)
    n_workers = args.workers
    say(f"encoding {indices.size} stress images on {platform.machine()} "
        f"workers={n_workers}")
    thetas, deltas, elapsed = mods.encode_local.encode_indices(
        x_train, indices, active_indices,
        steps=mods.encoder_gate.ENCODER_STEPS,
        n_workers=n_workers)
    thetas, deltas, indices = assert_stress_row_contract(
        thetas, deltas, indices, label="x86-encode")

    summary = {
        "phase": "x86-encode",
        "platform": platform_summary(),
        "encoder_steps": int(mods.encoder_gate.ENCODER_STEPS),
        "n_images": int(indices.size),
        "encode_elapsed_s": float(elapsed),
        "indices_sha256": indices_sha256(indices),
        "n_nonfinite_theta": int(np.count_nonzero(~np.isfinite(thetas))),
        "n_nonfinite_delta": int(np.count_nonzero(~np.isfinite(deltas))),
        "max_delta": float(np.max(deltas)) if deltas.size else float("nan"),
    }
    if summary["n_nonfinite_theta"] or summary["n_nonfinite_delta"]:
        raise Protocol1Fail("non-finite values in x86 encode output")

    def compute():
        return {
            "thetas_505": thetas.astype(np.float64),
            "deltas": deltas.astype(np.float64),
            "indices": indices.astype(np.int64),
            "active_indices": active_indices.astype(np.int64),
            "summary_json": np.array(_dumps(summary)),
        }

    x86_name = _obj(mods, KIND_ENCODED_X86, "npz")
    ensure_npz(mods, bucket, work_dir, x86_name, compute,
               fingerprint=fp, no_upload=args.no_upload)
    say(f"objects: {x86_name}")
    print(X86_OK_SENTINEL, flush=True)
    return 0


def _stage_kmnist(mods, bucket, repo_root):
    dest_dir = os.path.join(repo_root, KMNIST_SUBDIR)
    os.makedirs(dest_dir, exist_ok=True)
    for filename, kind in sorted(KMNIST_FILES.items()):
        name = mods.gcs.object_path(
            stage=KMNIST_STAGING_STAGE, condition=None, kind=kind,
            ext=KMNIST_EXT, split=SPLIT)
        dest = os.path.join(dest_dir, filename)
        if os.path.isfile(dest):
            continue
        mods.gcs.consume_validated(name, dest, bucket=bucket,
                                   require_manifest=False)
        say(f"downloaded {name} -> {dest}")
    return dest_dir


def _load_topologies(mods, bucket, work_dir, no_upload=False):
    topo_name = mods.gcs.object_path(
        stage=KMNIST_STAGING_STAGE, condition=None, kind="topologies",
        ext="npz", split=SPLIT)
    local = local_path_for(work_dir, topo_name)
    # Prefer any local stage2b results copy.
    if bucket is None or no_upload:
        candidates = [local]
        # also try stage2b results layouts used elsewhere
        alt = os.path.join(os.path.dirname(work_dir), "results",
                           os.path.basename(topo_name))
        candidates.append(alt)
        for path in candidates:
            if os.path.isfile(path):
                with np.load(path, allow_pickle=False) as handle:
                    return {k: handle[k] for k in handle.files}, path
    if bucket is None:
        raise Protocol1Fail("topologies.npz unavailable offline")
    mods.gcs.consume_validated(topo_name, local, bucket=bucket,
                               require_manifest=False)
    with np.load(local, allow_pickle=False) as handle:
        return {k: handle[k] for k in handle.files}, local


def _evolve_arch(mods, theta0, topo, ref_idx, arch, work_dir, bucket, fp,
                 parents, no_upload):
    if mods.batched_evolve_on_graph_jax is None or mods.jax is None:
        raise Protocol1Fail("JAX evolve stack is required for --phase propagate")
    n = int(theta0.shape[0])
    chunk = min(EVOLVE_CHUNK, n)
    features = {}
    # pre-evolution features
    features[mods.conditions.PRE_EVOLUTION] = np.stack(
        [mods.core.reference_node_features(theta0[i], ref_idx) for i in range(n)])
    kind_theta = KIND_THETA_ARM if arch == "arm" else KIND_THETA_X86
    kind_feat = KIND_FEATURES_ARM if arch == "arm" else KIND_FEATURES_X86

    # publish pre features
    def compute_pre(X=features[mods.conditions.PRE_EVOLUTION]):
        return {
            "X": np.asarray(X, dtype=np.float64),
            "indices": np.asarray([], dtype=np.int64),  # filled by caller overlay
        }

    # indices filled after return; publish happens with indices attached below
    evolved = {}
    for graph in mods.conditions.EVOLVED_GRAPHS:
        W_np = np.asarray(topo[f"W_{graph}"], dtype=np.float64)

        def compute(W_np=W_np, graph=graph):
            W = mods.jnp.asarray(W_np)
            thetas, flags = [], []
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                t0 = time.time()
                theta_T, success = mods.batched_evolve_on_graph_jax(
                    mods.jnp.asarray(theta0[lo:hi]), W)
                mods.jax.block_until_ready(theta_T)
                success_np = np.asarray(success)
                say(f"evolve/{arch}/{graph} rows {lo}:{hi} "
                    f"ok={int(success_np.sum())}/{hi - lo} "
                    f"({time.time() - t0:.2f}s)")
                thetas.append(np.asarray(theta_T))
                flags.append(success_np)
            theta_cat = np.concatenate(thetas)
            success_cat = np.concatenate(flags)
            return {
                "theta_T": theta_cat.astype(np.float64),
                "success": success_cat.astype(bool),
            }

        name = _obj(mods, kind_theta, "npz",
                    condition=mods.conditions.path_segment(graph))
        loaded, _ = ensure_npz(
            mods, bucket, work_dir, name, compute, fingerprint=fp,
            parents=parents, no_upload=no_upload)
        success = np.asarray(loaded["success"])
        n_failed = int(np.count_nonzero(~success))
        if n_failed:
            raise Protocol1Fail(
                f"{arch}/{graph}: {n_failed}/{success.size} evolve failures")
        theta_ref, diag = mods.core.evolve_on_graph(theta0[0], W_np)
        if theta_ref is None or diag.get("failed", False):
            raise Protocol1Fail(
                f"{arch}/{graph}: CPU reference evolve failed on image 0")
        evolved[graph] = np.asarray(loaded["theta_T"], dtype=np.float64)
        features[graph] = np.stack(
            [mods.core.reference_node_features(evolved[graph][i], ref_idx)
             for i in range(n)])

    # publish features per condition (indices attached by caller)
    return features, evolved, kind_feat


def phase_propagate(mods, args, repo_root, work_dir, bucket, fp, record):
    audit = mods.audit
    run_id = record["run"]["run_id"]
    say(f"propagate run_id={run_id}")

    arm_name = _obj(mods, KIND_ENCODED_ARM, "npz")
    x86_name = _obj(mods, KIND_ENCODED_X86, "npz")
    arm, _ = consume_npz(mods, bucket, work_dir, arm_name, no_upload=args.no_upload)
    x86, _ = consume_npz(mods, bucket, work_dir, x86_name, no_upload=args.no_upload)

    thetas_arm, deltas_arm, indices_arm = assert_stress_row_contract(
        arm["thetas_505"], arm["deltas"], arm["indices"], label="arm")
    thetas_x86, deltas_x86, indices_x86 = assert_stress_row_contract(
        x86["thetas_505"], x86["deltas"], x86["indices"], label="x86")

    if not np.array_equal(indices_arm, indices_x86):
        raise Protocol1Fail("arm/x86 stress indices differ byte-for-byte")
    indices = indices_arm
    sha_arm = indices_sha256(indices_arm)
    # optional summary cross-check
    for blob, label in ((arm, "arm"), (x86, "x86")):
        if "summary_json" in blob:
            try:
                s = json.loads(blob["summary_json"].item())
                if "indices_sha256" in s and s["indices_sha256"] != sha_arm:
                    raise Protocol1Fail(
                        f"{label} summary indices_sha256 mismatch")
            except Protocol1Fail:
                raise
            except Exception:  # noqa: BLE001
                pass

    # B meta from stress_indices artifact when present
    idx_name = _obj(mods, KIND_STRESS_INDICES, "npz")
    try:
        stress_blob, _ = consume_npz(
            mods, bucket, work_dir, idx_name, no_upload=args.no_upload)
        stress_summary = json.loads(stress_blob["summary_json"].item())
        construction_seed = stress_summary.get("construction", {})
        b_meta = construction_seed.get("component_b", {
            "true_count": 0, "cap": TAIL_CAP, "cap_applied": False, "n_used": 0})
    except Exception as exc:  # noqa: BLE001
        say(f"stress_indices summary unavailable ({exc}); reconstructing B")
        construction_seed = {}
        b_meta = {"true_count": 0, "cap": TAIL_CAP, "cap_applied": False,
                  "n_used": 0}

    kmnist_dir = args.kmnist_dir or os.path.join(repo_root, KMNIST_SUBDIR)
    if bucket is not None and not os.path.isdir(kmnist_dir):
        kmnist_dir = _stage_kmnist(mods, bucket, repo_root)
    _x_train, y_train, _x_test, _y_test = mods.load_mnist(kmnist_dir, gz=False)
    labels = np.asarray(y_train, dtype=np.int64)
    official_indices = np.arange(N_OFFICIAL_TRAIN, dtype=np.int64)

    # Rank A from dual-arch encodings of the provisional set.
    ranked = audit.rank_discrepancy_indices(indices, thetas_arm, thetas_x86)
    A = ranked[:COMPONENT_A_SIZE]
    say(f"component A regenerated: top-{COMPONENT_A_SIZE} of {ranked.size}")

    # Rebuild B from ARM production deltas via the same rule (cap inert).
    encoded_name = _obj(mods, f"encoded_train_s{ENCODER_STEPS}", "npz")
    encoded_local = os.path.join(
        repo_root, "experiments", "stage2b_denoising", "results",
        f"stage3_encoded_train_s{ENCODER_STEPS}.npz")
    try:
        encoded_full, _ = consume_npz(
            mods, bucket, work_dir, encoded_name, no_upload=args.no_upload,
            fallback_local=encoded_local)
        positive, b_meta_fresh = audit.capped_positive_delta_indices(
            np.asarray(encoded_full["train_indices"], dtype=np.int64),
            np.asarray(encoded_full["deltas"], dtype=np.float64),
            tail_cap=TAIL_CAP)
        b_meta = dict(b_meta_fresh)
        b_meta["n_used"] = int(positive.size)
    except Exception as exc:  # noqa: BLE001
        say(f"re-deriving B from full encode failed ({exc}); using stress B empty")
        positive = np.asarray([], dtype=np.int64)

    final = audit.build_stress_indices(
        official_indices, labels, A, positive,
        seed=COMPONENT_D_SEED, tail_cap=TAIL_CAP)
    indices_refined = not np.array_equal(final, indices)
    if indices_refined:
        say(f"indices_refined=true: provisional n={indices.size} -> final n={final.size}")
        thetas_arm = join_rows(thetas_arm, indices, final, audit)
        thetas_x86 = join_rows(thetas_x86, indices, final, audit)
        deltas_arm = join_rows(deltas_arm, indices, final, audit)
        deltas_x86 = join_rows(deltas_x86, indices, final, audit)
        indices = final
        thetas_arm, deltas_arm, indices = assert_stress_row_contract(
            thetas_arm, deltas_arm, indices, label="arm-refined")
        thetas_x86, deltas_x86, indices_x = assert_stress_row_contract(
            thetas_x86, deltas_x86, indices, label="x86-refined")
        if not np.array_equal(indices, indices_x):
            raise Protocol1Fail("post-refine arm/x86 indices diverge")
    else:
        say("indices_refined=false: provisional equals final (expected regenerate path)")

    enc_max = audit.max_abs_difference(thetas_arm, thetas_x86)
    say(f"encoding-stage max-abs={enc_max:.6e}")
    if enc_max > ENCODING_SANITY_MAX_ABS:
        print(
            f"PROTOCOL1_WARN encoding max-abs={enc_max:.3e} exceeds "
            f"{ENCODING_SANITY_MAX_ABS:.0e} — likely index misalignment; "
            f"refusing evolve",
            flush=True)
        raise Protocol1Fail(
            f"encoding sanity gate: max-abs {enc_max:.3e} > "
            f"{ENCODING_SANITY_MAX_ABS:.0e}")

    construction = build_construction_record(
        b_meta=b_meta, n_stress=int(indices.size), indices=indices,
        indices_refined=indices_refined)
    record["construction"] = construction
    record["platforms"] = {
        "propagate": platform_summary(),
        "arm_encode": _summary_platform(arm),
        "x86_encode": _summary_platform(x86),
    }

    topo, _topo_path = _load_topologies(
        mods, bucket, work_dir, no_upload=args.no_upload)
    active_indices = np.asarray(
        arm.get("active_indices", topo["active_indices"]), dtype=np.int64)
    if active_indices.size != EXPECTED_N_ACTIVE:
        raise Protocol1Fail(
            f"active_indices size {active_indices.size} != {EXPECTED_N_ACTIVE}")
    ref_idx = int(audit.GAUGE_NODE)
    if ref_idx != EXPECTED_REF_IDX:
        raise Protocol1Fail(
            f"GAUGE_NODE {ref_idx} != expected {EXPECTED_REF_IDX}")
    if "ref_idx" in topo and int(np.asarray(topo["ref_idx"]).reshape(-1)[0]) != ref_idx:
        # topologies may store median-degree node under a different key; soft
        say("topologies ref_idx present and differs — continuing with GAUGE_NODE")
    say(f"ref_idx={ref_idx} (== GAUGE_NODE)")

    parents = None
    features = {}
    for arch, theta0 in (("arm", thetas_arm), ("x86", thetas_x86)):
        feats, _evolved, kind_feat = _evolve_arch(
            mods, theta0, topo, ref_idx, arch, work_dir, bucket, fp,
            parents, args.no_upload)
        # attach indices and publish features
        for condition, X in feats.items():
            X = np.asarray(X, dtype=np.float64)
            if X.shape != (indices.size, EXPECTED_FEATURE_DIM):
                raise Protocol1Fail(
                    f"{arch}/{condition} features {X.shape} != "
                    f"{(indices.size, EXPECTED_FEATURE_DIM)}")
            if not np.all(np.isfinite(X)):
                raise Protocol1Fail(f"{arch}/{condition} non-finite features")

            def compute(X=X, indices=indices):
                return {
                    "X": X.astype(np.float64),
                    "indices": indices.astype(np.int64),
                }

            name = _obj(mods, kind_feat, "npz",
                        condition=mods.conditions.path_segment(condition))
            ensure_npz(mods, bucket, work_dir, name, compute,
                       fingerprint=fp, parents=parents,
                       no_upload=args.no_upload)
        features[arch] = feats

    # Production alphas
    tag = grid_tag(mods.ridge.ALPHA_GRID)
    if tag != EXPECTED_RIDGE_GRID_TAG:
        raise Protocol1Fail(
            f"ALPHA_GRID tag {tag} != production {EXPECTED_RIDGE_GRID_TAG}")
    ridge_final_name = _obj(mods, f"ridge_final_{tag}", "npz")
    ridge_local = local_path_for(work_dir, ridge_final_name)
    if bucket is not None:
        mods.gcs.consume_validated(ridge_final_name, ridge_local, bucket=bucket)
    elif not os.path.isfile(ridge_local):
        raise Protocol1Fail(f"missing {ridge_final_name} locally")
    with np.load(ridge_local, allow_pickle=False) as handle:
        ridge_summary = json.loads(handle["summary_json"].item())
    alphas = ridge_summary["alphas"]
    say(f"production alphas loaded from {ridge_final_name}")

    # Clean Y for all 60k, active columns; stress slice by official index.
    images_01 = np.asarray(
        mods.load_mnist(kmnist_dir, gz=False)[0], dtype=np.float64) / 255.0
    Y_full = images_01.reshape(N_OFFICIAL_TRAIN, FULL_GRID)[:, active_indices]
    Y_stress = Y_full[indices]
    del images_01

    fits = {}
    ridge_arrays = {}
    pred = {"arm": {}, "x86": {}}
    mse = {"arm": {}, "x86": {}}
    ridge_meta = {"alphas": {}, "run_id": run_id, "grid_tag": tag}

    for condition in mods.conditions.ALL_CONDITIONS:
        alpha = float(alphas[condition])
        feat_name = _obj(
            mods, "features", "npz",
            condition=mods.conditions.path_segment(condition))
        feat_local = local_path_for(work_dir, feat_name)
        if bucket is not None:
            mods.gcs.consume_validated(feat_name, feat_local, bucket=bucket)
        elif not os.path.isfile(feat_local):
            raise Protocol1Fail(f"missing production features {feat_name}")
        with np.load(feat_local, allow_pickle=False) as handle:
            X_train_cond = np.asarray(handle["X"], dtype=np.float64)
        say(f"fit_final/{condition}: X={X_train_cond.shape} alpha={alpha:g}")
        fit, scaler = mods.ridge.fit_final(X_train_cond, Y_full, alpha)
        fits[condition] = {"fit": fit, "scaler": scaler, "alpha": alpha}
        # W/b are (n_alpha=1, ...)
        ridge_arrays[f"W_{condition}"] = np.asarray(fit["W"][0], dtype=np.float64)
        ridge_arrays[f"b_{condition}"] = np.asarray(fit["b"][0], dtype=np.float64)
        ridge_arrays[f"mean_{condition}"] = np.asarray(scaler.mean_, dtype=np.float64)
        ridge_arrays[f"scale_{condition}"] = np.asarray(scaler.scale_, dtype=np.float64)
        ridge_arrays[f"alpha_{condition}"] = np.asarray(alpha, dtype=np.float64)
        ridge_meta["alphas"][condition] = alpha

        for arch in ("arm", "x86"):
            X_s = features[arch][condition]
            pred_arch = mods.ridge.ridge_predict(
                fit, scaler.transform(np.asarray(X_s, dtype=np.float64)), 0)
            pred[arch][condition] = pred_arch
            mse[arch][condition] = mods.ridge.clipped_per_image_mse(
                pred_arch, Y_stress)
        del X_train_cond, fit, scaler

    ridge_arrays["summary_json"] = np.array(_dumps(ridge_meta))
    ridge_kind = synthesis_kind_ridge(run_id)
    ridge_obj = _obj(mods, ridge_kind, "npz")

    def compute_ridge():
        return ridge_arrays

    ensure_npz(mods, bucket, work_dir, ridge_obj, compute_ridge,
               fingerprint=fp, no_upload=args.no_upload)
    record["artifacts"]["ridge_frozen"] = ridge_obj

    delta_g = {
        "arm": audit.audit_deltas(mse["arm"]),
        "x86": audit.audit_deltas(mse["x86"]),
    }
    stages = audit.propagation_stage_maxima(
        theta_arm=thetas_arm, theta_x86=thetas_x86,
        features_arm=features["arm"], features_x86=features["x86"],
        pred_arm=pred["arm"], pred_x86=pred["x86"],
        mse_arm=mse["arm"], mse_x86=mse["x86"],
        delta_g_arm=delta_g["arm"], delta_g_x86=delta_g["x86"],
    )
    halt = audit.evaluate_propagation_halt(stages["delta_g"])
    record["stages"] = stages
    record["halt"] = halt

    report = {
        "run_id": run_id,
        "verdict": HALT_SENTINEL if halt["halt_triggered"] else OK_SENTINEL,
        "construction": construction,
        "stages": stages,
        "halt": halt,
        "platforms": record["platforms"],
        "fingerprint": {
            "source_manifest_digest": fp.get("source_manifest_digest") if fp else None,
            "config_digest": fp.get("config_digest") if fp else None,
        },
        "artifacts": {
            "stress_indices": idx_name,
            "encoded_arm": arm_name,
            "encoded_x86": x86_name,
            "ridge_frozen": ridge_obj,
            "report": _obj(mods, synthesis_kind_report(run_id), "json"),
        },
        "framing": FRAMING,
        "component_a_source": "regenerated",
        "encoding_sanity_max_abs": float(enc_max),
        "threshold": float(audit.CONTRAST_THRESHOLD),
        "sequencing_deviation": (
            "COMPANION_PROTOCOLS.md's consequence rule specifies interpretation "
            "review before Stage 4. Stage 4 has already run and is locked. A "
            "Protocol 1 result here is therefore disclosed as post-hoc relative "
            "to that ordering, following the same sequencing-deviation precedent "
            "already established for the amendment-impact audit."
        ),
    }
    record["report"] = report
    report_kind = synthesis_kind_report(run_id)
    report_obj = _obj(mods, report_kind, "json")

    def compute_report():
        return report

    ensure_json(mods, bucket, work_dir, report_obj, compute_report,
                no_upload=args.no_upload)
    # optional latest convenience copy (local only, not a GCS object)
    latest = results_mirror(work_dir, "protocol1_propagation_report_latest.json")
    try:
        with open(latest, "w", encoding="utf-8") as handle:
            handle.write(_dumps(report))
    except Exception as exc:  # noqa: BLE001
        say(f"latest copy skipped: {exc}")

    record["artifacts"]["report"] = report_obj
    verdict = report["verdict"]
    record["verdict"] = verdict
    say(f"report object: {report_obj}")
    say(f"ridge object:  {ridge_obj}")
    say(f"halt_triggered={halt['halt_triggered']} "
        f"exceeding={halt['exceeding_graphs']}")
    print(verdict, flush=True)
    return 0


def _summary_platform(blob):
    if "summary_json" not in blob:
        return {}
    try:
        s = json.loads(blob["summary_json"].item())
        return s.get("platform", s.get("phase", {}))
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------- CLI


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", default="arm-construct",
                   choices=("arm-construct", "x86-encode", "propagate", "preflight"))
    p.add_argument("--preflight", action="store_true",
                   help="alias for --phase preflight")
    p.add_argument("--bucket", default=None)
    p.add_argument("--credentials", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--kmnist-dir", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-upload", action="store_true",
                   help="local files only; tests / dry-run")
    return p


def main(argv=None):
    # Colab `exec -f` transmits text with no argv; phase comes from env.
    if argv is None and os.environ.get(ENV_COMMIT):
        phase = os.environ.get("PROTOCOL1_PHASE", "x86-encode")
        argv = ["--phase", phase]
    args = build_parser().parse_args(argv)
    if args.preflight:
        args.phase = "preflight"

    # Colab remote bootstrap when BONSAI_COMMIT is set (x86-encode path).
    commit = os.environ.get(ENV_COMMIT)
    record = new_record()
    mods = None
    bucket = None
    fp = None
    try:
        if commit:
            info = bootstrap_repo(commit)
            record["run"].update(info)
            record["run"]["driver_identity"] = verify_driver_identity(
                CLONE_DIR, os.environ.get(ENV_DRIVER_SHA))
            repo_root, work_dir = CLONE_DIR, REMOTE_WORK_DIR
        else:
            _mode, repo_root, work_dir = resolve_runtime()
        os.makedirs(work_dir, exist_ok=True)
        mods = load_modules(repo_root)
        bucket = get_bucket(mods, args)

        fp_config = {
            "protocol": 1,
            "phase": args.phase,
            "ladder_stage": LADDER_STAGE,
            "split": SPLIT,
            "encoder_steps": ENCODER_STEPS,
            "component_a_size": COMPONENT_A_SIZE,
            "component_d_seed": COMPONENT_D_SEED,
            "tail_cap": TAIL_CAP,
            "framing": FRAMING,
            "dtype": "float64",
        }
        if args.phase != "preflight":
            fp = build_fingerprint(
                mods, repo_root, fp_config, require_clean=not args.allow_dirty)
            say(f"fingerprint config {fp['config_digest'][:16]}... "
                f"sources {fp['source_manifest_digest'][:16]}...")

        if args.phase == "preflight":
            return phase_preflight(mods, args, repo_root, work_dir, bucket)
        if args.phase == "arm-construct":
            rc = phase_arm_construct(mods, args, repo_root, work_dir, bucket, fp)
        elif args.phase == "x86-encode":
            rc = phase_x86_encode(mods, args, repo_root, work_dir, bucket, fp)
        elif args.phase == "propagate":
            rc = phase_propagate(mods, args, repo_root, work_dir, bucket, fp,
                                 record)
        else:
            raise Protocol1Fail(f"unknown phase {args.phase!r}")

        if fp is not None:
            try:
                mods.fingerprint.revalidate_after_execution(fp, repo_root)
            except Exception as exc:  # noqa: BLE001
                say(f"fingerprint revalidate note: {exc}")
        return rc
    except Protocol1Fail as exc:
        traceback.print_exc()
        print(f"{FAIL_SENTINEL} {exc}", flush=True)
        return 1
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"{FAIL_SENTINEL} {type(exc).__name__}: {exc}", flush=True)
        return 1

