"""The cloud-runtime half of Stage 2B's GCS round-trip check: a plain
Python script that runs on a Colab runtime, writes one small probe object
to the bucket through `stage2b_gcs`'s own transport, and prints a sentinel
line saying so.

It is driven by `tests/test_stage2b_gcs_roundtrip.py`, which uploads this
file's inputs into the session and then executes it there
(`mighty-colab exec --file`). The local half of that test reads the object
back afterwards; nothing here asserts the round trip, because a check run
inside the session cannot establish that anything outside it can see the
result.

## Parameters arrive as a file, not as arguments or environment

Every input is read from a JSON file at `PARAMS_PATH`, uploaded into the
session before this script runs:

    object_name        the object to write (built by `stage2b_gcs.object_path`)
    nonce              makes this run's content unique to this run
    bucket             bucket name
    project            GCP project
    module_dir         directory on the runtime holding `stage2b_gcs.py`
    credentials_path   the service-account key's path on the runtime
    local_path         where on the runtime the artifact is written

The key's location is passed explicitly rather than through the
environment, so this script depends on no remote environment state at all.
Its contents are never read here, printed, or included in any error.

## Why it writes through `ensure_artifact`

`ensure_artifact` is the primitive every Stage 2B run script uses, so the
probe exercises the real production write path rather than a bespoke one.
The object name is unique per run, so the step must genuinely produce and
upload -- `skipped=True` would mean the name collided with something
already in the bucket, and is treated as a failure.

## Output

`PROBE_OK <object_name>` on success, `PROBE_FAIL <reason>` otherwise, in
both cases on stdout. The driving test keys on those sentinels for a
legible failure message, but its real assertion is the object it reads
back from outside the session.
"""
import json
import os
import sys

PARAMS_PATH = "/content/bonsai_stage2b_probe_params.json"

OK_SENTINEL = "PROBE_OK"
FAIL_SENTINEL = "PROBE_FAIL"

PROBE_HEADER = "stage2b gcs colab-to-local round-trip probe"

REQUIRED_PARAMS = (
    "object_name",
    "nonce",
    "bucket",
    "project",
    "module_dir",
    "credentials_path",
    "local_path",
)


def probe_text(nonce):
    """The exact content this run writes, as a function of its nonce.

    Defined here and imported by the driving test, so the bytes the local
    side compares against are built by the same code that wrote them --
    not by a second copy of the format that could drift out of step with
    this one."""
    if not isinstance(nonce, str) or not nonce:
        raise ValueError(f"nonce must be a non-empty string, got {nonce!r}")
    return f"{PROBE_HEADER}\nnonce={nonce}\n"


def _fail(reason):
    print(f"{FAIL_SENTINEL} {reason}", flush=True)
    return 1


def main(params_path=PARAMS_PATH):
    try:
        with open(params_path, "r", encoding="utf-8") as handle:
            params = json.load(handle)
    except Exception as exc:
        return _fail(f"could not read probe parameters from {params_path}: "
                     f"{type(exc).__name__}: {exc}")

    if not isinstance(params, dict):
        return _fail(f"probe parameters at {params_path} are a "
                     f"{type(params).__name__}, not a JSON object")
    missing = [name for name in REQUIRED_PARAMS if not params.get(name)]
    if missing:
        return _fail(f"missing param(s): {', '.join(missing)}")

    # Imported here rather than at module scope: the driving test imports
    # this module for `probe_text` on a machine that has neither
    # `stage2b_gcs` on its path under this name nor the runtime's layout.
    sys.path.insert(0, params["module_dir"])
    try:
        import stage2b_gcs as gcs
    except Exception as exc:
        return _fail(f"could not import stage2b_gcs from {params['module_dir']}: "
                     f"{type(exc).__name__}: {exc}")

    text = probe_text(params["nonce"])

    def produce(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    try:
        bucket = gcs.get_bucket(bucket_name=params["bucket"],
                                credentials=params["credentials_path"],
                                project=params["project"])
        result = gcs.ensure_artifact(params["object_name"], params["local_path"],
                                     produce=produce, bucket=bucket)
    except Exception as exc:
        return _fail(f"writing {params['object_name']} failed: {type(exc).__name__}: {exc}")

    print(f"probe step: {result.summary()}", flush=True)
    if result.skipped or not result.produced or not result.uploaded:
        return _fail(f"the step did not produce and upload {params['object_name']} "
                     f"(summary above); a per-run unique object name should never "
                     f"already exist in the bucket")

    try:
        present = gcs.object_exists(params["object_name"], bucket=bucket)
    except Exception as exc:
        return _fail(f"existence check on {params['object_name']} failed: "
                     f"{type(exc).__name__}: {exc}")
    if not present:
        return _fail(f"{params['object_name']} does not exist after a reportedly "
                     f"successful upload")

    print(f"{OK_SENTINEL} {params['object_name']}", flush=True)
    return 0


# Runs when executed as a script, and also when a remote kernel gives this
# code some other `__name__` -- in which case the presence of the uploaded
# parameters file is what says "this is the runtime, go". Importing this
# module locally satisfies neither condition, so `probe_text` can be
# imported without side effects.
#
# A successful run exits normally rather than through `sys.exit(0)`: under
# a kernel, `SystemExit` surfaces as a raised exception, and a success that
# looks like one to the caller is worse than a missing exit code. Failure
# does exit non-zero, on top of the sentinel line.
if __name__ == "__main__" or os.path.exists(PARAMS_PATH):
    _status = main()
    if _status:
        sys.exit(_status)
