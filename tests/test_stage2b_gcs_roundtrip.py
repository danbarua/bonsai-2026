"""The one Stage 2B check that needs a real cloud runtime and a real
bucket: does an object written from a Colab session come back correctly
when read from outside that session?

Everything else covering `stage2b_gcs` verifies internal consistency of
code running on one machine -- `tests/test_stage2b_gcs.py` tests every
path, prefix, guard and credential decision offline, against an in-memory
fake bucket. None of it can establish that a cloud session can actually
write to the bucket and that the result is visible elsewhere, which is the
single thing the whole GCS layer exists for. That is this file's entire
purpose, and it deliberately checks nothing else.

    uv run --group gpu pytest tests/test_stage2b_gcs_roundtrip.py -m slow

Marked `slow` and so excluded from the default `-m "not slow"` suite: it
creates real cloud resources and consumes real compute, and must be asked
for explicitly.

## The flow

1. A CPU Colab session, named uniquely per run so two concurrent
   invocations cannot collide.
2. `google-cloud-storage` installed on the runtime.
3. The service-account key, `stage2b_gcs.py`, and a JSON parameters file
   uploaded into the session.
4. `experiments/stage2b_denoising/colab_gcs_roundtrip_probe.py` executed
   there as a plain Python script -- it writes one small text object whose
   name and content are both unique to this run, through
   `stage2b_gcs.ensure_artifact`.
5. **The assertion this test exists for**: the local pytest process, wholly
   outside the session, downloads that object and compares its bytes
   against what the probe was asked to write.
6. The object and the session are removed.

Because both the object name and its content carry a fresh nonce, a stale
object left by an earlier run cannot make this pass.

The readback uses the same service-account credentials rather than an
anonymous client. "Outside the session" here is a statement about the
process and the machine, not about the credentials; whether the bucket's
public-read grant works is a separate question this test does not ask.

## Scope of what is written

Training-side path space only, under a per-run `roundtrip_probe_<id>`
condition segment that no real artifact uses, at ladder stage 1.
`allow_test_split=True` is never passed. This is a transport check; it has
nothing to do with the Stage 2B test corpus.

## Credential states

Three states, deliberately distinguishable, because "did I actually
configure this right" should surface here rather than three steps into a
ladder run:

- `BONSAI_GCS_CREDENTIALS` unset or empty  -> skip
- set, but no file at that path            -> fail, naming the path
- set, file present, not a well-formed service-account key -> fail, saying so

`classify_credentials` is that logic, and the fast tests at the bottom of
this file exercise all three. Nothing here reads, prints, or otherwise
exposes the key's contents -- the validator reports field *names* and
parse positions only.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
import warnings
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_gcs as gcs                        # noqa: E402
import colab_gcs_roundtrip_probe as probe        # noqa: E402

_GCS_MODULE_FILE = _STAGE2B_DIR / "stage2b_gcs.py"
_PROBE_SCRIPT = _STAGE2B_DIR / "colab_gcs_roundtrip_probe.py"


# ---- What gets written, and where ----

PROBE_STAGE = 1                       # feasibility ladder stage 1, training side
PROBE_CONDITION_STEM = "roundtrip_probe"
PROBE_KIND = "probe"
PROBE_EXT = "txt"

REMOTE_DIR = "/content"
REMOTE_KEY_PATH = f"{REMOTE_DIR}/bonsai-colab-storage-key.json"
REMOTE_MODULE_PATH = f"{REMOTE_DIR}/stage2b_gcs.py"
REMOTE_PARAMS_PATH = probe.PARAMS_PATH
REMOTE_ARTIFACT_PATH = f"{REMOTE_DIR}/bonsai_stage2b_probe.txt"


# ---- Timeouts ----
#
# Every one of these is generous on purpose: an honest failure naming the
# step that ran out of time is worth more than a fast, confusing one. The
# CLI's own `exec --timeout` defaults to 30 seconds, which is short enough
# to be a hazard for anything involving a package install.

NEW_TIMEOUT_S = 300.0        # Colab runtime allocation; can queue for minutes
INSTALL_TIMEOUT_S = 900.0    # pip resolve + install on a cold VM; `install` has no --timeout
UPLOAD_TIMEOUT_S = 180.0     # three small files (key, module, params), each well under 100KB
EXEC_CODE_TIMEOUT_S = 300.0  # passed to `exec --timeout`: client construction + one small upload
EXEC_TIMEOUT_S = EXEC_CODE_TIMEOUT_S + 120.0   # the CLI's own timeout should fire first
STOP_TIMEOUT_S = 180.0


# ---- Credential states ----

SKIP = "skip"
FAIL = "fail"
OK = "ok"

# The fields `google.oauth2.service_account` requires of a key file. Named
# so a malformed key can be described precisely without its contents ever
# being read into a message.
REQUIRED_KEY_FIELDS = ("type", "project_id", "private_key", "private_key_id",
                       "client_email", "token_uri")


def describe_key_problem(path):
    """`None` if `path` holds a well-formed service-account key, else a
    short description of what is wrong with it.

    Reports field names, parse positions and error types only. No part of
    the file's contents is ever returned, printed, or logged -- a key is a
    secret, and a validator that leaks one while checking it is worse than
    no validator."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except UnicodeDecodeError:
        return "it is not valid UTF-8 text, so it is not a JSON key file"
    except json.JSONDecodeError as exc:
        return (f"it is not valid JSON ({exc.msg}, at line {exc.lineno} "
                f"column {exc.colno})")
    except OSError as exc:
        return f"it could not be read ({type(exc).__name__}: {exc.strerror})"

    if not isinstance(payload, dict):
        return f"its top level is a {type(payload).__name__}, not a JSON object"
    missing = [field for field in REQUIRED_KEY_FIELDS
               if not isinstance(payload.get(field), str) or not payload[field].strip()]
    if missing:
        return ("these service-account fields are missing, empty, or not strings: "
                + ", ".join(missing))
    if payload["type"] != "service_account":
        return "its 'type' field is not 'service_account'"
    return None


def classify_credentials(env=None):
    """`(verdict, message, path)` for the credential configuration.

    Verdict is `SKIP` when the override is unset -- a machine without the
    key simply cannot run this -- and `FAIL` when it is set to something
    that cannot work, because that is a configuration mistake and silently
    skipping it would let it resurface later as a confusing failure
    somewhere else.

    Empty counts as unset, matching `stage2b_gcs.credentials_path`, which
    falls back to its default on any falsy value."""
    env = os.environ if env is None else env
    raw = env.get(gcs.CREDENTIALS_ENV_VAR)
    if not raw:
        return (SKIP,
                f"{gcs.CREDENTIALS_ENV_VAR} is unset or empty. This test needs the Stage 2B "
                f"service-account key; point {gcs.CREDENTIALS_ENV_VAR} at it (the usual "
                f"location is {gcs.DEFAULT_CREDENTIALS_PATH}) to run it.",
                None)

    path = gcs.credentials_path(env)
    if not os.path.exists(path):
        return (FAIL,
                f"{gcs.CREDENTIALS_ENV_VAR} is set to {raw!r}, which resolves to {path}, but "
                f"no file exists there. This is a configuration error, not a missing "
                f"capability: unset {gcs.CREDENTIALS_ENV_VAR} to skip this test instead.",
                path)

    problem = describe_key_problem(path)
    if problem is not None:
        return (FAIL,
                f"{gcs.CREDENTIALS_ENV_VAR} points at {path}, but {problem}. A service-account "
                f"key that will not parse here will not authenticate on the Colab runtime "
                f"either. (The file's contents are never read into this message.)",
                path)
    return (OK, f"service-account key at {path} is well-formed", path)


def require_credentials(env=None):
    """Applies `classify_credentials`, returning the key's path when it is
    usable and skipping or failing the calling test otherwise."""
    verdict, message, path = classify_credentials(env)
    if verdict == SKIP:
        pytest.skip(message)
    if verdict == FAIL:
        pytest.fail(message, pytrace=False)
    return path


# ---- Driving the CLI ----

class ColabStepError(AssertionError):
    """A named pipeline step failed, timed out, or exited non-zero. Carries
    the step's name so the report says which one, rather than leaving a
    partial failure to be reconstructed."""


def _mighty_colab_executable():
    found = shutil.which("mighty-colab")
    if found is None:
        pytest.skip("the `mighty-colab` CLI is not on PATH. It ships in this project's `gpu` "
                    "dependency group -- run under `uv run --group gpu` to enable this test.")
    return found


def _require_storage_library():
    try:
        gcs._storage_module()
    except ImportError as exc:
        pytest.skip(f"google-cloud-storage is needed locally to read the object back from "
                    f"outside the Colab session, and is not installed. It ships in this "
                    f"project's `gpu` dependency group -- run under `uv run --group gpu` to "
                    f"enable this test. ({exc.__class__.__name__})")


def _evidence(line):
    """Report one step's evidence.

    Plain `print`, deliberately: under `-s` it streams live, and without
    `-s` pytest captures it and replays it on failure. Both are useful.
    What it must never become is a logger that a default-suite run has to
    configure away -- this test is deselected from that run entirely, so
    there is nothing to be quiet for."""
    print(f"[roundtrip] {line}", flush=True)


def _run_cli(step, args, timeout, executable):
    """One `mighty-colab` invocation, with an explicit timeout and a
    failure message that names the step."""
    try:
        result = subprocess.run([executable, *args], capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ColabStepError(
            f"timed out after {timeout:.0f}s trying to {step}. Nothing was concluded about "
            f"the GCS round trip; this is a timeout on that step alone.") from None
    if result.returncode != 0:
        raise ColabStepError(
            f"failed to {step}: `mighty-colab {' '.join(args)}` exited "
            f"{result.returncode}.\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}")
    return result


class _Roundtrip:
    """What the test needs to know about this run: its session, the object
    it expects to find, and the client that will look for it."""

    def __init__(self, *, executable, session, bucket, object_name, prefix, nonce,
                 key_path, params_file):
        self.executable = executable
        self.session = session
        self.bucket = bucket
        self.object_name = object_name
        self.prefix = prefix
        self.nonce = nonce
        self.key_path = key_path
        self.params_file = params_file


@pytest.fixture
def roundtrip(tmp_path):
    """A live CPU Colab session, plus the run's unique object identity, torn
    down unconditionally.

    Setup order is load-bearing. The credential gate runs first, so a
    misconfiguration is reported without any cloud resource being touched
    and is reachable on a machine with no `gpu` group installed at all. The
    local client is built next, before the session exists, so a client that
    cannot be constructed never strands a running session. Only then is the
    session created -- and the flag that authorises teardown is set
    *before* the call, because a `new` that times out may well have created
    a session anyway.

    Everything from the session's creation onward sits inside `try/finally`
    in the fixture body, so both cleanups run whether the test passes,
    fails, or the fixture itself raises partway through. Cleanup problems
    are reported, never swallowed."""
    key_path = require_credentials()
    _require_storage_library()
    executable = _mighty_colab_executable()

    bucket = gcs.get_bucket(credentials=key_path)

    stamp = uuid.uuid4().hex[:12]
    condition = f"{PROBE_CONDITION_STEM}_{stamp}"
    object_name = gcs.object_path(stage=PROBE_STAGE, condition=condition, kind=PROBE_KIND,
                                  ext=PROBE_EXT, split="train")
    prefix = gcs.condition_prefix(stage=PROBE_STAGE, condition=condition, split="train")
    nonce = uuid.uuid4().hex
    session = f"bonsai-stage2b-roundtrip-{stamp}"

    params_file = tmp_path / "probe_params.json"
    params_file.write_text(json.dumps({
        "object_name": object_name,
        "nonce": nonce,
        "bucket": gcs.bucket_name(),
        "project": gcs.GCS_PROJECT,
        "module_dir": REMOTE_DIR,
        "credentials_path": REMOTE_KEY_PATH,
        "local_path": REMOTE_ARTIFACT_PATH,
    }, indent=2), encoding="utf-8")

    session_may_exist = False
    try:
        session_may_exist = True
        _run_cli(f"create the CPU Colab session {session!r}",
                 ["new", "-s", session], NEW_TIMEOUT_S, executable)
        yield _Roundtrip(executable=executable, session=session, bucket=bucket,
                         object_name=object_name, prefix=prefix, nonce=nonce,
                         key_path=key_path, params_file=params_file)
    finally:
        problems = []
        try:
            gcs.delete_prefix(prefix, bucket=bucket, force_non_test_prefix=True)
        except Exception as exc:
            problems.append(f"could not delete {prefix!r} from the bucket: "
                            f"{type(exc).__name__}: {exc}")
        if session_may_exist:
            try:
                _run_cli(f"stop the Colab session {session!r}",
                         ["stop", "-s", session], STOP_TIMEOUT_S, executable)
            except Exception as exc:
                problems.append(f"could not stop the Colab session {session!r} -- it may still "
                                f"be running and consuming compute: {type(exc).__name__}: {exc}")
        if problems:
            report = "teardown did not fully succeed:\n  " + "\n  ".join(problems)
            print(report, flush=True)
            warnings.warn(report, stacklevel=1)
            if sys.exc_info()[0] is None:
                # Nothing is already propagating, so failing here reports the
                # teardown problem instead of hiding it behind a green run.
                pytest.fail(report, pytrace=False)


@pytest.mark.slow
def test_object_written_from_colab_is_readable_from_outside(roundtrip, tmp_path):
    """A Colab session writes one object; this process, outside it, reads
    the same bytes back.

    Both the object's name and its content are unique to this run, so a
    stale object cannot satisfy either half.

    **Run this with `-s`.** It reports each step's evidence as it goes,
    and on a rarely-run, explicitly-requested slow test that transcript
    is the point: a bare green PASS tells you the assertions held, not
    what actually happened on the wire. Without `-s`, pytest captures the
    output and shows it only if the test fails -- which is precisely
    backwards for a check whose whole job is to demonstrate that real
    infrastructure works. Verbosity costs nothing here; nobody is
    scrolling past this in a default suite run, because it is deselected
    from one."""
    ctx = roundtrip
    _evidence(f"session         : {ctx.session} (CPU runtime)")
    _evidence(f"object          : {ctx.object_name}")

    _run_cli("install google-cloud-storage on the Colab runtime",
             ["install", "google-cloud-storage", "-s", ctx.session],
             INSTALL_TIMEOUT_S, ctx.executable)
    _run_cli("upload the service-account key into the session",
             ["upload", str(ctx.key_path), REMOTE_KEY_PATH, "-s", ctx.session],
             UPLOAD_TIMEOUT_S, ctx.executable)
    _run_cli("upload stage2b_gcs.py into the session",
             ["upload", str(_GCS_MODULE_FILE), REMOTE_MODULE_PATH, "-s", ctx.session],
             UPLOAD_TIMEOUT_S, ctx.executable)
    _run_cli("upload the probe parameters into the session",
             ["upload", str(ctx.params_file), REMOTE_PARAMS_PATH, "-s", ctx.session],
             UPLOAD_TIMEOUT_S, ctx.executable)

    run = _run_cli("run the probe script on the Colab runtime",
                   ["exec", "-s", ctx.session, "-f", str(_PROBE_SCRIPT),
                    "--timeout", str(EXEC_CODE_TIMEOUT_S)],
                   EXEC_TIMEOUT_S, ctx.executable)
    transcript = f"--- stdout ---\n{run.stdout}\n--- stderr ---\n{run.stderr}"
    assert probe.FAIL_SENTINEL not in run.stdout, (
        f"the probe reported a failure on the Colab runtime.\n{transcript}")
    assert f"{probe.OK_SENTINEL} {ctx.object_name}" in run.stdout, (
        f"the probe did not report writing {ctx.object_name}. Either it never ran, or it "
        f"never reached its success sentinel.\n{transcript}")
    _evidence("probe on the VM :\n" + textwrap.indent(run.stdout.strip(), "    | "))

    # The assertion this whole test exists for: the object is visible, and
    # correct, from outside the session that wrote it.
    assert gcs.object_exists(ctx.object_name, bucket=ctx.bucket), (
        f"{ctx.object_name} was reported written from the Colab session but is not visible "
        f"to this process.\n{transcript}")

    expected = probe.probe_text(ctx.nonce).encode("utf-8")
    readback = tmp_path / "probe_readback.txt"
    gcs.download_file(ctx.object_name, str(readback), bucket=ctx.bucket)
    assert readback.read_bytes() == expected, (
        f"{ctx.object_name} read back from outside the session does not match what the probe "
        f"was asked to write.")
    _evidence(f"authenticated   : {len(expected)} bytes, exact match, read by this process")

    # The bucket is public-read, and Stage 2A's precedent depends on that
    # (a Colab driver pulling inputs with no credentials at all). The
    # authenticated readback above cannot show it: it proves the object
    # left the session, not that an anonymous consumer can fetch it. A
    # separate anonymous client is the only thing that distinguishes
    # "readable from outside this session" from "readable without
    # credentials", and those are different claims.
    anon_bucket = gcs.get_bucket(anonymous=True)
    anon_readback = tmp_path / "probe_readback_anon.txt"
    gcs.download_file(ctx.object_name, str(anon_readback), bucket=anon_bucket)
    assert anon_readback.read_bytes() == expected, (
        f"{ctx.object_name} is readable with credentials but not anonymously -- the bucket's "
        f"public-read grant is not doing what the Stage 2A precedent assumes.")
    _evidence(f"anonymous       : {len(expected)} bytes, exact match, no credentials used")


# ---- Fast checks on the credential gate itself (no network, no cloud) ----
#
# The gate decides whether the slow test above skips or fails, and it is the
# thing that turns a misconfiguration into an early, legible message. These
# run in the default suite.

def _well_formed_key_payload():
    """A syntactically valid service-account key with no real secret in it.
    Nothing here authenticates against anything."""
    return {
        "type": "service_account",
        "project_id": "not-a-real-project",
        "private_key_id": "0" * 40,
        "private_key": "-----BEGIN PRIVATE KEY-----\nNOT-A-REAL-KEY\n-----END PRIVATE KEY-----\n",
        "client_email": "nobody@not-a-real-project.iam.gserviceaccount.com",
        "client_id": "000000000000000000000",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


def _write_key(tmp_path, payload, name="key.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_unset_override_is_a_skip():
    verdict, message, path = classify_credentials(env={})
    assert verdict == SKIP
    assert path is None
    assert gcs.CREDENTIALS_ENV_VAR in message


def test_empty_override_is_a_skip():
    """`stage2b_gcs.credentials_path` falls back to its default on any falsy
    value, so an empty override means 'unset' there and must mean the same
    here -- otherwise the gate and the module disagree about one input."""
    verdict, _, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: ""})
    assert verdict == SKIP


def test_override_pointing_at_a_missing_file_is_a_failure(tmp_path):
    missing = tmp_path / "no-such-key.json"
    verdict, message, path = classify_credentials(
        env={gcs.CREDENTIALS_ENV_VAR: str(missing)})
    assert verdict == FAIL
    assert path == str(missing)
    assert str(missing) in message


def test_override_expands_a_tilde_before_deciding():
    """Resolution goes through `stage2b_gcs.credentials_path`, so `~` is
    expanded rather than treated as a literal directory name."""
    _, message, path = classify_credentials(
        env={gcs.CREDENTIALS_ENV_VAR: "~/definitely-not-a-key.json"})
    assert not path.startswith("~")
    assert os.path.expanduser("~") in path


def test_unparseable_key_is_a_failure(tmp_path):
    path = tmp_path / "key.json"
    path.write_text("{not json at all", encoding="utf-8")
    verdict, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == FAIL
    assert "not valid JSON" in message


def test_non_utf8_key_is_a_failure(tmp_path):
    path = tmp_path / "key.json"
    path.write_bytes(b"\xff\xfe\x00binary nonsense")
    verdict, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == FAIL
    assert "UTF-8" in message


def test_key_missing_required_fields_is_a_failure(tmp_path):
    payload = _well_formed_key_payload()
    del payload["private_key"]
    del payload["client_email"]
    path = _write_key(tmp_path, payload)
    verdict, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == FAIL
    assert "private_key" in message
    assert "client_email" in message


def test_key_with_empty_required_field_is_a_failure(tmp_path):
    payload = _well_formed_key_payload()
    payload["client_email"] = "   "
    path = _write_key(tmp_path, payload)
    verdict, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == FAIL
    assert "client_email" in message


def test_key_of_the_wrong_type_is_a_failure(tmp_path):
    payload = _well_formed_key_payload()
    payload["type"] = "authorized_user"
    path = _write_key(tmp_path, payload)
    verdict, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == FAIL
    assert "service_account" in message


def test_json_list_is_a_failure(tmp_path):
    path = tmp_path / "key.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    verdict, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == FAIL
    assert "not a JSON object" in message


def test_well_formed_key_passes(tmp_path):
    path = _write_key(tmp_path, _well_formed_key_payload())
    verdict, _, resolved = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    assert verdict == OK
    assert resolved == str(path)
    assert describe_key_problem(path) is None


def test_no_message_ever_carries_the_keys_contents(tmp_path):
    """The validator opens the key. Every message it can produce must
    describe the file without quoting any of it."""
    payload = _well_formed_key_payload()
    payload["private_key"] = "-----BEGIN PRIVATE KEY-----\nSENTINEL-SECRET\n-----END-----\n"
    payload["private_key_id"] = "SENTINEL-KEY-ID"
    del payload["client_email"]
    payload["type"] = "authorized_user"
    path = _write_key(tmp_path, payload)

    _, message, _ = classify_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)})
    for secret in ("SENTINEL-SECRET", "SENTINEL-KEY-ID", payload["private_key"]):
        assert secret not in message
    assert "client_email" in message


def test_require_credentials_skips_when_unset():
    with pytest.raises(pytest.skip.Exception):
        require_credentials(env={})


def test_require_credentials_fails_when_misconfigured(tmp_path):
    with pytest.raises(pytest.fail.Exception):
        require_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(tmp_path / "nope.json")})


def test_require_credentials_returns_the_path_when_usable(tmp_path):
    path = _write_key(tmp_path, _well_formed_key_payload())
    assert require_credentials(env={gcs.CREDENTIALS_ENV_VAR: str(path)}) == str(path)


def test_probe_text_is_unique_per_nonce_and_importable_without_side_effects():
    """Importing the probe module locally must not try to run it -- the local
    side needs `probe_text` to build the bytes it compares against, and that
    import happens on a machine with no runtime, no key, and no parameters
    file."""
    first, second = probe.probe_text("aaaa"), probe.probe_text("bbbb")
    assert first != second
    assert "aaaa" in first and "bbbb" in second
    assert not os.path.exists(probe.PARAMS_PATH)


def test_the_probe_writes_training_side_paths_only():
    """The object this test writes is training-side, at ladder stage 1, under
    a condition segment no real artifact uses."""
    condition = f"{PROBE_CONDITION_STEM}_deadbeefcafe"
    name = gcs.object_path(stage=PROBE_STAGE, condition=condition, kind=PROBE_KIND,
                           ext=PROBE_EXT, split="train")
    assert name.startswith(gcs.TRAIN_ROOT + "/")
    assert not gcs.is_test_split_path(name)
    prefix = gcs.condition_prefix(stage=PROBE_STAGE, condition=condition, split="train")
    assert name.startswith(prefix + "/")
