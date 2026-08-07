"""Stage 2B's artifact transport: the Google Cloud Storage object-path
scheme, credential resolution, and the idempotent-step primitive its run
scripts use to survive a dead cloud session.

DESIGN.md, "Computational strategy": generation, features, and statistics
run entirely in the cloud environment, and artifacts are pushed to Google
Cloud Storage from within it -- never round-tripped through local upload
(Stage 2A hit Colab's ~6-15MB upload limit with a 242MB artifact once
already).

## Why an artifact lands in GCS at the END of every step

PROJECT_MEMORY.md Part 4: an ephemeral cloud session, and the agent
driving it, can die mid-task with nothing recoverable -- no results on
the VM, none locally, none in git. The response encoded here is that
every pipeline step names its output object up front, skips itself if
that object already exists, and uploads on completion. A session that
dies loses at most the one step that was in flight; a fresh runtime
re-attaches by downloading what is already there.

`ensure_artifact` is that primitive, and the whole pattern is one line in
a run script:

    r = stage2b_gcs.ensure_artifact(path, local, produce=build_it, bucket=bucket)

where `build_it(local_path)` writes the file. Nothing about resumability
is left to the caller remembering to check first.

## Surviving a death mid-transfer, in both directions

A step is the unit of resumption above; a transfer is the unit below it,
and a gigabyte artifact is large enough that losing one in flight matters
on its own.

`download_file` writes to a `.part` sidecar and renames it into place
only once the transfer completes, so a death mid-download cannot leave a
truncated file that the next run mistakes for a finished artifact.

`upload_file_chunked` is the other direction: the file goes up as
numbered part objects under `{name}.parts/`, each recorded in a
`{local_path}.upload.json` checkpoint only after the remote confirms it
arrived, and the parts are composed into the object at the end. A re-run
resumes after the last confirmed part rather than sending the whole
artifact again. `upload_file` remains the single-request path, and
`ensure_artifact` chooses between the two on the artifact's size --
`should_chunk` is the rule -- rather than on the call site having
remembered to ask. `chunked=True` and `chunked=False` force either route.

## Content integrity, on both transfer routes

Atomicity is not correctness: a `.part` rename guarantees the whole
transfer landed, not that the bytes are the artifact's. Stage 2B pulls
down multi-gigabyte feature arrays and runs statistics on them, so a
silently corrupted download is the worst failure available -- it
produces numbers rather than an error.

The digest is `crc32c`, base64-encoded exactly as GCS reports it, and it
is GCS's own: the service computes and records one for every object it
stores, a composed one included. `md5_hash` is not populated for
composite objects, so a consumer verifying MD5 would behave differently
depending on which upload route produced what it is reading; `crc32c`
gives one uniform check for both, and covers objects written before any
of this existed.

Every transfer is verified by default. An upload compares the object's
recorded checksum against the local file's, and an object that fails is
deleted rather than left behind -- `ensure_artifact` treats an object's
existence as proof its step is done, and a known-wrong object must not
make that claim. A download verifies the `.part` sidecar BEFORE renaming
it, so a file that fails never reaches the destination path.
`verify_content=False` opts out, at the call site, visibly. An object
carrying no checksum raises `ChecksumMissingError`: GCS records one for
everything it stores, so an absent field means the content could not be
checked, not that it is intact.

Computing the local side needs `google-crc32c`, a hard dependency of
`google-cloud-storage` and so present wherever a real transfer happens.
Where neither is installed -- this project's local development
environment -- a pure-Python CRC32C stands in, so verification still runs
under the fake buckets the tests inject rather than quietly becoming a
no-op. It is correspondingly slow, and never on the path of a real
gigabyte transfer.

## Lazy import, deliberately

`google.cloud.storage` is imported inside `_storage_module()`, never at
module import time. Everything in this module that decides a path, a
prefix, or a credential location is pure and runs -- and is unit-tested
-- in an environment where `google-cloud-storage` is not installed and
there is no network. Only the functions that genuinely need a live client
touch the library. The local development environment is exactly that
environment, so this is a property the test suite checks rather than a
preference.

## Object-path scheme

    {root}/{split}/stage{n}/{condition}/{kind}.{ext}

    stage2b/train/stage1/common/encoder_gate.json
    stage2b/train/stage2/evolved_T/features.npz
    stage2b/testsplit/stage4/evolved_T/predictions.npz

Four segments, one per thing that distinguishes an artifact: the
feasibility-ladder stage (1-4), the split side, the condition or graph,
and the artifact kind. `condition=None` renders as the reserved segment
`common` for artifacts that are not condition-specific (corruption
diagnostics, the encoder gate, the partition manifest), which keeps every
object at the same depth so a prefix listing is uniform.

`object_path` is a pure function of its arguments -- no client, no
network, no filesystem -- so the scheme is testable on its own.

## Test-split namespacing and the guard

Test-side objects live under their own root, `stage2b/testsplit/`, and
never share a prefix with training-side ones. Two consequences, both
intended:

- Reaching a test-side object path -- building it, uploading to it,
  downloading from it, checking it, deleting it -- requires
  `allow_test_split=True`, exactly the idiom
  `stage2b_corruption._check_split_allowed` uses, and for the same
  reason. DESIGN.md locks that no Stage 2B test-side result is accessed
  during feasibility stages 1-3; that lock is structural here, not a
  convention. Stage 4's single confirmatory evaluation is the one place
  that passes the flag, deliberately and visibly. Because the lock is
  about stages 1-3, `split="test"` is additionally refused at any stage
  other than 4.
- Everything test-side is removable in one call. `delete_prefix` is the
  bulk delete, and it refuses any prefix outside `stage2b/testsplit/`
  unless separately forced -- an accidental wipe of training-side
  artifacts is the obvious catastrophic misuse of a delete-by-prefix
  helper. It refuses any prefix outside `stage2b/` unconditionally,
  forced or not: the bucket is shared with other stages' cached data and
  nothing here has any business deleting that.

## Test-use scope

DESIGN.md's scope, in effect: the official KMNIST test images were used
extensively by Stage 2A and are not project-unseen. What is locked is
that the prespecified Stage 2B corrupted test corpus, test features,
model predictions, and denoising scores are generated and inspected in
one final confirmatory evaluation only. Nothing here should be described
as producing a project-unseen test set.

## Credentials

The service-account key defaults to
`~/.config/colab-cli/bonsai-colab-storage-key.json`, the path it is
uploaded to on a cloud runtime, and the `BONSAI_GCS_CREDENTIALS`
environment variable overrides it. `credentials_path()` resolves and
returns that path and does not open it. Nothing in this module reads,
logs, or echoes the key's contents; the only filesystem question it asks
is whether the path exists.

The bucket is public-read, so a script that only downloads needs no key
at all -- `anonymous=True` builds a credential-free client. Anonymous
clients cannot write.
"""
import base64
import hashlib
import json
import os
import re
from typing import NamedTuple

# ---- Infrastructure constants (settled; not configuration to rediscover) ----
GCS_PROJECT = "bonsai-504422"

# The bucket resolves like the credentials path below: a default here, an
# environment variable that overrides it, and a function that does the
# resolving. There is deliberately no module-level `GCS_BUCKET` constant --
# a name holding the default would be read at import time by anything that
# referenced it, silently bypassing the override and leaving a caller
# operating on a different bucket than the one it was told to use
# (CLAUDE.md principle 16: the helper is correct, the glue around it is
# what goes wrong). Call `bucket_name()`.
BUCKET_ENV_VAR = "BONSAI_GCS_BUCKET"
DEFAULT_GCS_BUCKET = "bonsai-2026-stage2b-cache"   # public read (anonymous objectViewer)

CREDENTIALS_ENV_VAR = "BONSAI_GCS_CREDENTIALS"
DEFAULT_CREDENTIALS_PATH = "~/.config/colab-cli/bonsai-colab-storage-key.json"

# ---- The object-path scheme ----
ROOT_PREFIX = "stage2b"
TRAIN_ROOT = "stage2b/train"
TEST_SPLIT_ROOT = "stage2b/testsplit"
COMMON_CONDITION = "common"       # reserved segment for non-condition-specific artifacts

VALID_SPLITS = ("train", "test")
LADDER_STAGES = (1, 2, 3, 4)      # DESIGN.md's feasibility ladder
TEST_SPLIT_STAGE = 4              # the one locked confirmatory evaluation

# Condition and kind tokens are validated by shape, not against a fixed
# vocabulary: DESIGN.md's condition names include `evolved_T`, so the
# charset is case-sensitive, and pinning a list here would invent design
# facts this module has no business owning. What the pattern does buy is
# that no token can introduce a path separator, a parent-directory hop, or
# a leading dot.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_EXT_RE = re.compile(r"^[A-Za-z0-9]+$")
# GCS bucket naming, restricted to the unambiguous subset: lowercase only,
# no leading/trailing punctuation. Domain-named buckets are not in play.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")

# ---- Chunked upload ----
CHUNK_SIZE_DEFAULT = 64 * 1024 * 1024   # 64 MiB: one part is one request, held in memory once
COMPOSE_MAX_SOURCES = 32                # GCS composes at most 32 sources per request
MAX_PARTS = 1024                        # this module's own cap; see `upload_file_chunked`
PART_SUFFIX = ".parts"
CHECKPOINT_SUFFIX = ".upload.json"
CHECKPOINT_FORMAT = "stage2b-chunked-upload/1"
_OCTET_STREAM = "application/octet-stream"

# ---- Content integrity ----
CHECKSUM_FIELD = "crc32c"               # the GCS object property both upload routes land in
DIGEST_BLOCK_SIZE = 4 * 1024 * 1024     # a gigabyte artifact is digested in blocks, not read whole
_CRC32C_POLYNOMIAL = 0x82F63B78         # CRC-32C (Castagnoli), reflected
_CRC32C_TABLE = None


class ChunkedUploadError(OSError):
    """A part upload returned without the object arriving intact.

    Deliberately not a `FileNotFoundError`: a missing local file and a
    part that did not land are different failures, and a caller catching
    one should not silently swallow the other."""


class ChecksumMismatchError(OSError):
    """An object's own checksum and the local bytes disagree.

    Which side is wrong is not knowable from here -- the message names
    the object and both digests so it can be worked out. What is knowable
    is that the transfer must not be treated as good."""


class ChecksumMissingError(OSError):
    """An object carries no checksum to verify against.

    Separate from a mismatch because the remedy differs: a mismatch means
    something is corrupt, this means nothing could be checked. GCS records
    a `crc32c` for every object it stores, so this should not happen
    against the real bucket at all."""


# ---- Credentials: resolved, never read ----

def credentials_path(env=None):
    """The service-account key's path: `BONSAI_GCS_CREDENTIALS` if set to
    a non-empty value, otherwise `DEFAULT_CREDENTIALS_PATH`, in both cases
    with `~` expanded.

    Returns the path. Does not open, read, or validate the file -- the key
    is a secret and this module never touches its contents.

    `env` overrides the environment mapping consulted, for tests."""
    env = os.environ if env is None else env
    raw = env.get(CREDENTIALS_ENV_VAR) or DEFAULT_CREDENTIALS_PATH
    return os.path.expanduser(raw)


def bucket_name(env=None):
    """The bucket to operate on: `BONSAI_GCS_BUCKET` if set to a non-empty
    value, otherwise `DEFAULT_GCS_BUCKET`.

    The name is validated the same way an object-path token is, and for
    the same reason: a name carrying a `/` would turn every object path
    built against it into a different path than the one requested, and a
    stray empty or whitespace value should fail loudly here rather than
    produce a confusing 404 from the API.

    `env` overrides the environment mapping consulted, for tests."""
    env = os.environ if env is None else env
    # Stripped *before* the fallback, so a variable set to whitespace --
    # which is what `BONSAI_GCS_BUCKET=` or a stray quote in a Makefile
    # produces -- falls back to the default rather than raising. Matches
    # `credentials_path`'s treatment of an empty override.
    name = (env.get(BUCKET_ENV_VAR) or "").strip() or DEFAULT_GCS_BUCKET
    if not _BUCKET_RE.match(name):
        raise ValueError(
            f"{BUCKET_ENV_VAR}={name!r} is not a valid GCS bucket name: expected 3-63 "
            f"characters of lowercase letters, digits, dashes, underscores and dots, "
            f"starting and ending alphanumeric.")
    return name


def credentials_available(env=None):
    """Whether the resolved key path exists. An existence check only; the
    file is not opened."""
    return os.path.exists(credentials_path(env))


# ---- The path scheme: pure, no client, no network ----

def _check_split(split):
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS!r}, got {split!r}")


def _check_split_allowed(split, allow_test_split, *, stage=None):
    """DESIGN.md's test-side lock, mirroring
    `stage2b_corruption._check_split_allowed`: feasibility stages 1-3
    cannot reach a test-side object by accident, typo, or a loop variable
    that ran one iteration too far."""
    _check_split(split)
    if split != "test":
        return
    if not allow_test_split:
        raise PermissionError(
            "refusing to touch a Stage 2B TEST-side object: DESIGN.md locks 'no Stage 2B "
            "test-side result is accessed during stages 1-3'. Only the single confirmatory "
            "evaluation at feasibility stage 4 may pass allow_test_split=True, and it must "
            "do so deliberately.")
    if stage is not None and stage != TEST_SPLIT_STAGE:
        raise ValueError(
            f"split='test' is defined only at feasibility ladder stage {TEST_SPLIT_STAGE} "
            f"(DESIGN.md: 'ONE locked evaluation on the official 10,000-image test corpus'); "
            f"got stage={stage!r}. allow_test_split=True opts into stage "
            f"{TEST_SPLIT_STAGE}, not into test-side work at the earlier stages.")


def split_root(split):
    """The prefix root for a split. Test-side objects get their own,
    `stage2b/testsplit/`, so that nothing training-side shares a prefix
    with them and the whole test side is removable in one call.

    Pure: no guard here, because naming the root is not reaching an
    object. `object_path`, `stage_prefix`, and the transport functions
    apply the guard."""
    _check_split(split)
    return TEST_SPLIT_ROOT if split == "test" else TRAIN_ROOT


def is_test_split_path(name):
    """Whether an object name or prefix lies under the test-split root.
    Matches the root itself and anything beneath it, and nothing that
    merely starts with the same characters."""
    name = str(name)
    return name == TEST_SPLIT_ROOT or name.startswith(TEST_SPLIT_ROOT + "/")


def _check_token(value, what):
    if not isinstance(value, str) or not _TOKEN_RE.match(value):
        raise ValueError(
            f"{what} must match {_TOKEN_RE.pattern} (letters, digits, underscore, dash; "
            f"no separators, no leading dot), got {value!r}")


def _check_stage(stage):
    # `stage in LADDER_STAGES` alone would accept `1.0` and `True`, both of
    # which render as "stage1" and neither of which anyone meant to write.
    # The type is checked before the membership.
    if not isinstance(stage, int) or isinstance(stage, bool) or stage not in LADDER_STAGES:
        raise ValueError(f"stage must be one of {LADDER_STAGES!r} as an int (DESIGN.md's "
                         f"feasibility ladder), got {stage!r}")
    return int(stage)


def stage_prefix(*, stage, split, allow_test_split=False):
    """The prefix holding one ladder stage's artifacts on one split, e.g.
    `stage2b/train/stage2`. Listing or deleting under it covers every
    condition at that stage."""
    stage = _check_stage(stage)
    _check_split_allowed(split, allow_test_split, stage=stage)
    return f"{split_root(split)}/stage{stage}"


def condition_prefix(*, stage, condition, split, allow_test_split=False):
    """The prefix holding one condition's artifacts at one ladder stage,
    e.g. `stage2b/train/stage2/evolved_T`. `condition=None` gives the
    reserved `common` segment used by artifacts that are not
    condition-specific."""
    prefix = stage_prefix(stage=stage, split=split, allow_test_split=allow_test_split)
    if condition is None:
        return f"{prefix}/{COMMON_CONDITION}"
    _check_token(condition, "condition")
    if condition == COMMON_CONDITION:
        raise ValueError(
            f"{COMMON_CONDITION!r} is the reserved segment for artifacts that are not "
            f"condition-specific -- pass condition=None to use it, so the intent is "
            f"visible at the call site rather than spelled as a condition name.")
    return f"{prefix}/{condition}"


def object_path(*, stage, condition, kind, ext, split, allow_test_split=False):
    """The full object name for one artifact:

        {root}/{split}/stage{n}/{condition}/{kind}.{ext}

    A pure function of its arguments -- it constructs no client and
    touches no network or filesystem, so the scheme is testable on its
    own. Every argument is keyword-only with no default: which ladder
    stage, which condition, which split and which kind an artifact belongs
    to is never a thing this module should guess.

    `condition=None` renders the reserved `common` segment. `split="test"`
    requires `allow_test_split=True` and `stage=4`."""
    _check_token(kind, "kind")
    if not isinstance(ext, str) or not _EXT_RE.match(ext):
        raise ValueError(f"ext must match {_EXT_RE.pattern} and exclude the dot, got {ext!r}")
    prefix = condition_prefix(stage=stage, condition=condition, split=split,
                              allow_test_split=allow_test_split)
    return f"{prefix}/{kind}.{ext}"


# =====================================================================
# Artifact class: which names are write-once
# =====================================================================
#
# Generation pinning guarantees "the consumer reads the object that was
# committed" only while that generation is the surviving one. Object
# versioning is OFF on this bucket (measured, `PHASE_B_PLAN.md`), so a
# supersession destroys the pinned generation -- the soft-delete window
# keeps the bytes for seven days but a pinned READ still 404s, which is a
# forensic affordance and not availability.
#
# So semantic naming plus manifest pinning is only acceptable under a
# genuinely write-once policy for anything a consumer pins. If an ordinary
# overwrite path remained open for those artifacts, content-addressed
# naming would become mandatory. It does not remain open: the policy is
# enforced here rather than described in a document.
#
# Two classes:
#
#   LINEAGE     everything scientific -- anything that is or can be a
#               PARENT in a manifest, anything whose generation a consumer
#               pins. Strictly create-once. `force=True` on one of these
#               is a REFUSAL, not a republish, so an accidental or
#               external replacement surfaces as a loud availability
#               failure rather than as silent corruption.
#
#   RUN_SCOPED  reports and timing summaries: deliberately regenerated,
#               never a parent, never pinned. The drivers give these
#               run-scoped names so they are create-once in practice too;
#               the class exists so that if one ever is overwritten, it
#               still cannot enter a lineage as a parent.
#
# The default for an undeclared kind is LINEAGE, deliberately. The
# dangerous mistake is treating a pinned artifact as disposable, so the
# default errs toward refusing a write rather than allowing one.

LINEAGE = "lineage"
RUN_SCOPED = "run_scoped"

# Kind tokens that are run-scoped. Matched as prefixes, because the
# drivers append a run id (`stage2_report_20260807T160000Z`).
RUN_SCOPED_KIND_PREFIXES = (
    "stage1_report", "stage2_report", "stage3_report",
    "probe", "chunked_probe",          # smoke-check scratch, never consumed
)


class WriteOnceViolation(OSError):
    """A write would have replaced a committed lineage artifact."""


def artifact_kind(name):
    """The kind token of an object path: the basename without its
    extension. `object_path` builds `{prefix}/{kind}.{ext}`, so this is
    the inverse of that last segment and nothing more."""
    base = str(name).rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def artifact_class(name):
    """`LINEAGE` or `RUN_SCOPED` for an object path."""
    kind = artifact_kind(name)
    for prefix in RUN_SCOPED_KIND_PREFIXES:
        if kind == prefix or kind.startswith(prefix + "_"):
            return RUN_SCOPED
    return LINEAGE


def _check_object_path_allowed(name, allow_test_split):
    """The same lock applied to an object name the caller assembled itself
    -- `object_path` is not the only way to produce a string, so the
    transport functions re-check rather than trusting that it was used."""
    name = str(name)
    if not name:
        raise ValueError("object path must not be empty")
    if is_test_split_path(name) and not allow_test_split:
        raise PermissionError(
            f"refusing to touch {name!r}: it lies under the Stage 2B test-split root "
            f"{TEST_SPLIT_ROOT!r}, and DESIGN.md locks 'no Stage 2B test-side result is "
            f"accessed during stages 1-3'. Pass allow_test_split=True only from the stage 4 "
            f"confirmatory evaluation.")
    return name


# ---- The lazy import, in one place ----

def _storage_module():
    """`google.cloud.storage`, imported on demand.

    Kept out of module scope on purpose: every path, prefix, and
    credential decision above must remain importable and testable where
    `google-cloud-storage` is not installed and there is no network, which
    is exactly this project's local development environment. The library
    lives in the `gpu` dependency group with the rest of the cloud-runtime
    dependencies."""
    try:
        from google.cloud import storage
    except ImportError as exc:                      # pragma: no cover - needs the library absent
        raise ImportError(
            "google-cloud-storage (>=2.19.0) is required for Stage 2B's GCS transport but "
            "is not installed. It is expected on the cloud runtime, not in the local "
            "development environment -- which is why it is imported here rather than at "
            "module scope."
        ) from exc
    return storage


def make_client(*, credentials=None, project=GCS_PROJECT, anonymous=False):
    """A `google.cloud.storage.Client`.

    `credentials` is the service-account key's path, defaulting to
    `credentials_path()`. Its existence is checked; its contents are never
    read by this module and never appear in an error message.

    `anonymous=True` builds a credential-free client instead. The bucket
    is public-read, so that is enough to download and to check existence
    -- it cannot upload or delete."""
    storage = _storage_module()
    if anonymous:
        return storage.Client.create_anonymous_client()
    path = credentials_path() if credentials is None else os.path.expanduser(str(credentials))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"service-account key not found at {path}. On a cloud runtime the key is uploaded "
            f"into the session at {DEFAULT_CREDENTIALS_PATH}; set {CREDENTIALS_ENV_VAR} to "
            f"override the location. For read-only work pass anonymous=True instead -- the "
            f"bucket is public-read.")
    return storage.Client.from_service_account_json(path, project=project)


def get_bucket(*, name=None, client=None, credentials=None,
               project=GCS_PROJECT, anonymous=False):
    """The bucket handle every transport function below takes.

    Build it once per script and pass it down: the transport functions
    require `bucket` as a keyword argument with no default, so a run
    script constructs exactly one client and a test can inject a stand-in
    without the library being installed at all.

    `name` defaults to `bucket_name()`, resolved on each call rather than
    bound at import: a default argument would freeze whatever the
    environment held when this module was first imported."""
    if name is None:
        name = bucket_name()
    if client is None:
        client = make_client(credentials=credentials, project=project, anonymous=anonymous)
    return client.bucket(name)


# ---- Content integrity: the digest, and the comparison it feeds ----

def _google_crc32c():
    """`google_crc32c`, imported on demand, or None where it is absent.

    Lazily imported for the same reason `google.cloud.storage` is: it is
    a hard dependency of that library and ships with it, so it is present
    on the cloud runtime and absent locally, and nothing here may make
    the module unimportable in the second case."""
    try:
        import google_crc32c
    except ImportError:
        return None
    return google_crc32c


def checksum_backend():
    """Which CRC32C implementation is in use: `google-crc32c` (the
    compiled one GCS's own client uses) or `python` (the fallback). Worth
    reporting from a script that verifies a real transfer."""
    return "python" if _google_crc32c() is None else "google-crc32c"


def _crc32c_table():
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        table = []
        for index in range(256):
            value = index
            for _ in range(8):
                value = (value >> 1) ^ (_CRC32C_POLYNOMIAL if value & 1 else 0)
            table.append(value)
        _CRC32C_TABLE = tuple(table)
    return _CRC32C_TABLE


def _crc32c_python(data, value=0):
    """CRC-32C over `data`, continuing a running `value`.

    The fallback for an environment without `google-crc32c`, which is
    this project's local one. Pinned against the algorithm's published
    check value (`123456789` -> 0xE3069283) and cross-checked against
    `google_crc32c` wherever that is installed, because a checksum that
    is merely self-consistent would agree with itself while disagreeing
    with the service (CLAUDE.md principle 16)."""
    table = _crc32c_table()
    value ^= 0xFFFFFFFF
    for byte in data:
        value = table[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


class _RunningCrc32c:
    """A CRC32C accumulated over successive blocks, in whichever
    implementation is available, reported in GCS's own encoding."""

    def __init__(self):
        module = _google_crc32c()
        self._checksum = None if module is None else module.Checksum()
        self._value = 0

    def update(self, data):
        if self._checksum is None:
            self._value = _crc32c_python(data, self._value)
        else:
            self._checksum.update(data)

    def encoded(self):
        digest = (self._value.to_bytes(4, "big") if self._checksum is None
                  else self._checksum.digest())
        return base64.b64encode(digest).decode("ascii")


def crc32c_of_bytes(data):
    """The CRC32C of some bytes, base64-encoded as GCS reports it -- so a
    comparison against `blob.crc32c` is a string comparison with no
    decoding step to get wrong."""
    running = _RunningCrc32c()
    running.update(bytes(data))
    return running.encoded()


def crc32c_of_file(path):
    """The CRC32C of a file's contents, read in blocks. A Stage 2B
    artifact does not fit comfortably in memory and is not read whole to
    be checked."""
    running = _RunningCrc32c()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(DIGEST_BLOCK_SIZE), b""):
            running.update(block)
    return running.encoded()


def _blob_checksum(blob):
    """The object's recorded `crc32c`, or None if the field is not
    populated. A freshly-built handle carries no metadata until it is
    reloaded, which is why the reload is here rather than assumed -- the
    same shape as `_remote_size`."""
    value = getattr(blob, CHECKSUM_FIELD, None)
    if value is None:
        reload_ = getattr(blob, "reload", None)
        if callable(reload_):
            reload_()
            value = getattr(blob, CHECKSUM_FIELD, None)
    return None if value is None else str(value)


def _compare_checksum(blob, name, local_digest, *, action, local_label):
    """Raises unless the object's checksum and the local bytes agree.
    Returns the digest they agreed on."""
    recorded = _blob_checksum(blob)
    if recorded is None:
        raise ChecksumMissingError(
            f"{name!r} carries no {CHECKSUM_FIELD}. GCS records one for every object it "
            f"stores, composite ones included, so an absent field means the content could "
            f"not be checked -- not that it is intact. Pass verify_content=False to proceed "
            f"without a check, deliberately.")
    if recorded != local_digest:
        raise ChecksumMismatchError(
            f"{action} {name!r} does not match: the object's {CHECKSUM_FIELD} is {recorded}, "
            f"the {local_label}'s is {local_digest}.")
    return recorded


def _discard_object(bucket, name):
    """Removes an object that failed verification.

    `ensure_artifact` treats an object's existence as proof its step is
    done and skips the step on the next run. An object already known to
    be wrong must not be left behind making that claim. Best-effort: the
    verification failure is the error worth raising, not whatever the
    cleanup delete does."""
    try:
        bucket.blob(name).delete()
    except Exception:                                # noqa: BLE001 - cleanup, not the failure
        pass


def _verify_uploaded(bucket, name, local_digest):
    """Checks what actually landed, and removes it if it is wrong.

    Built on a fresh handle rather than the one the upload returned, so
    the service is asked again about the stored object instead of being
    taken at its word about the write. One extra metadata request, next
    to a transfer measured in gigabytes."""
    try:
        return _compare_checksum(bucket.blob(name), name, local_digest,
                                 action="the upload to", local_label="local file")
    except (ChecksumMismatchError, ChecksumMissingError):
        _discard_object(bucket, name)
        raise


def object_checksum(name, *, bucket, allow_test_split=False):
    """The `crc32c` GCS recorded for an object, or None if the field is
    not populated. Metadata only -- the object is not downloaded."""
    name = _check_object_path_allowed(name, allow_test_split)
    return _blob_checksum(bucket.blob(name))


def verify_object(name, local_path, *, bucket, allow_test_split=False):
    """Checks a local file against the object's own checksum without
    transferring anything, and returns the digest they agreed on.

    The check `download_file` and the upload paths run, callable on its
    own -- for a script confirming a local copy is still the artifact, or
    demonstrating that verification is live rather than vacuous."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"nothing to verify: {local_path} is not a file")
    return _compare_checksum(bucket.blob(name), name, crc32c_of_file(local_path),
                             action="the object at", local_label="local file")


# ---- Transport ----

def object_exists(name, *, bucket, allow_test_split=False):
    """Whether an object is already in the bucket. This is what makes a
    step skippable, and so what makes a pipeline resumable after a session
    dies."""
    name = _check_object_path_allowed(name, allow_test_split)
    return bool(bucket.blob(name).exists())


class GenerationPreconditionError(OSError):
    """A write was refused because the object was not in the generation the
    writer expected -- someone else wrote it in between."""


def _precondition(generation_match):
    """Keyword arguments carrying a generation precondition, or none.

    `0` means "only if this object does not exist"; a positive integer
    means "only if it is still exactly this generation". Both are
    server-side compare-and-set: the service refuses the write rather than
    the client checking first and racing between the check and the write.
    """
    if generation_match is None:
        return {}
    return {"if_generation_match": int(generation_match)}


def _precondition_failed_types():
    """The exception a precondition violation raises, looked up lazily.

    `google.api_core` is a cloud-only import, like `google.cloud.storage`
    itself -- this module must stay importable on a machine that has
    neither. Returns an empty tuple when it is absent, in which case
    nothing is translated and the underlying error propagates unchanged."""
    try:
        from google.api_core import exceptions as api_exceptions
    except ImportError:
        return ()
    return (api_exceptions.PreconditionFailed,)


def current_generation(name, *, bucket, allow_test_split=False):
    """The object's current generation, or 0 if it does not exist.

    0 is the value a precondition uses to mean "must not exist", so an
    absent object and the precondition that requires absence share a
    representation deliberately."""
    _check_object_path_allowed(name, allow_test_split)
    blob = bucket.blob(name)
    if not blob.exists():
        return 0
    blob.reload()
    return int(getattr(blob, "generation", 0) or 0)


def _write_guarded(name, generation_match, action):
    """Runs `action()`, translating a service-side precondition failure
    into this module's own named error.

    The message says what the guard is FOR, because a bare 412 from the
    client library reads as a transient fault and this is not one: it means
    a second writer reached the same object, which for an artifact bucket
    is a correctness problem rather than something to retry."""
    try:
        return action()
    except _precondition_failed_types() as exc:
        raise GenerationPreconditionError(
            f"refusing to write {name!r}: the object is not in the generation this "
            f"writer expected (required generation {generation_match}). Another "
            f"writer changed it in between -- most likely a second session that "
            f"was presumed dead. The write did NOT happen, which is the intended "
            f"outcome: an artifact silently replaced mid-run is worse than one "
            f"that failed loudly.") from exc


def upload_file(local_path, name, *, bucket, allow_test_split=False, verify_content=True,
                generation_match=None):
    """Uploads a local file to an object path. Returns the object path.

    `verify_content` (on by default) compares the object's own `crc32c`
    against the local file's once the upload returns, and deletes the
    object if they disagree rather than leaving a known-wrong artifact in
    the bucket for the next run to skip past. It costs one more read of
    the local file, which is disk against a transfer that is network.

    `generation_match` is the no-overwrite-race guard Freeze 4 requires --
    `0` to create only, an integer to replace exactly that generation. It
    is a compare-and-set at the service, not a check here followed by a
    write: the gap between checking and writing is the race."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"nothing to upload: {local_path} is not a file")
    _write_guarded(name, generation_match, lambda: bucket.blob(name).upload_from_filename(
        local_path, **_precondition(generation_match)))
    if verify_content:
        _verify_uploaded(bucket, name, crc32c_of_file(local_path))
    return name


# ---- Chunked upload: a transfer that survives the session dying ----

def part_prefix(name):
    """Where one object's in-flight parts live: `{name}.parts`.

    Adjacent to the object itself, deliberately. A part of a test-side
    object is itself test-side, so it lands under the test-split root and
    both the guard and `delete_test_split_artifacts` cover it with no
    special case."""
    return f"{str(name)}{PART_SUFFIX}"


def part_name(name, index):
    """One part's object name. Zero-padded so the parts sort in the order
    they are composed in, which makes a half-finished upload legible in a
    bucket listing."""
    return f"{part_prefix(name)}/{int(index):06d}"


def checkpoint_path(local_path):
    """The upload checkpoint's sidecar, `{local_path}.upload.json` -- next
    to the file being uploaded, in the manner of `download_file`'s
    `.part`."""
    return f"{str(local_path)}{CHECKPOINT_SUFFIX}"


def _check_chunk_size(chunk_size):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
    return chunk_size


def _check_chunked(chunked):
    if chunked is not None and not isinstance(chunked, bool):
        raise TypeError(
            f"chunked must be True, False, or None (decide on size), got {chunked!r}")
    return chunked


def should_chunk(size_bytes, *, chunked=None, chunk_size=CHUNK_SIZE_DEFAULT):
    """Whether an artifact of `size_bytes` goes up the chunked route.

    The rule `ensure_artifact` applies, as a pure function -- decidable
    without a bucket, a file, or a client, so it can be checked on its
    own the way `object_path` can.

    `chunked=None` decides on size: chunked as soon as the artifact needs
    more than one chunk, which at the default `chunk_size` is anything
    over 64 MiB. `chunked=True` and `chunked=False` force either route
    regardless of size, and are the only values accepted besides None --
    a truthy string or a stray `1` would otherwise select a transfer
    route by accident.

    ## Why the threshold is `chunk_size` and not `CHUNK_SIZE_DEFAULT`

    The two are the same number on a default call, and the two rules are
    observationally identical for any `chunk_size >= CHUNK_SIZE_DEFAULT`:
    `upload_file_chunked` sends anything that fits in one chunk through
    `upload_file` anyway, so a 242MB artifact at `chunk_size=256MiB`
    takes the single request under either rule. They differ only below
    the default -- 8 MiB chunks on a 32MB file -- where this rule sends
    four resumable parts and a fixed 64 MiB threshold would send one
    unresumable request. Deciding against the chunk size actually in use
    is the rule that says what it means: chunk when there is more than
    one chunk to send, and so something for a resume to pick up."""
    if _check_chunked(chunked) is not None:
        return chunked
    _check_chunk_size(chunk_size)
    return int(size_bytes) > chunk_size


def _file_identity(local_path):
    st = os.stat(local_path)
    return {"size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _chunk_count(size_bytes, chunk_size):
    return (size_bytes + chunk_size - 1) // chunk_size


def _chunk_length(index, size_bytes, chunk_size):
    return min(chunk_size, size_bytes - index * chunk_size)


def _remote_size(blob):
    """The remote object's byte count, or None if this blob type does not
    report one. A freshly-built handle carries no metadata until it is
    reloaded, which is why the reload is here rather than assumed."""
    size = getattr(blob, "size", None)
    if size is None:
        reload_ = getattr(blob, "reload", None)
        if callable(reload_):
            reload_()
            size = getattr(blob, "size", None)
    return size


def _part_is_intact(bucket, part, expected_size):
    blob = bucket.blob(part)
    if not blob.exists():
        return False
    size = _remote_size(blob)
    return size is None or int(size) == int(expected_size)


def _confirm_part(bucket, part, expected_size):
    """Ask the remote whether the part actually arrived, and at the right
    length. This is what a chunk is recorded on -- not on the upload call
    having returned. A checkpoint claiming a chunk that never landed is
    worse than no checkpoint: the resume would skip it and compose a hole
    into the object."""
    blob = bucket.blob(part)
    if not blob.exists():
        raise ChunkedUploadError(
            f"the upload of {part!r} returned but the object is not in the bucket; "
            f"refusing to record it as done")
    size = _remote_size(blob)
    if size is not None and int(size) != int(expected_size):
        raise ChunkedUploadError(
            f"the upload of {part!r} landed {int(size)} bytes, not {int(expected_size)}; "
            f"refusing to record it as done")


def _read_chunk(handle, index, chunk_size, size_bytes):
    handle.seek(index * chunk_size)
    return handle.read(_chunk_length(index, size_bytes, chunk_size))


def _checkpoint_state(name, local_path, identity, chunk_size, n_chunks, confirmed):
    return {
        "format": CHECKPOINT_FORMAT,
        "object_path": name,
        "part_prefix": part_prefix(name),
        "local_path": os.path.abspath(str(local_path)),
        "size_bytes": identity["size_bytes"],
        "mtime_ns": identity["mtime_ns"],
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "confirmed": [confirmed[i] for i in sorted(confirmed)],
    }


def _write_checkpoint(path, state):
    """Written whole and swapped into place, so the file on disk is either
    the previous consistent state or the new one -- never a half-written
    record of what has been confirmed."""
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _remove_checkpoint(path):
    for candidate in (path, path + ".tmp"):
        try:
            os.remove(candidate)
        except FileNotFoundError:
            pass


def _load_checkpoint(path, *, name, identity, chunk_size, n_chunks):
    """The confirmed chunks a previous run recorded, or `{}` if there is
    no usable checkpoint.

    Every field that describes what the checkpoint belongs to is compared,
    and any disagreement discards the whole thing rather than salvaging
    part of it. A stale checkpoint composing one artifact's parts into
    another object is the silent-corruption case this exists to prevent,
    and there is no version of it worth being clever about: re-uploading
    costs bandwidth, the alternative costs a wrong result.

    The identity checked is (object path, size, mtime, chunk size, chunk
    count). That catches a regenerated, resized, or truncated file and a
    checkpoint left by a different object or a different chunking. It does
    not catch an in-place edit that preserves both size and mtime --
    `verify_digests=True` is what covers that, at the cost of re-reading
    the local file."""
    try:
        with open(path) as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        return {}
    if (state.get("object_path") != name
            or state.get("chunk_size") != chunk_size
            or state.get("n_chunks") != n_chunks
            or state.get("size_bytes") != identity["size_bytes"]
            or state.get("mtime_ns") != identity["mtime_ns"]):
        return {}

    confirmed = {}
    entries = state.get("confirmed")
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            return {}
        index, size, digest = entry.get("index"), entry.get("size"), entry.get("sha256")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < n_chunks:
            return {}
        if size != _chunk_length(index, identity["size_bytes"], chunk_size):
            return {}
        if not isinstance(digest, str):
            return {}
        confirmed[index] = {"index": index, "size": size, "sha256": digest}
    return confirmed


def _reconcile_confirmed(confirmed, *, bucket, name, handle, chunk_size, size_bytes,
                         verify_digests):
    """Drops any recorded chunk that does not hold up: its part is gone
    from the bucket, its part is the wrong length, or -- under
    `verify_digests` -- the local bytes it was made from have changed.

    Per chunk rather than all-or-nothing, so an edit confined to one chunk
    does not cost the transfer the rest of them."""
    kept = {}
    for index in sorted(confirmed):
        entry = confirmed[index]
        if not _part_is_intact(bucket, part_name(name, index), entry["size"]):
            continue
        if verify_digests:
            digest = hashlib.sha256(
                _read_chunk(handle, index, chunk_size, size_bytes)).hexdigest()
            if digest != entry["sha256"]:
                continue
        kept[index] = entry
    return kept


def _compose(bucket, dest, sources, generation_match=None):
    """Server-side concatenation.

    `generation_match` applies to the DESTINATION only, and only the final
    one -- the intermediate merge targets under the part prefix are
    transient scratch this function owns end to end. The parts themselves
    deliberately carry no precondition either: a resumed upload rewrites a
    part it found damaged, which is an overwrite by design, and guarding it
    would break exactly the resume the chunked route exists for. The race
    Freeze 4 names is two writers replacing each other's ARTIFACT, and the
    artifact is the destination."""
    bucket.blob(dest).compose([bucket.blob(source) for source in sources],
                              **_precondition(generation_match))


def _compose_parts(bucket, name, sources, generation_match=None):
    """Concatenates the parts into the object, server-side.

    GCS composes at most `COMPOSE_MAX_SOURCES` sources per request, so
    anything longer is merged level by level into intermediate objects --
    which live under the part prefix too, and so are cleaned up with
    everything else."""
    prefix = part_prefix(name)
    current = list(sources)
    level = 0
    while len(current) > COMPOSE_MAX_SOURCES:
        merged = []
        for start in range(0, len(current), COMPOSE_MAX_SOURCES):
            target = f"{prefix}/_merge-L{level}-{len(merged):06d}"
            _compose(bucket, target, current[start:start + COMPOSE_MAX_SOURCES])
            merged.append(target)
        current = merged
        level += 1
    _write_guarded(name, generation_match,
                   lambda: _compose(bucket, name, current, generation_match))


def _delete_parts(bucket, name, allow_test_split):
    """Removes everything under this object's part prefix -- the parts
    just composed, the intermediate merges, and any surplus left by an
    earlier, longer attempt at the same object.

    Scoped to the prefix's children (`{name}.parts/`), because GCS prefix
    matching is plain string matching and an object whose name merely
    starts the same way is not this upload's to delete. The guard is
    re-applied to the prefix actually used rather than inherited from the
    caller."""
    prefix = _check_object_path_allowed(part_prefix(name), allow_test_split)
    for blob in list(bucket.list_blobs(prefix=prefix + "/")):
        blob.delete()


def upload_file_chunked(local_path, name, *, bucket, allow_test_split=False,
                        chunk_size=CHUNK_SIZE_DEFAULT, verify_digests=False,
                        verify_content=True, generation_match=None):
    """Uploads a local file in chunks, resuming where a previous run died.
    Returns the object path.

    `upload_file` is one request: a session that dies partway through a
    multi-gigabyte transfer loses all of it and starts again from zero on
    the next run. This uploads the file as numbered part objects under
    `{name}.parts/`, records each one on disk *after* the remote confirms
    it arrived, and composes the parts into the object at the end. A
    re-run picks up after the last confirmed part.

    ## Why parts and composition rather than a resumable session URI

    GCS's own resumable upload is the other candidate primitive, and it
    would put the checkpoint (a session URI plus a committed byte offset)
    somewhere that survives too. It is rejected here for one reason: it
    lives at the HTTP layer, below the `bucket.blob(...)` seam this module
    is written against, so it cannot be exercised without the library
    installed and a live endpoint to talk to. Everything in this module is
    testable against an injected stand-in bucket with
    `google-cloud-storage` absent, and a transfer path whose failure modes
    could only be checked against the real bucket would be the one piece
    of Stage 2B's transport that nothing verifies. Composition needs only
    the same four blob operations already in use, plus `compose`.

    One trade-off comes with that, stated rather than discovered later:
    the parts are real objects in the bucket until the upload finishes.
    A composed object also carries no MD5, which is why the content check
    is on `crc32c` -- GCS records that one for composite and
    single-request objects alike, so both routes leave a downloader the
    same thing to verify against.

    ## What is recorded, and when

    The checkpoint is `{local_path}.upload.json`, written whole and
    swapped into place after each confirmed part. Confirmation is a fresh
    remote check that the part exists and is the right length -- not the
    upload call having returned. Recording a chunk before that is the bug
    the whole mechanism exists to prevent, because the resume would skip
    it and compose a hole into the object.

    On the next run the checkpoint is discarded outright unless the object
    path, the local file's size and mtime, the chunk size and the chunk
    count all still agree, and each surviving entry is then checked
    against the bucket. `verify_digests=True` additionally re-reads the
    local file and compares each recorded chunk's SHA-256, which catches
    an in-place edit that preserved size and mtime; it costs a full read
    of the file.

    Order at the end is compose, verify, then delete the parts, then
    remove the checkpoint, so a death at any point leaves state a re-run
    can reconcile rather than state it must trust. `verify_content` (on
    by default, and distinct from `verify_digests` above -- that one
    re-reads local chunks when deciding what a checkpoint may still
    claim, this one checks the finished object) compares the composed
    object's `crc32c` against the whole local file's. It is the only
    check here that can see parts composed in the wrong order: every one
    of them exists, every one is the right length, and the object is the
    right size. A failure deletes the composed object and leaves the
    parts and the checkpoint in place, since they are the state a retry
    would want.

    A file that fits in one chunk is uploaded directly by `upload_file` --
    there is nothing to resume inside a single request -- and still clears
    any parts an earlier, larger attempt left behind.

    `MAX_PARTS` caps how many parts one object is assembled from. It is
    this module's own conservative limit, not a quoted API figure: the
    per-request source limit of 32 is documented, the ceiling on a
    composite object's total component count is not something this module
    can verify from here. At the default chunk size the cap is reached at
    64 GiB, well beyond anything Stage 2B moves.

    The test-split guard applies exactly as it does to `upload_file`, to
    the object and to its part prefix."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"nothing to upload: {local_path} is not a file")
    _check_chunk_size(chunk_size)

    identity = _file_identity(local_path)
    size_bytes = identity["size_bytes"]
    n_chunks = _chunk_count(size_bytes, chunk_size)
    if n_chunks > MAX_PARTS:
        raise ValueError(
            f"{local_path} would need {n_chunks} parts at chunk_size={chunk_size}, over this "
            f"module's cap of {MAX_PARTS}; use a larger chunk_size")

    checkpoint = checkpoint_path(local_path)
    if n_chunks <= 1:
        upload_file(local_path, name, bucket=bucket, allow_test_split=allow_test_split,
                    verify_content=verify_content)
        _delete_parts(bucket, name, allow_test_split)
        _remove_checkpoint(checkpoint)
        return name

    with open(local_path, "rb") as handle:
        confirmed = _load_checkpoint(checkpoint, name=name, identity=identity,
                                     chunk_size=chunk_size, n_chunks=n_chunks)
        if confirmed:
            confirmed = _reconcile_confirmed(
                confirmed, bucket=bucket, name=name, handle=handle, chunk_size=chunk_size,
                size_bytes=size_bytes, verify_digests=verify_digests)

        for index in range(n_chunks):
            if index in confirmed:
                continue
            data = _read_chunk(handle, index, chunk_size, size_bytes)
            part = part_name(name, index)
            bucket.blob(part).upload_from_string(data, content_type=_OCTET_STREAM)
            _confirm_part(bucket, part, len(data))
            confirmed[index] = {"index": index, "size": len(data),
                                "sha256": hashlib.sha256(data).hexdigest()}
            _write_checkpoint(checkpoint, _checkpoint_state(
                name, local_path, identity, chunk_size, n_chunks, confirmed))

    _compose_parts(bucket, name, [part_name(name, i) for i in range(n_chunks)],
                   generation_match=generation_match)
    if verify_content:
        _verify_uploaded(bucket, name, crc32c_of_file(local_path))
    _delete_parts(bucket, name, allow_test_split)
    _remove_checkpoint(checkpoint)
    return name


def download_file(name, local_path, *, bucket, allow_test_split=False,
                  verify_content=True, generation=None):
    """Downloads an object to a local path, creating the parent directory.

    The download goes to a `.part` sidecar and is renamed into place only
    once it completes, so a session that dies mid-transfer leaves no
    truncated file that the next run would mistake for a finished
    artifact. Returns the local path.

    `verify_content` (on by default) checks the sidecar against the
    object's own `crc32c` BEFORE the rename, so a transfer whose bytes
    are wrong never reaches the destination path -- the `.part`
    discipline covers a transfer that stopped, and this covers one that
    finished with the wrong contents. The sidecar is removed on failure
    too: nothing that failed verification is left lying next to the
    destination to be renamed by hand. An existing good file at
    `local_path` is untouched by a failed download.

    `generation` pins the read to one exact object version rather than
    "whatever this name points at now". A semantic name is a mutable
    pointer; a generation is not. Reading latest-by-name and then checking
    a digest catches a mismatch after the fact; reading the version the
    manifest COMMITTED means there is nothing to mismatch -- and if that
    version is gone, the read fails loudly instead of quietly succeeding
    against a replacement."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    parent = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(parent, exist_ok=True)
    partial = local_path + ".part"
    blob = (bucket.blob(name) if generation is None
            else bucket.blob(name, generation=generation))
    blob.download_to_filename(partial)
    if verify_content:
        try:
            _compare_checksum(blob, name, crc32c_of_file(partial),
                              action="the download of", local_label="downloaded file")
        except (ChecksumMismatchError, ChecksumMissingError):
            try:
                os.remove(partial)
            except OSError:
                pass
            raise
    os.replace(partial, local_path)
    return local_path


def list_objects(prefix, *, bucket, allow_test_split=False):
    """Object names under a prefix, sorted."""
    prefix = _check_object_path_allowed(prefix, allow_test_split)
    return sorted(blob.name for blob in bucket.list_blobs(prefix=prefix))


def delete_prefix(prefix, *, bucket, allow_test_split=False,
                  force_non_test_prefix=False, dry_run=False):
    """Deletes every object under `prefix`. Returns the names deleted (or,
    with `dry_run=True`, the names that would be).

    This is how the test-side artifacts get cleaned up once they are
    finished with, and it is guarded twice because a delete-by-prefix
    helper is the one function here whose misuse is unrecoverable:

    - Any prefix outside `stage2b/` is refused unconditionally, forced or
      not. The bucket holds other stages' cached data and nothing in
      Stage 2B has business deleting it.
    - A prefix under `stage2b/testsplit/` needs `allow_test_split=True`,
      like every other way of touching a test-side object.
    - Any other prefix -- that is, anything training-side -- needs
      `force_non_test_prefix=True`. Wiping the training side is a real
      operation, not a forbidden one, but it is not something a cleanup
      call should be able to do by having the wrong string passed to it.
    - Whatever the prefix *string* looks like, the objects it actually
      MATCHES are checked too: if any of them is test-side,
      `allow_test_split=True` is required. The two rules above reason
      about the string, and a truncated prefix makes the string and the
      matched set disagree -- `"stage2b/t"` is not under the test root and
      so needs only `force_non_test_prefix`, yet it matches the whole test
      side as well as the whole training side. The check on the matched
      names is what actually holds; the string checks are the early,
      legible half of it.

    Matching is plain string prefix, as the GCS API does it -- it is not
    segment-aware, so a prefix is only as precise as it is long. Build one
    with `stage_prefix` or `condition_prefix` rather than by hand.
    """
    prefix = str(prefix)
    if not prefix:
        raise ValueError("refusing to delete with an empty prefix")
    if not (prefix == ROOT_PREFIX or prefix.startswith(ROOT_PREFIX + "/")):
        raise PermissionError(
            f"refusing to delete {prefix!r}: it lies outside {ROOT_PREFIX + '/'!r}, and the "
            f"bucket {bucket.name!r} may hold other stages' cached artifacts. This "
            f"refusal is unconditional -- force_non_test_prefix does not lift it.")
    if is_test_split_path(prefix):
        _check_object_path_allowed(prefix, allow_test_split)
    elif not force_non_test_prefix:
        raise PermissionError(
            f"refusing to delete {prefix!r}: it is not under the test-split root "
            f"{TEST_SPLIT_ROOT!r}, so this would remove training-side artifacts. Pass "
            f"force_non_test_prefix=True to mean it.")

    names = sorted(blob.name for blob in bucket.list_blobs(prefix=prefix))
    reaches_test = [name for name in names if is_test_split_path(name)]
    if reaches_test and not allow_test_split:
        raise PermissionError(
            f"refusing to delete under {prefix!r}: it matches {len(reaches_test)} object(s) "
            f"under the test-split root {TEST_SPLIT_ROOT!r} (e.g. {reaches_test[0]!r}), even "
            f"though the prefix itself is not under it. Touching a test-side object needs "
            f"allow_test_split=True; force_non_test_prefix does not cover it.")

    if dry_run:
        return names
    for name in names:
        bucket.blob(name).delete()
    return names


def delete_test_split_artifacts(*, bucket, stage=None, allow_test_split=False, dry_run=False):
    """Removes the test-side artifacts -- all of them, or one ladder
    stage's -- once they are finished with. A named call for the intended
    cleanup, so no one has to hand-assemble a prefix for it."""
    prefix = (TEST_SPLIT_ROOT if stage is None
              else stage_prefix(stage=stage, split="test", allow_test_split=allow_test_split))
    return delete_prefix(prefix, bucket=bucket, allow_test_split=allow_test_split,
                         dry_run=dry_run)


# ---- The idempotent step ----

class StepResult(NamedTuple):
    """What `ensure_artifact` did, recorded rather than inferred -- a run
    log that says which steps were computed and which were resumed from
    GCS is the difference between a dead session costing one step and
    costing an afternoon (CLAUDE.md principle 7)."""
    object_path: str
    local_path: str
    skipped: bool          # the artifact was already in GCS
    produced: bool         # `produce` ran
    uploaded: bool
    downloaded: bool
    size_bytes: int
    # Provenance, recorded for the same reason the rest is: a run log that
    # says a step was resumed is only useful if it also says whether the
    # thing resumed from was checked. `validated` False on a skip means the
    # object was accepted under the named pre-contract opt-out.
    validated: bool = False            # the skip path checked a manifest
    manifest_published: bool = False   # the produce path wrote one

    def summary(self):
        return {
            "object_path": self.object_path,
            "local_path": self.local_path,
            "skipped": self.skipped,
            "produced": self.produced,
            "uploaded": self.uploaded,
            "downloaded": self.downloaded,
            "size_bytes": self.size_bytes,
            "validated": self.validated,
            "manifest_published": self.manifest_published,
        }


def ensure_artifact(name, local_path, *, produce, bucket, allow_test_split=False,
                    force=False, chunked=None, chunk_size=CHUNK_SIZE_DEFAULT,
                    verify_content=True, fingerprint=None, require_manifest=True,
                    expected_fingerprint=None, policy=None, parents=None):
    """Skip if already done, else compute and upload.

        r = stage2b_gcs.ensure_artifact(path, local, produce=build_it, bucket=bucket)

    `produce(local_path)` is called only when the object is not already in
    the bucket (or when `force=True`), and must write that path. On return
    the artifact exists both locally and in GCS in every branch, so the
    next step can read `r.local_path` without caring which happened.

    The point is that a dead cloud session loses at most the step that was
    in flight. Re-running the same script on a fresh runtime downloads
    what is already there and resumes.

    An existing local file is trusted when the object is already in the
    bucket: it is not re-downloaded and not compared against the remote
    copy. That is right for the case this exists for -- a fresh runtime
    starts with an empty disk, so there is nothing stale to trust -- but
    it does mean reusing one `local_path` across two different objects
    gives the second one the first one's file. Give each artifact its own
    local path. `verify_object` is the call for checking a local file
    that was never transferred by this function.

    `verify_content` (on by default) reaches every transfer this makes:
    the upload on either route, and the download taken when a previous
    session's object is already in the bucket. It does not reach the
    trusted-existing-local-file branch above, which transfers nothing.

    `force=True` recomputes and overwrites an existing object -- for the
    case where the artifact is known stale, not as a routine flag. It does
    NOT skip validation; there is nothing to validate on that path, because
    the bytes are freshly produced here rather than trusted from the
    bucket. What it must not do is leave a manifest describing the object
    it just replaced, and it does not: see "force and the stale sidecar".

    ## The skip branch is a CONSUME, and consumes are validated

    An object merely existing is never sufficient evidence that it is
    resumable. The skip branch goes through `consume_validated`, so an
    object without a manifest, or one whose manifest disagrees with the
    bytes, halts here instead of being read as a completed step.

    `require_manifest=False` is the named opt-out, and it is not a
    convenience: it is how a completed rung stays re-runnable. Ladder
    stages 1 and 2 wrote every one of their artifacts before this contract
    existed, and retrofitting manifests onto them would fabricate
    provenance rather than record it. Those call sites pass it explicitly,
    so each one greps. Everything written from stage 3 onward carries a
    manifest and takes the default.

    `expected_fingerprint` and `policy` are forwarded to
    `consume_validated`: a consumer that knows what produced its input can
    say so, and `CONTENT_ONLY` is how deliberate cross-stage reuse (stage
    3 reading stage 1's topologies) stays legal without weakening payload
    verification.

    ## force and the stale sidecar

    A forced overwrite replaces the payload. Any manifest already beside it
    then describes bytes that no longer exist -- and that is WORSE than no
    manifest, because the next `consume_validated` raises a mismatch, which
    reads as corruption rather than as "this was deliberately regenerated".

    So the produce path never leaves one: with `fingerprint`, it publishes
    a manifest for what it just wrote; without, it removes any manifest it
    finds. Passing `fingerprint` is what makes publication ATOMIC with the
    upload rather than a second call every driver has to remember -- the
    forgot-the-flag failure this function's `chunked` argument already
    exists to prevent.

    `chunked` selects the upload route and defaults to deciding on size
    (`should_chunk`): an artifact needing more than one chunk goes
    through `upload_file_chunked`, so a session that dies mid-upload
    resumes after the last confirmed chunk instead of sending the whole
    transfer again, and anything smaller takes the single request there
    is nothing to resume inside. `chunked=True` and `chunked=False`
    force either route regardless of size.

    The decision is made here rather than left to the caller because
    resumability must not depend on a call site remembering to ask for
    it. That is the failure shape PROJECT_MEMORY.md already records
    once: `mighty-colab exec --timeout`, a flag carried by every
    hand-run command and dropped when those commands were frozen into
    Makefile targets, so the targets could never once complete as
    written. A caller that forgot `chunked=True` on a 242MB artifact
    would get a non-resumable upload and nothing to indicate it. Stage
    1's ~8MB artifacts still go up in one request; stage 3's
    242MB-class ones chunk without being asked."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    if not callable(produce):
        raise TypeError("produce must be a callable taking the local path and writing it")
    # Both route arguments are checked here rather than where they are
    # read, which is after `produce` has run: a step that computes for an
    # hour on a GPU and then raises on a malformed argument loses the
    # hour, and uploads nothing for the next run to resume from. The
    # `produce` check above is here for the same reason.
    _check_chunked(chunked)
    if chunked is None:
        _check_chunk_size(chunk_size)

    # The write-once invariant, enforced rather than described. A lineage
    # artifact's generation is pinned by its manifest and by any manifest
    # that names it as a parent; with versioning off, replacing it destroys
    # a reference that something else is holding. There is no supported
    # overwrite for these, `force` included.
    if force and artifact_class(name) == LINEAGE:
        raise WriteOnceViolation(
            f"refusing to overwrite {name!r}: it is a LINEAGE artifact and lineage "
            f"artifacts are create-once. Its generation is pinned by its own "
            f"manifest, and possibly by another manifest naming it as a parent; "
            f"object versioning is off on this bucket, so replacing it destroys a "
            f"reference something else is holding. To regenerate, write a NEW "
            f"name -- as the Phase A regeneration did, producing "
            f"`encoded_train_s1200` rather than overwriting `encoded_fit_s1200`. "
            f"`force` remains available only for run-scoped artifacts, which "
            f"nothing pins and nothing may adopt as a parent.")

    if not force and object_exists(name, bucket=bucket, allow_test_split=allow_test_split):
        # THE trust point. Everything this function does for a resumed run
        # hangs off this branch, so it is the one place a provenance check
        # has to sit -- and it is a consume like any other.
        manifest, downloaded = consume_validated(
            name, local_path, bucket=bucket, allow_test_split=allow_test_split,
            verify_content=verify_content, require_manifest=require_manifest,
            expected_fingerprint=expected_fingerprint, policy=policy)
        return StepResult(object_path=name, local_path=local_path, skipped=True,
                          produced=False, uploaded=False, downloaded=downloaded,
                          size_bytes=os.path.getsize(local_path),
                          validated=manifest is not None)

    parent = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(parent, exist_ok=True)
    produce(local_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"produce() returned without writing {local_path!r}; nothing to upload to "
            f"{name!r}. A step whose artifact is missing must fail here rather than record "
            f"itself as complete -- otherwise the next run would skip it.")
    # Freeze 4's no-overwrite-race guard, computed here rather than asked
    # of the caller. A fresh step must CREATE (generation 0); a forced
    # overwrite must replace exactly the generation it read. Reading the
    # generation and then requiring it is not a check-then-write race: the
    # requirement travels to the service, which refuses if anything moved
    # in between.
    gen = current_generation(name, bucket=bucket,
                             allow_test_split=allow_test_split) if force else 0
    if should_chunk(os.path.getsize(local_path), chunked=chunked, chunk_size=chunk_size):
        upload_file_chunked(local_path, name, bucket=bucket,
                            allow_test_split=allow_test_split, chunk_size=chunk_size,
                            verify_content=verify_content, generation_match=gen)
    else:
        upload_file(local_path, name, bucket=bucket, allow_test_split=allow_test_split,
                    verify_content=verify_content, generation_match=gen)

    # The generation THIS write produced, captured before the sidecar is
    # written, so the manifest commits that exact version. Reading it back
    # is safe only because every writer here is precondition-guarded: no
    # one else can move the object between our successful write and this
    # read without violating a precondition of their own. And if the
    # capture were wrong anyway, it is detected rather than silent -- the
    # consumer pins to this number and a pinned fetch of a version that is
    # not there fails loudly.
    written_generation = current_generation(name, bucket=bucket,
                                            allow_test_split=allow_test_split)

    # After the payload verifies, never before: a sidecar that exists is a
    # sidecar describing a complete object.
    published = False
    if fingerprint is not None:
        # The sidecar is written under its own precondition: create when
        # there is none, replace exactly the one we found when forcing.
        # Until it lands the payload is UNCOMMITTED and no consumer may
        # accept it -- that is the crash-window semantic, and it is what
        # makes the gap between these two writes safe rather than
        # ambiguous.
        sidecar = manifest_object_name(name)
        publish_manifest(local_path, name, bucket=bucket, fingerprint=fingerprint,
                         parents=parents, allow_test_split=allow_test_split,
                         payload_generation=written_generation,
                         generation_match=current_generation(
                             sidecar, bucket=bucket,
                             allow_test_split=allow_test_split))
        published = True
    else:
        stale = manifest_object_name(name)
        if bucket.blob(stale).exists():
            _discard_object(bucket, stale)
    return StepResult(object_path=name, local_path=local_path, skipped=False,
                      produced=True, uploaded=True, downloaded=False,
                      size_bytes=os.path.getsize(local_path),
                      manifest_published=published)


# =====================================================================
# Artifact manifests: the single validated consume path
# =====================================================================
#
# **An object merely existing is never sufficient evidence that it is
# resumable.**
#
# `ensure_artifact` above treats existence as proof a step is done, which
# is what makes a dead session cost one step rather than an afternoon. That
# same property silently accepts an artifact produced under a different
# configuration once configuration can change. The manifest is the missing
# evidence.
#
# ## Why a sidecar object rather than blob metadata
#
# Both were viable -- verified against the real bucket rather than assumed:
# custom metadata set on a destination handle BEFORE `compose()` survives
# composition, and `patch()` works afterwards, so the chunked route could
# have carried it. A sidecar was chosen anyway, for two reasons. GCS custom
# metadata is size-limited and a source manifest grows with the dependency
# set; and "valid only when payload and manifest both exist and agree" is
# expressed directly by two objects, where a half-published artifact is
# detectably incomplete rather than merely odd.
#
# ## Publication order is the atomicity mechanism
#
# Payload first, then the manifest recording that payload's digest AND the
# GCS generation it had at publication. A reader requires both, so a run
# that died between the two writes leaves an artifact that is refused, not
# one that is silently trusted. If the payload is later overwritten its
# generation changes and the recorded generation no longer matches, which
# is detected on the next consume.

FINGERPRINT_AVAILABLE = True
try:                                            # pragma: no cover - import shape
    import stage2b_fingerprint as _fp
except ImportError:                             # pragma: no cover
    FINGERPRINT_AVAILABLE = False
    _fp = None

MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_FORMAT = "stage2b-artifact-manifest/1"


class ManifestMissingError(OSError):
    """A payload exists with no manifest beside it.

    Distinct from a mismatch: this means the artifact carries no evidence
    of what produced it, which under the fingerprint contract is a refusal
    rather than a reason to fall back on trusting it."""


class ManifestMismatchError(OSError):
    """A manifest and its payload disagree, or the recorded fingerprint is
    not what this run expects."""


def manifest_object_name(payload_name):
    """The sidecar manifest's object name for a payload."""
    return f"{payload_name}{MANIFEST_SUFFIX}"


def is_manifest_name(name):
    return str(name).endswith(MANIFEST_SUFFIX)


def build_manifest(local_path, *, object_name, fingerprint, parents=None,
                   generation=None):
    """The manifest recorded beside a payload.

    `arrays` is per-array dtype/shape/SHA-256 for `.npz` payloads. It is
    recorded in addition to the whole-payload digest because the whole-file
    digest is the WRONG test for that container: `np.savez` writes a zip
    whose headers embed timestamps, so two runs producing bit-identical
    arrays produce different files. The per-array digests are what a
    regeneration is actually compared on."""
    # The other half of the class split. A run-scoped object is
    # overwritable by policy, so adopting one as a parent would put a
    # mutable thing inside an immutable lineage -- and the resulting
    # manifest would record a digest for bytes that are allowed to change
    # underneath it.
    for parent_name in (parents or {}):
        if artifact_class(parent_name) == RUN_SCOPED:
            raise WriteOnceViolation(
                f"refusing to record {parent_name!r} as a parent of "
                f"{object_name!r}: it is RUN_SCOPED, which means it may be "
                f"regenerated in place. A lineage may only be built from "
                f"create-once artifacts, or the parent digest describes bytes "
                f"that are permitted to change.")
    manifest = {
        "format": MANIFEST_FORMAT,
        "object_path": object_name,
        "payload_sha256": _fp.sha256_of_file(local_path),
        "payload_crc32c": crc32c_of_file(local_path),
        "payload_size": os.path.getsize(local_path),
        "payload_generation": generation,
        "fingerprint": fingerprint,
        "parents": dict(parents or {}),
    }
    if str(local_path).endswith(".npz"):
        try:
            manifest["arrays"] = _fp.array_manifest(local_path)
        except Exception as exc:               # noqa: BLE001 - recorded, not fatal
            manifest["arrays_error"] = f"{type(exc).__name__}: {exc}"
    return manifest


def publish_manifest(local_path, object_name, *, bucket, fingerprint,
                     parents=None, allow_test_split=False, generation_match=None,
                     payload_generation=None):
    """Write the manifest sidecar, COMMITTING one exact payload version.

    The manifest is the commit point, not a description. Write order is
    load-bearing: payload first under its own precondition, then the
    generation that write produced is captured, then this sidecar records
    that generation and is itself written under a precondition. Until the
    sidecar lands the payload is UNCOMMITTED, and no consumer may accept
    it -- which is what makes the crash window between the two writes safe
    rather than ambiguous.

    `payload_generation` is that captured generation. When omitted it is
    read back from the bucket, which is correct only for the backfill path
    (describing an object staged before this contract existed); the write
    path always passes it, so the manifest commits the version its own
    upload produced rather than whatever the name points at now."""
    generation = payload_generation
    if generation is None:
        blob = bucket.blob(object_name)
        reload_ = getattr(blob, "reload", None)
        if callable(reload_):
            try:
                reload_()
                generation = getattr(blob, "generation", None)
            except Exception:                  # noqa: BLE001 - generation is optional
                generation = None
    manifest = build_manifest(local_path, object_name=object_name,
                              fingerprint=fingerprint, parents=parents,
                              generation=generation)
    payload = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    name = manifest_object_name(object_name)
    _check_object_path_allowed(name, allow_test_split)
    _write_guarded(name, generation_match,
                   lambda: bucket.blob(name).upload_from_string(
                       payload.encode("utf-8"), content_type="application/json",
                       **_precondition(generation_match)))
    return manifest


def read_manifest(object_name, *, bucket, allow_test_split=False):
    """The recorded manifest for a payload, or None if there is none."""
    name = manifest_object_name(object_name)
    _check_object_path_allowed(name, allow_test_split)
    blob = bucket.blob(name)
    if not blob.exists():
        return None
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "manifest.json")
        blob.download_to_filename(path)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def validate_against_manifest(local_path, object_name, *, manifest,
                              expected_fingerprint=None, policy=None):
    """The one place a manifest is checked against a payload.

    Recomputes the payload digest from the local bytes rather than trusting
    anything recorded, compares the per-array manifest where present, and
    only then compares the producer fingerprint under `policy`. Raises
    `ManifestMismatchError` naming every disagreement; returns the manifest
    on success."""
    problems = []
    if manifest.get("format") != MANIFEST_FORMAT:
        problems.append(f"manifest format {manifest.get('format')!r} "
                        f"!= {MANIFEST_FORMAT!r}")
    recorded = manifest.get("payload_sha256")
    computed = _fp.sha256_of_file(local_path)
    if recorded != computed:
        problems.append(f"payload sha256: manifest {recorded!r} != local {computed!r}")
    if "arrays" in manifest and str(local_path).endswith(".npz"):
        problems += _fp.compare_array_manifests(manifest["arrays"],
                                                _fp.array_manifest(local_path))
    if problems:
        raise ManifestMismatchError(
            f"{object_name!r} does not match its recorded manifest. An object "
            f"merely existing is never sufficient evidence that it is resumable.\n  "
            + "\n  ".join(problems))
    if expected_fingerprint is not None:
        _fp.require_match(manifest.get("fingerprint") or {}, expected_fingerprint,
                          name=object_name, policy=policy or _fp.STRICT)
    return manifest


def verify_lineage(object_name, manifest, *, bucket, allow_test_split=False,
                   max_depth=32):
    """Walk an artifact's parents transitively; every one must be LINEAGE.

    Checking only the immediate parents would leave the property one link
    deep. A lineage is immutable only if every link in it is: a RUN_SCOPED
    object three hops up is still an overwritable thing inside a chain of
    digests that claim to be fixed.

    The walk is breadth-first over recorded manifests, visiting each object
    once, and stops at any artifact with no manifest -- a pre-contract
    input, which has no recorded parents to follow. `max_depth` bounds it
    against a cycle; a cycle is itself a corrupt lineage and is reported as
    one rather than hung on.

    Returns the set of ancestors visited, so a caller can log the lineage
    it actually verified rather than that it verified something."""
    seen, frontier, depth = set(), [(object_name, manifest)], 0
    while frontier:
        depth += 1
        if depth > max_depth:
            raise WriteOnceViolation(
                f"lineage walk from {object_name!r} exceeded depth {max_depth}; "
                f"the parent graph is cyclic or implausibly deep, and either way "
                f"it is not a lineage that can be verified.")
        nxt = []
        for name, current in frontier:
            for parent_name in sorted((current or {}).get("parents") or {}):
                if artifact_class(parent_name) == RUN_SCOPED:
                    raise WriteOnceViolation(
                        f"{parent_name!r} appears in {object_name!r}'s lineage (via "
                        f"{name!r}) but is RUN_SCOPED, which means it may be "
                        f"regenerated in place. A recorded parent digest must "
                        f"describe bytes that cannot change.")
                if parent_name in seen:
                    continue
                seen.add(parent_name)
                parent_manifest = read_manifest(parent_name, bucket=bucket,
                                                allow_test_split=allow_test_split)
                if parent_manifest is not None:
                    nxt.append((parent_name, parent_manifest))
        frontier = nxt
    return seen


def discard_uncommitted(object_name, *, bucket, allow_test_split=False):
    """Delete a payload that was written but never committed, so a halted
    run has a forward path. Returns True if it deleted something.

    ## The state this exists for

    A session that dies between the payload write and the sidecar write
    leaves an orphan: bytes under a LINEAGE name with no manifest. Every
    ordinary route out is closed, by design and correctly --
    `consume_validated` refuses it (no recorded provenance), the create
    path's `if_generation_match=0` fails against an existing object, and
    `force=True` raises `WriteOnceViolation`. Confirmed by constructing the
    state, not reasoned about: without this function the run cannot proceed
    unattended.

    ## Why this does not weaken write-once

    The invariant protects the COMMIT, not the byte-write. A lineage
    artifact is immutable because its generation is pinned -- by its own
    manifest, and possibly by another manifest naming it as a parent. An
    orphan has neither: no manifest committed it, so nothing can hold a
    reference to it. Deleting it removes bytes no consumer was ever
    permitted to read.

    That is exactly why the refusal below is the whole safety property, and
    why it re-reads the manifest at call time rather than trusting a
    listing taken earlier: between that listing and this call, the write
    that commits the payload may have landed.

    ## Why this is not automatic

    `ensure_artifact` does not call this on its own, and the driver halts
    and names it rather than invoking it. An orphan means a session died
    mid-step, which is a signal worth a human seeing. More concretely, the
    resumability story explicitly contemplates a second session resuming
    after the first is only PRESUMED dead: if that first session is alive
    and slow, auto-discarding its payload lets it commit a manifest
    pinning a generation that no longer exists, converting a clean halt
    into a lineage that must be diagnosed. A human deciding costs one
    command; the automatic version costs a confusing failure.

    ## The exact name, never a prefix

    `manifest_object_name` is the payload name plus a suffix, so
    `delete_prefix(payload_name, ...)` matches the manifest too and would
    strip a COMMITTED artifact's provenance while leaving its bytes.
    This deletes the one object it was given."""
    object_name = _check_object_path_allowed(object_name, allow_test_split)
    if is_manifest_name(object_name):
        raise ValueError(
            f"refusing to discard {object_name!r}: it is a manifest, not a payload. "
            f"Deleting a manifest uncommits an artifact whose bytes remain, which is "
            f"the corrupt state this function exists to clear, not to create.")
    manifest = read_manifest(object_name, bucket=bucket,
                             allow_test_split=allow_test_split)
    if manifest is not None:
        raise WriteOnceViolation(
            f"refusing to discard {object_name!r}: it carries a manifest at "
            f"{manifest_object_name(object_name)!r} and is therefore COMMITTED. This "
            f"function clears the orphan state -- a payload with no manifest, which "
            f"no consumer may accept -- and a committed artifact is create-once with "
            f"no supported delete. If these bytes are genuinely wrong, the answer is "
            f"a new name, as Phase A used when it wrote encoded_train_s1200 rather "
            f"than overwriting encoded_fit_s1200.")
    blob = bucket.blob(object_name)
    if not blob.exists():
        return False
    blob.delete()
    return True


def consume_validated(object_name, local_path, *, bucket, expected_fingerprint=None,
                      policy=None, allow_test_split=False, verify_content=True,
                      require_manifest=True):
    """THE central validated consume path. Every read of an existing
    artifact goes through here -- `ensure_artifact`'s skip branch, report
    steps, and the drivers' direct downloads alike.

    A raw `download_file` reaches bytes without ever consulting a manifest,
    which is exactly the bypass the contract exists to close; call sites
    that need an existing artifact call this instead.

    `require_manifest=False` is the named, greppable opt-out for artifacts
    written BEFORE the fingerprint contract existed (ladder stages 1 and 2).
    Their configurations are fixed history, recorded in their own run
    reports, and retrofitting manifests onto them would fabricate
    provenance rather than record it.

    ## The legacy policy, stated so a refusal reads as designed

    There is no grandfather allowlist and there will not be one. Every
    artifact under the stage-1 and stage-2 prefixes is completed-rung
    history; nothing from stage 3 onward consumes any of it except at three
    call sites that say so out loud -- the two `stage_kmnist` staging reads
    and stage 2's cross-rung corruption spot-check. Those pass
    `require_manifest=False` inline, next to a comment naming the rung.

    Everything else consumes under the contract. So if a future code path
    meets a `ManifestMissingError` on an un-manifested object, **the
    refusal is the correct behaviour, not an obstacle**: it means new code
    reached for pre-contract history, and the fix is to decide deliberately
    whether that reuse is legitimate -- adding the opt-out at that call
    site with a reason -- rather than to widen a policy centrally where
    nobody will see it."""
    _check_object_path_allowed(object_name, allow_test_split)
    if not object_exists(object_name, bucket=bucket, allow_test_split=allow_test_split):
        raise FileNotFoundError(f"{object_name!r} does not exist in {bucket.name!r}")

    manifest = read_manifest(object_name, bucket=bucket, allow_test_split=allow_test_split)
    if manifest is None:
        if require_manifest:
            raise ManifestMissingError(
                f"{object_name!r} exists but carries no manifest at "
                f"{manifest_object_name(object_name)!r}. Under the fingerprint "
                f"contract an artifact without recorded provenance is refused: an "
                f"object merely existing is never sufficient evidence that it is "
                f"resumable. Pass require_manifest=False, deliberately, only for "
                f"artifacts written before the contract existed.")
        downloaded = False
        if not os.path.isfile(local_path):
            download_file(object_name, local_path, bucket=bucket,
                          allow_test_split=allow_test_split,
                          verify_content=verify_content)
            downloaded = True
        return None, downloaded

    # Order is the contract, not a style choice.
    #
    #   1. the manifest has already been read, above;
    #   2. the PRODUCER is verified against what this consumer expects --
    #      before any payload bytes move, so a run consuming the wrong
    #      artifact pays nothing to find out;
    #   3. the payload is fetched AT THE COMMITTED GENERATION, not
    #      latest-by-name. A semantic name is a mutable pointer; the
    #      manifest committed one exact version, and that is the one read;
    #   4. the digest is recomputed on the fetched bytes.
    if expected_fingerprint is not None:
        _fp.require_match(manifest.get("fingerprint") or {}, expected_fingerprint,
                          name=object_name, policy=policy or _fp.STRICT)

    # Consume-side class check, and it is not redundant with the one in
    # `publish_manifest`. That one binds what THIS code writes; this one
    # binds what this code is willing to READ, including a manifest written
    # by an older commit, by a different driver, or by hand. The walk is
    # transitive: a lineage is only immutable if every link in it is.
    verify_lineage(object_name, manifest, bucket=bucket,
                   allow_test_split=allow_test_split)

    downloaded = False
    if not os.path.isfile(local_path):
        download_file(object_name, local_path, bucket=bucket,
                      allow_test_split=allow_test_split, verify_content=verify_content,
                      generation=manifest.get("payload_generation"))
        downloaded = True
    validate_against_manifest(local_path, object_name, manifest=manifest,
                              expected_fingerprint=expected_fingerprint, policy=policy)
    return manifest, downloaded
