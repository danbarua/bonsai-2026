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
import os
import re
from typing import NamedTuple

# ---- Infrastructure constants (settled; not configuration to rediscover) ----
GCS_PROJECT = "bonsai-504422"
GCS_BUCKET = "bonsai-2026-stage4a-cache"      # public read (anonymous objectViewer)

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


def get_bucket(*, bucket_name=GCS_BUCKET, client=None, credentials=None,
               project=GCS_PROJECT, anonymous=False):
    """The bucket handle every transport function below takes.

    Build it once per script and pass it down: the transport functions
    require `bucket` as a keyword argument with no default, so a run
    script constructs exactly one client and a test can inject a stand-in
    without the library being installed at all."""
    if client is None:
        client = make_client(credentials=credentials, project=project, anonymous=anonymous)
    return client.bucket(bucket_name)


# ---- Transport ----

def object_exists(name, *, bucket, allow_test_split=False):
    """Whether an object is already in the bucket. This is what makes a
    step skippable, and so what makes a pipeline resumable after a session
    dies."""
    name = _check_object_path_allowed(name, allow_test_split)
    return bool(bucket.blob(name).exists())


def upload_file(local_path, name, *, bucket, allow_test_split=False):
    """Uploads a local file to an object path. Returns the object path."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"nothing to upload: {local_path} is not a file")
    bucket.blob(name).upload_from_filename(local_path)
    return name


def download_file(name, local_path, *, bucket, allow_test_split=False):
    """Downloads an object to a local path, creating the parent directory.

    The download goes to a `.part` sidecar and is renamed into place only
    once it completes, so a session that dies mid-transfer leaves no
    truncated file that the next run would mistake for a finished
    artifact. Returns the local path."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    parent = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(parent, exist_ok=True)
    partial = local_path + ".part"
    bucket.blob(name).download_to_filename(partial)
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
            f"bucket {GCS_BUCKET!r} is shared with other stages' cached artifacts. This "
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

    def summary(self):
        return {
            "object_path": self.object_path,
            "local_path": self.local_path,
            "skipped": self.skipped,
            "produced": self.produced,
            "uploaded": self.uploaded,
            "downloaded": self.downloaded,
            "size_bytes": self.size_bytes,
        }


def ensure_artifact(name, local_path, *, produce, bucket, allow_test_split=False,
                    force=False):
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
    local path.

    `force=True` recomputes and overwrites an existing object -- for the
    case where the artifact is known stale, not as a routine flag."""
    name = _check_object_path_allowed(name, allow_test_split)
    local_path = str(local_path)
    if not callable(produce):
        raise TypeError("produce must be a callable taking the local path and writing it")

    if not force and object_exists(name, bucket=bucket, allow_test_split=allow_test_split):
        downloaded = False
        if not os.path.isfile(local_path):
            download_file(name, local_path, bucket=bucket, allow_test_split=allow_test_split)
            downloaded = True
        return StepResult(object_path=name, local_path=local_path, skipped=True,
                          produced=False, uploaded=False, downloaded=downloaded,
                          size_bytes=os.path.getsize(local_path))

    parent = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(parent, exist_ok=True)
    produce(local_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"produce() returned without writing {local_path!r}; nothing to upload to "
            f"{name!r}. A step whose artifact is missing must fail here rather than record "
            f"itself as complete -- otherwise the next run would skip it.")
    upload_file(local_path, name, bucket=bucket, allow_test_split=allow_test_split)
    return StepResult(object_path=name, local_path=local_path, skipped=False,
                      produced=True, uploaded=True, downloaded=False,
                      size_bytes=os.path.getsize(local_path))
