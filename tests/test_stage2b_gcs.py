"""Tests for experiments/stage2b_denoising/stage2b_gcs.py -- Stage 2B's
GCS object-path scheme, credential resolution, test-split guard, and the
idempotent-step primitive.

Tier 1 only, and strictly so: no network, no credentials, no
`google-cloud-storage`. Everything that decides a path, a prefix, or a
credential location is pure and tested directly; everything that would
otherwise need a live client is exercised against an in-memory fake
bucket injected through the `bucket` keyword argument the transport
functions already require.

What is deliberately NOT tested here: that a real upload, download,
existence check, or delete works against the real bucket. That cannot be
verified without network and credentials, and a test that passed in their
absence would be claiming something it had not checked.
`experiments/stage2b_denoising/smoke_stage2b_gcs.py` is the manually-run
script for that, and it is not collected by pytest.

The subprocess test is the load-bearing one: it blocks the `google`
package outright and confirms the module still imports and its pure logic
still runs, so the lazy import cannot regress into a module-scope one
even on a machine where the library happens to be installed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE2B_DIR = _REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(_STAGE2B_DIR))

import stage2b_gcs as gcs  # noqa: E402


# ---- an in-memory stand-in for a GCS bucket ----

class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        # Unpopulated until the object is fetched or written, as on a real
        # `Blob`: `bucket.blob(name)` is a handle, not a read.
        self.size = None
        self.crc32c = None

    def exists(self):
        return self.name in self.bucket.objects

    def _populate(self):
        """Size and checksum as the real client fills them in -- from the
        object the service actually holds, never from what was sent."""
        data = self.bucket.objects.get(self.name, b"")
        self.size = len(data)
        self.crc32c = self.bucket._checksum(data)

    def reload(self):
        if self.name not in self.bucket.objects:
            raise FileNotFoundError(f"no such object: {self.name}")
        self._populate()

    def upload_from_filename(self, path):
        self.bucket._store_upload(self.name, Path(path).read_bytes())
        self._populate()

    def upload_from_string(self, data, content_type=None):
        self.bucket._store_upload(self.name, bytes(data))
        self._populate()

    def compose(self, sources):
        """Server-side concatenation, with the API's own source limit
        enforced -- a composition tree that exceeded it would be a real
        failure against the real bucket, so it must be one here."""
        if not 1 <= len(sources) <= 32:
            raise ValueError(f"compose takes 1-32 sources, got {len(sources)}")
        missing = [s.name for s in sources if s.name not in self.bucket.objects]
        if missing:
            raise FileNotFoundError(f"no such object(s): {missing}")
        ordered = self.bucket._compose_order(sources)
        self.bucket.objects[self.name] = b"".join(
            self.bucket.objects[s.name] for s in ordered)
        self.bucket.composes.append((self.name, [s.name for s in sources]))
        self._populate()

    def download_to_filename(self, path):
        if self.name not in self.bucket.objects:
            raise FileNotFoundError(f"no such object: {self.name}")
        Path(path).write_bytes(self.bucket._deliver(self.bucket.objects[self.name]))
        self.bucket.downloads.append(self.name)

    def delete(self):
        del self.bucket.objects[self.name]
        self.bucket.deletes.append(self.name)


class FakeBucket:
    """Only the blob operations `stage2b_gcs` uses, plus the prefix
    listing. `list_blobs` matches on a plain string prefix, as the real
    API does."""

    def __init__(self, name=gcs.DEFAULT_GCS_BUCKET):
        self.name = name
        self.objects = {}
        self.uploads = []
        self.downloads = []
        self.deletes = []
        self.composes = []

    def _store_upload(self, name, data):
        """The one place an upload lands, so a subclass can make a
        transfer die or silently vanish without touching `FakeBlob`."""
        self.objects[name] = bytes(data)
        self.uploads.append(name)

    def _checksum(self, data):
        """The service's own checksum of the bytes it holds -- computed
        here with the module's implementation, because what these tests
        exercise is whether stored bytes and local bytes agree, not
        whether the algorithm is CRC32C. That second question is pinned
        separately, by the standard check value and by the cross-check
        against `google_crc32c` where it is installed."""
        return gcs.crc32c_of_bytes(data)

    def _compose_order(self, sources):
        """The order the service concatenates in. A hook, so a subclass
        can compose the right parts wrongly."""
        return list(sources)

    def _deliver(self, data):
        """The bytes a download actually hands back, which are not
        necessarily the bytes the object holds."""
        return data

    def blob(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


class DyingBucket(FakeBucket):
    """A session that dies partway through a transfer -- PROJECT_MEMORY.md
    Part 4's failure mode, which is what the checkpoint exists for. The
    upload raises; nothing after it runs."""

    def __init__(self, die_after=None, **kwargs):
        super().__init__(**kwargs)
        self.die_after = die_after

    def _store_upload(self, name, data):
        if self.die_after is not None and len(self.uploads) >= self.die_after:
            raise ConnectionError("the session died mid-transfer")
        super()._store_upload(name, data)


class SilentlyDroppingBucket(FakeBucket):
    """An upload call that returns successfully and lands nothing.

    This is the case that separates 'recorded once the remote confirmed
    it' from 'recorded once the call returned': a checkpoint written on
    the second rule would claim a chunk that does not exist, and the
    resume would compose a hole into the object."""

    def __init__(self, drop_from=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_from = drop_from

    def _store_upload(self, name, data):
        if self.drop_from is not None and len(self.uploads) >= self.drop_from:
            self.uploads.append(name)          # the call "succeeded"
            return
        super()._store_upload(name, data)


class TruncatingBucket(FakeBucket):
    """An upload that lands a short object -- the call returns, the object
    exists, and its bytes are not the ones that were sent."""

    def __init__(self, truncate_from=None, **kwargs):
        super().__init__(**kwargs)
        self.truncate_from = truncate_from

    def _store_upload(self, name, data):
        if self.truncate_from is not None and len(self.uploads) >= self.truncate_from:
            data = bytes(data)[: max(0, len(data) // 2)]
        super()._store_upload(name, data)


class CorruptingDownloadBucket(FakeBucket):
    """A download that hands back bytes the object does not hold.

    The stored object and its checksum are both fine -- what goes wrong
    is between the service and the disk, which is precisely the failure
    a size check and a `.part` rename cannot see: the file is complete,
    it is the right length, and its contents are wrong."""

    def _deliver(self, data):
        mangled = bytearray(data)
        if mangled:
            mangled[len(mangled) // 2] ^= 0xFF
        return bytes(mangled)


class ChecksumlessBucket(FakeBucket):
    """An object with no checksum recorded against it -- what a consumer
    would face if the field were not populated for some object it meets."""

    def _checksum(self, data):
        return None


class MiscomposingBucket(FakeBucket):
    """Every part arrives intact and at the right length, and the
    composition assembles them in the wrong order.

    Nothing about existence or size can see this. The parts are all
    there, the composed object is exactly the right length, and its bytes
    are not the artifact's."""

    def _compose_order(self, sources):
        return list(reversed(sources))


@pytest.fixture
def bucket():
    return FakeBucket()


def _write(path, text="payload"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


TRAIN_ARGS = dict(stage=2, condition="evolved_T", kind="features", ext="npz", split="train")
TEST_ARGS = dict(stage=4, condition="evolved_T", kind="predictions", ext="npz", split="test")


# ---- infrastructure constants ----

def test_infrastructure_constants():
    assert gcs.GCS_PROJECT == "bonsai-504422"
    assert gcs.DEFAULT_GCS_BUCKET == "bonsai-2026-stage2b-cache"
    assert gcs.BUCKET_ENV_VAR == "BONSAI_GCS_BUCKET"
    assert not hasattr(gcs, "GCS_BUCKET"), (
        "a module-level GCS_BUCKET is read at import time, so anything referencing it "
        "silently ignores BONSAI_GCS_BUCKET and operates on the wrong bucket. Call "
        "bucket_name() instead.")
    assert gcs.CREDENTIALS_ENV_VAR == "BONSAI_GCS_CREDENTIALS"
    assert gcs.DEFAULT_CREDENTIALS_PATH == "~/.config/colab-cli/bonsai-colab-storage-key.json"
    assert gcs.ROOT_PREFIX == "stage2b"
    assert gcs.TRAIN_ROOT == "stage2b/train"
    assert gcs.TEST_SPLIT_ROOT == "stage2b/testsplit"
    assert gcs.LADDER_STAGES == (1, 2, 3, 4)
    assert gcs.TEST_SPLIT_STAGE == 4


def test_the_two_split_roots_do_not_nest():
    """Neither root is a prefix of the other, so a prefix listing or a
    bulk delete on one can never reach the other."""
    assert not gcs.TRAIN_ROOT.startswith(gcs.TEST_SPLIT_ROOT)
    assert not gcs.TEST_SPLIT_ROOT.startswith(gcs.TRAIN_ROOT)
    assert gcs.TRAIN_ROOT.startswith(gcs.ROOT_PREFIX + "/")
    assert gcs.TEST_SPLIT_ROOT.startswith(gcs.ROOT_PREFIX + "/")


# ---- the lazy import ----

def test_module_import_does_not_pull_in_google_cloud_storage():
    """Importing this module must not import the library.

    Checked in a fresh interpreter, not against this process's
    `sys.modules`: locally the latter is trivially google-free because the
    library is absent, so it would assert nothing about `stage2b_gcs`, and
    on a runtime that has the `gpu` group installed any unrelated import
    of `google.auth` by another test or plugin would fail it while this
    module was perfectly correct. `google` is deliberately NOT blocked
    here -- the point is that the module does not reach for it when it
    could."""
    code = (f"import sys; sys.path.insert(0, {str(_STAGE2B_DIR)!r});"
            "import stage2b_gcs;"
            "print(any(n == 'google' or n.startswith('google.') for n in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_pure_logic_runs_with_the_google_package_blocked():
    """The property stated above, enforced rather than observed: a fresh
    interpreter with `google` made unimportable still imports the module
    and still resolves paths, prefixes, and credentials. This keeps
    passing on a machine where google-cloud-storage IS installed, which is
    the case the previous test cannot cover."""
    code = f"""
import sys
class _Block:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name == "google" or name.startswith("google."):
            raise ImportError("google is blocked for this test")
        return None
sys.meta_path.insert(0, _Block())
sys.path.insert(0, {str(_STAGE2B_DIR)!r})
import stage2b_gcs as gcs
print(gcs.object_path(stage=2, condition="evolved_T", kind="features", ext="npz",
                      split="train"))
print(gcs.credentials_path(env={{}}))
print(gcs.stage_prefix(stage=1, split="train"))
try:
    gcs.object_path(stage=4, condition="evolved_T", kind="predictions", ext="npz",
                    split="test")
except PermissionError:
    print("guard-ok")
try:
    gcs._storage_module()
except ImportError:
    print("lazy-ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out[0] == "stage2b/train/stage2/evolved_T/features.npz"
    assert out[1].endswith("bonsai-colab-storage-key.json")
    assert out[2] == "stage2b/train/stage1"
    assert out[3] == "guard-ok"
    assert out[4] == "lazy-ok"


# ---- the bucket: resolved per call, never bound at import ----

def test_bucket_name_defaults_to_the_module_default():
    assert gcs.bucket_name(env={}) == gcs.DEFAULT_GCS_BUCKET


def test_bucket_name_is_overridden_by_the_environment_variable():
    assert gcs.bucket_name(env={gcs.BUCKET_ENV_VAR: "some-scratch-bucket"}) == \
        "some-scratch-bucket"


def test_an_empty_bucket_override_falls_back_to_the_default():
    assert gcs.bucket_name(env={gcs.BUCKET_ENV_VAR: ""}) == gcs.DEFAULT_GCS_BUCKET
    assert gcs.bucket_name(env={gcs.BUCKET_ENV_VAR: "   "}) == gcs.DEFAULT_GCS_BUCKET


@pytest.mark.parametrize("bad", [
    "has/a/slash",        # would silently redirect every object path built against it
    "UPPERCASE",          # GCS rejects it; fail here rather than at the API
    "-leading-dash",
    "trailing-dash-",
    "ab",                 # under the 3-character minimum
    "x" * 64,             # over the 63-character maximum
    "sp ace",
])
def test_an_invalid_bucket_override_is_rejected_loudly(bad):
    """A bad name must fail at resolution, not become a confusing 404 or --
    worse, for a name containing a separator -- a silently different path."""
    with pytest.raises(ValueError, match=gcs.BUCKET_ENV_VAR):
        gcs.bucket_name(env={gcs.BUCKET_ENV_VAR: bad})


def test_get_bucket_resolves_the_name_per_call_not_at_import(monkeypatch):
    """The regression a default argument would reintroduce: `get_bucket`
    must read the environment when it is called, so a variable set after
    this module was imported is still honoured."""
    seen = []

    class _Client:
        def bucket(self, name):
            seen.append(name)
            return name

    monkeypatch.setenv(gcs.BUCKET_ENV_VAR, "set-after-import")
    gcs.get_bucket(client=_Client())
    monkeypatch.delenv(gcs.BUCKET_ENV_VAR)
    gcs.get_bucket(client=_Client())

    assert seen == ["set-after-import", gcs.DEFAULT_GCS_BUCKET]


# ---- credentials: resolved, never read ----

def test_credentials_path_defaults_to_the_uploaded_key_location():
    assert gcs.credentials_path(env={}) == os.path.expanduser(gcs.DEFAULT_CREDENTIALS_PATH)
    assert "~" not in gcs.credentials_path(env={})


def test_credentials_path_is_overridden_by_the_environment_variable(tmp_path):
    override = tmp_path / "elsewhere" / "key.json"
    assert gcs.credentials_path(env={gcs.CREDENTIALS_ENV_VAR: str(override)}) == str(override)


def test_credentials_path_expands_a_tilde_in_the_override():
    resolved = gcs.credentials_path(env={gcs.CREDENTIALS_ENV_VAR: "~/somewhere/key.json"})
    assert "~" not in resolved
    assert resolved.endswith(os.path.join("somewhere", "key.json"))


def test_an_empty_override_falls_back_to_the_default():
    assert (gcs.credentials_path(env={gcs.CREDENTIALS_ENV_VAR: ""})
            == os.path.expanduser(gcs.DEFAULT_CREDENTIALS_PATH))


def test_credentials_path_does_not_require_the_file_to_exist_or_read_it(tmp_path):
    """Resolution returns a path. It does not open the key -- so a missing
    one resolves without error, and a present one is never read."""
    missing = str(tmp_path / "not-there.json")
    env = {gcs.CREDENTIALS_ENV_VAR: missing}
    assert gcs.credentials_path(env=env) == missing
    assert gcs.credentials_available(env=env) is False

    present = _write(tmp_path / "key.json", '{"secret": "value"}')
    assert gcs.credentials_available(env={gcs.CREDENTIALS_ENV_VAR: present}) is True


# ---- the object-path scheme ----

def test_object_path_is_the_documented_four_segment_scheme():
    assert gcs.object_path(**TRAIN_ARGS) == "stage2b/train/stage2/evolved_T/features.npz"
    assert gcs.object_path(stage=1, condition="pre_evolution", kind="ridge_alpha",
                           ext="json", split="train") == \
        "stage2b/train/stage1/pre_evolution/ridge_alpha.json"


def test_object_path_is_pure(tmp_path, monkeypatch):
    """No client, no network, no filesystem: the scheme is decidable on
    its own, which is what makes it testable here at all."""
    monkeypatch.chdir(tmp_path)
    before = sorted(os.listdir(tmp_path))
    gcs.object_path(**TRAIN_ARGS)
    gcs.stage_prefix(stage=3, split="train")
    assert sorted(os.listdir(tmp_path)) == before


def test_a_condition_of_none_uses_the_reserved_common_segment():
    """Artifacts that are not condition-specific -- corruption
    diagnostics, the encoder gate -- keep the same path depth."""
    assert gcs.object_path(stage=1, condition=None, kind="encoder_gate", ext="json",
                           split="train") == "stage2b/train/stage1/common/encoder_gate.json"


def test_common_cannot_be_passed_as_a_condition_name():
    with pytest.raises(ValueError, match="reserved segment"):
        gcs.object_path(stage=1, condition="common", kind="x", ext="json", split="train")


def test_stage_and_condition_prefixes_are_prefixes_of_the_object_path():
    path = gcs.object_path(**TRAIN_ARGS)
    stage = gcs.stage_prefix(stage=2, split="train")
    condition = gcs.condition_prefix(stage=2, condition="evolved_T", split="train")
    assert stage == "stage2b/train/stage2"
    assert condition == "stage2b/train/stage2/evolved_T"
    assert path.startswith(condition + "/")
    assert condition.startswith(stage + "/")


@pytest.mark.parametrize("stage", [0, 5, -1, "1", 1.0, True, None])
def test_object_path_rejects_a_stage_outside_the_ladder(stage):
    """`1.0` and `True` are both `== 1`, so a plain membership test would
    accept them and render "stage1" from something nobody wrote."""
    with pytest.raises(ValueError, match="feasibility"):
        gcs.object_path(stage=stage, condition="evolved_T", kind="features", ext="npz",
                        split="train")


@pytest.mark.parametrize("token", ["", "a/b", "../escape", ".hidden", "has space",
                                   "trailing/", 3])
def test_object_path_rejects_tokens_that_could_reshape_the_path(token):
    with pytest.raises(ValueError):
        gcs.object_path(stage=1, condition=token, kind="features", ext="npz", split="train")
    with pytest.raises(ValueError):
        gcs.object_path(stage=1, condition="evolved_T", kind=token, ext="npz", split="train")


def test_a_kind_of_none_is_rejected_even_though_a_condition_of_none_is_not():
    """`condition=None` is the documented way to reach the `common`
    segment; `kind=None` is just a missing filename."""
    with pytest.raises(ValueError, match="kind must match"):
        gcs.object_path(stage=1, condition=None, kind=None, ext="npz", split="train")


@pytest.mark.parametrize("ext", ["", ".npz", "np z", "a/b", None])
def test_object_path_rejects_a_malformed_extension(ext):
    with pytest.raises(ValueError, match="ext must match"):
        gcs.object_path(stage=1, condition="evolved_T", kind="features", ext=ext,
                        split="train")


def test_condition_names_may_carry_the_designs_capitalisation():
    """DESIGN.md's condition is literally `evolved_T`; the scheme must not
    quietly lowercase or reject it."""
    assert "evolved_T" in gcs.object_path(**TRAIN_ARGS)


def test_object_path_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split must be one of"):
        gcs.object_path(stage=1, condition="evolved_T", kind="features", ext="npz",
                        split="validation")


def test_every_argument_is_keyword_only_with_no_default():
    with pytest.raises(TypeError):
        gcs.object_path(2, "evolved_T", "features", "npz", "train")
    for missing in ("stage", "condition", "kind", "ext", "split"):
        kwargs = {k: v for k, v in TRAIN_ARGS.items() if k != missing}
        with pytest.raises(TypeError):
            gcs.object_path(**kwargs)


# ---- the test-split guard ----

def test_a_test_split_path_requires_the_explicit_opt_in():
    with pytest.raises(PermissionError, match="stages 1-3"):
        gcs.object_path(**TEST_ARGS)
    assert gcs.object_path(**TEST_ARGS, allow_test_split=True) == \
        "stage2b/testsplit/stage4/evolved_T/predictions.npz"


def test_the_opt_in_reaches_stage_4_only():
    """`allow_test_split=True` opts into the single confirmatory
    evaluation, not into test-side work at the feasibility stages."""
    for stage in (1, 2, 3):
        with pytest.raises(ValueError, match="stage 4"):
            gcs.object_path(stage=stage, condition="evolved_T", kind="features", ext="npz",
                            split="test", allow_test_split=True)


def test_the_guard_applies_to_prefixes_too():
    for fn, kwargs in ((gcs.stage_prefix, {}),
                       (gcs.condition_prefix, {"condition": "evolved_T"})):
        with pytest.raises(PermissionError):
            fn(stage=4, split="test", **kwargs)
        assert gcs.TEST_SPLIT_ROOT in fn(stage=4, split="test", allow_test_split=True,
                                         **kwargs)


def test_training_side_paths_never_land_under_the_test_root():
    for stage in gcs.LADDER_STAGES:
        path = gcs.object_path(stage=stage, condition=None, kind="manifest", ext="json",
                               split="train")
        assert not gcs.is_test_split_path(path)
        assert path.startswith(gcs.TRAIN_ROOT + "/")


@pytest.mark.parametrize("name,expected", [
    ("stage2b/testsplit", True),
    ("stage2b/testsplit/", True),
    ("stage2b/testsplit/stage4/evolved_T/predictions.npz", True),
    ("stage2b/testsplitting/x", False),
    ("stage2b/train/stage2/evolved_T/features.npz", False),
    ("stage2b", False),
    ("", False),
])
def test_is_test_split_path_matches_the_root_and_nothing_adjacent(name, expected):
    assert gcs.is_test_split_path(name) is expected


def test_transport_functions_recheck_a_hand_assembled_path(bucket, tmp_path):
    """`object_path` is not the only way to produce a string, so the
    guard sits on the transport functions as well as the builder."""
    raw = "stage2b/testsplit/stage4/evolved_T/predictions.npz"
    local = _write(tmp_path / "a.bin")
    with pytest.raises(PermissionError):
        gcs.object_exists(raw, bucket=bucket)
    with pytest.raises(PermissionError):
        gcs.upload_file(local, raw, bucket=bucket)
    with pytest.raises(PermissionError):
        gcs.download_file(raw, str(tmp_path / "b.bin"), bucket=bucket)
    with pytest.raises(PermissionError):
        gcs.list_objects(raw, bucket=bucket)
    with pytest.raises(PermissionError):
        gcs.ensure_artifact(raw, local, produce=lambda p: None, bucket=bucket)
    assert bucket.objects == {}
    assert bucket.uploads == []


# ---- transport, against the injected fake bucket ----

def test_upload_then_exists_then_download_round_trips(bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)
    assert gcs.object_exists(name, bucket=bucket) is False

    local = _write(tmp_path / "out" / "features.npz", "the payload")
    assert gcs.upload_file(local, name, bucket=bucket) == name
    assert gcs.object_exists(name, bucket=bucket) is True

    back = tmp_path / "restored" / "features.npz"
    assert gcs.download_file(name, back, bucket=bucket) == str(back)
    assert back.read_text() == "the payload"


def test_download_creates_the_parent_directory_and_leaves_no_partial_file(bucket, tmp_path):
    """The download lands on a `.part` sidecar and is renamed into place,
    so a session that dies mid-transfer cannot leave a truncated file the
    next run would treat as a finished artifact."""
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(_write(tmp_path / "src.bin", "x" * 64), name, bucket=bucket)
    target = tmp_path / "deep" / "nested" / "features.npz"
    gcs.download_file(name, target, bucket=bucket)
    assert target.is_file()
    assert not (target.parent / (target.name + ".part")).exists()


def test_upload_refuses_a_local_path_that_is_not_a_file(bucket, tmp_path):
    with pytest.raises(FileNotFoundError):
        gcs.upload_file(tmp_path / "missing.bin", gcs.object_path(**TRAIN_ARGS),
                        bucket=bucket)
    assert bucket.objects == {}


def test_list_objects_is_prefix_scoped_and_sorted(bucket, tmp_path):
    local = _write(tmp_path / "x.bin")
    for kind in ("features", "alphas", "manifest"):
        gcs.upload_file(local, gcs.object_path(stage=2, condition="evolved_T", kind=kind,
                                               ext="npz", split="train"), bucket=bucket)
    gcs.upload_file(local, gcs.object_path(stage=2, condition="lattice", kind="features",
                                           ext="npz", split="train"), bucket=bucket)
    gcs.upload_file(local, gcs.object_path(stage=1, condition=None, kind="gate", ext="json",
                                           split="train"), bucket=bucket)

    at_stage2 = gcs.list_objects(gcs.stage_prefix(stage=2, split="train"), bucket=bucket)
    assert len(at_stage2) == 4
    assert at_stage2 == sorted(at_stage2)

    only_T = gcs.list_objects(
        gcs.condition_prefix(stage=2, condition="evolved_T", split="train"), bucket=bucket)
    assert len(only_T) == 3
    assert all("/evolved_T/" in n for n in only_T)


# ---- delete-by-prefix: the refusals ----

def _populate(bucket, tmp_path):
    local = _write(tmp_path / "x.bin")
    train = [gcs.object_path(stage=s, condition="evolved_T", kind="features", ext="npz",
                             split="train") for s in (1, 2, 3)]
    test = [gcs.object_path(stage=4, condition=c, kind="predictions", ext="npz",
                            split="test", allow_test_split=True)
            for c in ("evolved_T", "pre_evolution")]
    for name in train + test:
        bucket.objects[name] = Path(local).read_bytes()
    return train, test


@pytest.mark.parametrize("prefix", ["", "stage2a", "stage2a/scratch", "/", "other",
                                    "stage2b_other"])
def test_delete_refuses_anything_outside_the_stage2b_root_even_when_forced(bucket, prefix):
    """Unconditional: the bucket is shared with other stages' cached
    artifacts, and force does not lift this."""
    with pytest.raises((PermissionError, ValueError)):
        gcs.delete_prefix(prefix, bucket=bucket, force_non_test_prefix=True,
                          allow_test_split=True)
    assert bucket.deletes == []


def test_delete_refuses_a_non_test_prefix_unless_separately_forced(bucket, tmp_path):
    train, test = _populate(bucket, tmp_path)
    with pytest.raises(PermissionError, match="force_non_test_prefix"):
        gcs.delete_prefix(gcs.TRAIN_ROOT, bucket=bucket)
    with pytest.raises(PermissionError, match="force_non_test_prefix"):
        gcs.delete_prefix(gcs.ROOT_PREFIX, bucket=bucket, allow_test_split=True)
    assert set(bucket.objects) == set(train + test)

    deleted = gcs.delete_prefix(gcs.TRAIN_ROOT, bucket=bucket, force_non_test_prefix=True)
    assert deleted == sorted(train)
    assert set(bucket.objects) == set(test)


def test_a_forced_delete_cannot_reach_test_objects_via_a_truncated_prefix(bucket, tmp_path):
    """The guard has to hold on the objects MATCHED, not on the prefix
    string. `stage2b/t` is not under the test root, so the string checks
    let it through on `force_non_test_prefix` alone -- and it matches the
    whole test side as well as the whole training side."""
    train, test = _populate(bucket, tmp_path)
    for prefix in ("stage2b/t", gcs.ROOT_PREFIX, "stage2b/"):
        with pytest.raises(PermissionError, match="test-split root"):
            gcs.delete_prefix(prefix, bucket=bucket, force_non_test_prefix=True)
        with pytest.raises(PermissionError, match="test-split root"):
            gcs.delete_prefix(prefix, bucket=bucket, force_non_test_prefix=True,
                              dry_run=True)
    assert set(bucket.objects) == set(train + test)
    assert bucket.deletes == []


def test_a_prefix_spanning_both_sides_works_once_both_flags_are_given(bucket, tmp_path):
    """The refusal is a missing opt-in, not a prohibition: saying both
    things explicitly deletes both sides."""
    train, test = _populate(bucket, tmp_path)
    deleted = gcs.delete_prefix(gcs.ROOT_PREFIX, bucket=bucket, allow_test_split=True,
                                force_non_test_prefix=True)
    assert deleted == sorted(train + test)
    assert bucket.objects == {}


def test_deleting_the_test_prefix_needs_the_test_split_opt_in(bucket, tmp_path):
    train, test = _populate(bucket, tmp_path)
    with pytest.raises(PermissionError, match="stages 1-3"):
        gcs.delete_prefix(gcs.TEST_SPLIT_ROOT, bucket=bucket)
    with pytest.raises(PermissionError, match="stages 1-3"):
        gcs.delete_prefix(gcs.TEST_SPLIT_ROOT, bucket=bucket, force_non_test_prefix=True)
    assert set(bucket.objects) == set(train + test)


def test_test_split_artifacts_are_bulk_deletable_and_leave_training_alone(bucket, tmp_path):
    train, test = _populate(bucket, tmp_path)
    deleted = gcs.delete_test_split_artifacts(bucket=bucket, allow_test_split=True)
    assert deleted == sorted(test)
    assert set(bucket.objects) == set(train)


def test_test_split_deletion_can_be_scoped_to_one_ladder_stage(bucket, tmp_path):
    train, test = _populate(bucket, tmp_path)
    deleted = gcs.delete_test_split_artifacts(bucket=bucket, stage=4, allow_test_split=True)
    assert deleted == sorted(test)
    assert set(bucket.objects) == set(train)


def test_delete_test_split_artifacts_still_needs_the_opt_in(bucket, tmp_path):
    train, test = _populate(bucket, tmp_path)
    with pytest.raises(PermissionError):
        gcs.delete_test_split_artifacts(bucket=bucket)
    assert set(bucket.objects) == set(train + test)


def test_dry_run_reports_without_deleting(bucket, tmp_path):
    train, test = _populate(bucket, tmp_path)
    listed = gcs.delete_prefix(gcs.TEST_SPLIT_ROOT, bucket=bucket, allow_test_split=True,
                               dry_run=True)
    assert listed == sorted(test)
    assert bucket.deletes == []
    assert set(bucket.objects) == set(train + test)


# ---- the idempotent step ----

def _producer(text="computed", counter=None):
    def produce(local_path):
        if counter is not None:
            counter.append(local_path)
        Path(local_path).write_text(text)
    return produce


def test_ensure_artifact_computes_and_uploads_when_the_object_is_absent(bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)
    local = tmp_path / "work" / "features.npz"
    calls = []
    r = gcs.ensure_artifact(name, local, produce=_producer(counter=calls), bucket=bucket)
    assert len(calls) == 1
    assert (r.skipped, r.produced, r.uploaded, r.downloaded) == (False, True, True, False)
    assert r.object_path == name and r.local_path == str(local)
    assert r.size_bytes == len("computed")
    assert name in bucket.objects


def test_ensure_artifact_skips_a_step_that_already_landed(bucket, tmp_path):
    """The whole point: a dead session loses at most the step in flight."""
    name = gcs.object_path(**TRAIN_ARGS)
    local = tmp_path / "work" / "features.npz"
    calls = []
    gcs.ensure_artifact(name, local, produce=_producer(counter=calls), bucket=bucket)
    r = gcs.ensure_artifact(name, local, produce=_producer(counter=calls), bucket=bucket)
    assert len(calls) == 1
    assert (r.skipped, r.produced, r.uploaded, r.downloaded) == (True, False, False, False)


def test_ensure_artifact_on_a_fresh_runtime_downloads_instead_of_recomputing(bucket, tmp_path):
    """The resumption case: the object is in GCS from a previous session,
    and the local disk is empty because the runtime is new."""
    name = gcs.object_path(**TRAIN_ARGS)
    first = tmp_path / "session_a" / "features.npz"
    gcs.ensure_artifact(name, first, produce=_producer("from session a"), bucket=bucket)

    calls = []
    second = tmp_path / "session_b" / "features.npz"
    r = gcs.ensure_artifact(name, second, produce=_producer(counter=calls), bucket=bucket)
    assert calls == []
    assert (r.skipped, r.produced, r.uploaded, r.downloaded) == (True, False, False, True)
    assert second.read_text() == "from session a"


def test_ensure_artifact_leaves_the_artifact_present_both_places_in_every_branch(
        bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)
    for i, local in enumerate([tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "b.npz"]):
        r = gcs.ensure_artifact(name, local, produce=_producer(f"run{i}"), bucket=bucket)
        assert Path(r.local_path).is_file()
        assert gcs.object_exists(name, bucket=bucket)


def test_force_recomputes_and_overwrites(bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)
    local = tmp_path / "features.npz"
    gcs.ensure_artifact(name, local, produce=_producer("first"), bucket=bucket)
    r = gcs.ensure_artifact(name, local, produce=_producer("second"), bucket=bucket,
                            force=True)
    assert (r.skipped, r.produced, r.uploaded) == (False, True, True)
    assert bucket.objects[name] == b"second"


def test_a_step_whose_producer_writes_nothing_fails_instead_of_recording_completion(
        bucket, tmp_path):
    """It must not upload, and it must not leave the next run thinking the
    step is done."""
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(FileNotFoundError, match="without writing"):
        gcs.ensure_artifact(name, tmp_path / "features.npz", produce=lambda p: None,
                            bucket=bucket)
    assert name not in bucket.objects
    assert gcs.object_exists(name, bucket=bucket) is False


def test_a_producer_that_raises_leaves_no_object_behind(bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)

    def explode(local_path):
        raise RuntimeError("the step failed")

    with pytest.raises(RuntimeError, match="the step failed"):
        gcs.ensure_artifact(name, tmp_path / "features.npz", produce=explode, bucket=bucket)
    assert bucket.objects == {}


def test_ensure_artifact_requires_a_callable_producer(bucket, tmp_path):
    with pytest.raises(TypeError, match="callable"):
        gcs.ensure_artifact(gcs.object_path(**TRAIN_ARGS), tmp_path / "f.npz",
                            produce="not callable", bucket=bucket)


def test_step_result_summary_records_what_happened(bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)
    r = gcs.ensure_artifact(name, tmp_path / "features.npz", produce=_producer(),
                            bucket=bucket)
    assert r.summary() == {
        "object_path": name, "local_path": str(tmp_path / "features.npz"),
        "skipped": False, "produced": True, "uploaded": True, "downloaded": False,
        "size_bytes": len("computed"),
    }


def test_a_stage_4_test_side_step_works_once_opted_in(bucket, tmp_path):
    """The one place the flag is passed: stage 4's confirmatory
    evaluation. Everything downstream of the guard behaves normally."""
    name = gcs.object_path(**TEST_ARGS, allow_test_split=True)
    r = gcs.ensure_artifact(name, tmp_path / "predictions.npz", produce=_producer(),
                            bucket=bucket, allow_test_split=True)
    assert r.uploaded and gcs.is_test_split_path(r.object_path)
    assert gcs.object_exists(name, bucket=bucket, allow_test_split=True) is True


# ---- chunked upload with on-disk checkpointing ----
#
# `download_file` already survives a death mid-transfer (the `.part`
# sidecar). `upload_file` did not, and Stage 2B pushes gigabyte artifacts
# out of a session that this project has already watched die mid-task
# (PROJECT_MEMORY.md Part 4). These cover the resume: what the checkpoint
# is allowed to claim, when it must be thrown away, and that a resumed
# upload lands exactly the bytes an uninterrupted one would.

CHUNK = 100


def _payload_bytes(n):
    """A byte pattern whose period (256) is not a multiple of the chunk
    size, so a chunk composed out of order or a chunk left out shows up as
    a byte difference rather than passing unnoticed."""
    return bytes((i * 37 + 11) % 256 for i in range(n))


def _local_artifact(tmp_path, n_bytes=1000, name="features.npz"):
    path = tmp_path / "work" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_payload_bytes(n_bytes))
    return path


def _rewrite(path, data, *, mtime_ns=None):
    """Replace a local artifact's contents, optionally pinning its mtime
    so the test does not depend on the filesystem's clock resolution."""
    path.write_bytes(data)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _checkpoint(local):
    return json.loads(Path(gcs.checkpoint_path(local)).read_text())


def _confirmed_indices(local):
    return sorted(entry["index"] for entry in _checkpoint(local)["confirmed"])


def _part(name, index):
    return f"{gcs.part_prefix(name)}/{index:06d}"


def test_the_checkpoint_is_a_sidecar_next_to_the_local_file(tmp_path):
    """The `.part` convention `download_file` already uses, applied to the
    other direction of transfer."""
    local = _local_artifact(tmp_path)
    assert gcs.checkpoint_path(local) == str(local) + gcs.CHECKPOINT_SUFFIX
    assert Path(gcs.checkpoint_path(local)).parent == local.parent


def test_a_chunked_upload_lands_the_same_bytes_as_a_plain_one(bucket, tmp_path):
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    assert gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK) == name

    plain = FakeBucket()
    gcs.upload_file(local, name, bucket=plain)
    assert bucket.objects[name] == plain.objects[name] == local.read_bytes()


def test_a_completed_chunked_upload_leaves_no_checkpoint_and_no_parts(bucket, tmp_path):
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert list(bucket.objects) == [name]
    assert not Path(gcs.checkpoint_path(local)).exists()
    assert not Path(gcs.checkpoint_path(local) + ".tmp").exists()


@pytest.mark.parametrize("n_bytes", [0, 1, 50, CHUNK])
def test_a_file_within_one_chunk_is_uploaded_directly(bucket, tmp_path, n_bytes):
    """Nothing to resume from inside a single request, so there is no
    checkpoint, no part object and no composition for a small artifact --
    the gate artifacts and manifests keep going up exactly as before."""
    local = _local_artifact(tmp_path, n_bytes=n_bytes)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert bucket.objects[name] == local.read_bytes()
    assert bucket.uploads == [name]
    assert bucket.composes == []
    assert not Path(gcs.checkpoint_path(local)).exists()


def test_the_checkpoint_records_what_it_belongs_to(tmp_path):
    """Provenance, CLAUDE.md principle 7: a checkpoint that does not say
    which object, which file state and which chunking it describes cannot
    be checked against anything on the next run."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    state = _checkpoint(local)
    assert state["format"] == gcs.CHECKPOINT_FORMAT
    assert state["object_path"] == name
    assert state["chunk_size"] == CHUNK
    assert state["size_bytes"] == local.stat().st_size
    assert state["mtime_ns"] == local.stat().st_mtime_ns
    assert state["n_chunks"] == 10
    assert state["part_prefix"] == gcs.part_prefix(name)


def test_a_death_mid_transfer_leaves_only_the_confirmed_chunks_recorded(tmp_path):
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert _confirmed_indices(local) == [0, 1, 2]
    assert sorted(bucket.objects) == [_part(name, i) for i in (0, 1, 2)]
    assert name not in bucket.objects


def test_a_chunk_is_recorded_only_once_the_remote_confirms_it(tmp_path):
    """The bug this whole mechanism exists to avoid: a checkpoint claiming
    a chunk that never landed is worse than no checkpoint at all, because
    the resume would skip it and compose a hole into the object."""
    bucket = SilentlyDroppingBucket(drop_from=2)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(gcs.ChunkedUploadError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert _confirmed_indices(local) == [0, 1]
    assert name not in bucket.objects
    assert bucket.composes == []


def test_a_chunk_that_lands_short_is_not_recorded_either(tmp_path):
    """Existence is not enough: the object is there and its bytes are not
    the ones that were sent."""
    bucket = TruncatingBucket(truncate_from=2)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(gcs.ChunkedUploadError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert _confirmed_indices(local) == [0, 1]
    assert name not in bucket.objects


def test_a_resumed_upload_is_byte_identical_to_an_uninterrupted_one(tmp_path):
    """The property the whole design is for."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    bucket.die_after = None                     # a fresh run against a live bucket
    bucket.uploads.clear()
    assert gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK) == name

    uninterrupted = FakeBucket()
    gcs.upload_file_chunked(local, name, bucket=uninterrupted, chunk_size=CHUNK)
    assert bucket.objects[name] == uninterrupted.objects[name] == local.read_bytes()
    assert list(bucket.objects) == [name]
    assert not Path(gcs.checkpoint_path(local)).exists()


def test_a_resume_does_not_re_send_the_chunks_already_confirmed(tmp_path):
    """Otherwise the checkpoint buys nothing -- restarting from zero is
    exactly the behaviour being replaced."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert bucket.uploads == [_part(name, i) for i in range(3, 10)]


def test_a_checkpoint_from_a_differently_sized_file_is_discarded(tmp_path):
    """The stale-checkpoint corruption: parts of the previous artifact and
    parts of this one composed into one object. Every chunk must be sent
    again."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    _rewrite(local, _payload_bytes(1600)[::-1])
    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert bucket.objects[name] == local.read_bytes()
    assert bucket.uploads == [_part(name, i) for i in range(16)]
    assert list(bucket.objects) == [name]


def test_a_checkpoint_from_a_same_sized_but_rewritten_file_is_discarded(tmp_path):
    """Same length, different bytes, different mtime -- the regenerated
    artifact case. The mtime is pinned rather than left to the clock so
    the test asserts the check, not the filesystem's resolution."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    before = local.stat().st_mtime_ns

    _rewrite(local, _payload_bytes(1000)[::-1], mtime_ns=before + 10 ** 9)
    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert bucket.objects[name] == local.read_bytes()
    assert bucket.uploads == [_part(name, i) for i in range(10)]


def test_a_checkpoint_naming_a_different_object_is_discarded(tmp_path):
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    first = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, first, bucket=bucket, chunk_size=CHUNK)

    second = gcs.object_path(stage=2, condition="lattice", kind="features", ext="npz",
                             split="train")
    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, second, bucket=bucket, chunk_size=CHUNK)

    assert bucket.objects[second] == local.read_bytes()
    assert bucket.uploads == [_part(second, i) for i in range(10)]


def test_a_checkpoint_written_under_a_different_chunk_size_is_discarded(tmp_path):
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=250)
    assert bucket.objects[name] == local.read_bytes()
    assert bucket.uploads == [_part(name, i) for i in range(4)]


def test_a_corrupt_checkpoint_file_is_discarded_rather_than_fatal(tmp_path):
    bucket = FakeBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    Path(gcs.checkpoint_path(local)).write_text("{not json at all")
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert bucket.objects[name] == local.read_bytes()


def test_a_confirmed_chunk_that_vanished_from_the_bucket_is_sent_again(tmp_path):
    """The checkpoint is checked against the remote before it is trusted:
    a recorded part that is no longer there must not be composed in."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    del bucket.objects[_part(name, 1)]
    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert _part(name, 1) in bucket.uploads
    assert bucket.objects[name] == local.read_bytes()


def test_a_confirmed_chunk_whose_remote_size_disagrees_is_sent_again(tmp_path):
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    bucket.objects[_part(name, 2)] = b"short"
    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert _part(name, 2) in bucket.uploads
    assert bucket.objects[name] == local.read_bytes()


def test_digest_verification_catches_an_edit_the_metadata_check_cannot(tmp_path):
    """Same size, same mtime, different bytes -- what `verify_digests=True`
    is for, and the boundary of the cheap check it strengthens."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    pinned = local.stat().st_mtime_ns

    _rewrite(local, _payload_bytes(1000)[::-1], mtime_ns=pinned)
    assert local.stat().st_mtime_ns == pinned          # the check cannot see this edit

    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK,
                            verify_digests=True)
    assert bucket.objects[name] == local.read_bytes()
    assert bucket.uploads == [_part(name, i) for i in range(10)]


def test_digest_verification_keeps_the_chunks_that_still_match(tmp_path):
    """Per-chunk, not all-or-nothing: an edit confined to one chunk does
    not cost the transfer the rest of them."""
    bucket = DyingBucket(die_after=4)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    pinned = local.stat().st_mtime_ns

    edited = bytearray(local.read_bytes())
    edited[250] ^= 0xFF                                # inside chunk 2 only
    _rewrite(local, bytes(edited), mtime_ns=pinned)

    bucket.die_after = None
    bucket.uploads.clear()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK,
                            verify_digests=True)
    assert bucket.objects[name] == local.read_bytes()
    assert _part(name, 2) in bucket.uploads
    assert _part(name, 0) not in bucket.uploads
    assert _part(name, 1) not in bucket.uploads
    assert _part(name, 3) not in bucket.uploads


def test_parts_left_by_a_previous_larger_upload_are_not_composed_in(tmp_path):
    """A shorter artifact reusing the same object path leaves surplus part
    objects behind. They must not survive the upload, and they certainly
    must not end up in the composed object."""
    bucket = DyingBucket(die_after=5)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    _rewrite(local, _payload_bytes(300))
    bucket.die_after = None
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert bucket.objects[name] == local.read_bytes()
    assert list(bucket.objects) == [name]


def test_an_artifact_too_small_to_chunk_still_clears_an_earlier_attempts_parts(tmp_path):
    """The step is re-run, and this time its artifact fits in one request.
    The parts the dead attempt left must still go: nothing else will ever
    revisit that prefix, and they would sit under the condition prefix
    forever, in every listing and every dry-run delete."""
    bucket = DyingBucket(die_after=3)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(ConnectionError):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    _rewrite(local, _payload_bytes(40))
    bucket.die_after = None
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert bucket.objects[name] == local.read_bytes()
    assert list(bucket.objects) == [name]
    assert not Path(gcs.checkpoint_path(local)).exists()


def test_the_part_cleanup_does_not_reach_a_neighbouring_object(bucket, tmp_path):
    """Prefix matching is plain string matching, so the cleanup has to be
    scoped to the part prefix's children -- an object whose name merely
    starts the same way is not this upload's to delete."""
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    neighbour = gcs.part_prefix(name) + "-from-something-else"
    bucket.objects[neighbour] = b"not mine"

    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert bucket.objects[neighbour] == b"not mine"
    assert bucket.objects[name] == local.read_bytes()


def test_more_chunks_than_one_compose_call_takes_are_composed_in_a_tree(bucket, tmp_path):
    """GCS composes at most 32 sources per request, so a gigabyte artifact
    needs more than one level. The fake refuses an over-long source list,
    so this fails rather than passing on an untested assumption."""
    local = _local_artifact(tmp_path, n_bytes=10_000)          # 100 chunks
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)

    assert bucket.objects[name] == local.read_bytes()
    assert len(bucket.composes) > 1
    assert all(len(sources) <= 32 for _, sources in bucket.composes)
    assert list(bucket.objects) == [name]


def test_a_chunked_upload_to_a_test_side_path_still_needs_the_opt_in(tmp_path):
    """The guard sits in front of the chunking, not around it: no part
    object, no checkpoint, nothing."""
    bucket = FakeBucket()
    local = _local_artifact(tmp_path)
    raw = "stage2b/testsplit/stage4/evolved_T/predictions.npz"
    with pytest.raises(PermissionError, match="stages 1-3"):
        gcs.upload_file_chunked(local, raw, bucket=bucket, chunk_size=CHUNK)
    assert bucket.objects == {}
    assert bucket.uploads == []
    assert not Path(gcs.checkpoint_path(local)).exists()


def test_a_stage_4_test_side_chunked_upload_works_once_opted_in(bucket, tmp_path):
    """Including the part cleanup, which touches test-side objects itself."""
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TEST_ARGS, allow_test_split=True)
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK,
                            allow_test_split=True)
    assert bucket.objects[name] == local.read_bytes()
    assert list(bucket.objects) == [name]


def test_chunked_upload_refuses_a_local_path_that_is_not_a_file(bucket, tmp_path):
    with pytest.raises(FileNotFoundError):
        gcs.upload_file_chunked(tmp_path / "missing.bin", gcs.object_path(**TRAIN_ARGS),
                                bucket=bucket, chunk_size=CHUNK)
    assert bucket.objects == {}


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, None, "100"])
def test_chunked_upload_rejects_a_malformed_chunk_size(bucket, tmp_path, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        gcs.upload_file_chunked(_local_artifact(tmp_path), gcs.object_path(**TRAIN_ARGS),
                                bucket=bucket, chunk_size=chunk_size)


def test_chunked_upload_refuses_more_parts_than_composition_allows(bucket, tmp_path):
    """Better a refusal naming the chunk size than a transfer that uploads
    every part and then fails at the compose."""
    local = _local_artifact(tmp_path, n_bytes=2000)
    with pytest.raises(ValueError, match="chunk_size"):
        gcs.upload_file_chunked(local, gcs.object_path(**TRAIN_ARGS), bucket=bucket,
                                chunk_size=1)
    assert bucket.objects == {}


def test_the_default_chunk_size_is_a_sane_transfer_unit():
    assert gcs.CHUNK_SIZE_DEFAULT >= 8 * 1024 * 1024
    assert gcs.COMPOSE_MAX_SOURCES == 32
    assert gcs.MAX_PARTS >= 1024


def test_ensure_artifact_can_push_its_artifact_in_chunks(bucket, tmp_path):
    """The chunking has to be reachable from the primitive the run scripts
    actually call, or it protects nothing."""
    name = gcs.object_path(**TRAIN_ARGS)
    local = tmp_path / "work" / "features.npz"
    payload = _payload_bytes(1000)

    def produce(path):
        Path(path).write_bytes(payload)

    r = gcs.ensure_artifact(name, local, produce=produce, bucket=bucket, chunked=True,
                            chunk_size=CHUNK)
    assert (r.skipped, r.produced, r.uploaded) == (False, True, True)
    assert bucket.objects[name] == payload
    assert list(bucket.objects) == [name]
    assert len(bucket.composes) == 1


def test_a_chunked_ensure_artifact_still_needs_the_test_split_opt_in(bucket, tmp_path):
    raw = "stage2b/testsplit/stage4/evolved_T/predictions.npz"
    with pytest.raises(PermissionError):
        gcs.ensure_artifact(raw, tmp_path / "p.npz", produce=_producer(), bucket=bucket,
                            chunked=True, chunk_size=CHUNK)
    assert bucket.objects == {}


# ---- which route `ensure_artifact` takes, and who decides ----
#
# `chunked` used to default to False, so an artifact went up resumably
# only where the call site had remembered to ask -- the shape of failure
# PROJECT_MEMORY.md records for `mighty-colab exec --timeout`, a flag
# every hand-run command carried and no Makefile target did. The default
# is now the size decision, with the bool as an override in both
# directions.
#
# The route is pinned through `ensure_artifact` against the fake bucket
# rather than by watching which function gets called. Above the threshold
# the two routes are told apart by the composition; at or below it both
# send one request, and what separates them is that the chunked route
# clears any parts an earlier attempt left under `{name}.parts/` and the
# single-request route never touches them.


def _producer_bytes(n_bytes):
    def produce(local_path):
        Path(local_path).write_bytes(_payload_bytes(n_bytes))
    return produce


def _seed_stale_part(bucket, name):
    """A part object left behind by an earlier, dead chunked attempt. It
    survives a single-request upload and is cleared by a chunked one."""
    stale = _part(name, 0)
    bucket.objects[stale] = b"left by a dead attempt"
    return stale


def test_should_chunk_decides_on_size_at_the_chunk_boundary():
    """The rule, as a pure function: more than one chunk to send means
    there is something for a resume to pick up."""
    assert gcs.should_chunk(CHUNK, chunk_size=CHUNK) is False
    assert gcs.should_chunk(CHUNK + 1, chunk_size=CHUNK) is True
    assert gcs.should_chunk(0, chunk_size=CHUNK) is False


def test_the_default_threshold_is_the_default_chunk_size():
    """A call that names no chunk size decides at 64 MiB -- Stage 1's
    ~8MB artifacts in one request, Stage 3's 242MB-class ones chunked."""
    assert gcs.should_chunk(8 * 1024 * 1024) is False
    assert gcs.should_chunk(gcs.CHUNK_SIZE_DEFAULT) is False
    assert gcs.should_chunk(gcs.CHUNK_SIZE_DEFAULT + 1) is True
    assert gcs.should_chunk(242 * 1000 * 1000) is True


def test_the_explicit_override_beats_the_size_in_both_directions():
    assert gcs.should_chunk(0, chunked=True, chunk_size=CHUNK) is True
    assert gcs.should_chunk(10 ** 9, chunked=False, chunk_size=CHUNK) is False


@pytest.mark.parametrize("chunked", [1, 0, "yes", "", []])
def test_a_chunked_value_that_is_neither_a_bool_nor_none_is_refused(chunked):
    """`chunked` is tri-state, so a truthy stand-in for True would select
    a transfer route by accident rather than by decision."""
    with pytest.raises(TypeError, match="chunked"):
        gcs.should_chunk(0, chunked=chunked, chunk_size=CHUNK)


def test_ensure_artifact_chunks_an_artifact_over_the_threshold_unasked(bucket, tmp_path):
    """The point of the change: a caller that does not mention chunking
    still gets the resumable upload on a large artifact."""
    name = gcs.object_path(**TRAIN_ARGS)
    local = tmp_path / "work" / "features.npz"
    r = gcs.ensure_artifact(name, local, produce=_producer_bytes(CHUNK * 10), bucket=bucket,
                            chunk_size=CHUNK)
    assert (r.skipped, r.produced, r.uploaded) == (False, True, True)
    assert bucket.objects[name] == _payload_bytes(CHUNK * 10)
    assert bucket.composes                      # it went up as parts and was composed
    assert list(bucket.objects) == [name]


@pytest.mark.parametrize("n_bytes", [CHUNK // 2, CHUNK])
def test_ensure_artifact_sends_an_artifact_within_one_chunk_in_one_request(
        bucket, tmp_path, n_bytes):
    """The other side of the same threshold, at it and below it -- a size
    exactly on the boundary alone would read the same under a comparison
    flipped the other way. There is nothing to resume inside a single
    request, so the small artifacts keep going up exactly as they did,
    and the single-request route leaves an earlier attempt's parts alone,
    which is what distinguishes it here."""
    name = gcs.object_path(**TRAIN_ARGS)
    stale = _seed_stale_part(bucket, name)
    local = tmp_path / "work" / "features.npz"
    gcs.ensure_artifact(name, local, produce=_producer_bytes(n_bytes), bucket=bucket,
                        chunk_size=CHUNK)
    assert bucket.objects[name] == _payload_bytes(n_bytes)
    assert bucket.composes == []
    assert bucket.uploads == [name]
    assert stale in bucket.objects              # the chunked route would have cleared it


def test_chunked_true_forces_the_chunked_route_below_the_threshold(bucket, tmp_path):
    name = gcs.object_path(**TRAIN_ARGS)
    stale = _seed_stale_part(bucket, name)
    local = tmp_path / "work" / "features.npz"
    gcs.ensure_artifact(name, local, produce=_producer_bytes(CHUNK), bucket=bucket,
                        chunked=True, chunk_size=CHUNK)
    assert bucket.objects[name] == _payload_bytes(CHUNK)
    assert stale not in bucket.objects          # only the chunked route clears parts
    assert list(bucket.objects) == [name]


def test_chunked_false_forces_the_single_request_route_above_the_threshold(bucket, tmp_path):
    """The override has to hold in the direction the size decision would
    otherwise overrule, or it is not an override."""
    name = gcs.object_path(**TRAIN_ARGS)
    stale = _seed_stale_part(bucket, name)
    local = tmp_path / "work" / "features.npz"
    gcs.ensure_artifact(name, local, produce=_producer_bytes(CHUNK * 10), bucket=bucket,
                        chunked=False, chunk_size=CHUNK)
    assert bucket.objects[name] == _payload_bytes(CHUNK * 10)
    assert bucket.composes == []
    assert bucket.uploads == [name]             # one request, not ten parts
    assert stale in bucket.objects


def test_ensure_artifact_refuses_a_chunked_value_that_is_neither_a_bool_nor_none(
        bucket, tmp_path):
    calls = []
    with pytest.raises(TypeError, match="chunked"):
        gcs.ensure_artifact(gcs.object_path(**TRAIN_ARGS), tmp_path / "f.npz",
                            produce=_producer(counter=calls), bucket=bucket, chunked="yes")
    assert calls == []                          # refused before the step ran


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, None, "100"])
def test_the_size_decision_rejects_a_malformed_chunk_size(bucket, tmp_path, chunk_size):
    """`ensure_artifact` now reads `chunk_size` on every call rather than
    only when handing it to `upload_file_chunked`, so a malformed one
    fails the same way here as it does there -- and before `produce`
    runs, not after. A Stage 2B step computes for minutes to hours on a
    GPU; raising on a malformed argument once that is done would throw
    the work away and upload nothing for the next run to resume from."""
    calls = []
    with pytest.raises(ValueError, match="chunk_size"):
        gcs.ensure_artifact(gcs.object_path(**TRAIN_ARGS), tmp_path / "f.npz",
                            produce=_producer(counter=calls), bucket=bucket,
                            chunk_size=chunk_size)
    assert calls == []


def test_a_chunked_upload_and_its_resume_run_with_the_google_package_blocked():
    """The load-bearing constraint, applied to the new code path: a fresh
    interpreter where `google` cannot be imported still runs a chunked
    upload, dies partway, and resumes to the right bytes."""
    code = f"""
import sys
class _Block:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name == "google" or name.startswith("google."):
            raise ImportError("google is blocked for this test")
        return None
sys.meta_path.insert(0, _Block())
sys.path.insert(0, {str(_STAGE2B_DIR)!r})
sys.path.insert(0, {str(_REPO_ROOT / "tests")!r})
import tempfile, os
from pathlib import Path
import stage2b_gcs as gcs
from test_stage2b_gcs import DyingBucket, _payload_bytes

with tempfile.TemporaryDirectory() as d:
    local = Path(d) / "features.npz"
    local.write_bytes(_payload_bytes(1000))
    name = gcs.object_path(stage=2, condition="evolved_T", kind="features", ext="npz",
                           split="train")
    bucket = DyingBucket(die_after=3)
    try:
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=100)
    except ConnectionError:
        print("died-ok")
    print(os.path.exists(gcs.checkpoint_path(local)))
    bucket.die_after = None
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=100)
    print(bucket.objects[name] == local.read_bytes())
print(any(n == "google" or n.startswith("google.") for n in sys.modules))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["died-ok", "True", "True", "False"]


# ---- content integrity: the digest, and what is done with it ----
#
# The `.part` sidecar makes a download atomic -- either the whole
# transfer landed or nothing did. It says nothing about whether the bytes
# are the artifact's. Stage 2B downloads multi-gigabyte feature arrays
# and runs science on them, so a silently corrupted download is the worst
# available failure: it produces numbers rather than an error.
#
# The digest is `crc32c`, the checksum GCS computes for every object it
# stores -- including a composite one, which carries no `md5_hash` at
# all. One field, both upload routes, and nothing for a downloader to
# know about which route produced what it is reading.


def test_the_digest_field_is_the_one_gcs_populates_for_every_object():
    assert gcs.CHECKSUM_FIELD == "crc32c"


def test_crc32c_matches_the_standard_check_value():
    """CRC-32C's published check value over `123456789` is 0xE3069283.
    Pinned as the base64 form GCS reports, so the encoding is pinned with
    the algorithm rather than assumed alongside it."""
    assert gcs.crc32c_of_bytes(b"123456789") == "4waSgw=="
    assert gcs.crc32c_of_bytes(b"") == "AAAAAA=="


def test_crc32c_of_a_file_is_the_one_shot_value_of_its_bytes(tmp_path):
    """The file digest is computed in blocks -- a gigabyte artifact is not
    read into memory to check it -- so the blocking must not change the
    number."""
    payload = _payload_bytes(300_000)
    local = tmp_path / "big.bin"
    local.write_bytes(payload)
    assert gcs.crc32c_of_file(local) == gcs.crc32c_of_bytes(payload)


def test_crc32c_agrees_with_google_crc32c_where_it_is_installed():
    """The module falls back to a pure-Python CRC32C where
    `google-crc32c` is absent, which is the local development
    environment. Where the real library IS present -- the cloud runtime,
    which is where the digests that matter are computed -- the two must
    agree, or the fallback would be checking against a different
    algorithm than GCS uses (CLAUDE.md principle 16)."""
    google_crc32c = pytest.importorskip("google_crc32c")
    import base64
    for payload in (b"", b"a", b"123456789", _payload_bytes(1), _payload_bytes(4096),
                    _payload_bytes(70_001)):
        expected = base64.b64encode(google_crc32c.Checksum(payload).digest()).decode()
        assert gcs.crc32c_of_bytes(payload) == expected


def test_a_plain_upload_lands_a_verifiable_digest(bucket, tmp_path):
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(local, name, bucket=bucket)
    assert gcs.object_checksum(name, bucket=bucket) == gcs.crc32c_of_file(local)


def test_a_chunked_upload_lands_a_digest_in_the_same_field(bucket, tmp_path):
    """The point of choosing this field: a downloader does not need to
    know which route produced the object. A composed object carries no
    MD5, so a downloader verifying MD5 would behave differently for the
    two routes."""
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert len(bucket.composes) == 1                     # it really was composed
    assert gcs.object_checksum(name, bucket=bucket) == gcs.crc32c_of_file(local)


def test_a_corrupted_download_raises_naming_the_object_and_both_digests(tmp_path):
    bucket = CorruptingDownloadBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(local, name, bucket=bucket)

    with pytest.raises(gcs.ChecksumMismatchError) as excinfo:
        gcs.download_file(name, tmp_path / "back" / "features.npz", bucket=bucket)
    message = str(excinfo.value)
    assert name in message
    assert gcs.crc32c_of_file(local) in message          # what the object says it is
    assert gcs.crc32c_of_bytes(bucket._deliver(local.read_bytes())) in message


def test_a_corrupted_download_leaves_nothing_at_the_destination(tmp_path):
    """The `.part` discipline exists so a bad transfer is never mistaken
    for a good one. Verification happens before the rename, so a file
    that fails it never reaches the destination path at all -- and the
    sidecar does not survive either, to be renamed by hand later."""
    bucket = CorruptingDownloadBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(local, name, bucket=bucket)

    target = tmp_path / "back" / "features.npz"
    with pytest.raises(gcs.ChecksumMismatchError):
        gcs.download_file(name, target, bucket=bucket)
    assert not target.exists()
    assert not (target.parent / (target.name + ".part")).exists()


def test_a_corrupted_download_does_not_overwrite_a_good_local_file(tmp_path):
    bucket = CorruptingDownloadBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(local, name, bucket=bucket)

    target = tmp_path / "existing.npz"
    target.write_bytes(b"the good copy")
    with pytest.raises(gcs.ChecksumMismatchError):
        gcs.download_file(name, target, bucket=bucket)
    assert target.read_bytes() == b"the good copy"


def test_an_object_with_no_recorded_digest_is_refused_rather_than_trusted(tmp_path):
    """GCS populates `crc32c` for every object it stores, so a missing one
    means the field could not be read -- not that the object is fine.
    Refusing is what stops "no digest" degrading into "no check"."""
    bucket = ChecksumlessBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)

    with pytest.raises(gcs.ChecksumMissingError, match=gcs.CHECKSUM_FIELD):
        gcs.upload_file(local, name, bucket=bucket)

    bucket.objects[name] = local.read_bytes()
    target = tmp_path / "back" / "features.npz"
    with pytest.raises(gcs.ChecksumMissingError, match=gcs.CHECKSUM_FIELD):
        gcs.download_file(name, target, bucket=bucket)
    assert not target.exists()


def test_verification_is_on_by_default_and_can_be_switched_off(tmp_path):
    """Opt-out, not opt-in: a science run should not have to remember to
    ask for correctness. The escape hatch exists, and taking it is
    visible at the call site."""
    bucket = CorruptingDownloadBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(local, name, bucket=bucket)

    target = tmp_path / "back" / "features.npz"
    gcs.download_file(name, target, bucket=bucket, verify_content=False)
    assert target.is_file()
    assert target.read_bytes() != local.read_bytes()     # it really was corrupt


def test_a_plain_upload_that_lands_wrong_raises_and_removes_the_object(tmp_path):
    """`ensure_artifact` treats the object's existence as proof the step
    is done. An object that failed verification must not be left behind
    making that claim."""
    bucket = TruncatingBucket(truncate_from=0)
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)

    with pytest.raises(gcs.ChecksumMismatchError, match=name):
        gcs.upload_file(local, name, bucket=bucket)
    assert name not in bucket.objects
    assert gcs.object_exists(name, bucket=bucket) is False


def test_a_miscomposed_chunked_upload_is_caught_by_the_content_digest(tmp_path):
    """Every part landed, every part is the right length, and the object
    is the right size -- and its bytes are the artifact's in the wrong
    order. The digest is the only thing here that can see it."""
    bucket = MiscomposingBucket()
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)

    with pytest.raises(gcs.ChecksumMismatchError, match=name):
        gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert name not in bucket.objects
    assert Path(gcs.checkpoint_path(local)).exists()     # the transfer state is kept
    assert any(n.startswith(gcs.part_prefix(name) + "/") for n in bucket.objects)


def test_a_chunked_upload_that_composes_correctly_still_passes(bucket, tmp_path):
    """The mirror of the test above: the check has to pass on the right
    answer, or catching the wrong one proves nothing."""
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=CHUNK)
    assert bucket.objects[name] == local.read_bytes()
    assert list(bucket.objects) == [name]


def test_ensure_artifact_verifies_the_artifact_it_downloads(tmp_path):
    """The resumption path -- a fresh runtime pulling down what a dead
    session left in GCS -- is exactly where an unverified download would
    feed corrupt features into the science."""
    bucket = CorruptingDownloadBucket()
    name = gcs.object_path(**TRAIN_ARGS)
    first = tmp_path / "session_a" / "features.npz"
    gcs.ensure_artifact(name, first, produce=_producer("from session a"), bucket=bucket)

    second = tmp_path / "session_b" / "features.npz"
    with pytest.raises(gcs.ChecksumMismatchError):
        gcs.ensure_artifact(name, second, produce=_producer(), bucket=bucket)
    assert not second.exists()


def test_ensure_artifact_verifies_what_it_uploads_on_both_routes(tmp_path):
    """Each route gets the corruption its own earlier checks cannot see:
    the single-request one a short object, the chunked one a composition
    in the wrong order (parts truncated on that route are caught by the
    part-size check long before anything is composed)."""
    name = gcs.object_path(**TRAIN_ARGS)
    for chunked, bucket in ((False, TruncatingBucket(truncate_from=0)),
                            (True, MiscomposingBucket())):
        local = tmp_path / f"chunked_{chunked}" / "features.npz"
        with pytest.raises(gcs.ChecksumMismatchError):
            gcs.ensure_artifact(name, local, produce=_producer("computed" * 100),
                                bucket=bucket, chunked=chunked, chunk_size=CHUNK)
        assert name not in bucket.objects


def test_ensure_artifact_verification_can_be_switched_off(tmp_path):
    bucket = CorruptingDownloadBucket()
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.ensure_artifact(name, tmp_path / "a" / "features.npz",
                        produce=_producer("session a"), bucket=bucket)
    r = gcs.ensure_artifact(name, tmp_path / "b" / "features.npz", produce=_producer(),
                            bucket=bucket, verify_content=False)
    assert r.downloaded is True


def test_verify_object_reports_the_digest_it_matched(bucket, tmp_path):
    """The check is callable on its own, so a script can assert a local
    copy is still the object's without downloading it again."""
    local = _local_artifact(tmp_path)
    name = gcs.object_path(**TRAIN_ARGS)
    gcs.upload_file(local, name, bucket=bucket)
    assert gcs.verify_object(name, local, bucket=bucket) == gcs.crc32c_of_file(local)

    other = _local_artifact(tmp_path, n_bytes=1000, name="other.npz")
    other.write_bytes(_payload_bytes(1000)[::-1])
    with pytest.raises(gcs.ChecksumMismatchError):
        gcs.verify_object(name, other, bucket=bucket)


def test_content_verification_runs_with_the_google_package_blocked():
    """`google-crc32c` is a hard dependency of `google-cloud-storage`, so
    the compiled implementation is present wherever a real transfer
    happens. Where neither is installed -- this project's local
    environment -- verification must still run rather than silently
    becoming a no-op."""
    code = f"""
import sys
class _Block:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name in ("google", "google_crc32c") or name.startswith("google"):
            raise ImportError("google is blocked for this test")
        return None
sys.meta_path.insert(0, _Block())
sys.path.insert(0, {str(_STAGE2B_DIR)!r})
sys.path.insert(0, {str(_REPO_ROOT / "tests")!r})
import tempfile
from pathlib import Path
import stage2b_gcs as gcs
from test_stage2b_gcs import CorruptingDownloadBucket, _payload_bytes

print(gcs.checksum_backend())
print(gcs.crc32c_of_bytes(b"123456789"))
with tempfile.TemporaryDirectory() as d:
    local = Path(d) / "features.npz"
    local.write_bytes(_payload_bytes(1000))
    name = gcs.object_path(stage=2, condition="evolved_T", kind="features", ext="npz",
                           split="train")
    bucket = CorruptingDownloadBucket()
    gcs.upload_file_chunked(local, name, bucket=bucket, chunk_size=100)
    try:
        gcs.download_file(name, Path(d) / "back.npz", bucket=bucket)
    except gcs.ChecksumMismatchError:
        print("caught-ok")
    print((Path(d) / "back.npz").exists())
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["python", "4waSgw==", "caught-ok", "False"]


def test_bucket_is_a_required_keyword_argument_everywhere(tmp_path):
    """Transport takes an injected bucket rather than building a client
    itself -- which is why every function here is testable without the
    library, and why a run script constructs exactly one client."""
    name = gcs.object_path(**TRAIN_ARGS)
    with pytest.raises(TypeError):
        gcs.object_exists(name)
    with pytest.raises(TypeError):
        gcs.upload_file(_write(tmp_path / "x.bin"), name)
    with pytest.raises(TypeError):
        gcs.upload_file_chunked(_write(tmp_path / "x.bin"), name)
    with pytest.raises(TypeError):
        gcs.download_file(name, tmp_path / "y.bin")
    with pytest.raises(TypeError):
        gcs.list_objects(gcs.TRAIN_ROOT)
    with pytest.raises(TypeError):
        gcs.delete_prefix(gcs.TRAIN_ROOT)
    with pytest.raises(TypeError):
        gcs.ensure_artifact(name, tmp_path / "x.bin", produce=_producer())
