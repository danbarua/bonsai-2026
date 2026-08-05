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

    def exists(self):
        return self.name in self.bucket.objects

    def reload(self):
        if self.name not in self.bucket.objects:
            raise FileNotFoundError(f"no such object: {self.name}")
        self.size = len(self.bucket.objects[self.name])

    def upload_from_filename(self, path):
        self.bucket._store_upload(self.name, Path(path).read_bytes())
        self.size = len(self.bucket.objects.get(self.name, b""))

    def upload_from_string(self, data, content_type=None):
        self.bucket._store_upload(self.name, bytes(data))
        self.size = len(self.bucket.objects.get(self.name, b""))

    def compose(self, sources):
        """Server-side concatenation, with the API's own source limit
        enforced -- a composition tree that exceeded it would be a real
        failure against the real bucket, so it must be one here."""
        if not 1 <= len(sources) <= 32:
            raise ValueError(f"compose takes 1-32 sources, got {len(sources)}")
        missing = [s.name for s in sources if s.name not in self.bucket.objects]
        if missing:
            raise FileNotFoundError(f"no such object(s): {missing}")
        self.bucket.objects[self.name] = b"".join(
            self.bucket.objects[s.name] for s in sources)
        self.bucket.composes.append((self.name, [s.name for s in sources]))

    def download_to_filename(self, path):
        if self.name not in self.bucket.objects:
            raise FileNotFoundError(f"no such object: {self.name}")
        Path(path).write_bytes(self.bucket.objects[self.name])
        self.bucket.downloads.append(self.name)

    def delete(self):
        del self.bucket.objects[self.name]
        self.bucket.deletes.append(self.name)


class FakeBucket:
    """Only the blob operations `stage2b_gcs` uses, plus the prefix
    listing. `list_blobs` matches on a plain string prefix, as the real
    API does."""

    def __init__(self, name=gcs.GCS_BUCKET):
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
    assert gcs.GCS_BUCKET == "bonsai-2026-stage4a-cache"
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
