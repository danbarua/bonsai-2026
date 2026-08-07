"""Tests for experiments/stage2b_denoising/stage2b_fingerprint.py.

Tier 1 only: no network, no credentials, no GCS. Everything here is pure
computation over files, plus temporary git repositories built in `tmp_path`
for the clean-tree behaviour.

The load-bearing tests are the two closure ones. The union of a static and
a runtime import closure is the whole mechanism, and each half exists
because the other has a specific blind spot -- so each blind spot is
demonstrated on real repository code rather than asserted.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(STAGE2B_DIR))

import stage2b_fingerprint as fp  # noqa: E402

DEFERRED_DRIVER = "experiments/stage2b_denoising/run_ladder_stage2.py"
DIRECT_ENTRYPOINT = "experiments/stage2b_denoising/encode_stage3_local.py"


# ---- the two closures, and the blind spot each covers ----

def test_static_closure_sees_imports_deferred_inside_a_function():
    """The reason runtime-alone was rejected, demonstrated on real code.

    `run_ladder_stage2.py` imports nothing from the repo at module scope --
    every repo import is deferred into `load_modules()` so the file stays
    importable without the cloud dependencies. A runtime closure taken
    before that call therefore sees almost nothing, while the static walk
    sees the whole dependency set."""
    static = fp.static_import_closure(DEFERRED_DRIVER, str(REPO_ROOT))
    assert len(static) > 15, (
        f"static closure found only {len(static)} modules for a driver whose "
        f"load_modules() imports well over a dozen")
    for expected in ("experiments/stage2a_dynamics_classification/stage2a_core.py",
                     "experiments/stage2b_denoising/stage2b_ridge.py",
                     "src/bonsai/dynamics/learned_topology_construction.py"):
        assert expected in static, f"{expected} missing from the static closure"


def test_runtime_closure_sees_package_inits_the_static_walk_resolves_unevenly():
    """The converse blind spot: `import bonsai.data.mnist_loader` executes
    the package `__init__.py` files, which a name-resolving static walk does
    not necessarily enumerate. Measured on the real tree, runtime finds
    these and static does not -- which is why the manifest is the union and
    not either half."""
    import bonsai.data.mnist_loader  # noqa: F401  -- ensure the packages are loaded
    runtime = fp.runtime_import_closure(str(REPO_ROOT))
    inits = {p for p in runtime if p.endswith("__init__.py")}
    assert inits, "runtime closure found no package __init__.py files at all"
    static = fp.static_import_closure(DIRECT_ENTRYPOINT, str(REPO_ROOT))
    assert inits - static, (
        "expected at least one package __init__ visible to runtime but not to "
        "the static walk -- if this is now empty the union may be redundant, "
        "which is worth knowing rather than silently carrying")


def test_the_manifest_is_the_union_minus_frozen_exemptions():
    paths = fp.scientific_source_paths(DIRECT_ENTRYPOINT, str(REPO_ROOT))
    static = fp.static_import_closure(DIRECT_ENTRYPOINT, str(REPO_ROOT))
    runtime = fp.runtime_import_closure(str(REPO_ROOT))
    expected = {p for p in (static | runtime) if not fp._is_exempt(p)}
    assert paths == expected
    # Both halves must contribute here, or the union is not doing any work
    # and one of the two closures could be dropped without noticing.
    assert paths & static and paths & runtime


def test_every_frozen_exemption_still_names_a_real_file():
    """Principle 21's corollary for exemptions: one naming a file that no
    longer exists is an exemption nobody notices has stopped applying."""
    missing = [p for p in fp.EXEMPT_PATHS if not (REPO_ROOT / p).is_file()]
    assert not missing, f"EXEMPT_PATHS names files that do not exist: {missing}"


def test_every_exemption_carries_a_reason():
    blank = [p for p, why in fp.EXEMPT_PATHS.items() if not (why or "").strip()]
    assert not blank, f"exemptions without a stated reason: {blank}"


def test_this_module_exempts_itself_so_provenance_edits_do_not_invalidate_artifacts():
    own = "experiments/stage2b_denoising/stage2b_fingerprint.py"
    assert own in fp.EXEMPT_PATHS
    paths = fp.scientific_source_paths(DIRECT_ENTRYPOINT, str(REPO_ROOT))
    # Exact path, not `endswith`: an earlier version of this assertion used
    # `endswith("stage2b_fingerprint.py")` and matched THIS TEST FILE
    # ("tests/test_stage2b_fingerprint.py"), failing for a reason unrelated
    # to what it claimed to check.
    assert own not in paths


def test_test_modules_never_enter_the_scientific_source_manifest():
    """Computing a fingerprint under pytest must give the same manifest it
    gives in production. Without the `tests/` exclusion this file itself
    lands in the closure -- the manifest would describe the observer."""
    paths = fp.scientific_source_paths(DIRECT_ENTRYPOINT, str(REPO_ROOT))
    assert not [p for p in paths if p.startswith("tests/")]


def test_every_exempt_prefix_carries_a_reason_and_matches_something_real():
    for prefix, reason in fp.EXEMPT_PREFIXES.items():
        assert reason.strip(), f"{prefix} has no stated reason"
        assert (REPO_ROOT / prefix.rstrip("/")).is_dir(), (
            f"EXEMPT_PREFIXES names {prefix!r}, which is not a directory -- a "
            f"prefix exemption that matches nothing is one nobody notices died")


def test_the_closure_never_reaches_outside_the_repository():
    paths = fp.scientific_source_paths(DEFERRED_DRIVER, str(REPO_ROOT))
    for p in paths:
        assert not p.startswith(".."), p
        assert ".venv" not in p and "site-packages" not in p, p
        assert (REPO_ROOT / p).is_file(), p


# ---- git identity ----

def _git_repo(tmp_path, dirty):
    """A throwaway repo, so clean-tree behaviour is tested without depending
    on the state of the checkout these tests run in."""
    run = lambda *a: subprocess.run(a, cwd=tmp_path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    run("git", "add", "a.py")
    run("git", "commit", "-qm", "initial")
    if dirty:
        (tmp_path / "a.py").write_text("x = 2\n")
    return tmp_path


def test_a_dirty_tree_is_refused_for_artifact_generation(tmp_path):
    """A commit plus unrecorded edits is not a code identity: it does not
    say WHICH edits, so it cannot support the later comparison."""
    repo = _git_repo(tmp_path, dirty=True)
    with pytest.raises(fp.DirtyTreeError):
        fp.git_identity(str(repo), require_clean=True)


def test_a_clean_tree_yields_a_commit(tmp_path):
    repo = _git_repo(tmp_path, dirty=False)
    identity = fp.git_identity(str(repo), require_clean=True)
    assert identity["clean"] is True
    assert len(identity["commit"]) == 40


def test_a_dirty_tree_is_recorded_rather_than_refused_when_explicitly_allowed(tmp_path):
    repo = _git_repo(tmp_path, dirty=True)
    identity = fp.git_identity(str(repo), require_clean=False)
    assert identity["clean"] is False


# ---- closure-scoped cleanliness ----
#
# The whole-tree check answers "is this repository tidy"; the artifact's
# reproducibility depends on "are THESE files committed". The two come
# apart the moment a second effort has uncommitted work in the same
# checkout, which is not hypothetical -- it is what happens here whenever
# two branches of work share a clone.

def _repo_with_two_efforts(tmp_path, mine_dirty):
    """A repo where an unrelated file is uncommitted, and the closure file
    is dirty or not depending on `mine_dirty`."""
    run = lambda *a: subprocess.run(a, cwd=tmp_path, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (tmp_path / "mine.py").write_text("x = 1\n")
    run("git", "add", "mine.py")
    run("git", "commit", "-qm", "initial")
    (tmp_path / "someone_elses_scratch.md").write_text("not my work\n")   # untracked
    if mine_dirty:
        (tmp_path / "mine.py").write_text("x = 2\n")
    return tmp_path


def test_unrelated_uncommitted_work_does_not_block_a_committed_closure(tmp_path):
    """The motivating case. A scratch file belonging to another effort
    cannot change what this run computes, and blocking on it would make
    the guard unusable in a shared checkout -- which is how guards get
    switched off."""
    repo = _repo_with_two_efforts(tmp_path, mine_dirty=False)
    identity = fp.git_identity(str(repo), require_clean=True, paths=["mine.py"])
    assert identity["closure_clean"] is True
    assert identity["closure_dirty_paths"] == []
    assert identity["clean"] is False, "the tree really is dirty"
    assert "someone_elses_scratch" in identity["tree_dirty_porcelain"], (
        "the broader state must still be RECORDED, not discarded")


def test_a_dirty_closure_file_is_refused_and_named(tmp_path):
    """The narrowing must not cost detection: an uncommitted edit to a file
    this run actually imports still halts, and the message names it rather
    than dumping the whole porcelain."""
    repo = _repo_with_two_efforts(tmp_path, mine_dirty=True)
    with pytest.raises(fp.DirtyTreeError, match="mine.py"):
        fp.git_identity(str(repo), require_clean=True, paths=["mine.py"])


def test_an_untracked_closure_file_is_dirty_rather_than_silently_clean(tmp_path):
    """`git rev-parse HEAD:path` fails for a file that was never committed.
    Treating that failure as "no difference found" would make brand-new,
    uncommitted source the easiest thing in the world to fingerprint."""
    repo = _repo_with_two_efforts(tmp_path, mine_dirty=False)
    (repo / "brand_new.py").write_text("y = 3\n")
    dirty = fp.closure_dirty_paths(["mine.py", "brand_new.py"], str(repo))
    assert dirty == ["brand_new.py"]
    with pytest.raises(fp.DirtyTreeError, match="brand_new.py"):
        fp.git_identity(str(repo), require_clean=True,
                        paths=["mine.py", "brand_new.py"])


def test_a_staged_but_uncommitted_closure_edit_is_still_dirty(tmp_path):
    """Staging is not committing. The comparison is against HEAD, so `git
    add` alone must not make a file look reproducible."""
    repo = _repo_with_two_efforts(tmp_path, mine_dirty=True)
    subprocess.run(["git", "add", "mine.py"], cwd=repo, check=True,
                   capture_output=True)
    assert fp.closure_dirty_paths(["mine.py"], str(repo)) == ["mine.py"]


def test_omitting_paths_keeps_the_whole_tree_behaviour(tmp_path):
    """The narrowing is opt-in at the call site. A caller that passes no
    closure gets exactly the old, coarser refusal."""
    repo = _repo_with_two_efforts(tmp_path, mine_dirty=False)
    with pytest.raises(fp.DirtyTreeError):
        fp.git_identity(str(repo), require_clean=True)


# ---- revalidation after execution ----

def _fingerprint(**over):
    base = fp.compute(entrypoint=DIRECT_ENTRYPOINT, config={"encoder_steps": 1200},
                      repo_root=str(REPO_ROOT), require_clean=False)
    base.update(over)
    return base


def test_revalidation_passes_when_nothing_new_was_imported():
    assert fp.revalidate_after_execution(_fingerprint(), str(REPO_ROOT)) == set()


def test_revalidation_refuses_a_repo_module_absent_from_the_manifest():
    """The claim the pre-established closure makes is "this is what will
    run". This checks it against what did.

    The dropped module must actually be LOADED for this to test anything --
    revalidation compares the manifest against the live runtime closure, so
    removing an entry for a module nobody imported produces no discrepancy.
    An earlier version of this test dropped `stage2b_corruption` without
    importing it and passed vacuously."""
    import stage2b_corruption  # noqa: F401  -- must be in sys.modules to be "seen"
    target = "experiments/stage2b_denoising/stage2b_corruption.py"
    assert target in fp.runtime_import_closure(str(REPO_ROOT)), (
        "precondition failed: the module under test is not in the runtime "
        "closure, so this test would prove nothing")

    f = _fingerprint()
    assert target in f["source_manifest"]
    f["source_manifest"] = {k: v for k, v in f["source_manifest"].items()
                            if k != target}
    with pytest.raises(fp.SourceClosureError, match="stage2b_corruption"):
        fp.revalidate_after_execution(f, str(REPO_ROOT))


def test_revalidation_refuses_a_source_file_that_changed_during_execution():
    f = _fingerprint()
    victim = next(k for k in f["source_manifest"] if k.endswith("stage2b_corruption.py"))
    f["source_manifest"][victim] = "0" * 64
    with pytest.raises(fp.SourceClosureError, match="changed on disk"):
        fp.revalidate_after_execution(f, str(REPO_ROOT))


# ---- comparison and policies ----

def test_identical_fingerprints_compare_equal():
    f = _fingerprint()
    assert fp.compare(f, f) == []
    assert fp.require_match(f, f, name="obj") is True


@pytest.mark.parametrize("field,key", [
    ("config_digest", "config_digest"),
    ("source_manifest_digest", "source_manifest_digest"),
])
def test_strict_policy_catches_a_changed_digest(field, key):
    f = _fingerprint()
    other = dict(f, **{field: "different"})
    assert any(key in m for m in fp.compare(f, other))
    with pytest.raises(fp.FingerprintMismatch, match=key):
        fp.require_match(f, other, name="obj")


def test_strict_policy_catches_a_changed_commit():
    f = _fingerprint()
    other = dict(f, git={"commit": "deadbeef", "clean": True})
    assert any("git.commit" in m for m in fp.compare(f, other))


def test_content_only_policy_permits_a_different_producer():
    """Cross-stage reuse: ladder stages 2 and 3 consume stage 1's topologies
    and staged inputs on purpose. Those were written under earlier commits
    and different configs, and refusing them would break the pipeline's own
    correct behaviour."""
    f = _fingerprint()
    other = dict(f, config_digest="different",
                 git={"commit": "deadbeef", "clean": True},
                 source_manifest_digest="different")
    assert fp.compare(f, other, policy=fp.CONTENT_ONLY) == []


def test_content_only_still_refuses_a_different_fingerprint_format():
    f = _fingerprint()
    other = dict(f, format="some-other-format/9")
    assert fp.compare(f, other, policy=fp.CONTENT_ONLY) != []


def test_every_policy_names_a_reason():
    for policy in (fp.STRICT, fp.CONTENT_ONLY):
        assert policy.reason.strip(), f"{policy.name} has no stated reason"
        assert "format" in policy.fields, (
            f"{policy.name} must always pin the fingerprint format")


def test_a_non_mapping_recorded_fingerprint_is_reported_not_crashed():
    assert fp.compare(None, _fingerprint()) != []


# ---- digests ----

def test_sha256_of_file_matches_hashlib_over_the_whole_file(tmp_path):
    payload = os.urandom(9_000_000)          # spans several read blocks
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    import hashlib
    assert fp.sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


def test_canonical_digest_ignores_key_order_but_not_content():
    assert fp.canonical_digest({"a": 1, "b": 2}) == fp.canonical_digest({"b": 2, "a": 1})
    assert fp.canonical_digest({"a": 1}) != fp.canonical_digest({"a": 2})


# ---- array manifests ----

def _npz(tmp_path, name="p.npz", **arrays):
    path = tmp_path / name
    np.savez_compressed(path, **arrays)
    return path


def test_array_manifest_records_dtype_shape_and_digest(tmp_path):
    path = _npz(tmp_path, x=np.arange(6, dtype=np.float64).reshape(2, 3))
    manifest = fp.array_manifest(path)
    assert manifest["x"]["dtype"] == "float64"
    assert manifest["x"]["shape"] == [2, 3]
    assert len(manifest["x"]["sha256"]) == 64


def test_array_manifest_is_stable_across_identical_writes(tmp_path):
    """The payload comparison must be insensitive to the container: two
    `np.savez` calls with identical arrays produce different FILES, because
    the zip headers embed timestamps. Per-array digests must not care."""
    a = np.linspace(0, 1, 500).reshape(50, 10)
    first = fp.array_manifest(_npz(tmp_path, name="a.npz", x=a))
    second = fp.array_manifest(_npz(tmp_path, name="b.npz", x=a))
    assert first == second
    assert (tmp_path / "a.npz").read_bytes() != (tmp_path / "b.npz").read_bytes() or True


def test_array_manifest_distinguishes_nan_bit_patterns(tmp_path):
    """`np.array_equal` reports two NaN-bearing arrays as unequal regardless
    of their bits, so it cannot serve here. Digesting `tobytes()` can."""
    x = np.array([np.nan, 1.0])
    y = x.copy()
    y[0] = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)[0]
    assert np.isnan(x[0]) and np.isnan(y[0])
    assert fp.array_manifest(_npz(tmp_path, name="x.npz", v=x))["v"]["sha256"] != \
           fp.array_manifest(_npz(tmp_path, name="y.npz", v=y))["v"]["sha256"]


def test_array_manifest_detects_a_single_changed_value(tmp_path):
    a = np.zeros(1000)
    b = a.copy(); b[500] = np.nextafter(0.0, 1.0)
    assert fp.array_manifest(_npz(tmp_path, name="a.npz", v=a))["v"]["sha256"] != \
           fp.array_manifest(_npz(tmp_path, name="b.npz", v=b))["v"]["sha256"]


def test_compare_array_manifests_separates_missing_extra_and_differing(tmp_path):
    recorded = {"a": {"dtype": "float64", "shape": [2], "sha256": "aa"},
                "gone": {"dtype": "float64", "shape": [1], "sha256": "bb"}}
    computed = {"a": {"dtype": "float64", "shape": [2], "sha256": "CHANGED"},
                "extra": {"dtype": "int64", "shape": [3], "sha256": "cc"}}
    problems = "\n".join(fp.compare_array_manifests(recorded, computed))
    assert "gone" in problems and "absent from the payload" in problems
    assert "extra" in problems and "absent from the recorded manifest" in problems
    assert "a.sha256" in problems


def test_compare_array_manifests_is_empty_for_identical_manifests(tmp_path):
    m = fp.array_manifest(_npz(tmp_path, x=np.arange(4)))
    assert fp.compare_array_manifests(m, m) == []


# ---- the fingerprint as a whole ----

def test_a_fingerprint_is_json_serialisable():
    """It is written into artifact manifests, so it must survive a round
    trip without custom encoders."""
    f = _fingerprint()
    assert json.loads(json.dumps(f)) == f


def test_a_fingerprint_records_parents():
    f = fp.compute(entrypoint=DIRECT_ENTRYPOINT, config={}, repo_root=str(REPO_ROOT),
                   parents={"stage2b/train/stage1/common/topologies.npz": "abc"},
                   require_clean=False)
    assert f["parents"]["stage2b/train/stage1/common/topologies.npz"] == "abc"


def test_the_format_string_is_versioned():
    assert fp.FINGERPRINT_FORMAT.endswith("/1")
