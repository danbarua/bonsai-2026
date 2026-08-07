"""The Makefile and `stage2b_gcs` must agree about the bucket.

Moving the bucket name into a Makefile variable makes it overridable
without editing Python, which is what it is for -- and creates a second
place the name is written down, which is what this file is for. The two
declarations are independent strings in different languages, so nothing
but a test stops them drifting apart, and the failure mode is quiet: a
target exports one bucket, a script that was invoked some other way
defaults to the other, and artifacts land in two places with no error
anywhere.

That is the same shape as the bug that prompted this rename. The name
lived in `stage2b_gcs.py`, `README.md` and a test assertion; the test
pinned it, so it looked covered, but the pin simply restated the module
and would have gone on passing had any *other* copy been wrong.

CLAUDE.md principle 20: this is the executable form of a fact that would
otherwise live in a Makefile comment nobody re-checks. Tier 1 throughout
-- parses two files, touches no network and provisions nothing.
"""
import ast
import sys
from pathlib import Path

from _makefile import REPO_ROOT, make_var as _make_var, recipes as _recipes

sys.path.insert(0, str(REPO_ROOT / "experiments" / "stage2b_denoising"))

import stage2b_gcs as gcs  # noqa: E402


def test_the_makefile_declares_the_same_default_bucket_as_the_module():
    declared = _make_var("BONSAI_GCS_BUCKET")
    print(f"\n[bucket] Makefile BONSAI_GCS_BUCKET = {declared!r}")
    print(f"[bucket] stage2b_gcs.DEFAULT_GCS_BUCKET = {gcs.DEFAULT_GCS_BUCKET!r}")
    assert declared is not None, (
        "the Makefile no longer declares BONSAI_GCS_BUCKET; the GCS targets would fall "
        "back to the module default silently")
    assert declared == gcs.DEFAULT_GCS_BUCKET, (
        f"the Makefile exports {declared!r} but the module defaults to "
        f"{gcs.DEFAULT_GCS_BUCKET!r}. A script run through a target and the same script "
        f"run directly would use different buckets.")


def test_the_declared_bucket_is_a_name_the_resolver_accepts():
    """Guards against a rename that is valid Make and invalid GCS -- which
    would only fail at the point of touching the real bucket."""
    declared = _make_var("BONSAI_GCS_BUCKET")
    assert gcs.bucket_name(env={gcs.BUCKET_ENV_VAR: declared}) == declared


def test_the_stage2b_test_target_lists_every_stage2b_test_file():
    """`make stage2b-test` runs an explicit file list, so a new test file
    is picked up by the whole-suite target and silently skipped by the
    Stage-2B one -- green either way, and only the glob was ever proving
    anything.

    That is the `stage2a-verify` no-op-gate shape again: a target that
    looks like it covers a thing and does not. Both
    `test_stage2b_contracts.py` and this file were missing from the list
    when the check was written."""
    declared = _make_var("STAGE2B_TEST_FILES")
    assert declared is not None, "STAGE2B_TEST_FILES is no longer declared"
    listed = {Path(tok).name for tok in declared.split() if tok.endswith(".py")}
    on_disk = {p.name for p in (REPO_ROOT / "tests").glob("test_stage2b_*.py")}

    print(f"\n[bucket] STAGE2B_TEST_FILES lists {len(listed)} files, "
          f"{len(on_disk)} on disk")
    missing = sorted(on_disk - listed)
    for name in sorted(on_disk):
        print(f"[bucket]   {'ok  ' if name in listed else 'MISS'} {name}")

    assert not missing, (
        f"these Stage 2B test files exist but `make stage2b-test` does not run them: "
        f"{missing}")
    stale = sorted(listed - on_disk)
    assert not stale, (
        f"STAGE2B_TEST_FILES names files that do not exist, so the target fails to "
        f"collect: {stale}")



def _builds_a_live_gcs_client(path):
    """Whether this file constructs a real GCS client, by reading its AST.

    Discovered rather than listed. An allowlist of known GCS scripts would
    not catch the next one -- it would simply not look at it, and pass
    vacuously for whatever target runs it. The ladder driver is exactly
    that next one.

    `get_bucket` is the single chokepoint: every transport function takes
    an already-built `bucket`, so a file that reaches GCS calls it. The
    discriminator is whether the call passes `client=` -- a caller
    injecting a stand-in (as the unit tests do) never opens a socket,
    while one that does not gets a live client built from credentials or
    an anonymous session. That is the real semantic, not a proxy for it.

    Direct `google.cloud` imports count too, so a file bypassing this
    module entirely is still seen.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "get_bucket":
                if not any(kw.arg == "client" for kw in node.keywords):
                    return True
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("google.cloud") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("google.cloud"):
                return True
    return False


def _live_gcs_files():
    """Stage 2B files that reach real GCS, discovered by AST.

    `stage2b_gcs.py` is excluded: it is the module that *defines* the
    client construction, so it necessarily imports the library.
    """
    candidates = sorted(
        list((REPO_ROOT / "experiments" / "stage2b_denoising").glob("*.py"))
        + list((REPO_ROOT / "tests").glob("test_stage2b_*.py")))
    return [p for p in candidates
            if p.name != "stage2b_gcs.py" and _builds_a_live_gcs_client(p)]


# Runs on the Colab VM, uploaded and executed by the round-trip test
# rather than invoked by a target here. It is covered transitively, so it
# is not expected to appear in any recipe. Every other live-GCS file must.
REMOTE_EXECUTED = {"colab_gcs_roundtrip_probe.py"}


def test_every_target_running_a_live_gcs_file_exports_the_bucket():
    """`GCS_ENV` carries both the bucket and the credentials path. A target
    that reaches GCS must pass them explicitly rather than inheriting
    whatever the ambient environment holds -- otherwise `make` and a bare
    `uv run` of the same script disagree about which bucket they mean.

    A recipe that runs its script on a Colab runtime satisfies this with
    `GCS_EXEC_ENV` instead, because `GCS_ENV` sets variables in the LOCAL
    make shell and a remote kernel never sees them. The two forms are not
    interchangeable and the test does not treat them as such: the remote
    form counts only for a recipe that actually execs. A locally-run script
    still has to export `GCS_ENV`, so widening this check does not let the
    original requirement be satisfied by the wrong mechanism."""
    gcs_env = _make_var("GCS_ENV")
    assert gcs_env is not None and "BONSAI_GCS_BUCKET" in gcs_env, (
        f"GCS_ENV should carry the bucket; got {gcs_env!r}")
    gcs_exec_env = _make_var("GCS_EXEC_ENV")
    assert gcs_exec_env is not None and "BONSAI_GCS_BUCKET" in gcs_exec_env, (
        f"GCS_EXEC_ENV should carry the bucket; got {gcs_exec_env!r}")
    assert "BONSAI_GCS_CREDENTIALS" in gcs_exec_env, (
        f"GCS_EXEC_ENV should carry the credentials path the key was uploaded to; "
        f"got {gcs_exec_env!r}")

    live = _live_gcs_files()
    print(f"\n[bucket] files building a live GCS client (AST-discovered):")
    for path in live:
        print(f"[bucket]   {path.relative_to(REPO_ROOT)}")
    assert live, (
        "discovered no files building a live GCS client -- the detector has gone stale "
        "and every check below is vacuous")

    recipes = _recipes()
    offenders, orphans = [], []
    for path in live:
        if path.name in REMOTE_EXECUTED:
            print(f"[bucket] {path.name}: runs on the VM, covered by the round trip")
            continue
        naming = {t: body for t, body in recipes.items() if path.name in body}
        if not naming:
            orphans.append(path.name)
            continue
        for target, body in sorted(naming.items()):
            # Which form is acceptable is decided by what the recipe DOES,
            # not by which file it names -- so a target cannot claim the
            # remote form without actually running anything remotely.
            remote = ") exec " in body
            required = "$(GCS_EXEC_ENV)" if remote else "$(GCS_ENV)"
            ok = required in body
            print(f"[bucket] {target} -> {path.name}: "
                  f"{'exports' if ok else 'DOES NOT export'} {required} "
                  f"({'remote exec' if remote else 'local'})")
            if not ok:
                offenders.append(f"{target} (runs {path.name}, needs {required})")

    assert not offenders, (
        f"these targets reach GCS without exporting the bucket and credentials: "
        f"{offenders}. They would use whatever the ambient environment holds.")
    assert not orphans, (
        f"these files build a live GCS client but no target runs them: {orphans}. "
        f"Either add a target that exports $(GCS_ENV) (or $(GCS_EXEC_ENV), if it "
        f"execs the script on a runtime), or -- if it is executed remotely like the "
        f"round-trip probe -- add it to REMOTE_EXECUTED with a reason.")


def test_the_remote_executed_exemption_does_not_rot():
    """An exemption naming a file that no longer exists is an exemption
    nobody notices is dead."""
    names = {p.name for p in (REPO_ROOT / "experiments" / "stage2b_denoising").glob("*.py")}
    missing = sorted(REMOTE_EXECUTED - names)
    assert not missing, f"REMOTE_EXECUTED names files that no longer exist: {missing}"


# =====================================================================
# One way in: nothing consumes GCS bytes except through consume_validated
# =====================================================================
#
# `consume_validated` is only "THE central validated consume path" if
# nothing routes around it. A raw `download_file` reaches bytes without
# ever consulting a manifest, which is precisely the bypass the contract
# exists to close -- and a bypass is invisible from a green suite, because
# the code still works. It just works unvalidated.
#
# Discovered by AST rather than listed. An allowlist of "scripts we know
# download things" would not catch the next driver; it would simply not
# look at it. That is the failure this project has now produced five
# times.

# Files permitted to call `download_file` directly, each with a reason.
# A test below asserts every entry still exists AND still contains such a
# call -- an exemption for a file that stopped needing it is an exemption
# hiding the next real bypass.
RAW_TRANSPORT_EXEMPT = {
    "stage2b_gcs.py":
        "defines both functions; consume_validated calls download_file by "
        "construction",
    "smoke_stage2b_gcs.py":
        "deliberately exercises RAW transport against a real bucket -- "
        "round-trip, corruption and chunk-resume checks whose subject is "
        "download_file itself. Routing it through the validated path would "
        "test the wrong function",
}


def _calls_download_file(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "download_file":
                return True
    return False


def _stage2b_scripts():
    return sorted((REPO_ROOT / "experiments" / "stage2b_denoising").glob("*.py"))


def test_no_stage2b_script_downloads_around_the_validated_consume_path():
    offenders = []
    for path in _stage2b_scripts():
        if not _calls_download_file(path):
            continue
        if path.name in RAW_TRANSPORT_EXEMPT:
            print(f"[consume] {path.name}: raw download, exempt "
                  f"({RAW_TRANSPORT_EXEMPT[path.name][:40]}...)")
            continue
        offenders.append(path.name)
    print(f"[consume] {len(_stage2b_scripts())} scripts scanned, "
          f"{len(offenders)} bypassing consume_validated")
    assert not offenders, (
        f"these scripts call download_file directly, reaching bytes without "
        f"consulting a manifest: {offenders}. Use consume_validated -- it "
        f"downloads when the local file is absent, so there is nothing a raw "
        f"download adds except the bypass. If the raw call is genuinely the "
        f"subject of the code, add it to RAW_TRANSPORT_EXEMPT with a reason.")


def test_the_scanner_would_actually_find_a_bypass():
    """The vacuity guard. Every assertion above is over a discovered set,
    and a scanner that finds nothing passes it trivially -- so check that
    the detector fires on a file known to contain the call."""
    smoke = REPO_ROOT / "experiments" / "stage2b_denoising" / "smoke_stage2b_gcs.py"
    assert smoke.exists()
    assert _calls_download_file(smoke), "the AST scanner no longer detects the call"
    scanned = _stage2b_scripts()
    assert len(scanned) >= 10, f"only {len(scanned)} scripts discovered"


def test_every_raw_transport_exemption_still_names_a_file_that_needs_it():
    """Both directions. An exemption naming a deleted file, or one that no
    longer calls download_file, is dead -- and a dead exemption is where
    the next real bypass hides."""
    for name, reason in RAW_TRANSPORT_EXEMPT.items():
        assert reason, f"{name} is exempt with no reason"
        path = REPO_ROOT / "experiments" / "stage2b_denoising" / name
        assert path.exists(), f"exemption {name} names a file that no longer exists"
        assert _calls_download_file(path), (
            f"{name} is exempted from the consume-path rule but no longer calls "
            f"download_file -- remove the exemption rather than leaving it to "
            f"cover something else later")
