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
import re
import sys
from pathlib import Path

import pytest

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


def test_the_makefile_declares_the_same_project_as_the_module():
    """Two INDEPENDENT sources, which the project id did not have.

    It was pinned by `test_infrastructure_constants` as
    `gcs.GCS_PROJECT == "bonsai-504422"` -- the constant against a copy of
    itself in the test. That can only fail if someone edits the module and
    forgets the test, and anyone changing a project id greps for the old
    value and fixes both. `tools/gates/gate_inventory.py` names this exact
    shape in `break_demonstrated`: causal evidence the test fails when the
    PRODUCTION value changes, "not merely when the constant literal is
    edited, which tests that the literal equals itself".

    Now the same comparison the bucket has had all along.
    """
    declared = _make_var("BONSAI_GCP_PROJECT")
    print(f"\n[project] Makefile BONSAI_GCP_PROJECT = {declared!r}")
    print(f"[project] stage2b_gcs.GCS_PROJECT      = {gcs.GCS_PROJECT!r}")
    assert declared is not None, (
        "the Makefile no longer declares BONSAI_GCP_PROJECT; infra/ and the "
        "science module would have no common source to agree with")
    assert declared == gcs.GCS_PROJECT, (
        f"the Makefile declares {declared!r} but the module uses "
        f"{gcs.GCS_PROJECT!r}. A client constructed through a make target "
        f"and the same client constructed directly would address different "
        f"projects.")


def test_terraform_targets_the_same_project_as_the_makefile():
    """The third source. `infra/` provisions CI into a project, and the
    science module reads and writes buckets in one; if those ever differ,
    CI would be guarding infrastructure nobody uses.

    Reads the default out of `infra/variables.tf` rather than a tfvars file,
    because the default is what applies when nobody passes `-var`.
    """
    variables = REPO_ROOT / "infra" / "variables.tf"
    if not variables.exists():
        pytest.skip("infra/ is not present in this checkout")

    text = variables.read_text()
    block = re.search(r'variable\s+"project_id"\s*\{(.*?)\n\}', text, re.S)
    assert block, "infra/variables.tf no longer declares a project_id variable"
    default = re.search(r'default\s*=\s*"([^"]+)"', block.group(1))
    assert default, (
        "the project_id variable has no default, so `terraform apply` with "
        "no -var would prompt -- and nothing pins which project CI targets")

    declared = _make_var("BONSAI_GCP_PROJECT")
    print(f"\n[project] infra/variables.tf default = {default.group(1)!r}")
    print(f"[project] Makefile BONSAI_GCP_PROJECT = {declared!r}")
    assert default.group(1) == declared, (
        f"Terraform defaults to project {default.group(1)!r} but the "
        f"Makefile declares {declared!r}. CI would be provisioned into a "
        f"different project from the one the science code uses.")


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


# =====================================================================
# The Makefile exports the bucket variable; does anything actually read it?
# =====================================================================
#
# Every check above is a property of the MAKEFILE alone (does a target
# export $(GCS_ENV)/$(GCS_EXEC_ENV)) or of the MODULE alone (does
# `stage2b_gcs.DEFAULT_GCS_BUCKET` agree with the Makefile's declared
# default). None of it asks whether a DRIVER -- the script the target
# actually runs -- ever looks at the name the Makefile exports.
#
# `run_abs_conv_eps_sensitivity.py` (Stage 2B Protocol 2) is the concrete
# counterexample this section exists for. `make stage2b-protocol2`
# exported BONSAI_GCS_BUCKET correctly, every check above stayed green,
# and the driver still published every artifact LOCALLY: its own bucket
# resolution read `STAGE2B_BUCKET` and `BUCKET`, two names nothing in
# this repository ever sets, instead of `BONSAI_GCS_BUCKET` -- the one
# name the Makefile and `stage2b_gcs.bucket_name()` actually agree on. A
# guard checking only that name is DECLARED cannot see a consumer that
# spelled its own, different name for the same idea.
#
# GENERALISATION: a guard spanning two artefacts must assert the
# RELATION between them, not a property of one of them. Checking that a
# Makefile exports a name says nothing about whether anything reads it.


def _get_bucket_calls(tree):
    """Calls of the form `<expr>.get_bucket(...)` or a bare `get_bucket(...)`
    that build a LIVE client -- the same discriminator
    `_builds_a_live_gcs_client` uses, so a caller injecting a stand-in (as
    the unit tests do) is not asked to have read anything."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "get_bucket" and not any(kw.arg == "client" for kw in node.keywords):
            calls.append(node)
    return calls


def _is_environ_get_call(node):
    """Whether `node` is `os.environ.get(...)` or `environ.get(...)`
    (after `from os import environ`)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "get"):
        return False
    base = func.value
    return ((isinstance(base, ast.Attribute) and base.attr == "environ")
            or (isinstance(base, ast.Name) and base.id == "environ"))


def _module_string_constants(tree):
    """Names assigned a plain string literal at module scope, e.g.
    `ENV_BUCKET = "BONSAI_GCS_BUCKET"` in `run_arm_x86_propagation_stress.py`
    -- so a driver that names its own local alias for the env var is
    recognised too, not only one that spells the literal inline or reaches
    through `gcs.BUCKET_ENV_VAR` directly."""
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _environ_get_matches(node, env_var, constants):
    """Whether an `os.environ.get(...)` call's key names `env_var` --
    literally, through a module-level constant alias resolved by
    `_module_string_constants`, or through an attribute access ending in
    `.BUCKET_ENV_VAR` (the name `stage2b_gcs` itself uses, under whatever
    alias a driver imported the module as)."""
    if not node.args:
        return False
    key = node.args[0]
    if isinstance(key, ast.Constant) and key.value == env_var:
        return True
    if isinstance(key, ast.Attribute) and key.attr == "BUCKET_ENV_VAR":
        return True
    if isinstance(key, ast.Name) and constants.get(key.id) == env_var:
        return True
    return False


def _build_parent_map(tree):
    """`{child: parent}` over the whole tree, so a scope can be found by
    walking up from any node rather than down from the root."""
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_scope(node, parents, module):
    """The nearest enclosing function, or the module itself if `node` is
    at module scope. Variable resolution below is scoped to this --
    `bname` inside `def main(): ...` is not the same `bname` as one in a
    different function, and a name with no assignment found in scope is
    left unresolved rather than searched for elsewhere."""
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return module


def _find_assignment_values(name_id, scope):
    """Every RHS a plain `name_id = <value>` assigns within `scope`,
    at any nesting depth inside it (an `if`, a `try`, ...). Returns all
    of them rather than "the last one": which branch actually ran is not
    decidable statically, and missing a feed would fail this check open
    when it should fail closed."""
    values = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name_id:
                    values.append(node.value)
    return values


def _collect_feeding_environ_calls(value_node, scope, seen_names=None):
    """The `os.environ.get(...)` calls that can actually feed
    `value_node`'s result -- the two shapes this repository uses:

        get_bucket(name=os.environ.get(...))               -- inline
        bname = args.bucket or os.environ.get(...)
        get_bucket(name=bname)                              -- via a local

    A `Name` reference is expanded through `_find_assignment_values`
    within `scope` (recursively, so a chain of two local variables is
    still traced) and any environ.get calls reachable that way are
    included. A name with NO assignment found in scope -- a function
    parameter, an attribute, a dict subscript, anything bound outside
    `scope` -- is left unresolved: this function does not guess, so an
    unrelated `os.environ.get("BONSAI_DEBUG")` sitting elsewhere in the
    same module, with no dataflow connection to `name=`, is never
    collected here at all. `seen_names` stops infinite recursion on a
    name that (however contrived) refers to itself."""
    if seen_names is None:
        seen_names = set()
    found = []
    for node in ast.walk(value_node):
        if _is_environ_get_call(node):
            found.append(node)
        elif isinstance(node, ast.Name) and node.id not in seen_names:
            seen_names = seen_names | {node.id}
            for assigned in _find_assignment_values(node.id, scope):
                found.extend(_collect_feeding_environ_calls(assigned, scope, seen_names))
    return found


def _resolves_bucket_from_env(source_text, env_var):
    """Whether a module's bucket resolution actually reaches `env_var` --
    the relation the Makefile-only and module-only checks above cannot see
    between them.

    A file with no `get_bucket(...)` call at all (one that reaches GCS
    only through `anonymous=True` or a raw `google.cloud` import) is not
    this check's business, and passes. Nor is a `get_bucket(...)` call
    with no `name=` keyword -- it delegates entirely to
    `stage2b_gcs.bucket_name()`'s own internal resolution.

    For a call that DOES pass `name=`, only the environment reads that
    actually FEED that argument matter -- traced by
    `_collect_feeding_environ_calls`, not "any `os.environ.get(...)`
    found anywhere in the file". That distinction is the point: a module
    reading some unrelated variable (a debug flag, a feature switch) on
    its way to a `name=` that resolves correctly is not an offender, and
    an earlier, file-granularity version of this check got that wrong --
    caught in review before being committed, by feeding it exactly such
    a module and watching it flag a driver that was already correct (see
    `test_the_bucket_read_scanner_catches_the_historical_defect`).

    If no environment read feeds `name=` at all (e.g. `name=args.bucket`,
    where `--bucket` defaults to `None` and the call falls through to
    `bucket_name()`'s default), that call passes too -- it invents no
    competing name, so there is nothing here for it to get wrong. Only a
    call whose feeding reads are non-empty AND none of them name
    `env_var` is an offender -- exactly the historical Protocol 2 shape:
    `bname = args.bucket or os.environ.get("STAGE2B_BUCKET") or
    os.environ.get("BUCKET")` feeds `name=bname` with two real
    `os.environ.get(...)` calls, neither of which is `BONSAI_GCS_BUCKET`."""
    tree = ast.parse(source_text)
    calls = _get_bucket_calls(tree)
    if not calls:
        return True
    constants = _module_string_constants(tree)
    parents = _build_parent_map(tree)
    for call in calls:
        name_kw = next((kw for kw in call.keywords if kw.arg == "name"), None)
        if name_kw is None:
            continue
        scope = _enclosing_scope(call, parents, tree)
        feeds = _collect_feeding_environ_calls(name_kw.value, scope)
        if feeds and not any(_environ_get_matches(n, env_var, constants) for n in feeds):
            return False
    return True


def test_every_driver_reads_the_bucket_variable_the_makefile_exports():
    """`run_abs_conv_eps_sensitivity.py` (Stage 2B Protocol 2) exported
    BONSAI_GCS_BUCKET through `make stage2b-protocol2` correctly and
    still silently published every artifact locally, because its bucket
    resolution read `STAGE2B_BUCKET` and `BUCKET` -- names nothing sets
    -- instead of the one name the Makefile and `stage2b_gcs.bucket_name()`
    actually agree on. Every check above this one was green throughout.

    GENERALISATION: a guard spanning two artefacts must assert the
    RELATION between them, not a property of one of them. Checking that
    a Makefile exports a name says nothing about whether anything reads
    it."""
    bucket_var = gcs.BUCKET_ENV_VAR
    print(f"\n[bucket-read] canonical bucket variable = {bucket_var!r}")

    live = _live_gcs_files()
    assert live, "discovered no files building a live GCS client -- the detector has gone stale"

    offenders = []
    for path in live:
        ok = _resolves_bucket_from_env(path.read_text(), bucket_var)
        print(f"[bucket-read] {path.name}: "
              f"{'reads ' if ok else 'DOES NOT read '}{bucket_var}")
        if not ok:
            offenders.append(path.name)

    assert not offenders, (
        f"these files read from the environment to resolve a GCS bucket but never read "
        f"{bucket_var!r} anywhere while doing it: {offenders}. The Makefile can export "
        f"{bucket_var!r} correctly (checked above) and these would still ignore it -- "
        f"route bucket resolution through stage2b_gcs.get_bucket()/bucket_name(), or read "
        f"{bucket_var!r} (or stage2b_gcs.BUCKET_ENV_VAR) directly, rather than inventing a "
        f"locally-scoped environment-variable name for the same thing.")


# The historical defect, reconstructed rather than merely asserted about
# (CLAUDE.md principle 21's corollary: a guard you have not seen fail is
# not yet a guard). PROTOCOL2_BROKEN_SNIPPET is the actual body
# `run_abs_conv_eps_sensitivity.py` carried before this fix (its
# `main()`, lines 333-343 at the commit this test was added), wrapped in
# a standalone function so it parses on its own.
# PROTOCOL2_FIXED_SNIPPET is what replaced it. Both are hardcoded rather
# than fetched from git history, so this stays a live regression check
# after the fix is committed and HEAD no longer holds the broken form.
PROTOCOL2_BROKEN_SNIPPET = '''
import os


def resolve_bucket(args):
    bucket = None
    if not args.no_upload:
        bname = args.bucket or os.environ.get("STAGE2B_BUCKET") or os.environ.get("BUCKET")
        if bname:
            try:
                import stage2b_gcs as gcs
                creds = args.credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                bucket = gcs.get_bucket(name=bname, credentials=creds)
            except Exception as exc:  # noqa: BLE001
                print(f"[protocol2] bucket get failed ({exc}); local only")
                bucket = None
    return bucket
'''

PROTOCOL2_FIXED_SNIPPET = '''
import os


def resolve_bucket(args):
    bucket = None
    if not args.no_upload:
        try:
            import stage2b_gcs as gcs
            creds = args.credentials or os.environ.get(gcs.CREDENTIALS_ENV_VAR)
            bname = args.bucket or os.environ.get(gcs.BUCKET_ENV_VAR) or None
            bucket = gcs.get_bucket(name=bname, credentials=creds)
        except Exception as exc:  # noqa: BLE001
            print(f"[protocol2] bucket get failed ({exc}); local only")
            bucket = None
    return bucket
'''



# A SECOND wrong name, distinct from the historical STAGE2B_BUCKET/BUCKET
# pair, so passing this proves the scanner GENERALISES -- it is not just
# pattern-matching the two literal strings the historical bug happened to
# use (VACUOUS_TESTS category M: "a fix validated only against the case
# that prompted it").
GENERALISED_WRONG_NAME_SNIPPET = '''
import os
import stage2b_gcs as gcs


def get_bucket():
    bname = os.environ.get("MY_OWN_BUCKET_NAME")
    return gcs.get_bucket(name=bname)
'''

# The false-positive this test file shipped with in its first review round:
# an environment read that has NOTHING to do with the bucket, on a driver
# that resolves the bucket correctly by deferring to `args.bucket` (which
# defaults to None, so `get_bucket` falls through to `bucket_name()`'s own
# resolution). An earlier, file-granularity version of `_resolves_bucket_
# from_env` flagged this: it asked "does the file contain ANY os.environ.
# get call that doesn't match BONSAI_GCS_BUCKET", and BONSAI_DEBUG here
# does not, despite never feeding the bucket name at all. A guard with
# false positives like this gets switched off by the next author -- that
# is the failure mode this fixture pins against, not cosmetics.
UNRELATED_ENV_READ_SNIPPET = '''
import os
import stage2b_gcs as gcs


def f(args):
    if os.environ.get("BONSAI_DEBUG"):
        print("debug")
    return gcs.get_bucket(name=args.bucket)
'''


def test_the_bucket_read_scanner_catches_the_historical_defect():
    """The vacuity guard, in both failing directions, plus the one
    precision case that must NOT fail -- CLAUDE.md principle 21's
    corollary: a guard you have not seen fail (and not seen fail to fail)
    is not yet a guard.

    Four fixtures:
    - `PROTOCOL2_BROKEN_SNIPPET` / `PROTOCOL2_FIXED_SNIPPET`: the actual
      historical defect and its fix, REJECTED and ACCEPTED respectively.
      Without this, `offenders` being empty in the test above would be
      equally consistent with the scanner working and with it never
      finding anything.
    - `GENERALISED_WRONG_NAME_SNIPPET`: a DIFFERENT wrong env-var name,
      REJECTED. Proves the scanner checks the relation (does a real feed
      name the right thing) rather than pattern-matching the two literal
      strings STAGE2B_BUCKET/BUCKET happened to be.
    - `UNRELATED_ENV_READ_SNIPPET`: a correct driver that happens to read
      an unrelated env var, ACCEPTED. Proves the scanner is precise about
      WHICH reads it is checking -- a whole-file "any env read must
      match" version of this check passes the first three fixtures and
      fails this one, which is exactly the bug an earlier draft had."""
    bucket_var = gcs.BUCKET_ENV_VAR
    assert not _resolves_bucket_from_env(PROTOCOL2_BROKEN_SNIPPET, bucket_var), (
        "the scanner accepted the historical Protocol 2 defect (reads STAGE2B_BUCKET/"
        "BUCKET instead of BONSAI_GCS_BUCKET) -- it has gone vacuous")
    assert _resolves_bucket_from_env(PROTOCOL2_FIXED_SNIPPET, bucket_var), (
        "the scanner rejected the fixed form of Protocol 2's bucket resolution")
    assert not _resolves_bucket_from_env(GENERALISED_WRONG_NAME_SNIPPET, bucket_var), (
        "the scanner accepted a bucket name fed by MY_OWN_BUCKET_NAME, a name that is "
        "neither BONSAI_GCS_BUCKET nor anything historical -- it is pattern-matching the "
        "old bug's literal strings rather than checking the relation")
    assert _resolves_bucket_from_env(UNRELATED_ENV_READ_SNIPPET, bucket_var), (
        "the scanner rejected a driver that resolves its bucket correctly (defers to "
        "args.bucket / bucket_name()'s default) merely because the module also reads an "
        "unrelated environment variable -- this is the false-positive shape that gets a "
        "guard switched off")
