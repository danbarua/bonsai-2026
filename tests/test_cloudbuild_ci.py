"""The two CI guards, and proof that each one fails when it should.

`cloudbuild.yaml` rests on two claims: that a build cannot invoke a target
which spends, and that a green build cannot be a mass-skip. Both are
enforced by code under `tools/ci/`, and CLAUDE.md principle 21's corollary
is the reason this file is longer than that code: a guard nobody has
watched fail is not yet a guard. Every check below that asserts a clean
result has a partner that breaks what it watches and asserts the specific
expected complaint.

The spend guard is checked against SYNTHETIC Makefile recipes as well as
the real one. Testing a narrowing through the broader thing it narrows --
running the real Makefile past a check and calling that evidence the check
discriminates -- is the second half of principle 21, and the reason
`STAGE2B_TEST_FILES`'s gap stayed invisible under `pytest tests/`.

Nothing here runs `make`, `gcloud`, or a network call.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUDBUILD = REPO_ROOT / "cloudbuild.yaml"


def _load(name):
    path = REPO_ROOT / "tools" / "ci" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ci_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ci_targets = _load("ci_targets")
vacuity = _load("check_suite_not_vacuous")
credentials = _load("assert_no_cloud_credentials")


# --------------------------------------------------------------- fixtures

# A minimal stand-in for the Makefile: one billing target, one bucket
# target, one honest test runner. Written out rather than sliced from the
# real file so a check can be shown to discriminate between them.
SYNTHETIC_RECIPES = {
    "test": '\tcd $(REPO_ROOT) && uv run pytest tests/ -m "not slow" -q',
    "burn-money": "\t$(MIGHTY_COLAB) new -s x --gpu A100",
    "touch-bucket": "\t$(GCS_ENV) uv run python upload.py",
}


def junit_xml(cases):
    """A JUnit report from (classname, name, skip_reason_or_None) triples."""
    body = []
    skipped = sum(1 for _, _, reason in cases if reason is not None)
    for classname, name, reason in cases:
        if reason is None:
            body.append(f'<testcase classname="{classname}" name="{name}"/>')
        else:
            body.append(f'<testcase classname="{classname}" name="{name}">'
                        f'<skipped message="{reason}"/></testcase>')
    return (f'<testsuite name="pytest" errors="0" failures="0" '
            f'skipped="{skipped}" tests="{len(cases)}">' + "".join(body) + "</testsuite>")


@pytest.fixture
def report(tmp_path):
    def write(cases):
        path = tmp_path / "junit.xml"
        path.write_text(junit_xml(cases))
        return path
    return write


@pytest.fixture
def baseline(tmp_path):
    def write(keys):
        path = tmp_path / "baseline.txt"
        path.write_text("# test baseline\n" + "\n".join(keys) + "\n")
        return path
    return write


# ------------------------------------------------- the spend guard, clean

def test_the_real_build_config_invokes_only_allowlisted_targets():
    assert ci_targets.check(CLOUDBUILD) == []


def test_the_derived_spending_set_is_not_empty():
    """If the Makefile parser ever returns nothing, every disjointness check
    below passes for the wrong reason. That is the shape of a vacuous gate,
    so the derivation's own output is asserted before it is trusted."""
    spending = ci_targets.spending_targets()
    assert len(spending) > 10, (
        f"only {len(spending)} billable targets derived from the Makefile. The "
        f"parser or the marker list has drifted; a near-empty spending set makes "
        f"the disjointness check meaningless.")
    assert "stage2b-ladder-stage3" in spending
    assert "stage2b-verify-gpu" in spending


def test_every_allowlisted_target_exists_and_only_runs_pytest():
    import _makefile

    recipes = _makefile.recipes()
    for name in ci_targets.CI_INVOCABLE_TARGETS:
        assert name in recipes, f"allowlisted target `{name}` is not in the Makefile"
        assert "uv run pytest" in recipes[name]


def test_allowlist_and_derived_spending_set_are_disjoint():
    assert not (ci_targets.CI_INVOCABLE_TARGETS & ci_targets.spending_targets())


# ------------------------------------------- the spend guard, broken on purpose

def test_a_new_billing_target_on_the_allowlist_is_caught(tmp_path):
    """The case principle 21 says a hand-written denylist would miss: a
    target that did not exist when the list was written."""
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps:\n  - args:\n      - |\n        make burn-money\n")
    problems = ci_targets.check(
        config, allowlist=frozenset({"test", "burn-money"}), recipes=SYNTHETIC_RECIPES)
    assert any("billable infrastructure" in p for p in problems), problems


def test_a_new_test_suite_that_ci_never_runs_is_caught(tmp_path, monkeypatch):
    """The under-coverage direction, and the one an allowlist cannot see.

    An allowlist answers "may CI run this?" and fails safe -- forget a
    target and nothing spends. It fails UNSAFE the other way: add
    `stage3-test` tomorrow, forget the allowlist, and CI silently never runs
    that suite. Under-coverage wearing a green badge, which is the shape of
    `STAGE2B_TEST_FILES`. So the safe set is derived from recipe text and a
    derived runner must be invoked or dispositioned.
    """
    recipes = dict(SYNTHETIC_RECIPES)
    recipes["stage3-test"] = "\tcd $(REPO_ROOT) && uv run pytest tests/x.py -v\n"
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps:\n  - args:\n      - |\n        make test\n")
    monkeypatch.setattr(ci_targets, "NOT_RUN_IN_CI", {})
    problems = ci_targets.check(
        config, allowlist=frozenset({"test"}), recipes=recipes)
    assert any("stage3-test" in p and "neither invoked" in p
               for p in problems), problems


def test_a_new_test_suite_that_is_explicitly_declined_is_not_caught(
        tmp_path, monkeypatch):
    """Non-vacuity for the case above: it would pass on a check that
    complained about every derived runner unconditionally."""
    recipes = dict(SYNTHETIC_RECIPES)
    recipes["stage3-test"] = "\tcd $(REPO_ROOT) && uv run pytest tests/x.py -v\n"
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps:\n  - args:\n      - |\n        make test\n")
    monkeypatch.setattr(ci_targets, "NOT_RUN_IN_CI",
                        {"stage3-test": "covered by `test`"})
    problems = ci_targets.check(
        config, allowlist=frozenset({"test"}), recipes=recipes)
    assert not any("stage3-test" in p for p in problems), problems


def test_declining_a_target_that_stopped_being_a_test_runner_is_caught(
        tmp_path, monkeypatch):
    """An exemption concealing whatever the target became.

    The same rule the registration guard applies to its own exemptions: an
    entry for something that changed shape excuses exactly the thing worth
    catching.
    """
    recipes = dict(SYNTHETIC_RECIPES)
    recipes["stage3-test"] = "\t$(MIGHTY_COLAB) exec -f driver.py\n"
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps:\n  - args:\n      - |\n        make test\n")
    monkeypatch.setattr(ci_targets, "NOT_RUN_IN_CI",
                        {"stage3-test": "covered by `test`"})
    problems = ci_targets.check(
        config, allowlist=frozenset({"test"}), recipes=recipes)
    assert any("no longer looks like one" in p for p in problems), problems


def test_a_build_config_invoking_an_unlisted_target_is_caught(tmp_path):
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps:\n  - args:\n      - |\n        make touch-bucket\n")
    problems = ci_targets.check(
        config, allowlist=frozenset({"test"}), recipes=SYNTHETIC_RECIPES)
    assert any("non-allowlisted" in p for p in problems), problems


def test_naming_a_spending_target_even_in_a_comment_is_caught(tmp_path):
    """The raw-text check cannot tell a comment from a command, and that is
    the safe direction for a spend guard: over-flagging costs a doc edit,
    under-flagging costs an A100."""
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("# someday this should also run burn-money\nsteps: []\n")
    problems = ci_targets.check(
        config, allowlist=frozenset({"test"}), recipes=SYNTHETIC_RECIPES)
    assert any("mentions spending target" in p for p in problems), problems


def test_an_allowlisted_target_that_stops_being_a_test_runner_is_caught(tmp_path):
    """Guards the other direction: the allowlist is unchanged, and the
    target underneath it turned into a driver invocation."""
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps: []\n")
    recipes = dict(SYNTHETIC_RECIPES, test="\t$(PYTHON) experiments/run_something.py")
    problems = ci_targets.check(config, allowlist=frozenset({"test"}), recipes=recipes)
    assert any("no longer runs" in p for p in problems), problems
    assert any("$(PYTHON)" in p for p in problems), problems


def test_a_gpu_target_appended_to_the_real_makefile_is_derived_as_billable(
        tmp_path, monkeypatch):
    """Drives the real parser, not a hand-built recipes dict.

    The synthetic cases above prove `check` discriminates; this proves the
    derivation reaches a target that did not exist when the allowlist was
    written, which is the property principle 21 says a hand-maintained
    denylist would not have. The Makefile on disk is not touched -- a copy
    with one target appended is parsed instead.
    """
    import _makefile

    copied = tmp_path / "Makefile"
    copied.write_text(
        _makefile.MAKEFILE.read_text()
        + "\n.PHONY: stage2b-ladder-stage4\n"
          "stage2b-ladder-stage4:  ## a target nobody has written yet\n"
          "\t$(MIGHTY_COLAB) new -s ladder4 --gpu A100 --timeout $(EXEC_TIMEOUT)\n")
    monkeypatch.setattr(_makefile, "MAKEFILE", copied)

    spending = ci_targets.spending_targets()
    assert "stage2b-ladder-stage4" in spending, (
        "a newly added GPU target was not derived as billable -- the marker list "
        "or the parser has drifted, and the guard would let CI invoke it")

    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps:\n  - args:\n      - |\n        make stage2b-ladder-stage4\n")
    problems = ci_targets.check(config)
    assert any("non-allowlisted" in p for p in problems), problems


def test_an_allowlist_naming_a_renamed_target_is_caught(tmp_path):
    config = tmp_path / "cloudbuild.yaml"
    config.write_text("steps: []\n")
    problems = ci_targets.check(
        config, allowlist=frozenset({"test", "test-renamed-away"}),
        recipes=SYNTHETIC_RECIPES)
    assert any("do not exist in the Makefile" in p for p in problems), problems


# ------------------------------------------------ the vacuity guard, clean

def test_a_run_matching_its_baseline_passes(report, baseline):
    path = report([("tests.test_a", "test_one", None),
                   ("tests.test_a", "test_two", "no datasets")])
    base = baseline(["tests.test_a::test_two"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 0


def test_the_committed_baseline_names_only_test_files_that_exist():
    """A baseline entry pointing at a renamed module records a gap that no
    longer has the shape it claims, and would be reported as `vanished` on
    the next full run anyway -- catching it here says which entry."""
    keys = vacuity.read_baseline(vacuity.DEFAULT_BASELINE)
    assert keys, "the committed baseline is empty -- it would pass every build"
    for key in keys:
        module = key.split("::", 1)[0]
        path = REPO_ROOT / (module.replace(".", "/") + ".py")
        assert path.exists(), f"baseline entry {key} names a module that is gone"


def test_the_committed_baseline_covers_every_capability_ci_lacks():
    """The measured skip set, read back as a claim about why CI skips.

    Not a count: a count would let one reason quietly replace another. Each
    of these is a capability the build environment does not have, and if a
    whole class of them disappeared from the baseline that is a change in
    what CI covers, not a tidy-up.

    REMOVED 2026-08-08: `google_crc32c`. This guard fired when its baseline
    entry went, and its own message named the two possibilities -- CI gained
    the capability, or the baseline was regenerated where it should not have
    been. It is the first: `google-crc32c` moved into the `dev` group, which
    CI installs, so `test_crc32c_agrees_with_google_crc32c_where_it_is_installed`
    now RUNS in CI rather than skipping.

    Recorded here rather than deleted silently, because deleting an entry
    from a list to make a guard pass is the move this whole file exists to
    catch. The distinction that makes it legitimate: the list is a claim
    about what CI lacks, and CI stopped lacking this -- deliberately, in a
    change whose reasoning is in `pyproject.toml` beside the dependency.
    The library is a checksum wheel: no credentials, no cost, no cloud CLI,
    and nothing about the credential-free profile changes by having it.
    """
    text = vacuity.DEFAULT_BASELINE.read_text()
    for reason in ("datasets/kmnist not present",
                   "not present locally",
                   "mighty-colab"):
        assert reason in text, (
            f"no baseline entry mentions {reason!r}. Either CI gained that "
            f"capability -- in which case say so -- or the baseline was "
            f"regenerated somewhere it should not have been.")


# ----------------------------------------- the vacuity guard, broken on purpose

def test_a_newly_skipping_test_fails_the_build(report, baseline):
    """The headline case: the suite is green and covers less than it did."""
    path = report([("tests.test_a", "test_one", "credentials missing"),
                   ("tests.test_a", "test_two", "no datasets")])
    base = baseline(["tests.test_a::test_two"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 1


def test_a_swap_that_leaves_the_counts_identical_still_fails(report, baseline):
    """One test starts skipping while another stops. Passed and skipped
    totals are unchanged, so a floor-and-ceiling guard reports OK. This is
    why the check compares identities."""
    path = report([("tests.test_a", "test_one", "credentials missing"),
                   ("tests.test_a", "test_two", None)])
    base = baseline(["tests.test_a::test_two"])
    outcomes, totals = vacuity.parse_junit(path)
    assert totals["skipped"] == 1
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 1


def test_a_stale_baseline_entry_that_now_runs_fails(report, baseline):
    path = report([("tests.test_a", "test_one", None),
                   ("tests.test_a", "test_two", None)])
    base = baseline(["tests.test_a::test_two"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 1


def test_a_baseline_entry_whose_test_vanished_fails(report, baseline):
    path = report([("tests.test_a", "test_one", None)])
    base = baseline(["tests.test_a::test_gone"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 1


def test_an_empty_report_fails(report, baseline):
    path = report([])
    base = baseline(["tests.test_a::test_two"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 1


def test_a_subset_run_is_scoped_to_the_modules_it_collected(report, baseline):
    """The fast tier runs some files and not others. A baseline entry from
    an uncollected module is out of scope, not missing -- otherwise one
    measured baseline could not serve both tiers and the second copy would
    drift from the first."""
    path = report([("tests.test_a", "test_one", None),
                   ("tests.test_a", "test_two", "no datasets")])
    base = baseline(["tests.test_a::test_two", "tests.test_b::test_three"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base) == 0


def test_a_module_that_stops_importing_is_caught_by_its_collection_skip():
    """The shape two files in this suite could take on Linux.

    `tests/test_stage2b_encode_stage3_local.py` and
    `tests/test_stage2b_compare_stage3.py` both `pytest.importorskip` at
    module scope. Measured directly: pytest emits a testcase with an EMPTY
    classname and the module's basename as the name, carrying
    `message="collection skipped"`. That key is not in the baseline, so it
    lands in `unexpected` and fails the build -- while the module's
    individual tests vanish from collection entirely.

    Pinned because the guard holds by a property of pytest's reporting, not
    by anything this code arranged, and a reporting change would turn a
    caught failure into a silent one.
    """
    cases = [("", "test_some_module", "collection skipped"),
             ("tests.test_other", "test_still_here", None)]
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "junit.xml"
        report_path.write_text(junit_xml(cases))
        base_path = Path(tmp) / "baseline.txt"
        base_path.write_text("tests.test_other::test_gone_missing\n")
        outcomes, totals = vacuity.parse_junit(report_path)
        collection_skips = [o for o in outcomes if o.skipped and o.key.startswith("::")]
        assert collection_skips, "a collection skip must survive parsing as a key"
        assert vacuity.check(outcomes, totals,
                             vacuity.read_baseline(base_path), base_path) == 1


def test_full_tier_rejects_a_baseline_module_that_was_not_collected(report, baseline):
    """The residual hole module-scoping leaves, closed for the tier that
    runs everything. Without the flag this passes, which is correct for the
    fast tier and wrong for the full one."""
    path = report([("tests.test_a", "test_one", None)])
    base = baseline(["tests.test_a::test_one_skipped", "tests.test_b::test_two"])
    outcomes, totals = vacuity.parse_junit(path)
    loose = vacuity.check(outcomes, totals, vacuity.read_baseline(base), base)
    strict = vacuity.check(outcomes, totals, vacuity.read_baseline(base), base,
                           require_all_modules=True)
    assert strict == 1
    # `tests.test_a::test_one_skipped` is in scope and did not skip, so the
    # loose run fails too -- for the other reason. The flag's own effect is
    # the `tests.test_b` module, which only `strict` sees.
    assert loose == 1


def test_a_baseline_entry_that_ran_points_at_the_environment(report, baseline,
                                                             capsys):
    """The message must not accuse the baseline of being stale when the
    ENVIRONMENT changed under it.

    The live case, caught by reading before CI's first run: a Makefile target
    gained `uv run --group gpu` to fix a local skip; cloudbuild.yaml invokes
    that target, so the group installs in CI too, `google_crc32c` becomes
    importable, and a test the baseline correctly records as skipped now
    runs. Nothing about the baseline was wrong.

    The old message said only "remove them from the baseline", which is the
    one action that destroys a correct capability record to silence a
    complaint about something else. Two layers between symptom and cause is
    exactly when a guard has to name the cause.
    """
    path = report([("tests.test_a", "test_one", None)])
    base = baseline(["tests.test_a::test_one"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base),
                         base) == 1
    out = capsys.readouterr().out
    assert "BEFORE EDITING THE BASELINE" in out, (
        "the failure tells a reader to edit the baseline without warning "
        "that the environment is the likelier cause")
    assert "--group" in out and "cloudbuild" in out, (
        "the message does not name the mechanism -- a CI-invoked Makefile "
        "target acquiring a dependency group -- so a reader has to rediscover "
        "it from two layers away")


def test_full_tier_flag_does_not_fire_when_every_module_was_collected(report, baseline):
    path = report([("tests.test_a", "test_one", "no datasets"),
                   ("tests.test_b", "test_two", None)])
    base = baseline(["tests.test_a::test_one"])
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, vacuity.read_baseline(base), base,
                         require_all_modules=True) == 0


def test_failures_in_the_report_fail_the_check(tmp_path):
    """Unreachable while pytest's exit code stops the step first. Gated
    anyway: the checker is also run by hand against a saved report, where
    nothing else is watching."""
    path = tmp_path / "junit.xml"
    path.write_text('<testsuite name="pytest" errors="0" failures="1" skipped="0" '
                    'tests="1"><testcase classname="tests.test_a" name="test_one">'
                    '<failure message="boom"/></testcase></testsuite>')
    base = tmp_path / "baseline.txt"
    base.write_text("")
    outcomes, totals = vacuity.parse_junit(path)
    assert vacuity.check(outcomes, totals, set(), base) == 1


def test_a_missing_baseline_is_an_error_not_a_pass(tmp_path):
    with pytest.raises(SystemExit):
        vacuity.read_baseline(tmp_path / "nope.txt")


# --------------------------------------------- the credential-absence guard

def test_a_clean_environment_reports_nothing():
    """Both inputs injected. Probing the real interpreter here would red-fail
    on any machine that has synced `--group gpu` for a GPU target, which is a
    normal thing to have done and not a CI defect."""
    assert credentials.problems(env={}, installed=lambda _: False) == []


def test_a_mounted_credential_is_caught():
    for var in credentials.CREDENTIAL_ENV_VARS:
        found = credentials.problems(env={var: "/tmp/key.json"},
                                     installed=lambda _: False)
        assert any(var in p for p in found), (var, found)


def test_the_client_library_being_installed_is_caught():
    found = credentials.problems(env={}, installed=lambda _: True)
    assert any("google.cloud.storage" in p for p in found), found
    assert any("--group gpu" in p for p in found), found


def test_the_client_library_list_names_what_stage2b_actually_imports():
    """`stage2b_gcs` imports `google.cloud.storage` lazily. If that module
    path ever changes, this guard would pass while the library was present
    and importable under its new name."""
    source = (REPO_ROOT / "experiments" / "stage2b_denoising" / "stage2b_gcs.py").read_text()
    assert "google.cloud" in source
    assert any(lib.startswith("google.cloud") for lib in credentials.CLIENT_LIBRARIES)
