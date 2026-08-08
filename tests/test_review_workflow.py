"""The review workflow's wiring and its prose, both of which have bitten.

Two failures on the same afternoon, and neither was caught by anything:

1. `${{ steps.review.outputs.execution_file }}` was passed to a script
   expecting `structured_output`. The first is a PATH to the raw execution
   log; the second is the schema-validated JSON. The build went red on the
   first real run, which was the good outcome but a late one -- the
   documentation string says "Path to the ... file" and I had quoted it
   before wiring it as content.

2. The prompt asserted "eighteen tests" and "an eight-category taxonomy"
   about `docs/VACUOUS_TESTS.md` -- two quantitative claims, with no
   citation, hardcoded into a workflow, about a document that grows. They
   were already drifting when written: the catalogue said sixteen that
   morning. This is precisely what the citation verifier in
   `docs/PROVENANCE_CONTRACT.md` §5 exists to flag, committed by its author
   in a file the verifier does not scan.

Neither is exotic. Both are a value read under semantics its reader assumed,
which is the day's recurring shape.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "claude-code-review.yml"


@pytest.fixture(scope="module")
def text():
    assert WORKFLOW.exists(), f"{WORKFLOW} is gone"
    return WORKFLOW.read_text()


def _steps(text: str) -> list[dict]:
    """The job's steps, PARSED. A workflow is structured data.

    The first version of the two tests below matched substrings against the
    whole file and inspected the first line containing `publish_review.sh`.
    Both were shown to be inadequate by mutation, and the failure is
    instructive rather than embarrassing:

    - the wiring moved into `env:` when the value was un-interpolated to
      survive quotes in the JSON, so the `run:` line the test inspected can
      no longer contain `outputs.execution_file` under ANY input. That
      assertion had become unfalsifiable without changing shape.
    - the remaining whole-file check was carried entirely by there being
      exactly one mention of the string, in a 190-line file that is mostly
      commentary and whose comment block directly above the wiring discusses
      both output names. Reintroducing the real bug plus one comment naming
      `outputs.structured_output` passed all three assertions.

    Category A in docs/VACUOUS_TESTS.md -- matching source rather than
    evaluating the decision -- committed by a test written to pin a wiring
    bug, which is where it does the most damage. Found by the review this
    workflow runs, on the first range that contained tests.
    """
    doc = yaml.safe_load(text)
    jobs = doc.get("jobs") or {}
    assert jobs, "the workflow declares no jobs"
    steps = []
    for job in jobs.values():
        steps.extend(job.get("steps") or [])
    assert steps, "the workflow declares no steps"
    return steps


def _step_invoking(text: str, script: str) -> dict:
    """The single step whose `run` invokes `script`. Ambiguity is an error,
    not a first-match."""
    matches = [s for s in _steps(text) if script in (s.get("run") or "")]
    assert matches, f"no step runs `{script}` -- it is no longer invoked"
    assert len(matches) == 1, (
        f"{len(matches)} steps run `{script}`; this check cannot tell which "
        f"one carries the wiring")
    return matches[0]


def test_the_publisher_receives_structured_output_not_a_file_path(text):
    """The wiring bug, pinned to the step that actually carries it.

    `execution_file` and `structured_output` are both action outputs and only
    one is JSON. Passing the wrong one fails loudly, which is fortunate, but
    it should fail here rather than in a build.
    """
    step = _step_invoking(text, "publish_review.sh")
    env = step.get("env") or {}
    # Scoped to the variable the command ACTUALLY CONSUMES, not to the step.
    # Joining every env value into one string was the first attempt and it
    # repeated the defect one level down: over-broad, since an unrelated
    # `EXEC_LOG` on the same step would trip it, and simultaneously blind,
    # since a value naming the right output anywhere satisfied it.
    names = [a or b for a, b in re.findall(r"\$\{(\w+)\}|\$(\w+)",
                                           str(step.get("run") or ""))]
    assert names, "the publisher command passes no variable at all"
    for name in names:
        assert name in env, (
            f"the publisher command passes `${name}`, which the step's `env` "
            f"does not define. The shell expands it to an empty string and "
            f"the publisher fails the build for a missing review -- a red "
            f"build whose cause is a renamed variable")
    wired = " ".join(str(env[name]) for name in names)
    assert "outputs.structured_output" in wired, (
        "the publisher is not given `structured_output`. If it is receiving "
        "`execution_file`, that is a PATH and jq will reject it")
    assert "outputs.execution_file" not in wired, (
        "`execution_file` is wired into the value the publisher reads; it is "
        "a path, not JSON")


def test_the_execution_file_is_still_kept_as_an_artifact(text):
    """It is the right output for the artifact, and only for that. Keeping
    the two uses straight is the point of the test above.

    Scoped to the upload step for the same reason: as a whole-file substring
    this passed even with the upload deleted, so long as any comment
    mentioned the name -- and this file comments heavily on exactly that
    name.
    """
    uploads = [s for s in _steps(text)
               if "upload-artifact" in str(s.get("uses") or "")]
    assert uploads, (
        "no upload-artifact step: the durable copy of what the review did is "
        "gone, and the rendered report is the only remaining record")
    assert any("outputs.execution_file" in str(s.get("with", {}).get("path", ""))
               for s in uploads), (
        "an artifact is uploaded but it is not the execution record")


def _embedded_schema(text: str) -> dict:
    """The `--json-schema` payload, PARSED rather than pattern-matched.

    The first version of this helper regexed for `"required":[...]` and
    matched the wrong one -- the schema nests a `required` array inside
    `findings` items as well as declaring one at the top level, and
    `re.search` returns the first. It reported the top-level requirement as
    missing when it was present.

    That is the same failure as everything else this checker guards: a value
    read under semantics its reader assumed. A schema is structured data and
    gets parsed.
    """
    match = re.search(r"--json-schema\s+'(.*?)'\s*$", text,
                      re.DOTALL | re.MULTILINE)
    assert match, "no --json-schema found in claude_args"
    return json.loads(match.group(1))


def test_the_json_schema_requires_the_fields_the_publisher_reads(text):
    """A schema and a reader that disagree produce a confident empty report.

    The publisher branches on `no_tests_changed` and `files_examined`. If the
    schema stops REQUIRING them the model may omit them, and their absence
    reads as a review that examined nothing -- a false alarm on every run,
    which is how a real finding stops being believed.
    """
    schema = _embedded_schema(text)
    required = set(schema.get("required", []))
    for field in ("no_tests_changed", "files_examined", "findings"):
        assert field in schema.get("properties", {}), (
            f"the schema no longer declares `{field}`")
        assert field in required, (
            f"`{field}` is not REQUIRED at the top level of the schema, so "
            f"the model may omit it and the publisher will read its absence "
            f"as a review that examined nothing")


def test_each_finding_must_carry_its_taxonomy_category_and_reasoning(text):
    """The fields that make a finding actionable rather than an opinion."""
    schema = _embedded_schema(text)
    item = schema["properties"]["findings"]["items"]
    required = set(item.get("required", []))
    for field in ("file", "test", "category", "what_would_have_to_change"):
        assert field in required, (
            f"a finding need not carry `{field}`, so the review could report "
            f"a problem without saying where it is or why it is one")


# --- the prose ------------------------------------------------------------

_COUNT_WORD = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"\d+)\b[-\s]*(?:incident|categor|taxonom|test)", re.I)


def test_the_prompt_states_no_counts_about_the_catalogue(text):
    """A number about a growing document, hardcoded and uncited, is stale on
    arrival -- and this one was: it said eighteen while the catalogue moved
    from sixteen to eighteen the same morning.

    The document carries its own counts. The prompt points at it.
    """
    offenders = _COUNT_WORD.findall(text)
    matches = [m.group(0) for m in _COUNT_WORD.finditer(text)]
    assert not offenders, (
        f"the workflow states a count about the catalogue: {matches}. "
        f"docs/VACUOUS_TESTS.md grows; cite it rather than quantifying it. "
        f"This is the check docs/PROVENANCE_CONTRACT.md §5 makes for durable "
        f"documents, applied to the one file it does not scan.")


def test_the_prompt_still_points_at_the_catalogue(text):
    """Non-vacuity for the check above: deleting the reference entirely would
    satisfy it while removing the review's specification."""
    assert "docs/VACUOUS_TESTS.md" in text, (
        "the prompt no longer cites the catalogue, so the review has no "
        "specification and the count check above passes for the wrong reason")


def _allowlist(text: str) -> str:
    """The quoted tool allowlist, under either flag spelling.

    The action's examples write `--allowedTools`; this workflow writes
    `--allowed-tools`, which is the spelling with EVIDENCE behind it here --
    it is what took permission denials from twelve to four on run
    31255640813. Accepting only one spelling would make this helper return
    nothing the day someone adopts the other, and a test that cannot find
    the allowlist reports "no allowlist" rather than "the allowlist is
    wrong". Those need different fixes.

    A missing allowlist is a REAL failure and still fails here -- but as
    itself, not disguised as a content problem.
    """
    match = re.search(r"--allowed-?[Tt]ools\s+'([^']*)'", text)
    assert match, (
        "no tool allowlist found under either spelling. On a runner there is "
        "nobody to approve, so 'requires approval' means denied silently, "
        "and the review cannot read the PR at all")
    return match.group(1)


def test_the_review_can_read_files_and_publish(text):
    """A runner has nobody to approve, so "requires approval" means denied.

    Twelve denials on run 31255640813 and seven on 93106991131, all of them
    the review trying to reach context it had not been given. It needs two
    capabilities and no more: READ the checked-out files, and PUBLISH what it
    found. The changed-file list arrives in `<changed_files>`, so nothing
    needs to fetch it.

    Publishing is the one people forget. In tag mode the report IS the
    comment: without `update_claude_comment` the review runs, costs money,
    and says nothing anyone can see.
    """
    listed = _allowlist(text)
    for needed in ("Read", "mcp__github_comment__update_claude_comment"):
        assert needed in listed, (
            f"`{needed}` is not pre-approved. Without Read it cannot examine "
            f"a test; without the comment tool it cannot publish, and a "
            f"review nobody can read is indistinguishable from none")


def test_the_review_only_fires_on_prs_into_the_checkpoint_branch(text):
    """Scope, pinned, because widening it is silent and expensive.

    `branches` on a `pull_request` trigger filters the BASE. Without it the
    review fires on every pull request in the repository — including
    stage2b into main, where the work has already passed the checkpoint and
    reviewing it again is the repetition this design exists to remove.

    Deleting the filter produces no error and no red build. It produces more
    reviews, each costing minutes and dollars, and the only signal is a
    billing line.
    """
    doc = yaml.safe_load(text)
    trigger = (doc.get(True) or doc.get("on") or {}).get("pull_request") or {}
    assert trigger.get("branches") == ["stage2b-ci"], (
        f"the review's base-branch filter is {trigger.get('branches')!r}, "
        f"not ['stage2b-ci']. Without it every PR in the repository gets a "
        f"review, silently")
    assert trigger.get("paths") == ["tests/**"], (
        f"the path filter is {trigger.get('paths')!r}; a vacuous-test review "
        f"of a PR that changed no tests has nothing to say and still costs "
        f"tokens")


def test_track_progress_is_on(text):
    """The single line the whole design rests on.

    `src/modes/detector.ts`: supplying `prompt:` on a pull_request event
    selects AGENT mode, and `src/create-prompt/index.ts` makes agent mode
    return the prompt AND NOTHING ELSE -- no `<changed_files>`, no PR body,
    no comments. `track_progress: true` forces TAG mode, which injects all of
    it and appends the prompt as `<custom_instructions>`.

    Delete this one line and the workflow still runs, still costs money, and
    still reports success, while the review is blind to the PR and the prompt
    tells it to read a `<changed_files>` block that is not there. Nothing
    else in this file would notice, which is exactly why it is pinned.
    """
    doc = yaml.safe_load(text)
    steps = _steps(text)
    review = [s for s in steps
              if "claude-code-action" in str(s.get("uses") or "")]
    assert review, "the review step is gone"
    assert review[0].get("with", {}).get("track_progress") is True, (
        "track_progress is not enabled, so this workflow runs in AGENT mode "
        "and the model receives the prompt with no PR context whatsoever -- "
        "silently, and with the prompt still referring to <changed_files>")
    assert doc  # the file parsed at all


def test_the_review_does_not_reach_for_git(text):
    """The context is already in GitHub.

    Every commit-range design here failed by reconstructing, with git
    plumbing, what the platform already knows -- a PR's base, head and file
    list. An allowlist granting git invites that back, and the prompt now
    says not to. Both directions matter: the allowlist must not offer it and
    the prompt must not need it.
    """
    listed = _allowlist(text)
    assert "git " not in listed and "git)" not in listed, (
        f"git is pre-approved again: {listed!r}. A PR's diff comes from "
        f"`gh pr diff`; reaching for git inside a GitHub Action is the "
        f"reinvention that produced three broken range designs")


def test_the_review_is_not_granted_write_access(text):
    """Read-only by construction, not only by instruction.

    The prompt forbids modifying files and pushing. An allowlist that granted
    them anyway would leave the prohibition resting entirely on the model
    choosing to obey -- the same request-instead-of-mechanism error that put
    a path where JSON belonged earlier today.
    """
    listed = _allowlist(text)
    for forbidden in ("Write", "Edit", "git push", "git commit"):
        assert forbidden not in listed, (
            f"`{forbidden}` is pre-approved for a review that must only "
            f"read. Remove it: the prompt's prohibition is not a permission "
            f"boundary")


# --- the mutation, promoted out of a transcript ----------------------------
#
# Principle 21's corollary: a guard nobody has seen fail is not yet a guard.
# The previous version of the publisher test passed against a reintroduced
# wiring bug, and that was established by mutating the file in a throwaway
# script -- which is load-bearing scratch by principle 24, produced while
# building the tooling that exists to stop it. So the mutation lives here
# instead, where it re-runs and can fail.
#
# `MUTANT_KILLED_BY` names, for each mutation, the assertion that must
# reject it. A mutation nothing rejects is the defect being reintroduced.

def _mutate(text: str, old: str, new: str) -> str:
    assert old in text, f"mutation target absent, test is stale: {old!r}"
    return text.replace(old, new)


WIRING = "STRUCTURED_OUTPUT: ${{ steps.review.outputs.structured_output }}"
BUG = "STRUCTURED_OUTPUT: ${{ steps.review.outputs.execution_file }}"


# `match=` is not decoration. `pytest.raises(AssertionError)` around a test
# that parses and asserts its way to the real check will happily pass on an
# assertion from the SETUP -- a stale mutation target, a workflow with no
# steps -- and report a guard as working when the guard never ran. The
# expected message pins which assertion did the killing.
@pytest.mark.parametrize("mutate,expected,why", [
    (lambda t: _mutate(t, WIRING, BUG),
     "not given `structured_output`",
     "the exact wiring bug that turned the first real run red"),
    (lambda t: _mutate(_mutate(t, WIRING, BUG),
                       "# `structured_output` is the schema-validated JSON.",
                       "# steps.review.outputs.structured_output is the JSON."),
     "not given `structured_output`",
     "the same bug, plus one comment mentioning the string the old "
     "whole-file check relied on. THIS is the case that walked through"),
    (lambda t: _mutate(t, WIRING, "STRUCTURED_OUTPUT: ''"),
     "not given `structured_output`",
     "the publisher wired to nothing at all"),
    # The three above all die on the same assertion. These two exist because
    # of that: an assertion no mutation reaches is unexercised, which is the
    # defect this whole block was written to fix.
    (lambda t: _mutate(t, WIRING, BUG + "\n          REVIEW_JSON: "
                       "${{ steps.review.outputs.structured_output }}"),
     "not given `structured_output`",
     "INVERSION: the consumed variable carries the path while a SECOND, "
     "unconsumed variable carries the JSON. Whole-step matching passed this"),
    (lambda t: _mutate(t, WIRING, WIRING.rstrip()
                       + "${{ steps.review.outputs.execution_file }}"),
     "is wired into the value the publisher reads",
     "both outputs concatenated into the consumed value -- the only "
     "mutation that reaches the second assertion"),
    (lambda t: _mutate(t, "STRUCTURED_OUTPUT: ${{", "REVIEW_JSON: ${{"),
     "does not define",
     "the env key renamed while `run:` still passes $STRUCTURED_OUTPUT. The "
     "shell expands it to empty and the publisher fails the build for a "
     "missing review -- a red build blamed on the wrong thing"),
    # Supplied by the vacuous-test review on run 93106991131, as a worked
    # counterexample against the PREVIOUS version of this test, where it was
    # correct: "add a second, differently-named env entry sourced from
    # outputs.execution_file, and route the run line through that." It is a
    # genuinely distinct mutation from the ones above -- both the consumed
    # variable AND its source change together, so nothing is left undefined
    # and no name is reused. Pinned here rather than trusted, and kept
    # because a counterexample from outside is worth more than one invented
    # by the author of the check.
    (lambda t: _mutate(
        _mutate(t, WIRING, WIRING + "\n          EXECUTION_PATH: "
                "${{ steps.review.outputs.execution_file }}"),
        'run: bash tools/ci/publish_review.sh "$STRUCTURED_OUTPUT"',
        'run: bash tools/ci/publish_review.sh "$EXECUTION_PATH"'),
     "not given `structured_output`",
     "the reviewer's counterexample: one layer of indirection reintroduces "
     "the wiring bug with every string still present somewhere in the file"),
])
def test_a_broken_publisher_wiring_is_rejected(text, mutate, expected, why):
    with pytest.raises(AssertionError, match=re.escape(expected)):
        test_the_publisher_receives_structured_output_not_a_file_path(
            mutate(text))


@pytest.mark.parametrize("mutate,expected,why", [
    (lambda t: re.sub(r"      - name: Keep the execution record.*",
                      "", t, flags=re.DOTALL),
     "no upload-artifact step",
     "the upload step deleted outright -- which the whole-file substring "
     "check passed, because the name still appears in the commentary above"),
])
def test_a_missing_execution_artifact_is_rejected(text, mutate, expected, why):
    with pytest.raises(AssertionError, match=re.escape(expected)):
        test_the_execution_file_is_still_kept_as_an_artifact(mutate(text))


@pytest.mark.parametrize("mutate,why", [
    (lambda t: _mutate(t, "      - name: Announce it if nothing was published",
                       "      # publish_review.sh is invoked below\n"
                       "      - name: Announce it if nothing was published"),
     "a comment naming the script above the step is harmless and must stay "
     "harmless, which the old first-match `next()` could not promise"),
    (lambda t: _mutate(t, WIRING, WIRING + "\n          EXEC_LOG: "
                       "${{ steps.review.outputs.execution_file }}"),
     "an unrelated env var on the same step, naming the path for logging, "
     "is not a defect. Whole-step matching called it one"),
])
def test_harmless_changes_do_not_change_the_verdict(text, mutate, why):
    """The other direction: scoping must not have made the check brittle. A
    guard that fires on innocent edits gets routed around."""
    test_the_publisher_receives_structured_output_not_a_file_path(mutate(text))


@pytest.mark.parametrize("mutate,expected,why", [
    (lambda t: _mutate(t, "          track_progress: true\n", ""),
     "not enabled",
     "the line removed entirely -- the workflow still parses, still runs, "
     "still reports success, and the review goes blind"),
    (lambda t: _mutate(t, "track_progress: true", "track_progress: false"),
     "not enabled",
     "flipped to false, which is the action's own default and therefore "
     "where this lands if anyone 'tidies' an explicit setting"),
])
def test_losing_track_progress_is_caught(text, mutate, expected, why):
    """The guard, seen failing. Its absence has no runtime symptom at all --
    no error, no warning, just a review that cannot see the pull request."""
    with pytest.raises(AssertionError, match=re.escape(expected)):
        test_track_progress_is_on(mutate(text))


def test_the_review_keeps_one_comment_and_names_its_template(text):
    """Two halves, and they fail differently.

    `use_sticky_comment` is a MECHANISM: the action delivers one comment
    however the model behaves. The template is CONTENT: the prompt asks the
    model to maintain a defined structure inside it. Without the mechanism a
    perfectly-obedient model still appends a fresh comment every run; without
    the template the single comment has no state to carry forward.

    The template is a repository file rather than prompt text on purpose --
    this workflow is hand-synced to two branches, so anything inline costs a
    sync to change, and a structure nobody can cheaply improve does not get
    improved.
    """
    review = [s for s in _steps(text)
              if "claude-code-action" in str(s.get("uses") or "")]
    assert review and review[0].get("with", {}).get("use_sticky_comment") is True, (
        "use_sticky_comment is off, so each run appends a new comment. The "
        "newest review becomes the hardest to find, and every prior one is "
        "injected back into the next run's context")
    assert "docs/REVIEW_COMMENT_TEMPLATE.md" in text, (
        "the prompt no longer names the comment template, so the single "
        "comment has no defined structure to maintain")
    assert (REPO_ROOT / "docs" / "REVIEW_COMMENT_TEMPLATE.md").exists(), (
        "the prompt cites a template that is not in the repository")


def test_the_prompt_forbids_dropping_a_live_finding(text):
    """The risk the de-duplication introduces, pinned.

    Telling a reviewer not to repeat itself invites it to drop a finding it
    has already reported -- converting a defect into a silence, which is
    strictly worse than the repetition being removed. The instruction must
    survive any future trimming of this prompt.
    """
    lowered = text.lower()
    assert "only when it is fixed" in lowered, (
        "the prompt no longer states that a finding leaves the comment only "
        "when FIXED. Without it, 'do not repeat yourself' reads as "
        "permission to drop still-open findings")


def test_the_review_is_advisory_and_says_so(text):
    """The line that keeps an LLM verdict from gating a build. It has been
    the design's fixed point all day and is worth pinning."""
    lowered = text.lower()
    assert "advis" in lowered or "what gate" in lowered, (
        "the prompt no longer states that findings advise rather than gate. "
        "That line is the one keeping an LLM verdict off a build badge")

    # The tree prohibition is NOT asserted as prose here, deliberately.
    #
    # It used to be: the prompt said "do not modify files, do not push".
    # That is a request, and a request is the weakest form this repository
    # has a name for. It is now a MECHANISM -- the allowlist grants Read,
    # Grep, Glob, two comment tools and `gh pr comment`, so there is no
    # tool to modify a file with. `test_the_review_is_not_granted_write_access`
    # pins that, and pins it in both directions.
    #
    # Worth stating plainly because the prompt got shorter here and shorter
    # normally means weaker: the guarantee did not move into prose that was
    # then deleted, it moved OUT of prose into something that cannot be
    # disobeyed.
    listed = _allowlist(text)
    for forbidden in ("Write", "Edit", "git push"):
        assert forbidden not in listed, (
            f"`{forbidden}` is pre-approved. The prompt no longer forbids "
            f"modifying the tree in words, because the allowlist forbids it "
            f"in fact -- so this list is now the only thing standing there")
