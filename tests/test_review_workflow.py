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

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "claude-code-review.yml"


@pytest.fixture(scope="module")
def text():
    assert WORKFLOW.exists(), f"{WORKFLOW} is gone"
    return WORKFLOW.read_text()


def test_the_publisher_receives_structured_output_not_a_file_path(text):
    """The wiring bug, pinned.

    `execution_file` and `structured_output` are both action outputs and only
    one is JSON. Passing the wrong one fails loudly, which is fortunate, but
    it should fail here rather than in a build.
    """
    assert "publish_review.sh" in text, "the publisher is no longer invoked"
    assert "outputs.structured_output" in text, (
        "the publisher is not being given `structured_output`. If it is "
        "receiving `execution_file`, that is a PATH and jq will reject it")
    publisher_line = next(
        line for line in text.splitlines() if "publish_review.sh" in line)
    assert "outputs.execution_file" not in publisher_line, (
        "`execution_file` is being passed to the publisher; it is a path, "
        "not JSON")


def test_the_execution_file_is_still_kept_as_an_artifact(text):
    """It is the right output for the artifact, and only for that. Keeping
    both uses straight is the whole point of the test above."""
    assert "outputs.execution_file" in text, (
        "the execution record is no longer archived -- the durable copy of "
        "what the review did is gone")


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


def test_the_review_is_pre_approved_to_read_its_own_diff(text):
    """A runner has nobody to approve, so "requires approval" means denied.

    Measured on run 31255640813: twelve permission denials, all Bash, every
    one an attempt to read the diff -- `gh pr view`, `gh pr diff`,
    `git fetch refs/pull/N/head`, `git log --parents` -- retried across 25
    turns before the review found another route. It happened to succeed. Had
    that PR contained tests it could have reported examining nothing, and the
    publisher would have failed the build for a reason that was really this
    config.
    """
    assert "--allowed-tools" in text, (
        "no tool allowlist: the review will be denied Bash on a runner and "
        "cannot read its own diff")
    allowed = re.search(r"--allowed-tools\s+'([^']*)'", text)
    assert allowed, "--allowed-tools is present but not readable as a quoted list"
    listed = allowed.group(1)
    for needed in ("git diff", "git log", "gh pr diff"):
        assert needed in listed, (
            f"`{needed}` is not pre-approved; it was denied on the first real "
            f"run and is how the review sees what changed")


def test_the_review_is_not_granted_write_access(text):
    """Read-only by construction, not only by instruction.

    The prompt forbids modifying files and pushing. An allowlist that granted
    them anyway would leave the prohibition resting entirely on the model
    choosing to obey -- the same request-instead-of-mechanism error that put
    a path where JSON belonged earlier today.
    """
    allowed = re.search(r"--allowed-tools\s+'([^']*)'", text)
    assert allowed, "no allowlist to check"
    listed = allowed.group(1)
    for forbidden in ("Write", "Edit", "git push", "git commit"):
        assert forbidden not in listed, (
            f"`{forbidden}` is pre-approved for a review that must only "
            f"read. Remove it: the prompt's prohibition is not a permission "
            f"boundary")


def test_the_review_is_advisory_and_says_so(text):
    """The line that keeps an LLM verdict from gating a build. It has been
    the design's fixed point all day and is worth pinning."""
    lowered = text.lower()
    assert "comment only" in lowered or "advis" in lowered, (
        "the prompt no longer states the review is advisory")
    assert "do not modify files" in lowered or "do not push" in lowered, (
        "the prompt no longer forbids the review from changing the tree")
