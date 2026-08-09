"""`review_delta.sh` must not report "nothing changed" when tests LEAVE scope.

The bug these tests pin, found 2026-08-08 and confirmed against the live
GitHub compare API before a line was written:

    previous_filename: tests/test_archive_fidelity.py          <- in scope
    filename:          .claude/claude2claude/mailbox-tools/...  <- out of scope
    status:            renamed

`.files[].filename` is the path AT `after`. For a rename that is the
DESTINATION. The script filtered `.filename` against an in-scope pattern, so
a test file moved OUT of `tests/` matched nothing and the script answered
`mode=none` -- "no in-scope test files changed" -- for a push that took eight
tests off the reviewed surface. The review then examined nothing and posted a
clean comment.

That is a field-semantics failure: a filter correct about the field it reads
(`.filename` really is the current path), consumed as an answer to a
different question (which paths changed).

WHAT IS ACTUALLY UNDER TEST. These run the real `tools/ci/review_delta.sh`
with a stub `gh` on PATH replaying two captured responses, rather than
re-implementing the classification in Python. The defect lived in the jq
filter, so a Python reimplementation would have asserted against a copy of
the logic and passed while the shipped script stayed broken -- principle 16,
one layer down. The stub serves raw JSON precisely so the filtering under
test is the filtering that runs in CI.

`test_restoring_the_filename_only_filter_brings_the_bug_back` is the
break-confirmation this project requires of any new guard (principle 21
corollary: a guard you have not seen fail is not yet a guard). It mutates the
script back to the broken form and asserts the specific wrong answer returns.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "ci" / "review_delta.sh"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The real push that exposed the bug: stage2b-lead moving the mailbox transit
# check out of the science suite.
COMPARE_FIXTURE = FIXTURES / "compare_7a5dfaf_b1d1018.json"
TREE_FIXTURE = FIXTURES / "tree_b1d1018.json"

DEPARTED_PATH = "tests/test_archive_fidelity.py"
DEPARTED_DEST = ".claude/claude2claude/mailbox-tools/test_transit_integrity.py"


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _write_gh_stub(bin_dir: Path, compare: dict, tree: dict) -> None:
    """A `gh` that replays two captured API responses and nothing else.

    It routes on the request path rather than on argument position, because
    the script passes the endpoint as a single argument and the shape of that
    call is not what these tests are about. Any unrecognised call exits
    non-zero, so a script that starts making a THIRD request fails here
    rather than silently reaching the network from CI.
    """
    (bin_dir / "compare.json").write_text(json.dumps(compare))
    (bin_dir / "tree.json").write_text(json.dumps(tree))

    stub = bin_dir / "gh"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            for arg in "$@"; do
              case "$arg" in
                */compare/*)   cat {bin_dir / "compare.json"}; exit 0 ;;
                */git/trees/*) cat {bin_dir / "tree.json"};    exit 0 ;;
              esac
            done
            echo "stub gh: unexpected call: $*" >&2
            exit 1
            """
        )
    )
    stub.chmod(0o755)


def _run_delta(
    tmp_path: Path,
    compare: dict,
    tree: dict,
    *,
    script: Path | None = None,
    before: str = "7a5dfaf",
    after: str = "b1d1018",
) -> tuple[dict[str, str], str]:
    """Run the script against replayed payloads; return (outputs, stderr)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_gh_stub(bin_dir, compare, tree)

    out_file = tmp_path / "github_output"
    out_file.write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["GITHUB_OUTPUT"] = str(out_file)

    proc = subprocess.run(
        ["bash", str(script or SCRIPT), before, after, "owner/repo"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, (
        "the script must always exit 0 -- it advises a review, it does not "
        f"gate a build.\nstderr:\n{proc.stderr}"
    )
    return _parse_github_output(out_file.read_text()), proc.stderr


def _parse_github_output(raw: str) -> dict[str, str]:
    """Parse `key=value` and `key<<EOF ... EOF` as Actions itself would."""
    result: dict[str, str] = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "<<" in line:
            key, delim = line.split("<<", 1)
            body: list[str] = []
            i += 1
            while i < len(lines) and lines[i] != delim:
                body.append(lines[i])
                i += 1
            result[key] = "\n".join(body)
        elif "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
        i += 1
    return result


@pytest.fixture(scope="module")
def real_compare() -> dict:
    return json.loads(COMPARE_FIXTURE.read_text())


@pytest.fixture(scope="module")
def real_tree() -> dict:
    return json.loads(TREE_FIXTURE.read_text())


def _tree_of(*paths: str) -> dict:
    return {"truncated": False, "tree": [{"path": p, "type": "blob"} for p in paths]}


# --------------------------------------------------------------------------
# The fixtures must carry the situation they are named for
# --------------------------------------------------------------------------


def test_the_captured_payloads_still_contain_the_departure_they_exist_for(
    real_compare: dict, real_tree: dict
) -> None:
    """Guard the ground truth, so a re-capture cannot quietly empty the test.

    If someone regenerates these fixtures from a different range, every
    assertion below would still pass vacuously against a payload with no
    departure in it. This is the check that the payload is still the payload.
    """
    entries = real_compare["files"]
    moved = [f for f in entries if f.get("previous_filename") == DEPARTED_PATH]
    assert len(moved) == 1, (
        f"the compare fixture no longer contains a move away from {DEPARTED_PATH}; "
        "every departure assertion in this file would pass for the wrong reason"
    )
    assert moved[0]["filename"] == DEPARTED_DEST

    tree_paths = {node["path"] for node in real_tree["tree"]}
    assert DEPARTED_PATH not in tree_paths, (
        "the departed path is present in the tree fixture, so it did not depart"
    )
    assert DEPARTED_DEST in tree_paths, "the destination is missing from the tree fixture"

    # And the in-scope filter really does reject where it landed -- the whole
    # premise is that the destination is OUT of scope.
    assert not DEPARTED_DEST.startswith("tests/")


# --------------------------------------------------------------------------
# The bug itself
# --------------------------------------------------------------------------


def test_a_test_renamed_out_of_scope_is_not_reported_as_nothing(
    tmp_path: Path, real_compare: dict, real_tree: dict
) -> None:
    """The exact push that shipped the bug must no longer answer `none`."""
    outputs, stderr = _run_delta(tmp_path, real_compare, real_tree)

    assert outputs["mode"] != "none", (
        "a push that moved eight tests out of `tests/` was reported as "
        "'no in-scope test files changed'. That is the collapse this test exists "
        f"to prevent.\nstderr: {stderr}"
    )
    assert outputs["mode"] == "incremental"
    assert DEPARTED_PATH in outputs["files"]
    assert DEPARTED_DEST in outputs["files"], (
        "a departure must say WHERE it went; otherwise the reviewer cannot tell "
        "a deletion from a move and cannot check the move was legitimate"
    )


def test_the_departed_section_tells_the_reviewer_not_to_read_the_files(
    tmp_path: Path, real_compare: dict, real_tree: dict
) -> None:
    """A departed path does not exist at `after`.

    Listing it as something to examine sends the reviewer at an ENOENT, so
    the section carries its own handling instructions -- it cannot rely on the
    workflow prompt, which enumerates only `incremental`/`none`/`full` and
    must stay byte-identical across two branches.
    """
    outputs, _ = _run_delta(tmp_path, real_compare, real_tree)
    files = outputs["files"]

    assert "DEPARTED" in files
    assert "NO LONGER EXIST" in files
    assert "do not try to read them" in files.lower()


def test_restoring_the_filename_only_filter_brings_the_bug_back(
    tmp_path: Path, real_compare: dict, real_tree: dict
) -> None:
    """Break-confirmation: this guard has been SEEN to fail.

    A guard nobody has watched fail is not yet a guard. Mutating the union of
    `filename` and `previous_filename` back to `filename` alone must restore
    the original wrong answer -- `mode=none` on a push that removed eight
    tests. If this passes, the assertions above are not pinning the mechanism
    they claim to.
    """
    original = SCRIPT.read_text()
    broken_line = "([ .files[] | .filename ]"
    fixed_line = "([ .files[] | .filename, (.previous_filename // empty) ]"
    assert original.count(fixed_line) == 1, (
        "the line this mutation targets has moved; the break-confirmation is "
        "no longer testing what it names"
    )

    mutant = tmp_path / "review_delta_mutant.sh"
    mutant.write_text(original.replace(fixed_line, broken_line))

    outputs, _ = _run_delta(tmp_path, real_compare, real_tree, script=mutant)

    assert outputs["mode"] == "none", (
        "restoring the `.filename`-only filter did NOT reproduce the original "
        "defect, so these tests are not pinning the field-semantics bug"
    )
    assert DEPARTED_PATH not in outputs["files"]


# --------------------------------------------------------------------------
# The derivation must be right for statuses nobody enumerated
# --------------------------------------------------------------------------


def test_a_copy_is_not_a_departure(tmp_path: Path) -> None:
    """`copied` carries `previous_filename`, but the source still exists.

    This is the case a hand-written status list gets wrong, and the reason
    the script derives departure from the tree at `after` instead of from
    `status`. Note the source is deliberately ABSENT from the compare payload:
    a copy does not modify its source, so the source need not appear as a
    changed file at all -- which is exactly how a set built from the compare
    alone would misread it as gone.
    """
    compare = {
        "files": [
            {
                "status": "copied",
                "filename": "tests/test_copy.py",
                "previous_filename": "tests/test_original.py",
            }
        ]
    }
    tree = _tree_of("tests/test_original.py", "tests/test_copy.py")

    outputs, _ = _run_delta(tmp_path, compare, tree)

    assert "DEPARTED" not in outputs["files"], (
        "a copy was classified as a departure -- the source still exists"
    )
    assert "tests/test_copy.py" in outputs["files"]
    assert outputs["mode"] == "incremental"


def test_an_outright_deletion_is_a_departure(tmp_path: Path) -> None:
    """A removed test has no `previous_filename`; its `filename` is the path.

    Handled by the same derivation, with no branch on status: the path is
    mentioned, it is in scope, and it is not in the tree.
    """
    compare = {
        "files": [
            {"status": "removed", "filename": "tests/test_deleted.py", "previous_filename": None}
        ]
    }
    tree = _tree_of("tests/test_kept.py")

    outputs, _ = _run_delta(tmp_path, compare, tree)

    assert outputs["mode"] == "incremental"
    assert "tests/test_deleted.py" in outputs["files"]
    assert "(gone)" in outputs["files"], (
        "a deletion with no destination should say so rather than claim a move"
    )


def test_an_ordinary_edit_still_reports_incremental_with_no_departed_section(
    tmp_path: Path,
) -> None:
    """The common case must be unchanged -- no new noise on every push."""
    compare = {
        "files": [
            {"status": "modified", "filename": "tests/test_x.py", "previous_filename": None},
            {"status": "added", "filename": "src/bonsai/thing.py", "previous_filename": None},
        ]
    }
    tree = _tree_of("tests/test_x.py", "src/bonsai/thing.py")

    outputs, _ = _run_delta(tmp_path, compare, tree)

    assert outputs["mode"] == "incremental"
    assert "tests/test_x.py" in outputs["files"]
    assert "src/bonsai/thing.py" not in outputs["files"], "out of scope, must not be listed"
    assert "DEPARTED" not in outputs["files"], (
        "a push with no departures must not grow a departed section"
    )


def test_a_push_that_touches_no_tests_still_reports_none(tmp_path: Path) -> None:
    """`none` must stay reachable, or the fix has traded one collapse for another.

    Non-significance is not equivalence: widening a signal until it always
    fires destroys it just as thoroughly as never firing.
    """
    compare = {
        "files": [{"status": "modified", "filename": "docs/README.md", "previous_filename": None}]
    }
    tree = _tree_of("docs/README.md", "tests/test_x.py")

    outputs, stderr = _run_delta(tmp_path, compare, tree)

    assert outputs["mode"] == "none"
    assert outputs["files"].strip() == ""
    assert "departed" in stderr.lower()


# --------------------------------------------------------------------------
# Fail-open, loudly
# --------------------------------------------------------------------------


def test_a_truncated_tree_fails_open_rather_than_inventing_departures(
    tmp_path: Path, real_compare: dict
) -> None:
    """A truncated tree cannot answer "does this path exist".

    Absence would be indistinguishable from being cut off, and every in-scope
    path would read as departed. Falling back to `full` is the only honest
    answer, and it must be said out loud.
    """
    truncated = {"truncated": True, "tree": []}

    outputs, stderr = _run_delta(tmp_path, real_compare, truncated)

    assert outputs["mode"] == "full"
    assert "truncated" in stderr.lower()
    assert "DEPARTED" not in outputs["files"]


def test_a_missing_jq_fails_open_loudly_rather_than_reviewing_nothing(
    tmp_path: Path, real_compare: dict, real_tree: dict
) -> None:
    """The dependency this script cannot work without, absent.

    This is not hypothetical: the Cloud Build image installed `git` and `make`
    and not `jq`, so on Linux the script fell straight through to `mode=full`
    on every invocation while every test here pinned a mode and went red.

    The behaviour was RIGHT -- fail open, review everything, say so on stderr.
    What was wrong was that nothing tested it, so the safe degradation looked
    identical to a logic bug, and the break-confirmation above could not tell
    the fixed script from the broken one because both returned `full`.

    `tests/test_ci_image_dependencies.py` stops the image losing a required
    command. This asserts what happens if one goes missing anyway.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_gh_stub(bin_dir, real_compare, real_tree)

    out_file = tmp_path / "github_output"
    out_file.write_text("")

    env = dict(os.environ)
    # ONLY the stub directory: `gh` resolves, `jq` does not.
    env["PATH"] = str(bin_dir)
    env["GITHUB_OUTPUT"] = str(out_file)

    # Resolve the interpreter before stripping PATH, or the child cannot even
    # find `bash` and the test fails for a reason unrelated to jq.
    bash = shutil.which("bash")
    assert bash, "no bash on PATH"

    proc = subprocess.run(
        [bash, str(SCRIPT), "7a5dfaf", "b1d1018", "owner/repo"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )

    assert proc.returncode == 0, (
        "a missing dependency must not fail the build -- this script advises a "
        f"review, it does not gate.\nstderr:\n{proc.stderr}"
    )
    outputs = _parse_github_output(out_file.read_text())
    assert outputs["mode"] == "full", (
        "without jq the script cannot classify anything, so the only honest "
        "answer is 'review everything in scope'"
    )
    assert "jq unavailable" in proc.stderr
    assert "reviewing everything in scope" in proc.stderr


def test_malformed_compare_json_fails_open_rather_than_reviewing_nothing(
    tmp_path: Path, real_tree: dict
) -> None:
    """A compare payload with no `.files` key makes `.files[]` a jq runtime
    error inside `classify`. Before the fix, neither call site checks jq's
    exit status, both come back empty, and the script reports `mode=none`
    -- exactly the silent fallback to reviewing nothing this script exists
    to avoid."""
    malformed_compare = {"url": "https://example.invalid/compare"}  # no "files"
    outputs, stderr = _run_delta(tmp_path, malformed_compare, real_tree)
    assert outputs["mode"] == "full"
    assert "jq failed" in stderr.lower()
    assert "reviewing everything in scope" in stderr


@pytest.mark.parametrize(
    "before, why",
    [
        ("", "no previous commit at all -- a first run"),
        ("0" * 40, "the zero SHA GitHub sends for a branch that did not exist"),
    ],
)
def test_no_usable_previous_commit_reviews_everything(
    tmp_path: Path, real_compare: dict, real_tree: dict, before: str, why: str
) -> None:
    outputs, stderr = _run_delta(tmp_path, real_compare, real_tree, before=before)

    assert outputs["mode"] == "full", why
    assert "reviewing everything in scope" in stderr


# --------------------------------------------------------------------------
# Two narrowings, and only one direction of disagreement is dangerous
# --------------------------------------------------------------------------


def _in_scope_pattern() -> str:
    """Read IN_SCOPE out of the script rather than restating it here.

    A copy of the pattern in the test would agree with itself forever while
    the script drifted -- the same shape of failure as the bug above.
    """
    for line in SCRIPT.read_text().splitlines():
        if line.startswith("IN_SCOPE="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise AssertionError("IN_SCOPE is no longer defined in review_delta.sh")


def _workflow_paths() -> list[str]:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "claude-code-review.yml").read_text()
    )
    # `on` is parsed as the boolean True by YAML 1.1 -- a field-semantics trap
    # of exactly the kind this file exists for.
    trigger = workflow.get("on", workflow.get(True))
    return list(trigger["pull_request"]["paths"])


def _matches_github_glob(path: str, pattern: str) -> bool:
    """GitHub path filters: `*` stops at `/`, `**` does not."""
    import re as _re

    regex = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        else:
            regex += _re.escape(pattern[i])
            i += 1
    return _re.fullmatch(regex, path) is not None


def test_every_file_the_delta_calls_in_scope_can_actually_trigger_the_workflow() -> None:
    """The workflow's `paths` must be a SUPERSET of the delta's `IN_SCOPE`.

    Two hand-maintained narrowings describe the reviewed surface: the
    workflow's `paths: ["tests/**"]` decides whether a review runs at all, and
    the script's `IN_SCOPE` decides what it looks at. They are not required to
    be equal, but the disagreement is only safe in one direction.

    - workflow WIDER than IN_SCOPE: a run fires and finds nothing in scope.
      Wasteful (a run costs real money and posts a "nothing new" comment) but
      it cannot produce a wrong answer. This is the state today: the workflow
      fires on `tests/test_publish_review.sh` and on `tests/fixtures/*.json`,
      which the review's declared scope of `tests/*.py` excludes.
    - IN_SCOPE WIDER than the workflow: the review never runs for a file it
      would have reviewed. Silent, and indistinguishable from a clean review.

    The corpus is enumerated from the filesystem rather than written down, so
    a file added under `tests/` is covered on the day it lands rather than
    whenever someone remembers this list.
    """
    import re as _re

    pattern = _re.compile(_in_scope_pattern())
    patterns = _workflow_paths()

    tests_dir = REPO_ROOT / "tests"
    candidates = [
        str(p.relative_to(REPO_ROOT))
        for p in tests_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    assert candidates, "no files found under tests/ -- the corpus is empty"

    in_scope = [c for c in candidates if pattern.match(c)]
    assert in_scope, (
        "IN_SCOPE matched nothing under tests/, so this test would pass "
        "vacuously for any workflow filter at all"
    )

    unreachable = [
        path for path in in_scope if not any(_matches_github_glob(path, p) for p in patterns)
    ]
    assert not unreachable, (
        "these files are in the delta's scope but cannot trigger the workflow, "
        "so a PR changing only them gets no review and no signal that it was "
        f"skipped: {unreachable}\nworkflow paths: {patterns}"
    )


def test_the_script_is_executable_and_has_no_git_calls() -> None:
    """The checkout may be shallow, and three range-based designs died here.

    The context is already in GitHub; reconstructing it with git plumbing is
    what produced the reviews that went green having read nothing.
    """
    source = SCRIPT.read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert not any(
        token in code for token in ("git rev-parse", "git diff", "git log", "git merge-base")
    ), "review_delta.sh must not shell out to git -- the checkout may be shallow"
    assert os.access(SCRIPT, os.X_OK), "review_delta.sh must be executable"
