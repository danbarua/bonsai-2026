#!/usr/bin/env python3
"""Tests for the briefing assembler.

Deliberately NOT in `tests/`. That directory verifies documented scientific
claims -- a FINDINGS number, a construction's byte-exact match against a
historical artifact. Comms tooling is not one, and putting it there has
already been reverted twice in this project. Lives beside the code it tests,
same as `.claude/claude2claude/mailbox-tools/`.

Run directly:  uv run pytest .claude/briefing/ -q

Every test here pins something that was OBSERVED failing, not something
imagined. The file-attribution test in particular pins a bug that produced
plausible output while silently attributing every changed path to the wrong
commit -- it corrupted the join haystack, not just the display, so nothing
about reading the briefing looked wrong.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from assemble_briefing import (  # noqa: E402
    Commit,
    Loop,
    bullets,
    digest_ts_to_iso,
    join_loops,
    read_commits,
    section,
    tokens_of,
    tree_evidence,
    _CLOSED_RE,
)


# --------------------------------------------------------------------------
# The file-attribution bug. Highest-value test in this file.
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A real git repo with two commits touching different, known files."""
    r = tmp_path / "repo"
    r.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "T")

    (r / "alpha.py").write_text("a\n")
    git("add", "alpha.py")
    git("commit", "-q", "-m", "first: add alpha", "-m", "body of first")

    (r / "beta.py").write_text("b\n")
    (r / "gamma.py").write_text("g\n")
    git("add", "beta.py", "gamma.py")
    git("commit", "-q", "-m", "second: add beta and gamma", "-m", "body of second")
    return r


def test_changed_files_attach_to_the_right_commit(tiny_repo: Path) -> None:
    """A trailing record separator pushes `--name-only` output into the NEXT
    record. That produced a briefing where every commit showed another
    commit's files -- and because `files` feeds the join haystack, loops were
    matched against the wrong commits with no visible symptom."""
    commits = read_commits(tiny_repo, "main", None)
    assert len(commits) == 2

    by_subject = {c.subject: c for c in commits}
    first = by_subject["first: add alpha"]
    second = by_subject["second: add beta and gamma"]

    assert first.files == ["alpha.py"]
    assert sorted(second.files) == ["beta.py", "gamma.py"]

    # The specific corruption: alpha must NOT appear under the second commit.
    assert "alpha.py" not in second.files
    assert "beta.py" not in first.files


def test_bodies_are_captured_not_swallowed_by_paths(tiny_repo: Path) -> None:
    """Bodies matter more than usual: this project routes durable reasoning
    into commit messages precisely because the mailbox is not committed."""
    commits = read_commits(tiny_repo, "main", None)
    bodies = {c.subject: c.body for c in commits}
    assert "body of first" in bodies["first: add alpha"]
    assert "body of second" in bodies["second: add beta and gamma"]


def test_empty_window_returns_empty_not_everything(tiny_repo: Path) -> None:
    """An empty range must yield zero commits. Falling back to 'all history'
    would emit a confident briefing over the wrong window."""
    assert read_commits(tiny_repo, "main..main", None) == []


# --------------------------------------------------------------------------
# Rarity is the evidence -- the rule that replaced a stopword list.
# --------------------------------------------------------------------------


def _commit(sha: str, subject: str, body: str = "", files: list[str] | None = None) -> Commit:
    return Commit(
        sha=sha * 8, short=sha, date="2026-08-08", author="T",
        subject=subject, body=body, files=files or [],
    )


def test_common_token_is_rejected_as_non_discriminating() -> None:
    """`main` hit 11 of 31 commits on the first real run: pure noise."""
    commits = [_commit(f"c{i}", f"work on main {i}") for i in range(10)]
    loop = Loop(text="something about `main`", tokens=["main"])
    join_loops([loop], commits)
    assert loop.matches == []
    assert "main" in loop.low_signal


def test_rare_token_is_accepted_as_a_candidate() -> None:
    commits = [_commit(f"c{i}", f"unrelated {i}") for i in range(10)]
    commits.append(_commit("hit", "fix ci_targets.py coverage"))
    loop = Loop(text="about `ci_targets.py`", tokens=["ci_targets.py"])
    join_loops([loop], commits)
    assert [c.short for _, c in loop.matches] == ["hit"]
    assert loop.low_signal == []


@pytest.mark.parametrize("window", [2, 3, 4, 8])
def test_a_single_hit_is_never_suppressed_however_small_the_window(window: int) -> None:
    """The fraction bound must not punish a window for being small.

    One hit in three commits is 0.33, over the 0.25 bound, so a unique and
    genuine match was discarded and its loop filed under "no candidate
    found". That fires as soon as the digests catch up and the cold window
    shrinks -- the steady state, not an edge case. The suite was silent on
    it because it pinned 2-of-2 (correctly ambiguous) and never 1-of-2.
    """
    commits = [_commit(f"c{i}", f"unrelated {i}") for i in range(window - 1)]
    commits.append(_commit("hit", "fix zeta.py"))
    loop = Loop(text="about `zeta.py`", tokens=["zeta.py"])
    join_loops([loop], commits)
    assert [c.short for _, c in loop.matches] == ["hit"]
    assert loop.low_signal == []


def test_fraction_bound_guards_a_short_window() -> None:
    """On a 2-commit window the absolute cap of 3 would admit everything,
    so the fractional bound has to carry the small-n end."""
    commits = [_commit("a", "touch zeta.py"), _commit("b", "touch zeta.py")]
    loop = Loop(text="about `zeta.py`", tokens=["zeta.py"])
    join_loops([loop], commits)
    assert loop.matches == []
    assert "zeta.py" in loop.low_signal


def test_the_join_searches_changed_paths_not_just_messages() -> None:
    """A commit that touches a file without naming it in the message is still
    the best candidate for a loop about that file."""
    commits = [_commit(f"c{i}", f"unrelated {i}") for i in range(8)]
    commits.append(_commit("hit", "tidy up", files=["tools/ci/publish_review.sh"]))
    loop = Loop(text="about `publish_review.sh`", tokens=["publish_review.sh"])
    join_loops([loop], commits)
    assert [c.short for _, c in loop.matches] == ["hit"]


# --------------------------------------------------------------------------
# Token extraction: a make target must join; a quotation must not.
# --------------------------------------------------------------------------


def test_make_target_is_joinable_despite_containing_a_space() -> None:
    """Rejecting every span with a space filed `make stage2b-gate-inventory`
    under NOT CHECKED -- reported as unsearchable, while being one of the
    most searchable strings in the digest."""
    assert "make stage2b-gate-inventory" in tokens_of(
        "Still open — `make stage2b-gate-inventory`."
    )


def test_prose_quotation_is_not_joinable() -> None:
    quoted = "`a real, passing test cited for an obligation it does not establish`"
    assert tokens_of(f"the rule says {quoted}") == []


def test_issue_numbers_join() -> None:
    assert "#22" in tokens_of("Still open — issue #22, Dan's call.")


def test_short_tokens_are_dropped() -> None:
    assert tokens_of("a `cd` and a `ls`") == []


# --------------------------------------------------------------------------
# Already-closed bullets must not be reported as open.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("marked", [
    "~~the proposal, unanswered~~ — **CLOSED**; already sent",
    "**CLOSED** — handled last window",
])
def test_closed_bullets_are_detected(marked: str) -> None:
    """A closed item carried through would be filed under 'no candidate
    found', stating the opposite of what the digest recorded."""
    assert _CLOSED_RE.search(marked)


def test_open_bullet_is_not_mistaken_for_closed() -> None:
    assert not _CLOSED_RE.search("**Still open — issue #22**, Dan's call.")


# --------------------------------------------------------------------------
# Tree evidence: the check with no window.
# --------------------------------------------------------------------------


def test_make_target_present_in_tree_is_found(tmp_path: Path) -> None:
    """`make stage2b-gate-inventory` was reported 'most likely genuinely
    open' while sitting at Makefile:656 -- added before the window opened.
    The git join is window-bounded; this check is not."""
    (tmp_path / "Makefile").write_text(
        ".PHONY: foo\n"
        "foo:\n"
        "\t@echo hi\n"
        ".PHONY: stage2b-gate-inventory\n"
        "stage2b-gate-inventory:  ## reconcile\n"
    )
    loop = Loop(text="x", tokens=["make stage2b-gate-inventory"])
    tree_evidence(tmp_path, loop)
    assert loop.present == [("make stage2b-gate-inventory", "Makefile:5")]


def test_absent_make_target_is_not_claimed_present(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("foo:\n\t@echo hi\n")
    loop = Loop(text="x", tokens=["make never-existed"])
    tree_evidence(tmp_path, loop)
    assert loop.present == []


def test_path_present_in_tree_is_found(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "thing.py").write_text("")
    loop = Loop(text="x", tokens=["tools/thing.py"])
    tree_evidence(tmp_path, loop)
    assert loop.present == [("tools/thing.py", "tools/thing.py")]


def test_absent_path_is_not_claimed_present(tmp_path: Path) -> None:
    loop = Loop(text="x", tokens=["tools/gone.py"])
    tree_evidence(tmp_path, loop)
    assert loop.present == []


# --------------------------------------------------------------------------
# Digest parsing.
# --------------------------------------------------------------------------


DIGEST = """# Mailbox summary — 2026-08-08T14-48-02Z

**Covers:** 9 messages

## What happened

Some prose.

## Decisions reached

- First decision.
- Second decision,
  wrapped onto a second line.

## Open loops

**Standing caveat:** this reports the mailbox, which lags the repository.

- **Still open — `make foo-bar`.** No message says otherwise.
- ~~Old thing~~ — **CLOSED**; handled.
"""


def test_section_extracts_only_its_own_body() -> None:
    body = section(DIGEST, "Decisions reached")
    assert "First decision" in body
    assert "Still open" not in body
    assert "Some prose" not in body


def test_bullets_keep_wrapped_continuation_lines() -> None:
    items = bullets(section(DIGEST, "Decisions reached"))
    assert len(items) == 2
    assert "wrapped onto a second line" in items[1]


def test_open_loops_parse_and_separate_closed_from_open() -> None:
    items = bullets(section(DIGEST, "Open loops"))
    open_items = [i for i in items if not _CLOSED_RE.search(i)]
    assert len(open_items) == 1
    assert "make foo-bar" in open_items[0]


def test_digest_timestamp_converts_to_git_iso() -> None:
    assert digest_ts_to_iso(
        "MAILBOX_SUMMARY_2026-08-08T14-48-02Z.md"
    ) == "2026-08-08T14:48:02Z"


def test_unparseable_digest_name_returns_none_rather_than_guessing() -> None:
    assert digest_ts_to_iso("MAILBOX_SUMMARY_garbage.md") is None
