"""Each transit-integrity heuristic, shown to fire on what it watches.

`check_transit_integrity.py`, beside this file, returns zero findings over both
real archives, and a check that has only ever returned zero is
indistinguishable from a check that cannot return anything else.

**Why this lives beside the mailboxes and not in `tests/`.** It used to be
in `tests/`, and it had no business there. That directory verifies
documented scientific claims — a FINDINGS number, a construction's
byte-exact match against a historical artifact. This file tests a mailbox
truncation heuristic for the agent-to-agent comms channel. Dan's ruling,
when a skip from it failed a CI build:

    our internal agent-to-agent comms tool has NOTHING TO DO with
    science. We don't care about that in CI.

The first fix attempted was a CI skip-baseline entry, which would have
gone green while asserting the c2gpt archive is a capability CI is
expected to lack — i.e. that it belongs in CI's world. The second was to
move only the archive-scanning case out and leave these behind. Both were
scoped to the symptom. The whole file was in the wrong tree.

So it sits with the mail it is about, under `.claude/claude2claude/`,
and no make target or CI build collects it. The mechanism by which
reviews reach this project is not part of the science.

Tier 1 throughout: synthetic fixtures, no capability required. The
real-archive scan is the make target, not a test — see the note at the
foot of this file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "check_transit_integrity.py"


# The archive-path resolution that used to live here moved with the scan
# it served, to `make c2c-archive-check`. It is worth stating what it had
# to handle, since the make target inherits the same problem: agents work
# in worktrees, and a worktree's own `.claude/claude2gpt/archive/` is
# created EMPTY -- the directories are tracked, their contents gitignored
# and local-only. So the mailbox lives in the main checkout, which
# `git rev-parse --git-common-dir` finds (its parent) and a relative path
# from the worktree does not.

spec = importlib.util.spec_from_file_location("_transit_integrity", MODULE)
transit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = transit
spec.loader.exec_module(transit)

DEFAULTS = ["terminal", "ordinal"]

HEALTHY = """\
<!-- from: chatgpt -->

## Reviewer ruling

### 1. First matter

Resolved as stated.

### 2. Second matter

See section 1 for the basis.

### 3. Third matter

This closes the outstanding interpretation.
"""


def _archive(tmp_path: Path, **files: str) -> Path:
    for name, text in files.items():
        (tmp_path / f"{name}.md").write_text(text)
    return tmp_path


def _scan(tmp_path: Path, enabled=None):
    return transit.scan([tmp_path], enabled or DEFAULTS)


def test_a_healthy_ruling_produces_no_findings(tmp_path):
    """Non-vacuity for every check below.

    Without it, a check that flagged every file would look identical to a
    working one in the tests that follow — they assert a finding appears,
    and a check that always fires satisfies that.
    """
    count, findings = _scan(_archive(tmp_path, ruling=HEALTHY))
    assert count == 1
    assert findings == []


def test_a_file_clipped_mid_sentence_is_caught(tmp_path):
    """The case citation resolution cannot see.

    The clipped text is the SHAPE of the live risk: a closing qualification
    referenced by nothing, so every internal citation still resolves and
    the file still ends on a complete-looking section.
    """
    clipped = HEALTHY + "\nI do **not** require an automated prose checker for"
    _, findings = _scan(_archive(tmp_path, ruling=clipped))
    assert any("terminal" in f for f in findings), findings


def test_a_clip_inside_inline_code_is_caught(tmp_path):
    """The real defect, pinned so it cannot come back.

    Clipping an actual ruling mid-sentence landed on ``For ` `` — the
    OPENING backtick of an inline code span — and the check passed,
    because backtick was in the terminator set for code fences. A
    truncation stopping inside inline code was invisible to the check
    built to catch truncation.

    Worth keeping as its own case rather than folding into the test above:
    the generic clip lands on a letter and would pass on any terminator
    set that merely excludes letters. This one only passes on a set that
    got backtick specifically right.
    """
    clipped = HEALTHY + "\nThe rule is stated in `"
    _, findings = _scan(_archive(tmp_path, ruling=clipped))
    assert any("terminal" in f for f in findings), findings


def test_a_dropped_middle_section_is_caught(tmp_path):
    """Mid-file loss where nothing references the missing section.

    Section 2 goes along with the sentence that pointed at 1, so a
    citation check would stay quiet and only the ordinal gap remains as a
    tell — which is the whole reason `ordinal` is a default.
    """
    dropped = HEALTHY.replace(
        "### 2. Second matter\n\nSee section 1 for the basis.\n\n", "")
    _, findings = _scan(_archive(tmp_path, ruling=dropped))
    assert any("ordinal" in f for f in findings), findings


def test_a_sign_off_line_is_not_a_truncation(tmp_path):
    """23 false positives came from this, mine among them.

    Mesh messages end on their own author line. A terminator set that
    calls that a clipped sentence fires on nearly every message in the
    code2code archive — and a check that cries wolf at that rate buries
    the one real finding it exists for.
    """
    _, findings = _scan(_archive(tmp_path, msg=HEALTHY + "\n— stage2b-lead\n"))
    assert findings == []


def test_widening_did_not_switch_the_guard_off(tmp_path):
    """The check on the fix, not on the code.

    The terminator set was widened twice to kill false positives, and
    widening a tolerance is how a guard gets disabled while looking
    healthier. So the damaged fixtures are re-run against the WIDENED
    set: a sign-off must pass and a clip must still fail, in the same
    test, or the two properties can drift apart unnoticed.
    """
    healthy_dir = tmp_path / "healthy"
    damaged_dir = tmp_path / "damaged"
    healthy_dir.mkdir()
    damaged_dir.mkdir()
    _archive(healthy_dir, a=HEALTHY + "\n— infra\n")
    _archive(damaged_dir, b=HEALTHY + "\nand the qualification that follows is")

    _, clean = _scan(healthy_dir)
    assert clean == [], "the widened set must not fire on a sign-off"
    _, broken = _scan(damaged_dir)
    assert any("terminal" in f for f in broken), (
        "the widened set no longer catches a real clip -- widening killed "
        "the guard rather than its false positives")


def test_citation_resolution_is_not_a_default(tmp_path):
    """Opt-in, and the reason is measured rather than stylistic.

    Run over the 37 c2gpt files it produced 28 findings, every one a
    legitimate CROSS-document reference — "Freeze 4", "requirement 4",
    each meaning the other side's numbering. In a two-party conversation
    that is the norm, so the check is meaningful only for self-contained
    documents that number their own sections.
    """
    assert "citation" not in DEFAULTS
    assert "citation" in transit.CHECKS, (
        "the check is kept, not deleted -- it is right for a self-contained "
        "document and the measurement only rules it out as a default")

    body = HEALTHY.replace("See section 1", "See section 9")
    _, off = _scan(_archive(tmp_path, ruling=body))
    assert off == []
    _, on = _scan(tmp_path, enabled=["citation"])
    assert any("citation" in f for f in on), on


# ---- the real-archive scan is the make target, not a test ------------
#
# `make c2c-mailbox-check` runs these tests and then scans the real
# archives. It is not a pytest case, because a pytest case would be
# collected by `make test` and would skip wherever the archives are
# absent -- which is every CI machine, since the mailboxes are gitignored
# and local-only.
#
# Nothing is lost by it not being a test. The anti-vacuity guard that its
# `count >= 30` assertion provided lives in the tool, which exits 2 on an
# empty scan rather than reporting a clean result, and exits 1 on
# findings. Principle 24 is satisfied by the generator being committed
# code either way.
