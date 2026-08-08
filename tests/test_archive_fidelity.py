"""Each transit-integrity heuristic, shown to fire on what it watches.

`tools/mailbox/check_transit_integrity.py` returns zero findings over both
real archives, and a check that has only ever returned zero is
indistinguishable from a check that cannot return anything else.

**Why the tests are here and the tool is there.** Two implementations of
this existed for about twenty minutes — one specifying the checks, one
implementing them — which is the drift risk this project spends its time
removing, one level up. The implementation that survived is the one with
measured false-positive rates behind its defaults and three defects found
by breaking it. What it did not have was tests. So: one tool, one test
file, and the fixture in `test_a_clip_inside_inline_code_is_caught` is the
real defect that killed the duplicate's terminator set.

Tier 1 throughout: synthetic fixtures. The real-archive scan is separate
and skips when the archive is absent, because both archives are local-only.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "tools" / "mailbox" / "check_transit_integrity.py"
C2GPT_ARCHIVE = REPO_ROOT / ".claude" / "claude2gpt" / "archive"

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


@pytest.mark.skipif(not C2GPT_ARCHIVE.is_dir(),
                    reason="the c2gpt archive is local-only, not committed")
def test_the_reviewer_archive_has_no_transit_tells():
    """The number quoted in `experiments/stage2b_denoising/gates.toml`.

    That file's provenance note states the scan covered every archive file
    with zero findings. Principle 24: the number anchors a durable record,
    so it is reproducible from committed code rather than from the heredoc
    that first produced it.

    REPORTS its evidence rather than only asserting — for a check whose
    normal output is "0", how many files were examined is most of what a
    reader needs. And the assertion it does NOT make is the point: a clean
    run means no DETECTABLE transit loss. Nothing in a file can prove it
    matches what was sent; only a transport-attested channel could, and
    this one is not.
    """
    count, findings = transit.scan([C2GPT_ARCHIVE], DEFAULTS)
    print(f"\nc2gpt archive files scanned: {count}")
    for finding in findings:
        print(f"  {finding}")
    assert count >= 30, (
        f"only {count} files scanned; the archive should hold the full "
        f"c2gpt history")
    assert findings == []
