"""Each truncation heuristic, shown to fire on what it watches.

`tools/provenance/check_archive_fidelity.py` returns zero findings over
the real archive, and a check that has only ever returned zero is
indistinguishable from a check that cannot return anything else. So every
one of the three is exercised against a deliberately truncated fixture,
and against a healthy one that must NOT trip it.

Tier 1 throughout: synthetic fixtures in `tmp_path`. The real-archive scan
is a separate test that skips when the archive is absent, because the
archive is local-only.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "tools" / "provenance" / "check_archive_fidelity.py"

spec = importlib.util.spec_from_file_location("_archive_fidelity", MODULE)
fidelity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fidelity
spec.loader.exec_module(fidelity)


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


def test_a_healthy_ruling_produces_no_findings(tmp_path):
    """Non-vacuity for all three: a fixture that trips nothing.

    Without this, a check that flagged every file would look identical to
    a working one in the tests below -- they only assert that a finding
    appears, and a check that always fires satisfies that.
    """
    report = fidelity.scan(_archive(tmp_path, ruling=HEALTHY))
    assert report.files_scanned == 1
    assert report.findings == []


def test_a_file_clipped_mid_sentence_is_caught(tmp_path):
    """The case citation resolution cannot see.

    The clipped text here is the SHAPE of the live risk: a closing
    qualification, referenced by nothing, so every internal citation still
    resolves and the file still ends on a complete-looking section.
    """
    clipped = HEALTHY + "\nI do **not** require an automated prose checker for"
    report = fidelity.scan(_archive(tmp_path, ruling=clipped))
    assert [f.kind for f in report.findings] == ["ends_mid_clause"]
    assert "prose checker for" in report.findings[0].detail


def test_a_dropped_middle_section_is_caught(tmp_path):
    """Mid-file loss where nothing references the missing section.

    Section 2 is removed along with the sentence that pointed at 1, so the
    citation check stays quiet and only the ordinal gap remains as a tell.
    """
    dropped = HEALTHY.replace(
        "### 2. Second matter\n\nSee section 1 for the basis.\n\n", "")
    report = fidelity.scan(_archive(tmp_path, ruling=dropped))
    kinds = [f.kind for f in report.findings]
    assert "ordinal_gap" in kinds, kinds
    assert "unresolved_reference" not in kinds, (
        "this fixture must isolate the ordinal check -- if the citation "
        "check also fires, the test does not show the gap was needed")


def test_a_reference_to_a_missing_section_is_caught(tmp_path):
    """The check c2c-implementation is building, kept honest here too."""
    body = HEALTHY.replace("See section 1 for the basis.",
                           "See section 9 for the basis.")
    report = fidelity.scan(_archive(tmp_path, ruling=body))
    assert [f.kind for f in report.findings] == ["unresolved_reference"]
    assert "'9'" in report.findings[0].detail


def test_a_file_with_no_sections_does_not_cry_wolf_on_references(tmp_path):
    """Most of the archive has no numbered headings.

    A reference in such a file points at ANOTHER document, and flagging it
    would produce a finding on the majority of the archive -- which trains
    a reader to ignore the check, the failure mode that makes a guard
    worse than none.
    """
    prose = "<!-- from: chatgpt -->\n\nSee section 5 of the design document.\n"
    report = fidelity.scan(_archive(tmp_path, note=prose))
    assert report.findings == []


def test_an_empty_file_is_a_finding_not_a_pass(tmp_path):
    report = fidelity.scan(_archive(tmp_path, ruling="   \n\n"))
    assert [f.kind for f in report.findings] == ["empty_file"]


def test_a_run_that_does_not_start_at_one_is_allowed(tmp_path):
    """A ruling continuing a prior message may open at section 3.

    Checking runs from wherever they start, rather than from 1, is what
    keeps this from firing on a legitimate continuation -- and the
    distinction is why the gap check tests contiguity, not numbering.
    """
    body = HEALTHY.replace("### 1. First", "### 4. Fourth") \
                  .replace("### 2. Second", "### 5. Fifth") \
                  .replace("### 3. Third", "### 6. Sixth") \
                  .replace("See section 1 for the basis.",
                           "See section 4 for the basis.")
    assert fidelity.scan(_archive(tmp_path, ruling=body)).findings == []


@pytest.mark.skipif(not fidelity.DEFAULT_ARCHIVE.is_dir(),
                    reason="the c2gpt archive is local-only, not committed")
def test_the_real_archive_has_no_truncation_tells():
    """The number quoted in `experiments/stage2b_denoising/gates.toml`.

    That file's provenance note states the scan covered every archive file
    with zero findings in all three checks. Principle 24: the number
    anchors a durable record, so it is reproducible from committed code
    rather than from the heredoc that first produced it.

    REPORTS its evidence rather than only asserting -- for a check whose
    entire output is normally "0", the count of files actually examined is
    most of what a reader needs.
    """
    report = fidelity.scan()
    print(f"\narchive files scanned: {report.files_scanned}")
    for finding in report.findings:
        print(f"  [{finding.kind}] {finding.file}: {finding.detail}")
    assert report.files_scanned >= 30, (
        f"only {report.files_scanned} files scanned; the archive should "
        f"hold the full c2gpt history")
    assert report.findings == []
