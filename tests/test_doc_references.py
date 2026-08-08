"""Every repo path a document names must exist.

A document that cites a file which is not there is a claim nobody checked,
and it fails in the direction that costs most: the reader who follows the
citation concludes the thing does not exist, or -- worse -- that they are
looking in the wrong place. Both were live when this was written, and both
came from ordinary maintenance rather than carelessness:

  * `docs/proposals/CLAUDE_MD_PRINCIPLE_24_AMENDMENT.md` cited the
    provenance contract at its pre-promotion path, and went on to describe
    the promotion as a future decision after it had happened.
  * `docs/GLOSSARY.md` cited `stage1b_pilot/FINDINGS.md`, missing the
    `experiments/` prefix -- a path that resolves in a shell run from the
    right directory and nowhere else.

Neither is exotic. A file moves, and the citations do not move with it. This
project generated a lot of documentation in a short time, and prose is the
one artifact nothing else in the suite reads.

WHY A TEST RATHER THAN A READ-THROUGH. Reading finds today's stale
citations and establishes nothing about tomorrow's. This is the same
argument the repository makes everywhere else: hand-verified functionality
becomes an executable test, or it lives only in a transcript.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCS = sorted(REPO_ROOT.glob("docs/**/*.md")) + [
    REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"]

# A backticked token shaped like a repo path. Backticks are the filter that
# keeps prose out: this repository writes paths in code spans, and a bare
# word with a dot in it is usually a sentence.
_FILE = re.compile(
    r"`([A-Za-z0-9_][A-Za-z0-9_./*-]*"
    r"\.(?:py|md|toml|yaml|yml|txt|sh|json|lock))`")
_DIR = re.compile(
    r"`((?:docs|tools|tests|src|experiments|benchmark_programme|datasets|"
    r"tarballs)/[A-Za-z0-9_./*-]*)`")

# Paths a doc may name that will NOT exist in a clean checkout. Each carries
# the reason, and each is tested in the other direction below -- an
# exemption whose justification has evaporated is a stale exemption, which
# is the shape principle 21 warns about.
EXEMPT = {
    # Gitignored local-only data. Their absence is the documented condition,
    # not a broken reference.
    "datasets/kmnist/":
        "gitignored dataset; PROJECT_MEMORY documents it as local-only",
    "experiments/stage0_simulator_calibration/results/stage1a_all_classes.pkl":
        "gitignored cached artifact, regenerable, never committed",
    # Specified but deliberately not built.
    "tests/test_provenance_citations.py":
        "the citation verifier is design-only in PROVENANCE_CONTRACT.md §5; "
        "the document says so in the section heading",
    # Paths inside ANOTHER repository, cited so a claim can be re-checked.
    "src/mcp/install-mcp-server.ts":
        "a path in anthropics/claude-code-action, not this repo",
}


def _strip_fenced(text: str) -> str:
    """Drop fenced code blocks before extracting citations.

    A path inside a fence is usually an EXAMPLE -- a template's placeholder,
    a sample command, a transcript -- and an example is not a claim that a
    file exists. `docs/REVIEW_COMMENT_TEMPLATE.md` shows the shape of a
    findings table using `tests/test_x.py`, and reading that as a broken
    citation is the checker misunderstanding the document rather than the
    document being wrong.

    The accepted cost, stated because it is real: a document whose ONLY
    mention of a tool is inside a fence is no longer covered. Prose is where
    a document makes claims about the repository, which is what this test
    is about, so the trade lands the right way -- but it is a trade.
    """
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def _cited_paths(doc: Path) -> set[str]:
    text = _strip_fenced(doc.read_text())
    found = set(_FILE.findall(text)) | set(_DIR.findall(text))
    # Globs are patterns, not citations; bare filenames name no location.
    return {c for c in found if "*" not in c and "/" in c}


def _all_citations() -> dict[str, set[str]]:
    return {d.relative_to(REPO_ROOT).as_posix(): _cited_paths(d)
            for d in DOCS if d.exists()}


def test_documents_were_actually_scanned():
    """Anti-vacuity, and the first thing to break if the doc tree moves.

    Without this, a glob that matched nothing would make every check below
    pass over an empty set -- which is category A in this repository's own
    catalogue, committed inside the checker for it.
    """
    citations = _all_citations()
    assert len(citations) >= 8, (
        f"only {len(citations)} documents found; the docs tree moved or the "
        f"glob stopped matching")
    total = sum(len(v) for v in citations.values())
    assert total >= 50, (
        f"only {total} path citations extracted across {len(citations)} "
        f"documents. The extraction is broken, not the documents")


def test_every_cited_path_exists():
    broken: list[str] = []
    for doc, paths in _all_citations().items():
        for path in sorted(paths):
            if path in EXEMPT or (REPO_ROOT / path).exists():
                continue
            broken.append(f"{doc} cites `{path}`, which does not exist")
    assert not broken, (
        "documents cite paths that are not there:\n  " + "\n  ".join(broken)
        + "\n\nA file moved and its citations did not. Fix the path, or -- "
          "if the absence is expected -- add it to EXEMPT with the reason.")


def test_every_exemption_is_still_needed():
    """The other direction. An exemption for a path that now EXISTS is
    silently switching the check off for a file it would otherwise verify,
    and reads to the next person as a considered decision."""
    resurrected = [p for p in EXEMPT if (REPO_ROOT / p).exists()]
    assert not resurrected, (
        f"these paths exist now and no longer need exempting: "
        f"{sorted(resurrected)}. Remove them so the check covers them again")


def test_every_exemption_is_still_cited():
    """An exemption for a path no document mentions is dead weight that
    looks like coverage."""
    cited = set().union(*_all_citations().values())
    orphaned = [p for p in EXEMPT if p not in cited]
    assert not orphaned, (
        f"exemptions name paths no document cites any more: "
        f"{sorted(orphaned)}. Remove them")


def test_an_example_inside_a_fence_is_not_a_citation(tmp_path):
    """The break that motivated fence-stripping, and its non-vacuity partner.

    A placeholder in a template's example block must not be read as a broken
    citation -- and prose outside the fence must still be read as one, or
    stripping fences would have switched the whole check off.
    """
    doc = tmp_path / "sample.md"
    doc.write_text(
        "Real prose cites `docs/PROJECT_MEMORY.md`.\n\n"
        "```markdown\n"
        "| 1 | `tests/test_totally_made_up.py::test_y` | A |\n"
        "```\n")
    cited = _cited_paths(doc)
    assert "tests/test_totally_made_up.py" not in cited, (
        "a placeholder inside a fenced example was read as a citation")
    assert "docs/PROJECT_MEMORY.md" in cited, (
        "stripping fences also removed the prose citations, which would "
        "make this whole check pass over nothing")


@pytest.mark.parametrize("sample,should_match", [
    ("see `docs/PROJECT_MEMORY.md` for detail", True),
    ("in `experiments/stage1b_pilot/FINDINGS.md`", True),
    ("the `tools/ci/` directory", True),
    ("run `pytest -m \"not slow\"` first", False),
    ("that is principle 21, i.e. derive it", False),
    ("`stage2b` moved under in-flight branches", False),
])
def test_the_extractor_finds_paths_and_not_prose(sample, should_match, tmp_path):
    """The guard, seen firing and seen not firing. An extractor that matched
    ordinary backticked prose would be turned off within a day."""
    doc = tmp_path / "sample.md"
    doc.write_text(sample)
    assert bool(_cited_paths(doc)) is should_match, (
        f"extractor {'missed' if should_match else 'false-positived on'}: "
        f"{sample!r}")
