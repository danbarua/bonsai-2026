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
import subprocess
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


def _is_git_ignored(path: str) -> bool:
    """Does git ignore this path?

    The DERIVED discriminator between the two kinds of absence in `EXEMPT`,
    and the reason it is derived rather than a second hand-labelled dict:
    a hand-split list is the artifact principle 21 says will silently
    under-cover, and it would need maintaining in step with `.gitignore`.

    A git-ignored path is local-only BY CONSTRUCTION, so its presence is a
    property of the machine rather than of the repository. `check-ignore`
    consults the ignore rules, not the filesystem, so it answers for absent
    paths too -- which is what the other direction needs.

    Self-correcting: if `datasets/` were ever committed, the path stops
    being ignored and its exemption correctly becomes stale.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path],
            capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        # No git, or no repository. Treat as not-ignored: the staleness
        # check then applies, which errs toward reporting rather than
        # toward silence.
        return False


def test_every_exemption_is_still_needed():
    """The other direction. An exemption for a path that now EXISTS is
    silently switching the check off for a file it would otherwise verify,
    and reads to the next person as a considered decision.

    Applies ONLY to paths git does not ignore, and that qualification was a
    real defect rather than a nicety. `EXEMPT` holds two kinds of absence,
    and this assertion was true of one and false of the other:

      environment-dependent  `datasets/kmnist/`, the cached `.pkl` -- absent
                             in CI, PRESENT on any working machine, which is
                             the intended state
      environment-independent  the unbuilt verifier, a path in another repo
                             -- absent everywhere

    Reported by stage2b-lead, who was running a red suite because of it
    while CI stayed green for lacking the data. It is the crc32c incident
    with the sign flipped, and their diagnosis is the line I had sent them
    that morning turned back on me: when one predicate serves two questions,
    it IS the shared blind spot. One `EXEMPT` dict, two questions, and the
    observable -- "the path exists" -- means opposite things for each.
    """
    resurrected = [p for p in EXEMPT
                   if (REPO_ROOT / p).exists() and not _is_git_ignored(p)]
    assert not resurrected, (
        f"these paths exist now, are tracked, and no longer need exempting: "
        f"{sorted(resurrected)}. Remove them so the check covers them again")


def test_the_discriminator_separates_the_two_kinds_of_absence():
    """The whole fix rests on `git check-ignore`, so check it discriminates
    rather than assuming it does.

    Both directions matter. If it returned True for everything the staleness
    check would be switched off entirely -- a guard passing over nothing,
    which is this file's own subject. If it returned False for everything we
    are back to the red suite that prompted the fix.
    """
    environment_dependent = [p for p in EXEMPT if _is_git_ignored(p)]
    environment_independent = [p for p in EXEMPT if not _is_git_ignored(p)]
    print(f"\n[exempt] git-ignored (local-only): {sorted(environment_dependent)}")
    print(f"[exempt] tracked (absent everywhere): {sorted(environment_independent)}")
    assert environment_dependent, (
        "no exemption is git-ignored, so the discriminator classifies "
        "everything as tracked and the staleness check is unchanged -- "
        "meaning the defect it was written for is still present")
    assert environment_independent, (
        "every exemption is git-ignored, so the staleness check now applies "
        "to nothing at all. It would pass over an empty set")


def test_a_stale_tracked_exemption_still_fires(tmp_path):
    """The guard, seen failing. A tracked file that exists and is exempted
    is exactly what this check exists to catch, and the fix must not have
    quietly excused it along with the local-only data."""
    tracked_and_present = "tests/test_doc_references.py"
    assert (REPO_ROOT / tracked_and_present).exists()
    assert not _is_git_ignored(tracked_and_present), (
        "this test's own premise is wrong -- the file it uses as a stand-in "
        "for a tracked path is git-ignored")
    resurrected = [p for p in {tracked_and_present: "pretend"}
                   if (REPO_ROOT / p).exists() and not _is_git_ignored(p)]
    assert resurrected, (
        "a tracked, existing, exempted path was NOT flagged. The "
        "git-ignore qualification has switched the staleness check off "
        "rather than narrowing it")


def test_local_only_data_present_on_a_working_machine_does_not_fire():
    """The false positive that was red on a machine with the datasets, and
    green in CI for lacking them. Presence of git-ignored data is the
    intended state of a working checkout, not a stale exemption."""
    for path in ("datasets/kmnist/",
                 "experiments/stage0_simulator_calibration/results/"
                 "stage1a_all_classes.pkl"):
        assert path in EXEMPT, f"{path} is no longer exempted; update this test"
        assert _is_git_ignored(path), (
            f"{path} is no longer git-ignored, so the staleness check now "
            f"applies to it and will fire on any machine that has the data")


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
