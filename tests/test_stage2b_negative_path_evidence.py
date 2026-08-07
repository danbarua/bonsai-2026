"""`NEGATIVE_PATH_EVIDENCE.md` cites tests by name. Keep the names real.

That document is the citation table the pre-Stage-4 package hands a
reviewer: five demanded negative paths, each mapped onto the test that
evidences it. Its entire value is that a reader can go look. A renamed or
deleted test turns a row into a claim about nothing -- and nothing about
the ordinary suite notices, because the test still passes under its new
name and the document is not code.

This is CLAUDE.md principle 21 applied to prose: the document is a
hand-maintained list standing in for a set that can be derived, so derive
the set and assert the list is contained in it.

**One direction only, and that is deliberate.** Every function the
document names must exist. The converse -- every test in `tests/` must
appear in the document -- is NOT asserted, because the document is a
curated subset by design: it cites the tests that evidence five specific
demands, not the suite. Asserting the reverse would make adding any
unrelated test fail this check.
"""
import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
EVIDENCE_DOC = (REPO_ROOT / "experiments" / "stage2b_denoising"
                / "NEGATIVE_PATH_EVIDENCE.md")

# A citation is a backticked token. Two shapes appear in the document: a
# path (optionally with a `:line` suffix, navigational only) and a bare
# test function name.
_FILE_CITATION = re.compile(r"`(tests/[A-Za-z0-9_./-]+\.py)(?::\d+)?`")
_FUNC_CITATION = re.compile(r"`(test_[a-z0-9_]+)`")

# Anything the regexes would read as a citation but which is not one.
# Each entry needs a reason, and a test below asserts each is still
# genuinely absent from `tests/` -- an exemption for something that now
# exists is an exemption hiding a real citation.
NOT_CITATIONS = {}


def _doc_text():
    return EVIDENCE_DOC.read_text()


def _declared_test_functions():
    """Every `def test_*` in `tests/`, derived from the AST rather than a
    grep, so a name inside a string or a comment cannot satisfy a citation."""
    found = {}
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    found.setdefault(node.name, []).append(
                        path.relative_to(REPO_ROOT).as_posix())
    return found


def test_the_evidence_document_exists_and_cites_something():
    """The vacuity guard. Every assertion below is over an extracted set;
    an extraction that silently returns nothing would pass all of them."""
    assert EVIDENCE_DOC.exists(), f"{EVIDENCE_DOC} is gone"
    funcs = set(_FUNC_CITATION.findall(_doc_text()))
    files = set(_FILE_CITATION.findall(_doc_text()))
    print(f"\n[evidence] {len(funcs)} function citations, {len(files)} file "
          f"citations extracted from {EVIDENCE_DOC.name}")
    assert len(funcs) >= 25, (
        f"only {len(funcs)} function citations found -- the extraction is "
        f"probably broken, not the document")
    assert len(files) >= 3, f"only {len(files)} file citations found"
    # A specific known citation, so a regex that matches the wrong thing in
    # roughly the right quantity still fails.
    assert "test_a_leak_never_masks_the_scientific_verdict" in funcs


def test_every_cited_test_function_still_exists():
    cited = set(_FUNC_CITATION.findall(_doc_text())) - set(NOT_CITATIONS)
    declared = _declared_test_functions()
    missing = sorted(name for name in cited if name not in declared)
    print(f"[evidence] {len(cited)} cited functions, all resolved"
          if not missing else f"[evidence] UNRESOLVED: {missing}")
    assert not missing, (
        "NEGATIVE_PATH_EVIDENCE.md cites tests that no longer exist:\n"
        + "\n".join(f"  {n}" for n in missing)
        + "\nEither the test was renamed (update the citation) or deleted "
          "(the demand lost its evidence and the row must say so).")


def test_every_cited_test_file_still_exists():
    cited = set(_FILE_CITATION.findall(_doc_text()))
    missing = sorted(p for p in cited if not (REPO_ROOT / p).exists())
    for p in sorted(cited):
        print(f"[evidence] file citation {p}: {'ok' if (REPO_ROOT / p).exists() else 'MISSING'}")
    assert not missing, f"cited test files that do not exist: {missing}"


def test_every_exemption_is_still_genuinely_not_a_test():
    """An exemption that names something real is an exemption concealing a
    citation the check was supposed to make."""
    declared = _declared_test_functions()
    for name, reason in NOT_CITATIONS.items():
        assert reason, f"exemption {name} carries no reason"
        assert name not in declared, (
            f"{name} is exempted as 'not a citation' but IS a test in "
            f"{declared[name]} -- remove the exemption")
