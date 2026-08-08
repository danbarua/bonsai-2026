"""The catalogue must not quantify itself in prose.

`docs/VACUOUS_TESTS.md` grows. Every count written into its prose is a claim
that goes stale silently, in the one document whose whole subject is claims
nobody re-checked.

Both instances this test was written for were live when it was added, and
the second is the instructive one:

  * "Sixteen incidents are catalogued below" -- the table held twenty.
  * "Deliberate breakage, nine of eighteen" -- NINE was correct and
    EIGHTEEN was stale. A sentence half of which is still true reads as
    checked, which is worse than one that is plainly wrong.

Found by the vacuous-test review reading the catalogue as its specification
and noticing the intro disagreed with the table beneath it -- an LLM review
catching a defect no deterministic check was watching for. This is the
deterministic check, added afterwards, so the next one fails here instead.

The rule is the same one CLAUDE.md principle 24 applies to numbers
generally: derive it, or do not state it. The table IS the count, and a
reader who wants the number can read the table -- which cannot disagree with
itself.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = REPO_ROOT / "docs" / "VACUOUS_TESTS.md"

_NUM = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|twenty-one|twenty-two|\d+")

# What counts as the catalogue quantifying ITSELF, as opposed to ordinary
# prose that happens to contain a number. The distinction is the whole
# design: a pattern that flagged "all three of the driver's joins" or "Two
# of the three assertions" -- both real sentences in this document,
# describing incidents rather than counting them -- would be edited out
# within a day, and an edited-out guard is no guard.
#
# Two shapes, because the two live instances had different ones:
#   A. a number attached to catalogue vocabulary  ("Sixteen incidents")
#   B. a number OF a number, terminal             ("nine of eighteen.")
# B requires the second number to end the clause, which is what separates
# it from "Two of the three assertions".
_COUNT_WORD = re.compile(
    rf"\b(?:{_NUM})\b(?:\s+of)?(?:\s+the)?\s+"
    rf"(?:incidents?|entries|categor\w+|taxonom\w+)"
    rf"|\b(?:{_NUM})\s+of\s+(?:the\s+)?(?:{_NUM})\b(?=[\s]*[.,;:)])",
    re.I)


def _text() -> str:
    assert CATALOGUE.exists(), f"{CATALOGUE} is gone"
    # Normalised: the file is hard-wrapped, so "nine of\neighteen" must read
    # as one phrase. A grep for the raw string finds neither instance.
    return " ".join(CATALOGUE.read_text().split())


def catalogue_ids() -> set[int]:
    """Incident numbers, from the table. Rows may name a RANGE (`6-7`)."""
    ids: set[int] = set()
    for line in CATALOGUE.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or not re.match(r"^\d", cells[0]):
            continue
        span = re.match(r"^(\d+)\s*-\s*(\d+)$", cells[0])
        if span:
            ids.update(range(int(span.group(1)), int(span.group(2)) + 1))
        else:
            ids.add(int(cells[0]))
    return ids


def test_the_table_parses_and_is_not_empty():
    """Anti-vacuity for everything below. If the table format changes, the
    derivation silently returns nothing and every count check passes over an
    empty set -- category A's own shape, in the checker for it."""
    ids = catalogue_ids()
    assert len(ids) >= 15, (
        f"only {len(ids)} incidents parsed from the catalogue table. Either "
        f"the table format changed or the parser stopped matching; an empty "
        f"derivation makes the checks below assert nothing")


def test_the_incident_numbers_are_contiguous():
    """A gap means a row was dropped or misnumbered, and a duplicate means
    two incidents share an identity that other documents cite by number."""
    ids = catalogue_ids()
    missing = set(range(1, max(ids) + 1)) - ids
    assert not missing, (
        f"the catalogue numbers jump: {sorted(missing)} absent below "
        f"#{max(ids)}. Other documents cite these by number")


def test_the_prose_states_no_count_of_itself():
    """The check the two live instances motivated."""
    offenders = [m.group(0) for m in _COUNT_WORD.finditer(_text())]
    assert not offenders, (
        f"the catalogue quantifies itself in prose: {offenders}. It grows, "
        f"so every such number goes stale silently -- and a half-stale one "
        f"('nine of eighteen', where nine was right) reads as checked. "
        f"Derive it or drop it; the table is the count, and it currently "
        f"holds {len(catalogue_ids())} incidents")


@pytest.mark.parametrize("sample,why", [
    ("Sixteen incidents are catalogued below",
     "the exact sentence that was stale by four"),
    ("Deliberate breakage, nine of eighteen.",
     "half-stale: the numerator was correct, which is why it survived"),
    ("across eight categories",
     "the taxonomy grows too"),
])
def test_the_count_check_catches_the_real_sentences(sample, why):
    """The guard, seen failing, on the exact strings that were live rather
    than on invented ones."""
    assert _COUNT_WORD.search(sample), (
        f"the pattern would not have caught: {sample!r} -- {why}")


@pytest.mark.parametrize("sample", [
    "CLAUDE.md principle 21 states it as",
    "spanning 2026-08-04 to 2026-08-08",
    "Incident #10 is the clearest demonstration",
    "one person can decline to write something plausible into it",
])
def test_the_count_check_does_not_fire_on_ordinary_prose(sample):
    """The other direction. A check that flagged principle numbers or dates
    would be edited out within a day, and an edited-out guard is no guard."""
    assert not _COUNT_WORD.search(sample), (
        f"false positive on ordinary prose: {sample!r}")
