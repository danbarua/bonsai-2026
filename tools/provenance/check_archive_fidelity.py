#!/usr/bin/env python3
"""Truncation heuristics for the c2gpt archive. Not an attestation.

`c2gpt-send` takes `from` as a routing PARAMETER, not an attestation of
authorship: a message written by the ChatGPT connector and one pasted by
hand are byte-identical in the archive, header included. Most replies
arrived by paste, because the connector is frequently unavailable.

Split the consequence, because only one half is at risk:

  * AUTHORITY is unaffected. Every relay had the authorizing human in the
    loop by construction -- a pasted ruling is the authorization pathway
    working, not a weaker version of it.
  * FIDELITY is attested by the relaying human, not by the channel. A
    copy-paste round trip can clip a clause or drop a trailing section,
    **and a partial ruling reads complete.**

That last property is why this file exists. Nothing here proves a ruling
arrived whole; these are three cheap checks for the ways a truncated one
betrays itself.

## Why three checks and not one

Internal citation resolution is the obvious check and it has a blind
spot that matters more than its coverage: it catches the loss of a
section something POINTS AT, and the final section of a ruling is
pointed at by nothing. That is also where qualifications live.

The live example, in the ruling this project's binding-clause inventory
depends on most (`2026-08-08T10-24-39Z`): its closing paragraph licenses
NOT building an automated prose checker for the 54 human-discharged
claims. Nothing in the file references that paragraph. Clipped in
transit, every internal citation would still resolve, the file would end
on section III looking complete -- and the forbidden work would have been
built in the belief that the ruling required it.

**A missing licence does not leave a gap you notice; it creates work you
invent.** So `terminal_completeness` and `ordinal_continuity` are not
belt-and-braces on the citation check; they cover the case it cannot see.

Usage:
    uv run python tools/provenance/check_archive_fidelity.py
    uv run python tools/provenance/check_archive_fidelity.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO_ROOT / ".claude" / "claude2gpt" / "archive"

# A sentence that survived transit ends in one of these. Deliberately
# permissive -- a ruling may end on a quote, a code span, a parenthetical
# or a colon introducing a block, and flagging those would train a reader
# to ignore the check, which is worse than not having it.
_TERMINATORS = (".", "!", "?", '"', "'", "`", ")", "]", ":", "*", "_")

# `## I.` / `### 3.` / `## Matter II` -- the forms these rulings use.
_ORDINAL_HEADING = re.compile(r"^#{2,4}\s+(?:matter\s+)?(\d+|[IVX]+)[.)]", re.I | re.M)

# A reference INTO the same document: "§5", "section 3", "matter III".
_INTERNAL_REF = re.compile(r"§\s*(\d+|[IVX]+)|\bsection\s+(\d+|[IVX]+)\b"
                           r"|\bmatter\s+(\d+|[IVX]+)\b", re.I)

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
          "VIII": 8, "IX": 9, "X": 10}


def _ordinal(token: str) -> int | None:
    token = token.strip().upper()
    if token.isdigit():
        return int(token)
    return _ROMAN.get(token)


@dataclass
class Finding:
    file: str
    kind: str
    detail: str


@dataclass
class Report:
    files_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"files_scanned": self.files_scanned,
                "findings": [f.__dict__ for f in self.findings]}


def terminal_completeness(name: str, text: str) -> list[Finding]:
    """End-of-file truncation -- the case with no inbound reference."""
    stripped = text.rstrip()
    if not stripped:
        return [Finding(name, "empty_file", "the file has no content")]
    if not stripped.endswith(_TERMINATORS):
        return [Finding(name, "ends_mid_clause",
                        f"last 60 chars: {stripped[-60:]!r}")]
    return []


def ordinal_continuity(name: str, text: str) -> list[Finding]:
    """Mid-file section loss, even where nothing references the section.

    A gap in `1. 2. 4.` is the tell a copy-paste dropped a block. Runs are
    checked from wherever they start rather than from 1, because a ruling
    may legitimately open at section 3 when it continues a prior message.
    """
    numbers = [_ordinal(m.group(1)) for m in _ORDINAL_HEADING.finditer(text)]
    numbers = [n for n in numbers if n is not None]
    if len(numbers) < 2:
        return []
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected:
        return [Finding(name, "ordinal_gap",
                        f"section numbers {numbers}, expected {expected}")]
    return []


def citation_resolution(name: str, text: str) -> list[Finding]:
    """Every section a ruling references must exist within the file.

    Only applied when the file HAS numbered headings: a reference in a
    file with no sections is pointing at another document, and treating
    that as unresolved would make the check cry wolf on the majority of
    the archive.
    """
    present = {n for n in (_ordinal(m.group(1))
                           for m in _ORDINAL_HEADING.finditer(text))
               if n is not None}
    if not present:
        return []
    findings = []
    for match in _INTERNAL_REF.finditer(text):
        token = next(g for g in match.groups() if g)
        target = _ordinal(token)
        if target is not None and target not in present:
            findings.append(Finding(
                name, "unresolved_reference",
                f"references section {token!r}, which is not in this file "
                f"(present: {sorted(present)})"))
    return findings


CHECKS = (terminal_completeness, ordinal_continuity, citation_resolution)


def scan(archive: Path = DEFAULT_ARCHIVE) -> Report:
    report = Report()
    for path in sorted(archive.glob("*.md")):
        report.files_scanned += 1
        text = path.read_text()
        for check in CHECKS:
            report.findings.extend(check(path.name, text))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.archive.is_dir():
        print(f"no archive at {args.archive}", file=sys.stderr)
        return 2

    report = scan(args.archive)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"files scanned : {report.files_scanned}")
        print(f"findings      : {len(report.findings)}")
        for finding in report.findings:
            print(f"  [{finding.kind}] {finding.file}: {finding.detail}")
        if not report.findings:
            print("\nNo truncation tells. This is NOT evidence the rulings "
                  "arrived whole -- it is the absence of the three ways a "
                  "truncated one betrays itself.")
    return 1 if report.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
