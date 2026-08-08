#!/usr/bin/env python3
"""Detect transit loss in relayed mailbox archives.

Some mail reaches an archive by hand rather than over a connector -- most
c2gpt rulings were pasted by Dan, the connector being frequently
unavailable. A paste can clip a clause or drop a trailing section, and the
failure is presence-shaped: **a partial ruling reads complete**. Nothing in
the file marks the loss, and a reader cannot see it from inside.

Three checks, each covering what the others cannot:

1. `citation` -- a self-reference (``section 5``, ``point 3``, ``matter II``)
   must resolve within the same file.
2. `terminal`  -- the file must end on a sentence terminator.
3. `ordinal`   -- numbered headings must run contiguously from 1.

The second and third exist because of a blind spot `stage2b-lead` found in
the first, and the reasoning is worth keeping: citation resolution only
catches truncation of a section that something POINTS AT, and the last
section of a document is exactly what nothing points at. It is also where
qualifications live. The live example was the final paragraph of the
Stage 2B inventory ruling -- "I do not require an artificial automated prose
checker" -- which nothing in its file references. Had it been clipped, every
citation would still have resolved, the file would have ended on the
preceding section looking complete, and the reader would have built a prose
checker they had been explicitly told not to build.

So the original check was strongest where loss is least dangerous and blind
where it is worst: **a losing clause creates work you invent rather than a
gap you notice.** Terminal completeness targets end-of-file truncation
directly -- the case with no inbound reference by construction.

All three are heuristics over what is IN the file. None can prove a file
matches what was sent; only a transport-attested channel could. A clean run
means no detectable transit loss, never "faithful".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A sentence may legitimately end on punctuation, a closing bracket or quote,
# a table row, or a list marker's own terminator. Anything else at
# end-of-file suggests the text stops mid-clause.
#
# A bare backtick is NOT here, and the break-test is why: clipping a real
# ruling mid-sentence landed on "For `" -- the opening backtick of an inline
# code span -- and the check passed, because backtick had been included for
# code fences. A truncation that stops inside inline code is exactly the
# case this check exists for, so the character that made it invisible had to
# go. Closing fences are handled separately below.
TERMINATORS = tuple('.!?:;)]}"\'|-')
FENCE = '```'

# Mesh messages often sign off on their own line ("— stage2b-lead", "-- infra").
# That is a convention, not a clipped sentence: measured over the 117-message
# code2code archive it produced 23 false positives before being handled. Skip
# trailing signature lines and judge the content line above them.
SIGNATURE_RE = re.compile(r'^\s*(?:[—–-]{1,2})\s*[\w@.`\- ]{1,40}\s*$')

# Self-references. Deliberately narrow: matching bare numbers would fire on
# every quantity in the corpus.
CITATION_RE = re.compile(
    r'(?:§\s*(\d+)'
    r'|\b(?:section|point|item|freeze|matter|part|requirement|question)\s+'
    r'(\d+|[IVX]+)\b)',
    re.IGNORECASE,
)

# Numbered headings/items at line start: "### 3." / "3. " / "## II."
ORDINAL_RE = re.compile(r'^#{0,4}\s*(\d+)\.\s+\S', re.MULTILINE)
ROMAN_RE = re.compile(r'^#{1,4}\s*([IVX]+)\.\s+\S', re.MULTILINE)

ROMAN_VALUES = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}


def _roman(token: str) -> int | None:
    return ROMAN_VALUES.get(token.upper())


def check_terminal(text: str) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ['file is empty']
    while len(lines) > 1 and SIGNATURE_RE.match(lines[-1]):
        lines.pop()
    last = lines[-1]
    if last.strip() == FENCE:
        return []
    if not last.endswith(TERMINATORS):
        return [f'ends mid-clause: ...{last[-60:]!r}']
    return []


def check_ordinal(text: str) -> list[str]:
    findings = []
    for label, pattern, conv in (
        ('arabic', ORDINAL_RE, int),
        ('roman', ROMAN_RE, _roman),
    ):
        seen = []
        for m in pattern.finditer(text):
            v = conv(m.group(1))
            if v is not None:
                seen.append(v)
        # Only sequences that actually start at 1 are treated as a numbered
        # run; a stray "2." in prose is not a gap.
        if not seen or seen[0] != 1:
            continue
        run = [seen[0]]
        for v in seen[1:]:
            if v == run[-1] + 1:
                run.append(v)
            elif v > run[-1] + 1:
                findings.append(
                    f'{label} ordinal gap: {run[-1]} -> {v} '
                    f'(missing {run[-1] + 1})'
                )
                run.append(v)
    return findings


def check_citations(text: str) -> list[str]:
    arabic_heads = {int(m.group(1)) for m in ORDINAL_RE.finditer(text)}
    roman_heads = {_roman(m.group(1)) for m in ROMAN_RE.finditer(text)}
    roman_heads.discard(None)
    findings = []
    for m in CITATION_RE.finditer(text):
        token = m.group(1) or m.group(2)
        value = int(token) if token.isdigit() else _roman(token)
        if value is None:
            continue
        if value in arabic_heads or value in roman_heads:
            continue
        # A reference may legitimately point at another document. Report it
        # rather than failing on it -- said explicitly so a reader does not
        # read this as proof of loss.
        findings.append(
            f'unresolved self-reference {m.group(0)!r} '
            f'(no matching numbered section in this file; '
            f'may point at another document)'
        )
    return findings


CHECKS = {
    'terminal': check_terminal,
    'ordinal': check_ordinal,
    'citation': check_citations,
}


def scan(paths: list[Path], enabled: list[str]) -> tuple[int, list[str]]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob('*.md')))
        elif p.is_file():
            files.append(p)
    report = []
    for f in files:
        text = f.read_text(encoding='utf-8', errors='replace')
        for name in enabled:
            for finding in CHECKS[name](text):
                report.append(f'{f.name}: [{name}] {finding}')
    return len(files), report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', nargs='+', type=Path,
                    help='archive directories or individual .md files')
    # `citation` is OPT-IN, and measurement is why. Run over the 37-message
    # c2gpt archive it produced 28 findings, every one a legitimate
    # CROSS-document reference -- a reply naming "Freeze 4" or "requirement 4"
    # means the other side's numbering, not its own. In a two-party
    # conversation that is the norm, not the exception, so the check cries
    # wolf at a rate that would bury a real finding. It is meaningful only for
    # self-contained documents that number their own sections. Keeping it
    # available and off by default beats deleting it or leaving it noisy.
    ap.add_argument('--checks', default='terminal,ordinal',
                    help='comma-separated subset of: terminal, ordinal, citation '
                         '(citation is opt-in: noisy on conversational corpora)')
    args = ap.parse_args()

    enabled = [c.strip() for c in args.checks.split(',') if c.strip()]
    unknown = set(enabled) - set(CHECKS)
    if unknown:
        print(f'unknown check(s): {sorted(unknown)}', file=sys.stderr)
        return 2

    n_files, report = scan(args.paths, enabled)

    # Exit 2 on an empty scan, following the gate-inventory precedent: a
    # checker over zero files reports nothing and looks immaculate, and here
    # that green would read as "no transit loss detected".
    if n_files == 0:
        print('no files scanned -- refusing to report a clean result',
              file=sys.stderr)
        return 2

    for line in report:
        print(line)
    print(f'\nscanned {n_files} file(s), {len(report)} finding(s)')
    if report:
        print('findings are transit-loss SIGNALS, not proof; and a clean run '
              'means no DETECTABLE loss, never that the file matches what '
              'was sent.')
    return 1 if report else 0


if __name__ == '__main__':
    raise SystemExit(main())
