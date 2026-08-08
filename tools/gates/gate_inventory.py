#!/usr/bin/env python3
"""Reconcile documented gates against the code that enforces them.

The mechanism half of the blocking readiness requirement: *every documented
MUST/HALT mapped to its enforcement point and its test; a documented gate
with no executable mapping itself fails readiness.*

The incident behind it: `DESIGN.md` froze the sentence "HALT for review if
any production condition selects 1e-6", the grid extension was implemented
without that halt, and a run that selected the floor reported OK — a verdict
meaning *no such gate exists*, not *the gate cleared*. Document and code had
the same author, in the same session, hours apart. There was no handoff to
blame, which is what rules out "be more careful" as the remedy.

**What is derived and what is not.** The clause set is derived from the
documents; the mapping from a clause to its enforcement site is not, and
cannot be — that correspondence is the human judgement this exists to
record. So the design puts every mechanical property under a check and
leaves exactly one thing to a person: which code enforces which sentence.

The trick that keeps the mapping honest is the clause id. It is a hash of
the clause's normalized TEXT, not its line number or an author-assigned
name. Edit a frozen MUST and its id changes, so its mapping orphans and
`reconcile` reports it. A mapping cannot quietly survive the rewording of
the requirement it claims to enforce — which is the specific way a
hand-maintained correspondence rots.

Usage:
    uv run python tools/gates/gate_inventory.py --inventory gates.toml
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Markers that make a sentence binding. A pattern list, and -- as with the
# scratch predicate -- there is no derivation available: no enumeration
# exists of "ways a document states a requirement". It is kept explicit,
# tested against a corpus, and extended by incident rather than by guess.
# Order matters only for reporting `kind`; matching is by any.
_MARKERS = (
    ("HALT", re.compile(r"\bHALT\b")),
    ("MUST_NOT", re.compile(r"\bMUST NOT\b|\bmust not\b")),
    ("MUST", re.compile(r"\bMUST\b|\bmust\b")),
    ("REFUSE", re.compile(r"\brefuses? to\b|\bREFUS")),
    ("NEVER", re.compile(r"\bNEVER\b|\bnever\b")),
)

# Lines that look binding but are not requirements about the system.
_SKIP_LINE = re.compile(r"^\s*(?:#|>|\||```|\d+\.\s*~~)")


@dataclass(frozen=True)
class Clause:
    """One binding sentence found in a document."""
    doc: str
    line: int
    kind: str
    text: str

    @property
    def clause_id(self) -> str:
        """Stable across reflowing, unstable across rewording -- deliberately.

        Derived from normalized text so that moving a clause down the file
        or rewrapping a paragraph does not orphan its mapping, while
        CHANGING what it requires does.
        """
        normalized = re.sub(r"\s+", " ", self.text).strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]


@dataclass
class Finding:
    kind: str
    detail: str
    clause: Clause | None = None


def _normalise_paragraph(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines).strip()


def derive_clauses(docs: list[Path], repo_root: Path = REPO_ROOT) -> list[Clause]:
    """Every binding sentence in `docs`, as a derived set.

    Paragraph-scoped rather than line-scoped: these documents are hard
    wrapped, so a requirement routinely spans three lines and a line-based
    scan would both miss clauses and split them. (The same hard-wrapping
    is why a `grep` for a backticked path in this repo can come up empty
    while the rendered text plainly contains it.)
    """
    clauses: list[Clause] = []
    for doc in docs:
        if not doc.exists():
            continue
        rel = doc.relative_to(repo_root).as_posix()
        paragraph: list[str] = []
        start = 0
        in_fence = False
        for number, raw in enumerate(doc.read_text().splitlines(), start=1):
            if raw.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if raw.strip():
                if not paragraph:
                    start = number
                paragraph.append(raw)
                continue
            if paragraph:
                clauses.extend(_clauses_in(paragraph, rel, start))
                paragraph = []
        if paragraph:
            clauses.extend(_clauses_in(paragraph, rel, start))
    return clauses


def _clauses_in(paragraph: list[str], doc: str, start: int) -> list[Clause]:
    if any(_SKIP_LINE.match(line) for line in paragraph[:1]):
        return []
    text = _normalise_paragraph(paragraph)
    for kind, pattern in _MARKERS:
        if pattern.search(text):
            return [Clause(doc=doc, line=start, kind=kind, text=text)]
    return []


def load_inventory(path: Path) -> dict:
    """The human half: clause id -> enforcement site, test, break evidence."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _defines(path: Path, name: str) -> bool:
    """Does `path` define `name` as a function, method or module constant?

    Resolved from the AST rather than by substring, so a name appearing in
    a comment or a string cannot satisfy a citation.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def _resolve(reference: str, repo_root: Path) -> tuple[bool, str]:
    """Check a `path/to/file.py::name` reference actually resolves."""
    if "::" not in reference:
        path = repo_root / reference
        return (path.exists(), f"{reference} does not exist")
    file_part, name = reference.split("::", 1)
    path = repo_root / file_part
    if not path.exists():
        return False, f"{file_part} does not exist"
    if not _defines(path, name):
        return False, f"{file_part} does not define {name}"
    return True, ""


def reconcile(clauses: list[Clause], inventory: dict,
              repo_root: Path = REPO_ROOT) -> list[Finding]:
    """Every way a documented gate and its enforcement can fail to agree.

    Checked in BOTH directions. A one-directional check passes happily
    while the documents grow requirements nobody enforced, or while the
    inventory keeps entries for requirements that no longer exist.
    """
    findings: list[Finding] = []
    mappings = inventory.get("gate", {})
    exemptions = inventory.get("exempt", {})
    by_id = {clause.clause_id: clause for clause in clauses}

    # Direction 1: every derived clause is mapped or exempted.
    for clause_id, clause in by_id.items():
        if clause_id in mappings or clause_id in exemptions:
            continue
        findings.append(Finding(
            "unmapped_clause",
            f"{clause.doc}:{clause.line} [{clause.kind}] has no enforcement "
            f"mapping (id {clause_id}): {clause.text[:110]}",
            clause))

    # Direction 2: every mapping and exemption still names a live clause.
    for clause_id in list(mappings) + list(exemptions):
        if clause_id not in by_id:
            findings.append(Finding(
                "orphaned_mapping",
                f"inventory entry {clause_id} matches no clause in the "
                f"documents -- the requirement was reworded or removed, and "
                f"its mapping no longer describes anything"))

    # The mapped references must resolve, and the break evidence must exist.
    for clause_id, entry in mappings.items():
        for field_name in ("enforcement", "test"):
            reference = entry.get(field_name)
            if not reference:
                findings.append(Finding(
                    "incomplete_mapping",
                    f"{clause_id} has no `{field_name}`"))
                continue
            ok, why = _resolve(reference, repo_root)
            if not ok:
                findings.append(Finding(
                    f"missing_{field_name}", f"{clause_id}: {why}"))
        if not entry.get("break_demonstrated"):
            findings.append(Finding(
                "unproven_test",
                f"{clause_id} does not record that its test was demonstrated "
                f"to FAIL without the gate. A cited test certifies spelling: "
                f"a grep for the enforcement call survives the branch being "
                f"disabled by `if False:`"))

        # What causes the test to RUN. A gate whose test only executes when
        # somebody remembers is closer to an unenforced requirement than an
        # enforced one -- the same "component correct, path to it wrong"
        # signature as a hook that never loads or a predicate never handed
        # its input, with `never run` as the mechanism. Observed: a correct,
        # self-deriving guard caught a file hours late because firing it
        # required a human to run the full suite in another session.
        trigger = entry.get("trigger")
        if not trigger:
            findings.append(Finding(
                "untriggered_test",
                f"{clause_id} does not record what RUNS its test"))
        elif trigger == "manual":
            # Reported, not fatal. Some gates legitimately cost too much to
            # run automatically; the point is that this is visible in the
            # inventory rather than discovered when one fires late.
            findings.append(Finding(
                "manual_trigger_only",
                f"{clause_id} is enforced only by a test somebody must "
                f"remember to run: {entry.get('test')}"))

    for clause_id, entry in exemptions.items():
        if not entry.get("reason"):
            findings.append(Finding(
                "unreasoned_exemption",
                f"{clause_id} is exempted with no reason"))
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--doc", type=Path, action="append", default=[],
                        help="document to scan; repeatable")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    clauses = derive_clauses(args.doc, args.root)
    inventory = load_inventory(args.inventory)
    findings = reconcile(clauses, inventory, args.root)

    print(f"clauses derived : {len(clauses)}")
    print(f"gates mapped    : {len(inventory.get('gate', {}))}")
    print(f"exempted        : {len(inventory.get('exempt', {}))}")
    if not clauses:
        # A run over zero clauses reports zero findings and looks clean.
        print("\nNO CLAUSES DERIVED -- the scan found nothing, which is not "
              "the same as everything being mapped.", file=sys.stderr)
        return 2
    print(f"\nfindings        : {len(findings)}")
    for finding in findings:
        print(f"  [{finding.kind}] {finding.detail}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
