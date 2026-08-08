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

# Candidate markers. **Deliberately over-inclusive**, and the asymmetry is
# the whole design: with an explicit-disposition model, an over-matched
# candidate costs one line saying "not binding, because ...", while an
# under-matched one is invisible and takes a green readiness signal with it.
#
# Measured on this project's frozen record (601 sentences across DESIGN.md,
# AUDIT_PROTOCOL.md, COMPANION_PROTOCOLS.md, STAGE3_PLAN.md): an RFC-2119
# marker list -- MUST / MUST NOT / HALT / NEVER, uppercase -- matched THREE.
# The record was written in prose, years of it, before any reconciler
# existed: `MUST` appears 0 times, lowercase `never` 23, `locked` 22,
# `frozen` 41. A narrow list would have derived 3 clauses, seen 3 mapped,
# and exited 0 over 2% coverage.
#
# So detection is not attempted. These patterns nominate; a human disposes.
_CANDIDATE_MARKERS = (
    ("MUST", re.compile(r"\bmust\b", re.I)),
    ("HALT", re.compile(r"\bhalts?\b|\bhalting\b", re.I)),
    ("REFUSE", re.compile(r"\brefus", re.I)),
    ("NEVER", re.compile(r"\bnever\b", re.I)),
    ("REQUIRED", re.compile(r"\brequire[ds]?\b", re.I)),
    ("LOCKED", re.compile(r"\block(?:ed|s)?\b", re.I)),
    ("FROZEN", re.compile(r"\bfrozen\b|\bfreeze[sd]?\b", re.I)),
    ("CANNOT", re.compile(r"\bcannot\b|\bcan not\b|\bmay not\b", re.I)),
    ("SHALL", re.compile(r"\bshall\b", re.I)),
    ("FORBIDDEN", re.compile(r"\bforbidden\b|\bnot permitted\b|\bprohibit", re.I)),
    ("ONLY", re.compile(r"\bonly ever\b|\bthe one place\b|\bexactly one\b", re.I)),
    ("ALWAYS", re.compile(r"\balways\b|\bunconditional", re.I)),
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
    for kind, pattern in _CANDIDATE_MARKERS:
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


# The Reviewer's six dimensions. Fail-closed completeness (6) falls out of
# every other field being required, so it is not a field of its own.
#
# `trigger` is deliberately NOT one of these. It records SCHEDULING -- what
# causes the test to run -- and a `manual` value is reported rather than
# failed. `production_reachability` is a different axis entirely: a correct
# predicate that production never invokes is not an implemented gate, and
# that is a failure. Collapsing the two would let "somebody runs it by hand"
# stand in for "production actually calls it".
_REQUIRED_DIMENSIONS = {
    "enforcement": "(1) the executable predicate",
    "production_reachability": "(2) the production path(s) that reach it, "
                               "INCLUDING wrappers and orchestration",
    "input_wiring": "(3) the concrete runtime values supplied, and their "
                    "provenance",
    "decision_consequence": "(4) the observable halt or branch on rejection "
                            "-- logging a failed predicate without preventing "
                            "continuation is insufficient unless the design "
                            "explicitly defines the gate as advisory",
    "test": "(5a) the test",
    "break_demonstrated": "(5b) evidence the test flips red under a "
                          "deliberate local disable -- a source grep, symbol "
                          "reference, or test that merely names the predicate "
                          "does not qualify",
    "trigger": "what schedules the test (separate axis from reachability)",
}


def check_ids_unique(clauses: list[Clause]) -> list[Finding]:
    """Two distinct clauses sharing an id would let one mapping cover both.

    A gap that looks mapped, which is the worst shape available here. The
    risk is not hypothetical in a record that repeats formulations --
    "locked", "frozen", "never overwritten" -- across several documents,
    and normalization (whitespace collapse + lowercasing) raises the odds.
    Collisions fail rather than merge.
    """
    findings: list[Finding] = []
    seen: dict[str, Clause] = {}
    for clause in clauses:
        prior = seen.get(clause.clause_id)
        if prior is not None and prior.text != clause.text:
            findings.append(Finding(
                "id_collision",
                f"{clause.clause_id} is shared by {prior.doc}:{prior.line} "
                f"and {clause.doc}:{clause.line} -- one disposition would "
                f"silently cover both"))
        else:
            seen[clause.clause_id] = clause
    return findings


def reconcile(clauses: list[Clause], inventory: dict,
              repo_root: Path = REPO_ROOT) -> list[Finding]:
    """Every way a documented gate and its enforcement can fail to agree.

    Checked in BOTH directions. A one-directional check passes happily
    while the documents grow requirements nobody enforced, or while the
    inventory keeps entries for requirements that no longer exist.
    """
    findings: list[Finding] = list(check_ids_unique(clauses))
    binding = inventory.get("binding", {})
    not_binding = inventory.get("not_binding", {})
    by_id = {clause.clause_id: clause for clause in clauses}

    # Direction 1: every candidate carries an explicit DISPOSITION.
    #
    # Not "is mapped" -- dispositioned. Detection of what is binding is a
    # semantic judgement with no derivation available, so it is not
    # attempted; candidates are nominated broadly and a human rules on each.
    # An unreviewed candidate is a finding rather than an absence, which is
    # what converts "silently missed" into "visibly unclassified" and is the
    # only move that survives a semantic boundary.
    for clause_id, clause in by_id.items():
        if clause_id in binding or clause_id in not_binding:
            continue
        findings.append(Finding(
            "undispositioned_candidate",
            f"{clause.doc}:{clause.line} [{clause.kind}] has no disposition "
            f"(id {clause_id}): {clause.text[:110]}",
            clause))

    # Direction 2: every disposition still names a live candidate.
    for clause_id in list(binding) + list(not_binding):
        if clause_id not in by_id:
            findings.append(Finding(
                "orphaned_disposition",
                f"inventory entry {clause_id} matches no candidate in the "
                f"documents -- the sentence was reworded or removed, and its "
                f"disposition no longer describes anything"))

    for clause_id, entry in binding.items():
        for dimension, description in _REQUIRED_DIMENSIONS.items():
            if not entry.get(dimension):
                findings.append(Finding(
                    f"missing_{dimension}",
                    f"{clause_id} has no `{dimension}` -- {description}"))
        # Only the two code references are machine-checkable; the rest are
        # prose claims a human makes and a reviewer reads. Checking what can
        # be checked is not the same as validating the row, and this tool
        # does not pretend otherwise.
        for reference_field in ("enforcement", "test"):
            reference = entry.get(reference_field)
            if reference:
                ok, why = _resolve(reference, repo_root)
                if not ok:
                    findings.append(Finding(
                        f"unresolved_{reference_field}", f"{clause_id}: {why}"))
        if entry.get("trigger") == "manual":
            # Reported, not failed -- some gates legitimately cost too much
            # to automate. This is SCHEDULING, and distinct from dimension 2:
            # "nobody has automated the test" is a weakness worth seeing,
            # while "production never invokes the predicate" is a failure and
            # is caught by `production_reachability` being required.
            findings.append(Finding(
                "manual_trigger_only",
                f"{clause_id} is enforced only by a test somebody must "
                f"remember to run: {entry.get('test')}"))

    for clause_id, entry in not_binding.items():
        if not entry.get("reason"):
            findings.append(Finding(
                "unreasoned_disposition",
                f"{clause_id} is dispositioned not-binding with no reason"))
    return findings


def coverage(clauses: list[Clause], inventory: dict) -> tuple[int, int]:
    """(dispositioned, total candidates). The floor a readiness run asserts."""
    dispositioned = set(inventory.get("binding", {})) | set(
        inventory.get("not_binding", {}))
    ids = {clause.clause_id for clause in clauses}
    return len(ids & dispositioned), len(ids)


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
    done, total = coverage(clauses, inventory)

    print(f"candidates derived : {total}")
    print(f"dispositioned      : {done}")
    print(f"  binding          : {len(inventory.get('binding', {}))}")
    print(f"  not binding      : {len(inventory.get('not_binding', {}))}")

    if not clauses:
        # A run over zero candidates reports zero findings and looks clean.
        print("\nNO CANDIDATES DERIVED -- the scan found nothing, which is "
              "not the same as everything being dispositioned.",
              file=sys.stderr)
        return 2

    # A zero-guard is not enough, and the reason is measured. A narrow
    # marker list derived THREE candidates from a 601-sentence record; three
    # dispositions would have exited 0 over 2% coverage. Three is not zero,
    # so the zero-guard never fires. The floor is the guard that does.
    if done < total:
        print(f"\nCOVERAGE {done}/{total} -- below the 100% floor. "
              f"Undispositioned candidates are unreviewed, not absent.",
              file=sys.stderr)

    print(f"\nfindings           : {len(findings)}")
    for finding in findings:
        print(f"  [{finding.kind}] {finding.detail}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
