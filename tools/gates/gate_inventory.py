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
# Across this project's frozen record -- DESIGN.md, AUDIT_PROTOCOL.md,
# COMPANION_PROTOCOLS.md, STAGE3_PLAN.md -- an uppercase RFC-2119 marker
# list matches a tiny fraction of what the broad set nominates. The record
# was written in prose, years of it, before any reconciler existed. A narrow
# list would have derived a handful of clauses, seen them all mapped, and
# exited 0 over a sliver of the record -- and the zero-guard would not fire,
# because a handful is not zero.
#
# The figures are GENERATED, not stated here:
# `tests/test_gate_inventory.py::test_the_real_record_is_mostly_lowercase_prose`
# measures them against the documents and prints them. An earlier version of
# this comment quoted a sentence count and a match count, both wrong -- wrong
# in unit as well as magnitude, since the derivation walks PARAGRAPHS and
# returns one clause each, never sentences. Neither had been measured by
# anyone here; both arrived from a peer's message. See
# `docs/PROVENANCE_CONTRACT.md` §3.2a.
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
    # "may not" is the obvious form; "NO metric MAY be added" and "may only"
    # are the ones that slipped. Found by a fixture written for an unrelated
    # test, which is a fair indication of how many more forms this record
    # uses that nobody has thought of -- and the reason disposition, not
    # detection, is what the coverage floor rests on.
    ("CANNOT", re.compile(r"\bcannot\b|\bcan not\b|\bmay not\b|"
                          r"\bno\s+\w+\s+may\b|\bmay only\b", re.I)),
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
        # Resolved against the root before use: the first thing anyone types
        # is the relative form, and `relative_to` raises on it.
        doc = doc if doc.is_absolute() else (repo_root / doc)
        if not doc.exists():
            continue
        try:
            rel = doc.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel = doc.as_posix()  # a document outside the tree still scans
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
    """Does `path` define `name` as a function, class or module constant?

    Resolved from the AST rather than by substring, so a name appearing in
    a comment or a string cannot satisfy a citation.

    **All four binding forms, and the omissions were a real defect.** The
    first version handled `Assign` only, so `ALPHA_BAR: float = 0.5` and
    `A, B = 1, 2` both failed to resolve -- and this repository uses
    annotated module constants, including in this very file. A citation to
    one would have reported `unresolved_enforcement`: a FALSE finding,
    sending a reader to hunt a problem that does not exist, and eroding
    trust in every true finding beside it.

    Found by applying a peer's rule to this function rather than agreeing
    with it: *checking the artifact is necessary and not sufficient -- know
    what its fields mean before you reason from them.* This walk assumed it
    knew what "defines a name" means in Python. It knew one of four ways.

    Deliberately NOT counted as defining: a name merely IMPORTED here. An
    enforcement citation should point at where the thing is defined, not at
    a module that re-exports it, and a re-export would let one gate be
    cited from anywhere that happens to import it.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return False

    def binds(target: ast.expr) -> bool:
        if isinstance(target, ast.Name):
            return target.id == name
        # `A, B = 1, 2` and `[A, B] = ...` bind through a sequence.
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(binds(element) for element in target.elts)
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        if isinstance(node, ast.Assign):
            if any(binds(target) for target in node.targets):
                return True
        # `ALPHA_BAR: float = 0.5` -- the form a frozen constant most often
        # takes in typed code, and the one originally missed.
        if isinstance(node, ast.AnnAssign) and binds(node.target):
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
# THE ANTI-GAMING CLAUSE, quoted verbatim from the Stage 2B reviewer ruling
# of 2026-08-08 and binding on every use of this schema:
#
#   "Classification itself is reviewable: a clause cannot be moved from
#    `binding_gate` to another kind merely because enforcement is absent."
#
# It closes the only loophole the trichotomy opens. Three kinds make the
# inventory honest about a record that is mostly promises -- and would also
# let an unimplemented gate be reclassified as a value or a claim until it
# stops failing. The kinds describe what a clause IS, not how much work its
# enforcement would be, and no check in this file can tell the difference.
# That one is enforced by a reviewer reading the classifications.
#
# It closes the loophole it names, and there are three. Reported by the
# instance filling the inventory, in the order of how honest each feels while
# you are making it:
#
#   1. reclassify a gate you cannot enforce -- named above, feels like
#      cheating, and is the one people expect to be watched
#   2. drop the clause to `not_binding` -- feels like judgement
#   3. cite ADJACENT EVIDENCE -- feels like doing the work
#
# So, binding as the clause above:
#
#   A citation must discharge the clause AS WRITTEN, not a nearby claim. A
#   real, passing test cited for an obligation it does not establish
#   satisfies every mechanical check in this file.
#
# The third is the dangerous one, and no check here can see it. The paid
# instance: a clause required that new per-image OOF MSEs reproduce the
# fold-aggregates STORED BY earlier runs. An existing OOF cross-check pinned
# the per-fold mean against a within-run recomputation on synthetic data --
# real, passing, same subject, right-sounding name, and a different claim.
# The whole point of the clause was pinning new machinery against numbers
# this project already trusts, and the citation would have pinned it against
# fresh expectations. `_resolve` confirms the reference EXISTS; existence was
# never the question.
#
# Nothing was reclassified, the row was `binding_gate` correctly, the test
# was genuine, and the reconciler would have reported nothing. It was caught
# by a person reading what the cited test actually asserted and noticing the
# docstring said "within-run" where the clause said "stored". This is the
# same instrument as VACUOUS_TESTS' "a citation of a gate is not evidence of
# a gate", turned on the inventory instead of on tests -- and naming the move
# is most of the defence against it, because the person making it experiences
# themselves as having found the test rather than as having lowered the bar.
#
# **Three kinds of binding, because a frozen protocol contains three kinds
# of promise.** A single seven-field schema forces every clause with no
# runtime decision into one of two lies -- `not_binding`, which is false for
# clauses that are among the most binding things in the record, or `binding`
# with invented `decision_consequence` prose for a clause that decides
# nothing. The second is the confabulation this design refused a model for;
# writing it by hand is not better.
#
# The evidence for the trichotomy is that filled rows land in all three
# kinds -- checkable by reading `gates.toml` -- not a count. An earlier
# version of this comment quantified the split three ways. Those figures
# reached this file from a mesh message rather than from anything runnable,
# which makes them load-bearing scratch by principle 24, and a reviewer
# then observed they coincided exactly with the MUST/FROZEN/rest marker
# histogram. Marker kind is demonstrably NOT semantic kind: LOCKED- and
# HALT-marked clauses are classified `binding_gate`. So the coincidence
# could not be dismissed and the figures could not be reproduced -- both
# reasons to state the structural argument and let `counts_by_kind` report
# the live numbers, which it does, from the file, on every run.
_REQUIRED_BY_KIND: dict[str, dict[str, str]] = {}

# Statuses, named once. They appear in three places -- the schema text a
# filler reads, the check that rejects an unknown one, and the tests -- and
# a hand-copied list in any of the three is the failure mode this file is
# about. Order is the order they are offered in.
_VALUE_STATUSES = ("enforced", "pending_consumer", "unresolved")
_CLAIM_STATUSES = ("discharged", "pending_package", "not_applicable",
                   "superseded", "unresolved")

# `superseded` exists because "discharged as amended" is not a thing.
#
# Reviewer ruling of 2026-08-09, on B10: a frozen clause that a later
# amendment validly replaced was NOT discharged -- the thing it required
# stopped being required. Recording that as a discharge "allows any violated
# frozen clause to appear green merely by redefining its meaning later",
# which is the escape this whole inventory exists to close, arriving through
# the status column instead of the kind column.
#
# So the history is represented rather than flattened: the original clause is
# `superseded`, names its successor and the amendment that made it, and the
# successor is separately inventoried as the surviving obligation. Execution
# evidence then demonstrates compliance with the SUCCESSOR, which is the true
# statement, rather than with an original the run did not satisfy.
#
# It fails readiness like every other non-discharged status. A superseded
# clause is not a closed one: somebody must still confirm the amendment was
# validly made, and that judgement is a reviewer's.

# Requirements that are CONDITIONAL on status, because demanding them
# unconditionally forces a lie.
#
# `discharged_in` and `evidence` name the package artifact and section where
# a claim's compliance is recorded. For a claim whose readiness package does
# not exist yet, there is no such artifact: the only ways to satisfy a
# required field are to invent a plausible-looking section in a package
# nobody has assembled -- the exact confabulation `triage_candidates.py`
# refuses a model for, and no better done by hand -- or to accept a finding
# per field per row, which buries the real findings. A findings list nobody
# can read is the green-that-means-nothing in the other direction.
#
# So they are required only where they can be true. The row still FAILS
# readiness via `pending_package`, and still carries a `pending_reason`
# naming what does not exist yet. Nothing is silently passed; what changes
# is that the row can be honest while failing, which `binding_value`'s
# `pending_consumer` already allows and this kind did not.
_REQUIRED_ONLY_WHEN_STATUS: dict[str, dict[str, frozenset[str]]] = {
    "binding_claim": {
        "discharged_in": frozenset({"discharged"}),
        "evidence": frozenset({"discharged"}),
        # A supersession with no successor is a deletion with better
        # manners: the obligation vanishes and nothing takes its place.
        # Both fields are what make the history traceable rather than
        # merely asserted.
        "superseded_by": frozenset({"superseded"}),
        "amendment_locator": frozenset({"superseded"}),
    },
}

# Fields that describe a row's RELATIONSHIP to another row rather than its
# own content. Both exist because the Reviewer's two cross-cutting rulings
# (2026-08-09) found the same structural gap from opposite directions.
#
#   canonical_clause  this row restates an obligation inventoried in full
#                     elsewhere. Part A, on A4: "'the binding content lives
#                     elsewhere' is valid only as DEDUPLICATION, not as a
#                     reason that the present text ceases to be binding.
#                     Without [a typed relationship] not_binding becomes
#                     structurally unsafe whenever the same obligation is
#                     restated in multiple places."
#   parent_clause     this row is one obligation of a paragraph that carries
#                     several. Part B: "split mixed clauses by obligation
#                     semantics rather than forcing the paragraph to have one
#                     kind. Paragraph identity can remain as provenance, but
#                     the inventory needs child obligations."
#
# A row carrying `canonical_clause` is exempt from its kind's field contract
# -- the canonical row holds the evidence, and demanding it twice would
# invite a second, divergent account of the same obligation -- and is
# reported on its own count line so it cannot inflate enforcement coverage.
_RELATIONSHIP_FIELDS = ("canonical_clause", "parent_clause")


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

_REQUIRED_BY_KIND["binding_gate"] = _REQUIRED_DIMENSIONS

# A frozen constant has no runtime decision and no production path to
# reach: nothing rejects anything. Its enforcement is a test asserting the
# value, and demanding `decision_consequence` for it would be demanding
# fiction.
_REQUIRED_BY_KIND["binding_value"] = {
    "value": "the authoritative source and the frozen value itself",
    "production_consumers": "EVERY production consumer of the value, or a "
                            "proved common propagation point -- one consumer "
                            "cited for a value read in four places is a row "
                            "that covers a quarter of what it claims",
    "enforcement": "the pinning/identity test",
    "break_demonstrated": "causal evidence the test fails when the PRODUCTION "
                          "value or its propagation is altered -- not merely "
                          "when the constant literal is edited, which tests "
                          "that the literal equals itself",
    "provenance_of_use": "evidence sufficient to establish the artifact or run "
                         "actually USED the frozen value, rather than that the "
                         "value was frozen somewhere",
    "trigger": "what schedules that test",
    "status": " | ".join(_VALUE_STATUSES),
}

# Binding on what may be CLAIMED, not on what the program does -- "must
# never be reported as a random sample", "no metric may be added or dropped
# after results exist". No code path can enforce these, and pretending one
# could is the failure this whole requirement exists to prevent.
_REQUIRED_BY_KIND["binding_claim"] = {
    "locator": "source document plus a STABLE locator -- a line number moves, "
               "a section or anchor does not",
    "obligation": "the normalized obligation: concisely, what must or must "
                  "not be claimed or done",
    "discharged_in": "the package artifact plus section/field where "
                     "compliance is discharged",
    "status": " | ".join(_CLAIM_STATUSES),
    "evidence": "a reviewer-checkable evidence pointer, or quoted local "
                "context, sufficient to VERIFY the discharge without "
                "searching the package. Point at an ARTIFACT, never at who "
                "checked it: an agent attribution is not reviewer-checkable, "
                "and a mesh `instance:` tag names a ROLE rather than a "
                "session -- an instance can truthfully deny sending a "
                "message that carries its tag, so it cannot carry provenance "
                "into a readiness package",
    "superseded_by": "the clause id of the obligation that REPLACED this "
                     "one. Required only at status=superseded, and it must "
                     "resolve to a row in a binding kind that is neither "
                     "superseded itself nor a duplicate -- without that, "
                     "'redefine the meaning later' is available one hop away",
    "amendment_locator": "where the amendment that superseded this clause is "
                         "disclosed, with its date. Required only at "
                         "status=superseded. A supersession whose amendment "
                         "cannot be found is indistinguishable from a clause "
                         "quietly abandoned",
}

# Negative obligations need more than a compliant example, and the reason is
# logical rather than procedural: pointing at one paragraph that says the
# right thing cannot establish that the prohibited claim is absent from
# everywhere else. So a `must not` / `never` row carries an attestation
# scoped to the output set it ranges over.
_NEGATIVE_OBLIGATION = re.compile(
    r"\bmust not\b|\bnever\b|\bmay not\b|\bcannot\b|\bno\s+\w+\s+may\b", re.I)

# Optional on `binding_claim`: a note that future tooling could plausibly
# enforce this one -- a doc-diff lint for "no metric added after results
# exist", say. It records a possibility so it is not lost.
#
# **It changes nothing.** A tagged row is still listed as unenforceable,
# still requires `discharged_in`, and is still discharged by a person. The
# constraint is deliberate: a tag that quietly promoted a row out of the
# human-reads-this list would convert "somebody must check this" into "the
# tool checks this" in everyone's head, which is worse than no tag at all,
# and a half-built lint firing on some cases is exactly how that happens.
# Same reason a `manual` trigger is reported rather than fatal.
_OPTIONAL_BY_KIND = {"binding_claim": ("mechanizable_candidate",)}


def _check_value(clause_id: str, entry: dict) -> list[Finding]:
    """`binding_value` statuses, including the not-yet-consumed case.

    Several frozen values in this record are consumed by code that does not
    exist yet -- `AUDIT_PROTOCOL.md` freezes values the unwritten audit
    driver will read. For those, `production_consumers` is not unknown, it
    is EMPTY, and will stay empty until the driver is written.

    That is not a schema defect: it is the inventory correctly reporting
    that a frozen value has no consumer yet, which is a true and useful
    readiness statement. But it needs a disposition that is neither a lie
    nor an escape hatch, so it gets a status of its own that FAILS
    readiness rather than a prose field somebody can satisfy by writing
    "none yet" into a required box.
    """
    findings: list[Finding] = []
    status = entry.get("status")
    if status == "pending_consumer":
        if not entry.get("pending_reason"):
            findings.append(Finding(
                "unreasoned_pending_consumer",
                f"{clause_id} is pending_consumer with no `pending_reason` "
                f"naming the code that does not exist yet"))
        findings.append(Finding(
            "value_has_no_production_consumer",
            f"{clause_id} is frozen but nothing consumes it yet: "
            f"{entry.get('pending_reason', '(no reason given)')}. Fails "
            f"readiness until the consumer exists -- absence of enforcement "
            f"is a finding, never a reclassification"))
    elif status == "unresolved":
        findings.append(Finding(
            "unresolved_value",
            f"{clause_id} is unresolved and fails readiness"))
    elif status != "enforced":
        findings.append(Finding(
            "unknown_status",
            f"{clause_id} has status {status!r}; expected one of "
            f"{', '.join(_VALUE_STATUSES)}"))
    return findings


def _check_claim(clause_id: str, entry: dict, clause: Clause | None) -> list[Finding]:
    """The `binding_claim` semantics that go beyond required fields."""
    findings: list[Finding] = []
    status = entry.get("status")

    if status == "unresolved":
        findings.append(Finding(
            "unresolved_claim",
            f"{clause_id} is unresolved. An unresolved obligation fails "
            f"readiness; it is not a state a package ships in"))
    elif status == "pending_package":
        # Honest, and still failing. The distinction from `unresolved` is
        # real and worth keeping separate: unresolved means nobody has
        # decided, pending_package means somebody decided and the artifact
        # to discharge into does not exist yet. They need different work --
        # a reviewer versus a package -- so they are different findings.
        if not entry.get("pending_reason"):
            findings.append(Finding(
                "unreasoned_pending_package",
                f"{clause_id} is pending_package with no `pending_reason` "
                f"naming the readiness artifact that does not exist yet"))
        findings.append(Finding(
            "claim_has_no_package",
            f"{clause_id} is binding and cannot be discharged yet: "
            f"{entry.get('pending_reason', '(no reason given)')}. Fails "
            f"readiness until the package exists -- absence of a place to "
            f"discharge into is a finding, never a reclassification"))
    elif status == "not_applicable" and not entry.get("not_applicable_reason"):
        findings.append(Finding(
            "unreasoned_not_applicable",
            f"{clause_id} is marked not_applicable with no reason tied to the "
            f"clause's TRIGGERING CONDITION. Not-applicable is a claim that "
            f"the condition never arose, not an escape hatch"))
    elif status == "superseded":
        # Visible, and still failing. Without its own finding a supersession
        # would be the quietest state in the file: a frozen clause retired
        # with no reader ever asked whether the amendment was legitimate.
        # The mechanical half -- successor exists, is binding, is not itself
        # superseded -- is `check_relationships`. This is the half no check
        # can settle.
        findings.append(Finding(
            "superseded_clause",
            f"{clause_id} was superseded rather than discharged, by "
            f"{entry.get('superseded_by', '(unnamed)')}. Whether the "
            f"amendment legitimately replaced what the freeze was FOR is a "
            f"reviewer's judgement and is open: "
            f"{entry.get('pending_reason', '(no reason given)')}"))
    elif status not in _CLAIM_STATUSES:
        findings.append(Finding(
            "unknown_status",
            f"{clause_id} has status {status!r}; expected one of "
            f"{', '.join(_CLAIM_STATUSES)}"))

    # Negative obligations need an attestation over the output set, because a
    # compliant paragraph cannot prove a prohibited claim is absent elsewhere.
    #
    # Not at `superseded`: the prohibition is no longer in force, so an
    # attestation that it is absent from the output set would be a claim
    # about a rule that stopped applying. The SUCCESSOR carries the
    # attestation for whatever survived -- which is exactly the distinction
    # the status exists to make, and demanding one here would quietly
    # re-assert the retired obligation.
    text = entry.get("obligation") or (clause.text if clause else "")
    if (status != "superseded"
            and _NEGATIVE_OBLIGATION.search(text)
            and not entry.get("negative_attestation")):
        findings.append(Finding(
            "missing_negative_attestation",
            f"{clause_id} is a must-not/never obligation and carries no "
            f"`negative_attestation` scoped to the relevant output set. "
            f"Pointing at one compliant passage cannot establish the "
            f"prohibited claim is absent from the rest"))
    return findings


_SEMANTIC_REVIEW_FIELDS = ("inventory_sha256", "reviewer", "scopes",
                           "findings", "findings_resolved")

# Kinds whose dispositions a machine cannot settle. The Reviewer's ruling of
# 2026-08-09 is that these, not "every row", are what human review is FOR:
#
#   "whether a not_binding disposition really is non-binding; whether a
#   superseded amendment preserves the material purpose of the original
#   freeze; whether canonical_clause genuinely restates rather than weakens
#   or adds to its target; whether a binding_claim is actually discharged by
#   the cited evidence. Those are semantic review decisions."
#
# And the converse, which is what replaced the old boolean: "machine-enforced
# binding_gate/binding_value rows with causal break evidence do not need
# ritual human rereading simply to turn a boolean green."
_SEMANTIC_SCOPES = ("not_binding", "superseded", "canonical_clause",
                    "binding_claim")


def inventory_digest(text: str) -> str:
    """SHA-256 of the inventory with its own review block removed.

    Circular otherwise: an attestation that covered itself would change the
    hash it records by recording it. Everything ABOVE `[semantic_review]` is
    the reviewed content, and the block is the signature over it.
    """
    # Line-anchored, not a bare substring: this file MENTIONS
    # `[semantic_review]` in prose above the table, and splitting on the
    # first occurrence cut the body short -- the attestation then hashed a
    # prefix and reported itself stale on the run that wrote it.
    body = re.split(r"^\[semantic_review\]", text, flags=re.M)[0]
    return hashlib.sha256(body.encode()).hexdigest()


def check_semantic_review(inventory: dict, text: str | None = None) -> list[Finding]:
    """The review attestation, versioned against what it reviewed.

    Replaces a naked `reviewed = true` boolean, on the Reviewer's ruling:
    "I would not use a naked global `reviewed = true` boolean at all unless
    its meaning is extremely explicit. Prefer a versioned review attestation
    tied to the inventory commit/hash, recording the reviewed scope and
    outcome."

    The boolean's defect was that it said nothing about WHAT was reviewed or
    WHEN, so it kept meaning "reviewed" across every subsequent edit. A hash
    makes it expire on the next change to the rows it covered, which is the
    only property that makes an attestation worth anything.
    """
    findings: list[Finding] = []
    review = inventory.get("semantic_review")
    if not isinstance(review, dict):
        findings.append(Finding(
            "no_semantic_review",
            "the inventory carries no `[semantic_review]` block. A draft -- "
            "machine-generated or half-finished -- cannot produce a clean run "
            "no matter what its rows contain"))
        return findings

    for field_name in _SEMANTIC_REVIEW_FIELDS:
        if review.get(field_name) in (None, "", []):
            findings.append(Finding(
                "incomplete_semantic_review",
                f"`[semantic_review]` has no `{field_name}`"))

    if text is not None and review.get("inventory_sha256"):
        actual = inventory_digest(text)
        if review["inventory_sha256"] != actual:
            findings.append(Finding(
                "stale_semantic_review",
                f"the review attests to inventory "
                f"{str(review['inventory_sha256'])[:12]} and the rows now "
                f"hash to {actual[:12]}. Rows changed after they were "
                f"reviewed; the attestation does not cover what is here"))

    missing = [k for k in _SEMANTIC_SCOPES if k not in (review.get("scopes") or [])]
    if missing:
        findings.append(Finding(
            "unreviewed_semantic_scope",
            f"`[semantic_review]` does not cover {missing}. These are the "
            f"dispositions a machine cannot settle, and an attestation that "
            f"omits one leaves exactly the judgements review exists for"))

    unresolved = (review.get("findings") or 0) - (review.get("findings_resolved") or 0)
    if unresolved > 0:
        findings.append(Finding(
            "unresolved_review_findings",
            f"{unresolved} of {review['findings']} review findings are "
            f"recorded as unresolved"))
    return findings


def _binding_rows(inventory: dict) -> dict[str, tuple[str, dict]]:
    """Every row in a binding kind, keyed by id."""
    rows: dict[str, tuple[str, dict]] = {}
    for kind in _REQUIRED_BY_KIND:
        for clause_id, entry in inventory.get(kind, {}).items():
            rows[clause_id] = (kind, entry)
    return rows


def check_relationships(inventory: dict) -> list[Finding]:
    """`superseded_by`, `canonical_clause` and `parent_clause` must land.

    Each of the three is a pointer out of one row into another, and each
    reopens the escape it was added to close if the pointer is allowed to
    terminate anywhere convenient. The chain rules are the load-bearing
    part:

    * a successor may not itself be `superseded` -- otherwise "redefine the
      meaning later" is available one hop away, which is precisely what the
      status was added to stop;
    * a successor may not be a duplicate, for the same reason one hop
      sideways;
    * a duplicate's canonical target may not itself be a duplicate, so a
      row's evidence is always exactly one dereference away rather than at
      the end of a chain nobody walks.

    The analogue of `test_every_binds_at_pointer_names_a_clause_in_a_binding_kind`
    one layer down, and added for the same reason: a pointer that resolves
    to nothing looks identical, in a green run, to an obligation that was
    handled.
    """
    findings: list[Finding] = []
    rows = _binding_rows(inventory)
    not_binding = inventory.get("not_binding", {})
    duplicates = {cid for cid, (_, e) in rows.items() if e.get("canonical_clause")}

    for clause_id, (kind, entry) in sorted(rows.items()):
        target = entry.get("canonical_clause")
        if target:
            if target not in rows:
                findings.append(Finding(
                    "canonical_target_missing",
                    f"{clause_id} [{kind}] restates {target}, which is not "
                    f"dispositioned in any binding kind. A duplicate whose "
                    f"canonical row does not exist is an undispositioned "
                    f"obligation wearing a pointer"))
            elif target in duplicates:
                findings.append(Finding(
                    "canonical_target_is_itself_a_duplicate",
                    f"{clause_id} restates {target}, which restates something "
                    f"else. Chains are refused: the evidence must be exactly "
                    f"one dereference away"))
            elif target == clause_id:
                findings.append(Finding(
                    "canonical_target_is_self",
                    f"{clause_id} names itself as its own canonical row"))
            elif (entry.get("status")
                  and entry["status"] != rows[target][1].get("status")):
                # Found by reading my own rows: a duplicate is skipped by the
                # field contract AND by the per-kind status checks, so any
                # status it declares is unchecked decoration -- and a reader
                # sees `discharged` on a row whose canonical is unresolved.
                # One source of truth: match it or omit it.
                findings.append(Finding(
                    "duplicate_status_disagrees",
                    f"{clause_id} declares status "
                    f"{entry['status']!r} while the row it restates "
                    f"({target}) is {rows[target][1].get('status')!r}. A "
                    f"duplicate carries no evidence of its own, so it cannot "
                    f"be in a different state from the row that does"))

        successor = entry.get("superseded_by")
        if successor:
            if successor not in rows:
                findings.append(Finding(
                    "successor_missing",
                    f"{clause_id} is superseded by {successor}, which is not "
                    f"dispositioned in any binding kind -- so the obligation "
                    f"was removed rather than replaced, and nothing carries "
                    f"what survived"))
            else:
                s_kind, s_entry = rows[successor]
                if s_entry.get("status") == "superseded":
                    findings.append(Finding(
                        "successor_is_itself_superseded",
                        f"{clause_id} is superseded by {successor}, which is "
                        f"itself superseded. A supersession chain lets a "
                        f"frozen clause be redefined one hop at a time, which "
                        f"is the escape this status exists to close"))
                if s_entry.get("canonical_clause"):
                    findings.append(Finding(
                        "successor_is_a_duplicate",
                        f"{clause_id} is superseded by {successor}, which is a "
                        f"duplicate rather than a row carrying its own "
                        f"evidence"))

        parent = entry.get("parent_clause")
        if parent and parent not in rows and parent not in not_binding:
            findings.append(Finding(
                "parent_missing",
                f"{clause_id} [{kind}] is a child obligation of {parent}, "
                f"which is dispositioned nowhere"))

    # One id, one row, across every kind. `check_ids_unique` covers COLLISIONS
    # in the derived corpus; this covers the inventory's own keyspace, which
    # now holds ids that no corpus candidate mints -- children.
    seen: dict[str, str] = {}
    for kind in list(_REQUIRED_BY_KIND) + ["not_binding"]:
        for clause_id in inventory.get(kind, {}):
            if clause_id in seen:
                findings.append(Finding(
                    "duplicate_inventory_id",
                    f"{clause_id} appears in both {seen[clause_id]} and "
                    f"{kind} -- one id, two dispositions, and no way to say "
                    f"which one governs"))
            else:
                seen[clause_id] = kind
    return findings


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
              repo_root: Path = REPO_ROOT,
              inventory_text: str | None = None) -> list[Finding]:
    """Every way a documented gate and its enforcement can fail to agree.

    Checked in BOTH directions. A one-directional check passes happily
    while the documents grow requirements nobody enforced, or while the
    inventory keeps entries for requirements that no longer exist.
    """
    findings: list[Finding] = list(check_ids_unique(clauses))

    # A draft must not be mistakable for a finished inventory. Without this,
    # the failure is obvious in hindsight: a machine-drafted file lands,
    # looks complete, somebody runs the reconciler against it and gets a
    # green. `reviewed` defaults to false, so a file that never says a human
    # read it cannot pass -- an artifact carrying evidence of the thing it
    # claims, rather than being trusted for looking finished.
    findings.extend(check_semantic_review(inventory, inventory_text))

    dispositions: dict[str, tuple[str, dict]] = {}
    for kind in _REQUIRED_BY_KIND:
        for clause_id, entry in inventory.get(kind, {}).items():
            dispositions[clause_id] = (kind, entry)
    not_binding = inventory.get("not_binding", {})
    binding = {cid: entry for cid, (_, entry) in dispositions.items()}
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
    #
    # Child obligations are exempt BY CONSTRUCTION, not by leniency: a child
    # is one obligation of a paragraph that carries several, and the
    # paragraph is the candidate. Its id is minted from its own normalized
    # obligation, so no corpus sentence will ever match it. `parent_clause`
    # is what keeps it anchored, and `check_relationships` requires that
    # parent to be dispositioned -- so a child cannot float free, it is
    # simply anchored to a paragraph rather than to a sentence.
    for clause_id in list(binding) + list(not_binding):
        if clause_id in by_id:
            continue
        entry = binding.get(clause_id) or not_binding.get(clause_id) or {}
        if entry.get("parent_clause"):
            continue
        findings.append(Finding(
            "orphaned_disposition",
            f"inventory entry {clause_id} matches no candidate in the "
            f"documents -- the sentence was reworded or removed, and its "
            f"disposition no longer describes anything"))

    findings.extend(check_relationships(inventory))

    for clause_id, (kind, entry) in dispositions.items():
        # A duplicate carries no evidence of its own: the canonical row holds
        # it, and demanding it twice invites two divergent accounts of one
        # obligation. `check_relationships` has already required the pointer
        # to land in a binding kind that is not itself a duplicate.
        if entry.get("canonical_clause"):
            continue
        for dimension, description in _REQUIRED_BY_KIND[kind].items():
            # A conditionally-required field is demanded only in the statuses
            # where it can be answered truthfully. Everywhere else the row's
            # status carries the finding instead -- see
            # `_REQUIRED_ONLY_WHEN_STATUS`.
            only_when = _REQUIRED_ONLY_WHEN_STATUS.get(kind, {}).get(dimension)
            if only_when is not None and entry.get("status") not in only_when:
                continue
            if not entry.get(dimension):
                findings.append(Finding(
                    f"missing_{dimension}",
                    f"{clause_id} [{kind}] has no `{dimension}` -- {description}"))
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
        if kind == "binding_claim":
            findings.extend(_check_claim(clause_id, entry, by_id.get(clause_id)))
        if kind == "binding_value":
            findings.extend(_check_value(clause_id, entry))

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


def counts_by_kind(clauses: list[Clause], inventory: dict) -> dict[str, int]:
    """Per-kind counts, and deliberately no aggregate percentage.

    Reviewer-mandated, not a presentation preference: *"No aggregate
    percentage that lets human obligations dilute or inflate executable
    coverage."* A single figure over a corpus dominated by human promises
    would report executable coverage as high by counting promises as
    covered -- the green-that-means-nothing at report scale. How dominated
    is a live number this function returns; it is not quoted here, because a
    quoted one goes stale in a file nobody rereads.
    """
    ids = {clause.clause_id for clause in clauses}
    out = {kind: len(ids & set(inventory.get(kind, {})))
           for kind in _REQUIRED_BY_KIND}
    out["not_binding"] = len(ids & set(inventory.get("not_binding", {})))
    out["undispositioned"] = len(ids) - sum(out.values())

    # Reported after the arithmetic above, and separately, because a
    # duplicate is dispositioned but carries no evidence of its own. Folding
    # it into its kind's count would report an obligation as enforced twice
    # on the strength of one row's evidence -- "avoid double-counting
    # coverage", the Reviewer's phrase, and the reason `canonical_clause`
    # exists rather than a second full row.
    #
    # Children are counted here too, and are NOT in the figures above by
    # construction: their ids are minted from their own obligation text, so
    # they never intersect the corpus candidate set. They are additional
    # rows, not additional coverage; the parent paragraph is what the corpus
    # asked about.
    rows = _binding_rows(inventory)
    out["  of which restate another row"] = sum(
        1 for cid, (_, e) in rows.items()
        if e.get("canonical_clause") and cid in ids)
    out["  child obligations (not corpus candidates)"] = sum(
        1 for _, e in rows.values() if e.get("parent_clause"))
    return out


def coverage(clauses: list[Clause], inventory: dict) -> tuple[int, int]:
    """(dispositioned, total candidates). The floor a readiness run asserts."""
    dispositioned = set(inventory.get("not_binding", {}))
    for kind in _REQUIRED_BY_KIND:
        dispositioned |= set(inventory.get(kind, {}))
    ids = {clause.clause_id for clause in clauses}
    return len(ids & dispositioned), len(ids)


def unenforceable(clauses: list[Clause], inventory: dict) -> list[Clause]:
    """Clauses dispositioned as promises no code can keep.

    Surfaced rather than netted out, and this is the point rather than a
    presentation choice. A reconciler reporting "all covered" while most of
    those are human promises has produced exactly the green-that-means-
    nothing the requirement exists to prevent. Their value is that somebody
    READS the list before a package goes out, not that a tool blessed them.
    """
    claims = set(inventory.get("binding_claim", {}))
    return [clause for clause in clauses if clause.clause_id in claims]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--doc", type=Path, action="append", default=[],
                        help="document to scan; repeatable")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    clauses = derive_clauses(args.doc, args.root)
    inventory = load_inventory(args.inventory)
    findings = reconcile(clauses, inventory, args.root,
                         inventory_text=args.inventory.read_text())
    done, total = coverage(clauses, inventory)

    claims = unenforceable(clauses, inventory)
    by_kind = counts_by_kind(clauses, inventory)
    print(f"candidates derived : {total}")
    print("counts by kind (no aggregate percentage -- see counts_by_kind):")
    for kind, n in by_kind.items():
        print(f"  {kind:<20} : {n}")

    if claims:
        # Printed in full, every time, and never folded into the coverage
        # figure above. These are promises a person keeps.
        print(f"\nUNENFORCEABLE BY CONSTRUCTION -- {len(claims)} clause(s) "
              f"bind what may be claimed, not what the program does. No code "
              f"gates these; a reader must check them before release:")
        for clause in claims:
            print(f"  {clause.doc}:{clause.line}  {clause.text[:100]}")

    if not clauses:
        # A run over zero candidates reports zero findings and looks clean.
        print("\nNO CANDIDATES DERIVED -- the scan found nothing, which is "
              "not the same as everything being dispositioned.",
              file=sys.stderr)
        return 2

    # A zero-guard is not enough. A narrow marker list derives a handful of
    # candidates from this record, all of which would be dispositioned,
    # exiting 0 over a sliver of it -- and a handful is not zero, so the
    # zero-guard never fires. The floor is the guard that does. Figures are
    # measured by tests/test_gate_inventory.py against the record itself,
    # never quoted here.
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
