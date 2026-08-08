"""Pin WHICH documents the Stage 2B binding-clause inventory ranges over.

`tools/gates/gate_inventory.py` derives the clause set from whatever
documents it is handed. That makes the document list itself load-bearing:
the Reviewer scoped requirement 4 to "the full 89-clause corpus", and 89
is a fact about these four files. Handed three, the reconciler derives a
smaller set, reports full coverage of it, and exits 0 -- a green over a
corpus quietly missing a document.

Which is principle 21 exactly: a hand-maintained list standing in for a
derivable set, verified with the broader tool. So the list is asserted
against the derived one -- every `.md` beside this file is either IN the
corpus or a NAMED exemption with a reason, checked in both directions, at
run time as well as in `tests/test_stage2b_gate_corpus.py`. A protocol
document added next month fails the make target on the day it lands
rather than whenever somebody remembers this list exists.

Usage (normally via `make stage2b-gate-inventory`):

    uv run python experiments/stage2b_denoising/gate_corpus.py \\
        --inventory experiments/stage2b_denoising/gates.toml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[1]

sys.path.insert(0, str(REPO_ROOT / "tools" / "gates"))

import gate_inventory  # noqa: E402

# The frozen protocol corpus. These four documents, and only these, state
# what Stage 2B has committed to; together they derive the 89 candidates.
PROTOCOL_DOCS = (
    "DESIGN.md",
    "AUDIT_PROTOCOL.md",
    "COMPANION_PROTOCOLS.md",
    "STAGE3_PLAN.md",
)

# Every other `.md` in the stage directory, each with the reason it is not
# a source of binding clauses. A reason, not a bare list: an exemption
# nobody can evaluate is an omission with extra steps.
#
# Note the sizes involved -- these four would contribute 115 further
# candidates between them, more than the corpus itself. That is the
# argument FOR exempting them rather than against: they are dense with
# words like `frozen` and `never` precisely because they narrate and cite
# the frozen protocol. Scanning them would bury 89 real obligations under
# a majority of quotations of those same obligations.
EXEMPT = {
    "FINDINGS.md":
        "reports what the runs measured. A findings document describes "
        "results; it does not commit the system to anything, and its "
        "obligations are quotations of DESIGN.md's.",
    "NEGATIVE_PATH_EVIDENCE.md":
        "an evidence table -- it DISCHARGES obligations stated elsewhere "
        "rather than stating any. It is a `discharged_in` target for "
        "rows in the inventory, which is the opposite role.",
    "PHASE_B_PLAN.md":
        "self-declared: 'Status: plan, not a frozen protocol. The frozen "
        "documents (DESIGN.md, AUDIT_PROTOCOL.md, COMPANION_PROTOCOLS.md, "
        "and STAGE3_PLAN.md's five freezes) are untouched by this and "
        "govern wherever they speak.' Should any part of it ever be "
        "frozen, that freeze lands in one of the four, not here.",
    "README.md":
        "orientation for a reader arriving cold -- a module map and a "
        "test table. It points at the protocol documents and restates "
        "none of their obligations bindingly.",
}


class CorpusDrift(Exception):
    """The declared corpus and the documents on disk disagree."""


def documents_on_disk(stage_dir: Path = _THIS_DIR) -> set[str]:
    return {path.name for path in stage_dir.glob("*.md")}


def check_corpus(stage_dir: Path = _THIS_DIR) -> None:
    """Both directions, because each catches a different failure.

    An undeclared document is the dangerous one -- clauses silently
    outside the inventory. A declared-but-absent document is the quieter
    one: a corpus entry naming a deleted file makes the count 89 mean
    something it no longer means, and an exemption for a file that is gone
    is a reason nobody can check.
    """
    declared = set(PROTOCOL_DOCS) | set(EXEMPT)
    on_disk = documents_on_disk(stage_dir)

    undeclared = sorted(on_disk - declared)
    if undeclared:
        raise CorpusDrift(
            f"{', '.join(undeclared)}: present in {stage_dir.name}/ and "
            f"neither in PROTOCOL_DOCS nor exempted. Add it to the corpus, "
            f"or exempt it with a reason -- an undeclared document's "
            f"clauses are outside the inventory and nothing reports them.")

    missing = sorted(declared - on_disk)
    if missing:
        raise CorpusDrift(
            f"{', '.join(missing)}: declared here and absent from "
            f"{stage_dir.name}/. A corpus entry naming a file that no "
            f"longer exists changes what the candidate count measures; an "
            f"exemption for one is a reason that can never be checked.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path,
                        default=_THIS_DIR / "gates.toml")
    parser.add_argument("--list-docs", action="store_true",
                        help="print the corpus and exit, checking it first")
    args = parser.parse_args(argv)

    # Before the reconciler runs, not after: a coverage figure over the
    # wrong corpus is worse than no figure, because it looks like an answer.
    #
    # Caught and printed rather than left to raise, because the message is
    # the point: a traceback reads as the tool being broken, when what has
    # actually happened is the tool correctly refusing.
    try:
        check_corpus()
    except CorpusDrift as drift:
        print(f"REFUSING: {drift}", file=sys.stderr)
        return 2

    docs = [_THIS_DIR / name for name in PROTOCOL_DOCS]
    rel = [doc.relative_to(REPO_ROOT).as_posix() for doc in docs]
    if args.list_docs:
        print("\n".join(rel))
        return 0

    print(f"corpus             : {len(rel)} document(s) -- {', '.join(PROTOCOL_DOCS)}")
    print(f"exempt             : {len(EXEMPT)} -- {', '.join(sorted(EXEMPT))}")
    argv_inner = ["--inventory", str(args.inventory)]
    for doc in docs:
        argv_inner += ["--doc", str(doc)]
    return gate_inventory.main(argv_inner)


if __name__ == "__main__":
    raise SystemExit(main())
