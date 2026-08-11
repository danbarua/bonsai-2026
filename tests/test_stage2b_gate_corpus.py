"""The Stage 2B gate corpus is a narrowing, so it needs its own test.

`gate_corpus.PROTOCOL_DOCS` names four of the eight `.md` files beside it.
Principle 21's second half is the part that bites here: verifying a
narrowing with the broader form proves the code works and says nothing
about the narrowing. Running the reconciler over all eight documents would
pass; it would also derive 204 candidates instead of 89 and describe a
corpus nobody scoped.

So these tests exercise the narrowing itself, in both directions, and each
guard is confirmed by breaking what it watches rather than by observing a
green suite.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"

spec = importlib.util.spec_from_file_location(
    "_gate_corpus", STAGE2B_DIR / "gate_corpus.py")
gate_corpus = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate_corpus
spec.loader.exec_module(gate_corpus)


def test_every_document_on_disk_is_corpus_or_named_exemption():
    """Direction 1: nothing escapes the inventory by not being listed.

    The dangerous direction. A protocol document added and not declared
    contributes no candidates, so its clauses are absent from the coverage
    figure rather than reported as undispositioned -- invisible from the
    very command used to check coverage.
    """
    declared = set(gate_corpus.PROTOCOL_DOCS) | set(gate_corpus.EXEMPT)
    on_disk = gate_corpus.documents_on_disk()
    assert on_disk - declared == set(), (
        "undeclared Stage 2B document(s); their clauses are outside the "
        "inventory and nothing reports them")


def test_every_declared_document_still_exists():
    """Direction 2: no entry names a file that is gone.

    Quieter, and still real: a corpus entry for a deleted document changes
    what the candidate count measures, and an exemption for one is a reason
    that can never be checked.
    """
    declared = set(gate_corpus.PROTOCOL_DOCS) | set(gate_corpus.EXEMPT)
    assert declared - gate_corpus.documents_on_disk() == set()


def test_the_corpus_and_the_exemptions_do_not_overlap():
    assert not set(gate_corpus.PROTOCOL_DOCS) & set(gate_corpus.EXEMPT)


def test_every_exemption_carries_a_substantive_reason():
    """An exemption nobody can evaluate is an omission with extra steps."""
    for name, reason in gate_corpus.EXEMPT.items():
        assert reason and len(reason.split()) >= 8, (
            f"{name} is exempted with no reason a reviewer can weigh")


def test_the_corpus_derives_candidates_at_all():
    """Anti-vacuity only. The `== 89` this used to assert is gone.

    89 was the number the Reviewer scoped requirement 4 to, pinned on the
    theory that it was the one quantity able to move silently. In practice
    it moved whenever a frozen document was legitimately edited -- task
    #49's `AUDIT_PROTOCOL.md` correction being the most recent -- and each
    move was resolved by setting the constant to whatever the derivation
    now returned. A snapshot of reviewer scope, frozen into the suite,
    that never caught a corpus shrink.

    What it was reaching for is covered without an integer:
    `test_no_clause_in_the_real_corpus_is_left_undispositioned` fails if a
    clause appears with no row, and the membership tests above fail if a
    document joins or leaves. A count adds only the ability to notice that
    prose was reflowed.

    The floor stays because deriving NOTHING is a different failure --
    `derive_clauses` says so itself ("NO CANDIDATES DERIVED -- the scan
    found nothing, which is not the same as everything being
    dispositioned"), and an empty derivation would let the disposition
    test pass over an empty set.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools" / "gates"))
    from gate_inventory import derive_clauses

    docs = [STAGE2B_DIR / name for name in gate_corpus.PROTOCOL_DOCS]
    assert len(derive_clauses(docs)) >= 1, (
        "the scan derived no candidates at all, which is not the same as "
        "everything being dispositioned")


def test_the_exempt_set_is_exactly_these_documents():
    """Membership, with no integers.

    This REPLACES `test_each_exemption_still_contributes_the_candidate_
    count_it_did`, deleted 2026-08-11 on Dan's ruling after an audit of
    blocking guards under `tests/`. That test pinned, per exempt document,
    how many MUST/HALT-shaped sentences it derived when it was exempted,
    on the theory that a changed count meant the exemption's reason had
    been written about different text.

    Its ledger: 15 commits moved a pinned count. README travelled
    21 -> 20 -> 21 -> 22 -> 23 -> 24 -> 25 -> 26 -> 28 -> 29, FINDINGS
    37 -> 46 -> 52 -> 59 -> 60 -> 61 -> 63. Every one was an honest
    narrative or module-map append, and the fix was identical each time:
    re-derive the integer, paste it in, paste a paragraph explaining that
    the new sentences still bind nothing. Zero named a real obligation
    smuggled into an exempt document. The test body had accumulated ~120
    lines of that changelog by the end, and the audit's own snapshot of
    the counts went stale twice while it was being reviewed.

    What it actually measured was the WORD COUNT of MUST-shaped English in
    documents whose entire job is narrative -- so it fired on exactly the
    commits that close real science, and trained a reflex of bumping
    integers to get the suite green. That reflex is the cost: the same
    operators must not rubber-stamp a genuine `allow_test_split` or
    undispositioned-clause failure.

    What it protected that is worth keeping is MEMBERSHIP: the exempt set
    cannot drift silently. That forks on substance -- a new exempt
    document is a deliberate decision needing a written reason -- and it
    needs no integers. Binding force lives in `PROTOCOL_DOCS` and
    `gates.toml`, where `test_no_clause_in_the_real_corpus_is_left_
    undispositioned` covers it.
    """
    assert set(gate_corpus.EXEMPT) == {
        "FINDINGS.md", "NEGATIVE_PATH_EVIDENCE.md", "PHASE_B_PLAN.md",
        "README.md",
    }, ("the exempt set changed; a document was exempted or un-exempted, "
        "which is a judgement needing a written reason, not a drift")


def test_every_binds_at_pointer_names_a_clause_in_a_binding_kind():
    """The `not_binding` reasons' pointers must terminate somewhere real.

    A reason saying a clause is narration and "binds at X (id)" is only
    honest if `id` is dispositioned as binding. If X is later dispositioned
    `not_binding` too, the obligation has been narrated away by a chain of
    rows each pointing at the next -- and `gate_inventory.py` cannot see
    it, because these reasons are prose to it.

    Scoped to the `binds at ... (id)` construction specifically. A bare id
    elsewhere in a reason is a cross-reference to a sibling row, which is
    legitimate: `02dbbe96e032` says it restates `ef1b61b7eac3`, and both
    are correctly narration.
    """
    inventory = tomllib.loads(
        (STAGE2B_DIR / "gates.toml").read_text())
    binding = set()
    for kind in ("binding_gate", "binding_value", "binding_claim"):
        binding |= set(inventory.get(kind, {}))

    pointer = re.compile(r"binds at [^(]*\(([0-9a-f]{12})\)")
    found = 0
    for clause_id, entry in inventory.get("not_binding", {}).items():
        for target in pointer.findall(entry["reason"]):
            found += 1
            assert target in binding, (
                f"{clause_id} says its binding content lives at {target}, "
                f"which is not dispositioned in any binding kind")
    assert found >= 4, (
        "no `binds at ... (id)` pointers found -- the convention this "
        "test enforces has been dropped or reworded, and the test is "
        "passing over an empty set")


def test_check_corpus_rejects_an_undeclared_document(tmp_path):
    """The guard fires on the failure it exists for -- not merely green.

    Built against a temporary directory rather than by writing a file into
    the real stage directory: a test that creates a `.md` beside the
    protocol documents and crashes before cleanup leaves the repository in
    a state where this same guard fails for the wrong reason.
    """
    for name in gate_corpus.PROTOCOL_DOCS:
        (tmp_path / name).write_text("stub\n")
    for name in gate_corpus.EXEMPT:
        (tmp_path / name).write_text("stub\n")
    gate_corpus.check_corpus(tmp_path)          # the declared set passes

    (tmp_path / "STAGE4_PROTOCOL.md").write_text("**Frozen.** ...\n")
    with pytest.raises(gate_corpus.CorpusDrift, match="STAGE4_PROTOCOL.md"):
        gate_corpus.check_corpus(tmp_path)


def test_check_corpus_rejects_a_declared_document_that_is_gone(tmp_path):
    for name in gate_corpus.PROTOCOL_DOCS[1:]:
        (tmp_path / name).write_text("stub\n")
    for name in gate_corpus.EXEMPT:
        (tmp_path / name).write_text("stub\n")
    with pytest.raises(gate_corpus.CorpusDrift,
                       match=gate_corpus.PROTOCOL_DOCS[0]):
        gate_corpus.check_corpus(tmp_path)


# ---- the inventory itself, not only the corpus it is derived against ----

def _reconcile_the_real_inventory():
    sys.path.insert(0, str(REPO_ROOT / "tools" / "gates"))
    from gate_inventory import derive_clauses, reconcile

    docs = [STAGE2B_DIR / name for name in gate_corpus.PROTOCOL_DOCS]
    inventory = tomllib.loads((STAGE2B_DIR / "gates.toml").read_text())
    return reconcile(derive_clauses(docs), inventory)


def test_no_clause_in_the_real_corpus_is_left_undispositioned():
    """`make stage2b-gate-inventory` is local and read-only, so nothing
    runs it. That made the coverage figure an artifact of somebody
    remembering.

    Measured before writing this: deleting an entire dispositioned row
    from `gates.toml` -- `binding_value.ca5c4da6d40e`, ~2,000 characters
    -- left all 64 tests in this file and `test_gate_inventory.py` green.
    The whole inventory could be emptied the same way. Every existing
    test either drives the reconciler on synthetic input (the mechanism)
    or derives candidate counts from the DOCUMENTS (the corpus); none
    read the dispositions.

    Asserts on the undispositioned count specifically, NOT on a clean
    reconciliation. The run is expected to have findings -- rows pending
    an audit driver that does not exist, and `reviewed = false`, which is
    a human's flag and stays red until a human reads all 89 rows.
    Requiring zero findings here would either fail permanently or invite
    someone to flip `reviewed` to make a test pass, which is the one
    thing that flag must never be worth doing.
    """
    undispositioned = [f for f in _reconcile_the_real_inventory()
                       if f.kind == "undispositioned_candidate"]
    assert not undispositioned, (
        f"{len(undispositioned)} clause(s) in the frozen protocol documents "
        f"carry no disposition: {[f.message[:70] for f in undispositioned]}")


def test_the_inventory_still_holds_a_disposition_for_every_kind():
    """Anti-vacuity for the guard above, and it needs a specific shape.

    An empty `gates.toml` derives 89 candidates and reports 89
    undispositioned, so the check above would catch it. What it would NOT
    catch is the reconciler silently deriving nothing -- then there are no
    candidates, no undispositioned ones, and the assertion passes over an
    empty set. That is the failure `derive_clauses` warns about in its own
    output ("NO CANDIDATES DERIVED -- the scan found nothing, which is not
    the same as everything being dispositioned").
    """
    sys.path.insert(0, str(REPO_ROOT / "tools" / "gates"))
    from gate_inventory import derive_clauses

    inventory = tomllib.loads((STAGE2B_DIR / "gates.toml").read_text())
    kinds = ("binding_gate", "binding_value", "binding_claim", "not_binding")
    for kind in kinds:
        assert inventory.get(kind), f"{kind} is empty or absent"

    # Counted against the CORPUS, not against every row in the file. Child
    # obligations mint their own ids and are additional rows rather than
    # additional coverage -- the parent paragraph is what the corpus asked
    # about.
    #
    # `len(answered) == 89` used to sit here and is gone, with the corpus
    # test's matching pin: it locksteps this file to the same frozen
    # snapshot of reviewer scope, so a legitimate edit to a protocol
    # document failed BOTH and was resolved by bumping BOTH. The property
    # worth asserting is that the inventory answers everything derived,
    # which is a relation and not a number -- and that is exactly
    # `test_no_clause_in_the_real_corpus_is_left_undispositioned`. What
    # stays here is the anti-vacuity that test cannot provide for itself.
    docs = [STAGE2B_DIR / name for name in gate_corpus.PROTOCOL_DOCS]
    candidate_ids = {c.clause_id for c in derive_clauses(docs)}
    answered = set()
    for kind in kinds:
        answered |= set(inventory.get(kind, {})) & candidate_ids
    assert answered, (
        "the inventory answers none of the derived candidates, so the "
        "disposition check above would pass over an empty set")

    children = [cid for kind in kinds for cid, e in inventory.get(kind, {}).items()
                if isinstance(e, dict) and e.get("parent_clause")]
    assert set(children).isdisjoint(candidate_ids), (
        "a child obligation collides with a corpus candidate id, so one "
        "disposition covers two different things")
