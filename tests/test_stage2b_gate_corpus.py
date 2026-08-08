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


def test_the_corpus_derives_the_scoped_candidate_count():
    """89 is the number the Reviewer scoped requirement 4 to.

    Pinned because it is the one quantity that would move silently: a
    document dropped from the corpus, or a frozen paragraph reflowed into
    two, changes it while every other test here still passes.

    Expected to change when the protocol documents genuinely change -- and
    that is the point. It fails, somebody looks, and updates it knowing
    what moved rather than discovering later that the corpus drifted.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools" / "gates"))
    from gate_inventory import derive_clauses

    docs = [STAGE2B_DIR / name for name in gate_corpus.PROTOCOL_DOCS]
    assert len(derive_clauses(docs)) == 89


def test_each_exemption_still_contributes_the_candidate_count_it_did():
    """The exemption that grows clauses is the one nothing else catches.

    Both direction tests are satisfied by a document that is DECLARED,
    and the 89-count pin only moves when the corpus changes -- so an
    exempt document quietly acquiring binding obligations is invisible to
    every other check here. Judging that a document states no obligations
    is as unmechanisable as judging a sentence non-binding; noticing that
    the judgement now covers different content is not.

    These counts are what each exempt document contributed when it was
    exempted. A change means the reason on that exemption was written
    about different text and needs re-reading -- not that anything is
    broken. Update the number with the re-read, never ahead of it.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools" / "gates"))
    from gate_inventory import derive_clauses

    at_exemption_time = {
        "FINDINGS.md": 37,
        "NEGATIVE_PATH_EVIDENCE.md": 19,
        "PHASE_B_PLAN.md": 38,
        "README.md": 21,
    }
    assert set(at_exemption_time) == set(gate_corpus.EXEMPT), (
        "an exemption was added or removed without a candidate count")
    for name, expected in at_exemption_time.items():
        actual = len(derive_clauses([STAGE2B_DIR / name]))
        assert actual == expected, (
            f"{name} now derives {actual} candidates, not {expected}: its "
            f"exemption reason was written about different text")


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
