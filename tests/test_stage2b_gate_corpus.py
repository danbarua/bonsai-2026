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

    # README.md moved 21 -> 20 at `8ad0ddd`, which removed the status
    # restatement from its header. Re-read before the number was changed,
    # which is the whole protocol here: the exemption reads "orientation
    # for a reader arriving cold ... restates none of their obligations
    # bindingly", and that is MORE true after the edit, not less. The
    # count fell because a candidate-generating paragraph left, not
    # because the judgement changed.
    #
    # FINDINGS.md moved 37 -> 46 across `9efac76`/`4bbd454`/`b0c382a`,
    # which appended the stage-4 closing section (the official result,
    # the floor-alpha caveat, the CNN comparison, the retraction of the
    # "raw pixel ~ identity" reading, and the audit-status correction).
    # Re-read all 9 new candidates individually (line > 1100) before
    # changing this number: every one reports a fact about an
    # already-frozen procedure's execution or result (the bootstrap CI,
    # the alpha that was already selected in stage 3, what the driver's
    # already-committed code does, what closed) using the same
    # vocabulary those procedures use ("locked", "frozen", "required")
    # -- none creates a new obligation this document did not already
    # have narrative license to describe. The count rose because
    # substantial new narrative content was added, not because the
    # judgement about FINDINGS.md's bindingness changed.
    # README.md moved 20 -> 21 at `3daea87`, which added the
    # run_audit.py/stage2b_audit.py module-map entry. Re-read: the new
    # paragraph describes what the code does (enforces the sequencing
    # gate) and names AUDIT_PROTOCOL.md as the actual authority -- the
    # same module-map genre as every other entry already counted here,
    # not a new obligation stated in this document's own voice.
    # README.md moved 21 -> 22 adding the run_abs_conv_eps_sensitivity.py
    # module-map entry (companion Protocol 2's driver). Re-read: the new
    # NEVER candidate is "never reimplemented" (`evaluate_rho_gate`
    # called unmodified) -- the same module-map genre as the run_audit.py
    # entry above, describing what the code does and naming
    # COMPANION_PROTOCOLS.md as the actual authority for the protocol
    # itself, not a new obligation this document states in its own voice.
    # FINDINGS.md moved 46 -> 52, appending the amendment-impact audit's
    # own closing section (the AUDIT_OK result, the stage-1/2 historical
    # cross-check, feature distances, the protocol-required scope
    # statement quoted verbatim, and the closed-investigation status) plus
    # a correction to the stage-4 section's now-stale "audit has not run"
    # paragraph. Re-read all 7 candidates individually (line > 1300)
    # before changing this number: each reports execution or a measured
    # result of an already-frozen procedure -- AUDIT_PROTOCOL.md's own
    # trigger definitions, its analytic resolution threshold, and its
    # required write-up scope statement (quoted, not restated in this
    # document's own voice) -- using the same vocabulary those frozen
    # documents already use. None creates a new obligation.
    # FINDINGS.md moved 52 -> 59, appending Companion Protocol 1
    # (PROTOCOL1_OK, five-stage maxima tables, construction sizes,
    # sequencing-deviation disclosure, encoding-only adversarial scope
    # limitation). Same genre: reports a measured run of an already-frozen
    # companion protocol; does not create new obligations.
    # README.md moved 22 -> 23 adding the generate_artifact_manifest.py
    # module-map entry. Re-read: the new LOCKED candidate ("the frozen
    # headline numbers behind Stage 2B's two locked results") describes
    # what the script does and points at the two results it indexes --
    # same module-map genre as every other entry, not a new obligation.
    # README.md moved 23 -> 24 adding the Protocol 1 driver module-map
    # entry (run_arm_x86_propagation_stress.py). Same genre: names the
    # driver, points at COMPANION_PROTOCOLS.md as authority, describes
    # phases — not a new obligation stated in README's own voice.
    # FINDINGS.md moved 59 -> 60 across `cf5ffca`/`c702bae` (Protocol 1/2
    # companion closure). The net +1 is four candidates added and three
    # removed, and the three removals are the point: two are the
    # "adversarial upper bound on cross-architecture propagation" framing
    # narrowed to "maximum observed within the 287-image provisional stress
    # set", and one is the run-id line superseded by a longer one naming
    # 20260810T151245Z as authoritative. Rewordings, not new obligations.
    # The one genuinely new candidate is Protocol 2's verdict: "Verdict at
    # locked ENCODER_STEPS=1200: INVARIANT across eps in {1e-10 .. 1e-13}.
    # No flip; HALT condition does not fire. ABS_CONV_EPS=1e-12 itself does
    # not change." Re-read: it reports the measured outcome of an
    # already-frozen procedure -- COMPANION_PROTOCOLS.md owns the protocol
    # and gates.toml owns the HALT -- in those documents' own vocabulary
    # ("locked", "HALT"). Saying a frozen gate did not fire is a result,
    # not a new obligation stated in FINDINGS.md's own voice. Same genre as
    # every FINDINGS.md bump above it.
    at_exemption_time = {
        "FINDINGS.md": 60,
        "NEGATIVE_PATH_EVIDENCE.md": 19,
        "PHASE_B_PLAN.md": 38,
        # README.md moved 24 -> 25 adding the measure_scaler_lipschitz.py
        # module-map entry. Re-read: the new candidate is the entry itself,
        # whose NEVER-shaped phrasing is "no GPU, no credentials, no refit,
        # no google-cloud-storage" -- a description of what the script does
        # not need in order to run, in the same module-map genre as every
        # entry counted above it. It names the file, says what it reads and
        # from where, and points at the axis-4 claim it measures against.
        # Not a new obligation stated in README's own voice.
        # README.md moved 25 -> 26 adding the cancellation sentence to the
        # measure_combined_operator_norm.py module-map entry. Re-read: the
        # new candidate is that sentence's NEVER -- "it never drops the
        # result below either factor alone" -- which is a statement of what
        # a MEASUREMENT turned out to show, not an obligation on anything.
        # Same module-map genre as every entry above it; a description of a
        # script's finding cannot bind the system.
        # README.md moved 26 -> 28 adding the plot_cnn_denoising.py and
        # animate_graph_dynamics.py module-map entries. Re-read, both: the
        # CNN entry's NEVER-shaped text is "untrained outputs are
        # unconstrained, not inert" and "no Stage 2B number is affected" --
        # a measured description of where a residual lands and an explicit
        # statement that it binds NOTHING. The animation entry's is "the CNN
        # has no comparable animation" and "features are read at the final
        # frame" -- a fact about what the pipeline's time axis covers, and a
        # restatement of the locked feature point rather than a new
        # obligation. Both are module-map entries in the same genre as every
        # one above; a script that renders pixels and computes no metric has
        # nothing to bind.
        "README.md": 28,
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
    # about. Written as a flat total first, which broke the moment the first
    # composite was split: 95 rows, 89 candidates, and the honest number is
    # neither of those on its own.
    docs = [STAGE2B_DIR / name for name in gate_corpus.PROTOCOL_DOCS]
    candidate_ids = {c.clause_id for c in derive_clauses(docs)}
    answered = set()
    for kind in kinds:
        answered |= set(inventory.get(kind, {})) & candidate_ids
    assert len(answered) == 89, (
        f"the inventory answers {len(answered)} of 89 derived candidates; "
        f"the corpus test pins the derivation, this pins that the inventory "
        f"still answers it")

    children = [cid for kind in kinds for cid, e in inventory.get(kind, {}).items()
                if isinstance(e, dict) and e.get("parent_clause")]
    assert set(children).isdisjoint(candidate_ids), (
        "a child obligation collides with a corpus candidate id, so one "
        "disposition covers two different things")
