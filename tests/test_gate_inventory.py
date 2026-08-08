"""The claim-vs-enforcement reconciler, on synthetic documents.

`tools/gates/gate_inventory.py` is the mechanism half of a blocking
readiness requirement: every documented MUST/HALT mapped to its enforcement
point and its test, with a documented gate lacking an executable mapping
failing readiness.

Fixtures are synthetic throughout. Testing it against this repository's real
frozen record would make the suite depend on documents another track is
actively editing, so a green run would mean "the documents happen to be
mapped today" rather than "the reconciler works" — and the second is the
only thing this file is entitled to claim.

The vacuity risk is specific and severe here: a reconciler that derives zero
clauses reports zero findings and looks perfectly clean. Several tests below
exist only to make that state impossible to mistake for success.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "tools" / "gates" / "gate_inventory.py"

spec = importlib.util.spec_from_file_location("_gate_inventory", MODULE)
gate_inventory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate_inventory
spec.loader.exec_module(gate_inventory)

derive_clauses = gate_inventory.derive_clauses
reconcile = gate_inventory.reconcile

DOC = """# Protocol

The driver MUST verify the digest before launching.

HALT for review if any production condition selects the floor.

This paragraph is ordinary prose with no binding force.
"""


def write_doc(tmp_path: Path, text: str = DOC, name: str = "DESIGN.md") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def kinds(findings):
    return sorted(f.kind for f in findings)


# --- derivation ------------------------------------------------------------

def test_binding_clauses_are_found_and_prose_is_not(tmp_path):
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)
    found = {c.kind for c in clauses}
    assert found == {"MUST", "HALT"}, f"got {found}"
    assert not any("ordinary prose" in c.text for c in clauses)


def test_a_clause_spanning_hard_wrapped_lines_is_one_clause(tmp_path):
    """These documents are hard wrapped, so a requirement routinely spans
    three lines. A line-based scan would split it, and each fragment would
    hash differently -- producing mappings that can never be satisfied."""
    doc = write_doc(tmp_path, "# P\n\nThe driver MUST verify the\ndigest "
                              "before launching the\nremote job.\n")
    (clause,) = derive_clauses([doc], tmp_path)
    assert "remote job" in clause.text
    assert "\n" not in clause.text


def test_fenced_code_is_not_scanned(tmp_path):
    """An example in a code block is illustration, not requirement."""
    doc = write_doc(tmp_path, "# P\n\n```\nassert x, 'MUST be true'\n```\n")
    assert derive_clauses([doc], tmp_path) == []


def test_clause_id_survives_reflowing_but_not_rewording(tmp_path):
    """The property the whole design rests on.

    Stable across reflowing so that rewrapping a paragraph does not orphan
    every mapping in the file. Unstable across rewording so that changing
    what a requirement SAYS cannot leave its mapping silently in place --
    the specific way a hand-maintained correspondence rots.
    """
    original = derive_clauses(
        [write_doc(tmp_path, "# P\n\nThe driver MUST verify the digest.\n")],
        tmp_path)[0]
    reflowed = derive_clauses(
        [write_doc(tmp_path, "# P\n\nThe driver MUST verify\nthe digest.\n")],
        tmp_path)[0]
    reworded = derive_clauses(
        [write_doc(tmp_path, "# P\n\nThe driver MUST verify the manifest.\n")],
        tmp_path)[0]

    assert original.clause_id == reflowed.clause_id, "reflowing orphaned it"
    assert original.clause_id != reworded.clause_id, (
        "rewording kept the same id -- a mapping would survive the change "
        "it was supposed to be invalidated by")


# --- reconciliation, both directions ---------------------------------------

def test_an_unmapped_clause_is_a_finding(tmp_path):
    """The incident this exists for: a frozen HALT with no implementation."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)
    findings = reconcile(clauses, {"reviewed": True, "binding_gate": {}}, tmp_path)
    assert kinds(findings) == ["undispositioned_candidate", "undispositioned_candidate"]


def test_a_mapping_for_a_clause_that_no_longer_exists_is_a_finding(tmp_path):
    """The other direction. Without it, editing a requirement leaves a
    mapping that describes nothing and nothing reports it."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)
    inventory = {"binding_gate": {c.clause_id: {"enforcement": "x.py::f",
                                        "test": "t.py::t",
                                        "break_demonstrated": "yes"}
                          for c in clauses}}
    inventory["binding_gate"]["deadbeef1234"] = {"enforcement": "x.py::f",
                                         "test": "t.py::t",
                                         "break_demonstrated": "yes"}
    findings = reconcile(clauses, inventory, tmp_path)
    assert "orphaned_disposition" in kinds(findings)


def complete_row(**overrides):
    """All six Reviewer dimensions plus scheduling. Missing ANY fails."""
    row = {
        "enforcement": "gate.py::verify_digest",
        "production_reachability": "run_ladder_stage3.py calls it via "
                                   "make stage2b-ladder-stage3",
        "input_wiring": "digest from BONSAI_DRIVER_SHA256, set at upload",
        "decision_consequence": "raises SystemExit before the kernel starts",
        "test": "test_gate.py::test_digest",
        "break_demonstrated": "disabled the branch, test failed",
        "trigger": "ci-fast",
    }
    row.update(overrides)
    return row


def value_row(**overrides):
    """A `binding_value` row to the reviewer's five requirements."""
    row = {
        "value": "ALPHA_BAR = 0.5, DESIGN.md 'Frozen parameters'",
        "production_consumers": "step7_ridge.py; proved common propagation "
                                "point is stage2b_conditions.ALPHA_BAR",
        "enforcement": "t.py::test_alpha",
        "break_demonstrated": "altered the propagated value in "
                              "stage2b_conditions; the test failed",
        "provenance_of_use": "run manifest records alpha_bar=0.5 for the run",
        "trigger": "ci-fast",
        "status": "enforced",
    }
    row.update(overrides)
    return row


def claim_row(**overrides):
    """A `binding_claim` row to the reviewer's six requirements."""
    row = {
        "locator": "DESIGN.md#reporting-constraints",
        "obligation": "the control is described as degree-preserving rewiring",
        "discharged_in": "FINDINGS.md, Scope and limits",
        "status": "discharged",
        "evidence": "quoted: 'a degree-preserving rewiring, not a random "
                    "sample from the space of graphs'",
    }
    row.update(overrides)
    return row


def test_a_fully_dispositioned_document_produces_no_findings(tmp_path):
    """Non-vacuity for every negative assertion above: they would all pass
    against a reconcile() that reported findings unconditionally."""
    doc = write_doc(tmp_path)
    (tmp_path / "gate.py").write_text("def verify_digest():\n    return True\n")
    (tmp_path / "test_gate.py").write_text("def test_digest():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True,
                 "binding_gate": {c.clause_id: complete_row() for c in clauses}}
    assert reconcile(clauses, inventory, tmp_path) == []


@pytest.mark.parametrize("dimension", [
    "production_reachability", "input_wiring", "decision_consequence"])
def test_each_reviewer_dimension_is_required(tmp_path, dimension):
    """The three dimensions beyond existence-and-test.

    `production_reachability` is the one that carries the most weight: a
    correct predicate production never invokes is not an implemented gate,
    and citing it without it would certify exactly the make-wrapped shape.
    """
    doc = write_doc(tmp_path, "# P\n\nThe driver MUST verify the digest.\n")
    (tmp_path / "gate.py").write_text("def verify_digest():\n    return True\n")
    (tmp_path / "test_gate.py").write_text("def test_digest():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    row = complete_row()
    del row[dimension]
    inventory = {"binding_gate": {clauses[0].clause_id: row}}
    assert f"missing_{dimension}" in kinds(reconcile(clauses, inventory, tmp_path))


def test_two_identical_sentences_in_different_documents_collide_loudly(tmp_path):
    """A shared id would let one disposition silently cover both.

    Not hypothetical in a record that repeats "locked", "frozen" and
    "never overwritten" across four documents.
    """
    same = "# P\n\nThe artifact is never overwritten.\n"
    a = write_doc(tmp_path, same, "A.md")
    b = write_doc(tmp_path, same.replace("never", "NEVER"), "B.md")
    clauses = derive_clauses([a, b], tmp_path)
    assert len(clauses) == 2
    assert clauses[0].clause_id == clauses[1].clause_id, "fixture is wrong"
    assert "id_collision" in kinds(reconcile(clauses, {"binding_gate": {}}, tmp_path))


def test_a_narrow_marker_list_would_have_certified_two_percent(tmp_path):
    """The measured near-miss, pinned as a property of the derivation.

    Across this project's real frozen record -- 601 sentences -- an
    RFC-2119 marker list matched THREE. Three dispositions would have exited
    0 over 2% coverage, and the zero-guard never fires because three is not
    zero. The candidate set must therefore catch lowercase prose forms, and
    this asserts it does rather than trusting that it does.
    """
    prose = ("# P\n\nThe manifest is frozen once written.\n\n"
             "Inputs are locked before the run.\n\n"
             "The artifact is never overwritten.\n\n"
             "A credential is required for upload.\n\n"
             "The driver refuses to launch on divergence.\n")
    clauses = derive_clauses([write_doc(tmp_path, prose)], tmp_path)
    assert len(clauses) == 5, (
        f"lowercase prose requirements went undetected: found "
        f"{[c.kind for c in clauses]}")


# --- the mapped references must actually resolve ---------------------------

def test_an_enforcement_site_that_does_not_exist_is_a_finding(tmp_path):
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    inventory = {"binding_gate": {clauses[0].clause_id: {
        "enforcement": "nonexistent.py::f", "test": "also_missing.py::t",
        "break_demonstrated": "yes"}}}
    assert "unresolved_enforcement" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_name_only_mentioned_in_a_comment_does_not_satisfy_a_citation(tmp_path):
    """Resolved from the AST, not by substring. A citation satisfied by a
    comment is the `endswith` failure from the vacuous-test catalogue."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "gate.py").write_text("# verify_digest is not defined here\n")
    inventory = {"binding_gate": {clauses[0].clause_id: {
        "enforcement": "gate.py::verify_digest", "test": "gate.py::verify_digest",
        "break_demonstrated": "yes"}}}
    assert "unresolved_enforcement" in kinds(reconcile(clauses, inventory, tmp_path))


@pytest.mark.parametrize("source,why", [
    ("ALPHA_BAR = 0.5\n", "plain assignment"),
    ("ALPHA_BAR: float = 0.5\n", "ANNOTATED assignment -- the form a frozen "
                                 "constant most often takes in typed code, "
                                 "and the one originally missed"),
    ("ALPHA_BAR, BETA = 0.5, 1.0\n", "bound through a tuple"),
    ("def ALPHA_BAR():\n    pass\n", "a function"),
    ("class ALPHA_BAR:\n    pass\n", "a class"),
])
def test_every_binding_form_resolves_a_citation(tmp_path, source, why):
    """A citation to a real definition must not report as unresolved.

    The failure direction matters: a false `unresolved_enforcement` sends a
    reader hunting a problem that does not exist, and erodes trust in every
    true finding beside it. This repository uses annotated module constants,
    including inside the reconciler itself, so the omission was live.
    """
    doc = write_doc(tmp_path, "# P\n\nALPHA_BAR is frozen at 0.5.\n")
    (tmp_path / "g.py").write_text(source)
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_value": {clauses[0].clause_id:
        value_row(enforcement="g.py::ALPHA_BAR")}}
    found = kinds(reconcile(clauses, inventory, tmp_path))
    assert "unresolved_enforcement" not in found, f"failed to resolve {why}"


def test_a_merely_imported_name_does_not_satisfy_a_citation(tmp_path):
    """Deliberate omission, stated as a choice rather than an oversight.

    An enforcement citation should point at where the gate is defined. A
    re-export would let one gate be cited from any module that happens to
    import it, which is a citation that resolves without locating anything.
    """
    doc = write_doc(tmp_path, "# P\n\nALPHA_BAR is frozen at 0.5.\n")
    (tmp_path / "g.py").write_text("from elsewhere import ALPHA_BAR\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_value": {clauses[0].clause_id:
        value_row(enforcement="g.py::ALPHA_BAR")}}
    assert "unresolved_enforcement" in kinds(
        reconcile(clauses, inventory, tmp_path))


def test_a_mapping_without_break_evidence_is_a_finding(tmp_path):
    """"Gate cited, test cited" certifies spelling.

    The precedent is concrete: a test that grepped source for
    `halt_reasons.append(...)` stayed green when the branch guarding it was
    disabled by `if False:`. It would have passed on exactly the artifact
    that failed.
    """
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    inventory = {"binding_gate": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f"}}}
    assert "missing_break_demonstrated" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_mapping_must_record_what_runs_its_test(tmp_path):
    """A gate whose test nobody runs is barely a gate.

    Fourth mechanism of the same signature: not-loaded, not-invoked,
    never-handed-the-input, and now never-RUN. Observed concretely — a
    correct, self-deriving guard caught an offending file hours late,
    because firing it required a human to run the full suite in another
    session.
    """
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    inventory = {"binding_gate": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f",
        "break_demonstrated": "yes"}}}
    assert "missing_trigger" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_manual_only_trigger_is_reported_but_not_treated_as_absent(tmp_path):
    """Some gates legitimately cost too much to run automatically.

    The point is that this is visible in the inventory rather than
    discovered when one fires late, so it is a distinct finding kind rather
    than either silence or a hard failure.
    """
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    inventory = {"binding_gate": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f",
        "break_demonstrated": "yes", "trigger": "manual"}}}
    found = kinds(reconcile(clauses, inventory, tmp_path))
    assert "manual_trigger_only" in found
    assert "missing_trigger" not in found


def test_a_frozen_value_is_not_asked_for_a_decision_consequence(tmp_path):
    """A constant has no runtime decision and no production path to reach.

    Measured over the real record, 21 of 89 candidates are frozen values.
    Demanding `decision_consequence` for them would demand fiction, and the
    only ways to satisfy a seven-field schema would be to call them
    not-binding (false -- they are among the most binding things there) or
    to invent the field, which is the confabulation a model was refused for.
    """
    doc = write_doc(tmp_path, "# P\n\nALPHA_BAR is frozen at 0.5.\n")
    (tmp_path / "t.py").write_text("def test_alpha():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_value": {clauses[0].clause_id: value_row()}}
    assert reconcile(clauses, inventory, tmp_path) == []


def test_a_process_promise_needs_no_code_fields_at_all(tmp_path):
    """54 of 89 candidates bind what may be CLAIMED, not what code does.

    "Must never be reported as a random sample" has no enforcement point,
    because the binding is on a human writing a paragraph. Forcing it into
    a code schema fabricates a mapping.
    """
    doc = write_doc(
        tmp_path, "# P\n\nThe control must never be reported as random.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {clauses[0].clause_id: claim_row()}}
    assert reconcile(clauses, inventory, tmp_path) == []


def test_unenforceable_promises_are_listed_not_netted_into_coverage(tmp_path):
    """The part that matters most.

    A reconciler reporting "89/89 covered" while 54 of those are human
    promises has produced exactly the green-that-means-nothing this
    requirement exists to prevent. Their value is that somebody READS them
    before release, so they must survive as a list.
    """
    doc = write_doc(
        tmp_path, "# P\n\nThe control must never be reported as random.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {clauses[0].clause_id: claim_row()}}
    listed = gate_inventory.unenforceable(clauses, inventory)
    assert [c.clause_id for c in listed] == [clauses[0].clause_id]
    # And it still counts as dispositioned -- the point is that it is
    # visible, not that it is excluded from review.
    assert gate_inventory.coverage(clauses, inventory) == (1, 1)


def test_tagging_a_claim_mechanizable_does_not_promote_it_out_of_the_list(tmp_path):
    """The tag records a possibility. It must not discharge anything.

    A tag that quietly moved a row out of the human-reads-this list would
    convert "somebody must check this" into "the tool checks this" in
    everyone's head -- worse than no tag, and exactly how a half-built lint
    firing on some cases does its damage.
    """
    doc = write_doc(
        tmp_path, "# P\n\nNo metric may be added after results exist.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {clauses[0].clause_id: claim_row(
            negative_attestation="checked all of FINDINGS.md and the tables",
            mechanizable_candidate="a doc-diff lint could compare metric "
                                   "lists across commits")}}
    assert reconcile(clauses, inventory, tmp_path) == []
    still_listed = gate_inventory.unenforceable(clauses, inventory)
    assert [c.clause_id for c in still_listed] == [clauses[0].clause_id], (
        "a mechanizable tag removed the row from the unenforceable listing")


def test_a_frozen_value_with_no_consumer_yet_fails_readiness(tmp_path):
    """The case where `production_consumers` is EMPTY, not unknown.

    Some frozen values here are consumed by code that does not exist yet --
    the audit protocol freezes values its unwritten driver will read. That
    is a true and useful readiness statement, but it needs a disposition
    that is neither a lie nor an escape hatch, so it fails rather than being
    satisfiable by writing "none yet" into a required prose box.
    """
    doc = write_doc(tmp_path, "# P\n\nM is frozen at 100 for the audit.\n")
    (tmp_path / "t.py").write_text("def test_m():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_value": {clauses[0].clause_id:
        value_row(status="pending_consumer",
                  pending_reason="the audit driver is unwritten",
                  production_consumers="none yet")}}
    found = kinds(reconcile(clauses, inventory, tmp_path))
    assert "value_has_no_production_consumer" in found


def test_pending_consumer_without_a_reason_is_also_a_finding(tmp_path):
    doc = write_doc(tmp_path, "# P\n\nM is frozen at 100 for the audit.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_value": {clauses[0].clause_id:
        value_row(status="pending_consumer", production_consumers="none yet")}}
    assert "unreasoned_pending_consumer" in kinds(
        reconcile(clauses, inventory, tmp_path))


def test_an_enforced_value_is_not_flagged(tmp_path):
    """Non-vacuity for the two cases above."""
    doc = write_doc(tmp_path, "# P\n\nALPHA_BAR is frozen at 0.5.\n")
    (tmp_path / "t.py").write_text("def test_alpha():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True,
                 "binding_value": {clauses[0].clause_id: value_row()}}
    found = kinds(reconcile(clauses, inventory, tmp_path))
    assert "value_has_no_production_consumer" not in found
    assert "unknown_status" not in found


def test_an_unresolved_claim_fails_readiness(tmp_path):
    """Reviewer-mandated: an unresolved obligation is not a state a package
    ships in."""
    doc = write_doc(tmp_path, "# P\n\nThe scope statement is required.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {
        clauses[0].clause_id: claim_row(status="unresolved")}}
    assert "unresolved_claim" in kinds(reconcile(clauses, inventory, tmp_path))


def test_not_applicable_without_a_reason_is_a_finding(tmp_path):
    """`not applicable` is a claim that the triggering condition never arose.
    Without a reason tied to it, it is an escape hatch."""
    doc = write_doc(tmp_path, "# P\n\nThe scope statement is required.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {
        clauses[0].clause_id: claim_row(status="not_applicable")}}
    assert "unreasoned_not_applicable" in kinds(
        reconcile(clauses, inventory, tmp_path))


def test_a_negative_obligation_needs_an_attestation_over_the_output_set(tmp_path):
    """Pointing at one compliant passage cannot establish that a prohibited
    claim is absent from everywhere else. That is a logical gap, not a
    procedural one, which is why a compliant `evidence` pointer does not
    discharge it."""
    doc = write_doc(
        tmp_path, "# P\n\nThe control must never be reported as random.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {
        clauses[0].clause_id: claim_row(
            obligation="must never be reported as a random sample")}}
    assert "missing_negative_attestation" in kinds(
        reconcile(clauses, inventory, tmp_path))


def test_a_positive_obligation_does_not_need_one(tmp_path):
    """Non-vacuity for the case above: it would pass on a check that
    demanded an attestation from every claim row."""
    doc = write_doc(tmp_path, "# P\n\nAn honest scope statement is required.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {
        clauses[0].clause_id: claim_row(
            obligation="the write-up carries an honest scope statement")}}
    assert "missing_negative_attestation" not in kinds(
        reconcile(clauses, inventory, tmp_path))


def test_counts_are_reported_per_kind_with_no_aggregate_percentage(tmp_path):
    """Reviewer-mandated: no single figure may let 54 human obligations
    dilute or inflate executable coverage."""
    doc = write_doc(tmp_path)
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True,
                 "binding_claim": {clauses[0].clause_id: claim_row()}}
    counts = gate_inventory.counts_by_kind(clauses, inventory)
    assert counts["binding_claim"] == 1
    assert counts["binding_gate"] == 0
    assert counts["undispositioned"] == 1
    assert all(isinstance(v, int) for v in counts.values()), (
        "counts_by_kind returned a non-integer -- a ratio or percentage "
        "would be exactly what the ruling forbids")


def test_a_gate_row_is_not_listed_as_unenforceable(tmp_path):
    """Non-vacuity for the case above: it would pass on an `unenforceable`
    that returned every clause."""
    doc = write_doc(tmp_path, "# P\n\nThe driver MUST verify the digest.\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True,
                 "binding_gate": {clauses[0].clause_id: complete_row()}}
    assert gate_inventory.unenforceable(clauses, inventory) == []


def test_relative_document_paths_work(tmp_path, monkeypatch):
    """The first thing anyone types is the relative form, and it crashed."""
    write_doc(tmp_path, "# P\n\nThe driver MUST verify the digest.\n")
    clauses = derive_clauses([Path("DESIGN.md")], tmp_path)
    assert len(clauses) == 1
    assert clauses[0].doc == "DESIGN.md"


def test_an_unreviewed_inventory_cannot_pass_however_complete_it_looks(tmp_path):
    """A machine-drafted file must not be mistakable for a finished one.

    The failure without this is obvious in hindsight: a draft lands, looks
    complete, somebody runs the reconciler and gets a green. `reviewed`
    defaults absent-is-false, so an inventory that never says a human read
    it fails on that alone.
    """
    doc = write_doc(tmp_path)
    (tmp_path / "gate.py").write_text("def verify_digest():\n    return True\n")
    (tmp_path / "test_gate.py").write_text("def test_digest():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    complete_but_undeclared = {
        "binding_gate": {c.clause_id: complete_row() for c in clauses}}
    assert "unreviewed_inventory" in kinds(
        reconcile(clauses, complete_but_undeclared, tmp_path))


def test_an_exemption_needs_a_reason(tmp_path):
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    inventory = {"binding_gate": {}, "not_binding": {clauses[0].clause_id: {}}}
    assert "unreasoned_disposition" in kinds(reconcile(clauses, inventory, tmp_path))


# --- the vacuity guard -----------------------------------------------------

def test_deriving_nothing_is_reported_as_an_error_not_as_success(tmp_path):
    """A reconciler over zero clauses reports zero findings and looks clean.

    That is the failure mode this whole repository keeps paying for, and it
    would be at its worst here: the tool's green result would be read as
    "every documented gate is enforced".
    """
    empty = write_doc(tmp_path, "# Nothing binding here.\n")
    inventory = tmp_path / "gates.toml"
    inventory.write_text("[gate]\n")
    exit_code = gate_inventory.main(
        ["--inventory", str(inventory), "--doc", str(empty),
         "--root", str(tmp_path)])
    assert exit_code == 2, "an empty derivation must not exit 0"


def test_findings_make_the_run_fail(tmp_path):
    doc = write_doc(tmp_path)
    inventory = tmp_path / "gates.toml"
    inventory.write_text("[gate]\n")
    exit_code = gate_inventory.main(
        ["--inventory", str(inventory), "--doc", str(doc),
         "--root", str(tmp_path)])
    assert exit_code == 1


def test_a_clean_reconciliation_exits_zero(tmp_path):
    """Non-vacuity for the two exit-code tests above."""
    doc = write_doc(tmp_path, "# P\n\nThe driver MUST verify the digest.\n")
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    (clause,) = derive_clauses([doc], tmp_path)
    inventory = tmp_path / "gates.toml"
    inventory.write_text(
        'reviewed = true\n'
        f'[binding_gate."{clause.clause_id}"]\n'
        f'enforcement = "g.py::f"\n'
        f'production_reachability = "called by the stage3 driver"\n'
        f'input_wiring = "digest from BONSAI_DRIVER_SHA256"\n'
        f'decision_consequence = "raises SystemExit before launch"\n'
        f'test = "g.py::f"\n'
        f'break_demonstrated = "disabled it, test failed"\n'
        f'trigger = "ci-fast"\n')
    exit_code = gate_inventory.main(
        ["--inventory", str(inventory), "--doc", str(doc),
         "--root", str(tmp_path)])
    assert exit_code == 0


# --- a claim whose package does not exist yet ------------------------------
#
# Raised as blocking by the instance FILLING the inventory, which is the
# useful direction for a schema defect to travel. Most of their claim rows
# were binding, reviewed, and not dischargeable because the readiness package
# to discharge into had not been assembled. The schema offered no way to say
# so: `discharged_in` and `evidence` were unconditionally required, so an
# honest row emitted three findings, and the only ways to silence them were
# to invent artifacts or to drown the real findings.

CLAIM_DOC = "# P\n\nResults MUST be reported with the seed.\n"


def test_a_claim_whose_package_does_not_exist_yet_is_not_asked_to_invent_one(tmp_path):
    """The blocking case: one finding about the true state, not three about
    fields that cannot be answered."""
    doc = write_doc(tmp_path, CLAIM_DOC)
    (clause,) = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {clause.clause_id: claim_row(
        status="pending_package",
        pending_reason="the Stage 2B readiness package is not assembled yet",
        discharged_in=None, evidence=None)}}
    findings = reconcile([clause], inventory, tmp_path)
    assert kinds(findings) == ["claim_has_no_package"], (
        "a pending_package row should report only that it cannot be "
        f"discharged yet; got {kinds(findings)}")
    assert "not assembled yet" in findings[0].detail, (
        "the finding does not carry the reason, so a reader cannot tell this "
        "row from one nobody has looked at")


def test_pending_package_without_a_reason_is_also_a_finding(tmp_path):
    """The escape hatch, closed. Without this, `pending_package` is a word
    that switches off two required fields and asserts nothing."""
    doc = write_doc(tmp_path, CLAIM_DOC)
    (clause,) = derive_clauses([doc], tmp_path)
    inventory = {"reviewed": True, "binding_claim": {clause.clause_id: claim_row(
        status="pending_package", discharged_in=None, evidence=None)}}
    assert "unreasoned_pending_package" in kinds(
        reconcile([clause], inventory, tmp_path))


def test_pending_package_still_fails_readiness(tmp_path):
    """An honest state, not a passing one. If this ever exits zero, a package
    that does not exist ships green."""
    doc = write_doc(tmp_path, CLAIM_DOC)
    (clause,) = derive_clauses([doc], tmp_path)
    inventory = tmp_path / "gates.toml"
    inventory.write_text(
        'reviewed = true\n'
        f'[binding_claim."{clause.clause_id}"]\n'
        'locator = "P.md#reporting"\n'
        'obligation = "results carry the seed"\n'
        'status = "pending_package"\n'
        'pending_reason = "no readiness package exists yet"\n')
    assert gate_inventory.main(
        ["--inventory", str(inventory), "--doc", str(doc),
         "--root", str(tmp_path)]) != 0


def test_a_discharged_claim_still_must_say_where_and_show_evidence(tmp_path):
    """Non-vacuity for the three above. The conditional must not have turned
    the requirement off everywhere: a `discharged` row asserts an artifact
    exists, so it is asked to name it."""
    doc = write_doc(tmp_path, CLAIM_DOC)
    (clause,) = derive_clauses([doc], tmp_path)
    for missing in ("discharged_in", "evidence"):
        inventory = {"reviewed": True, "binding_claim": {
            clause.clause_id: claim_row(**{missing: None})}}
        assert f"missing_{missing}" in kinds(
            reconcile([clause], inventory, tmp_path)), (
            f"a discharged claim with no `{missing}` was accepted; the "
            f"conditional requirement has switched the check off entirely")


def test_pending_package_is_distinct_from_unresolved(tmp_path):
    """Different findings, because they need different work: unresolved wants
    a reviewer, pending_package wants a package. Collapsing them hides
    which."""
    doc = write_doc(tmp_path, CLAIM_DOC)
    (clause,) = derive_clauses([doc], tmp_path)
    unresolved = reconcile([clause], {"reviewed": True, "binding_claim": {
        clause.clause_id: claim_row(status="unresolved")}}, tmp_path)
    pending = reconcile([clause], {"reviewed": True, "binding_claim": {
        clause.clause_id: claim_row(status="pending_package",
                                    pending_reason="no package yet",
                                    discharged_in=None,
                                    evidence=None)}}, tmp_path)
    assert kinds(unresolved) != kinds(pending)


# --- the conditional map itself, both directions ---------------------------
#
# Principle 21. A typo in `_REQUIRED_ONLY_WHEN_STATUS` fails SILENTLY and in
# the dangerous direction: a misspelled field name never matches, the field
# stays unconditionally required, and the defect this fix exists to remove is
# still present behind a green suite.


def _conditional_map_problems(conditional: dict) -> list[str]:
    """Every way a conditional-requirement map can be wrong.

    A function rather than assertions inline, so the checks can be run
    against a DELIBERATELY BROKEN map as well as the real one. Otherwise
    these are guards nobody has seen fail: they would pass identically
    against a validator that returned nothing at all.
    """
    valid = {"binding_claim": set(gate_inventory._CLAIM_STATUSES),
             "binding_value": set(gate_inventory._VALUE_STATUSES)}
    problems = []
    for kind, fields in conditional.items():
        if kind not in gate_inventory._REQUIRED_BY_KIND:
            problems.append(
                f"`{kind}` is not a real kind, so its conditions apply to "
                f"nothing")
            continue
        for field, statuses in fields.items():
            if field not in gate_inventory._REQUIRED_BY_KIND[kind]:
                problems.append(
                    f"`{kind}.{field}` is conditioned but is not a required "
                    f"field of that kind -- the condition matches nothing, "
                    f"and the field it was meant to relax is unaffected")
            unknown = set(statuses) - valid.get(kind, set())
            if unknown:
                problems.append(
                    f"`{kind}.{field}` is required only under "
                    f"{sorted(unknown)}, which the checker never assigns")
    return problems


def test_the_real_conditional_map_is_sound():
    assert _conditional_map_problems(
        gate_inventory._REQUIRED_ONLY_WHEN_STATUS) == []


@pytest.mark.parametrize("broken,why", [
    ({"binding_clam": {"discharged_in": frozenset({"discharged"})}},
     "a misspelled KIND matches nothing, so the field stays unconditionally "
     "required and the defect is silently still present"),
    ({"binding_claim": {"discharge_in": frozenset({"discharged"})}},
     "a misspelled FIELD is the same failure one level down"),
    ({"binding_claim": {"discharged_in": frozenset({"complete"})}},
     "a status the checker never assigns makes the field permanently "
     "optional -- the opposite of the intended effect"),
])
def test_a_broken_conditional_map_is_caught(broken, why):
    """The guard, seen failing. Each of these is a plausible typo whose
    effect is invisible from behaviour: the suite stays green either way."""
    assert _conditional_map_problems(broken), why


def test_the_offered_statuses_are_the_accepted_statuses():
    """The schema text a filler reads and the check that rejects an unknown
    status are one list. They were two hand-copied ones."""
    assert (gate_inventory._REQUIRED_BY_KIND["binding_claim"]["status"]
            == " | ".join(gate_inventory._CLAIM_STATUSES))
    assert (gate_inventory._REQUIRED_BY_KIND["binding_value"]["status"]
            == " | ".join(gate_inventory._VALUE_STATUSES))


def test_an_unknown_claim_status_is_still_rejected(tmp_path):
    """Non-vacuity: adding a status must not have opened the set."""
    doc = write_doc(tmp_path, CLAIM_DOC)
    (clause,) = derive_clauses([doc], tmp_path)
    assert "unknown_status" in kinds(reconcile([clause], {
        "reviewed": True, "binding_claim": {
            clause.clause_id: claim_row(status="pending_review")}}, tmp_path))
