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
    findings = reconcile(clauses, {"binding": {}}, tmp_path)
    assert kinds(findings) == ["undispositioned_candidate", "undispositioned_candidate"]


def test_a_mapping_for_a_clause_that_no_longer_exists_is_a_finding(tmp_path):
    """The other direction. Without it, editing a requirement leaves a
    mapping that describes nothing and nothing reports it."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)
    inventory = {"binding": {c.clause_id: {"enforcement": "x.py::f",
                                        "test": "t.py::t",
                                        "break_demonstrated": "yes"}
                          for c in clauses}}
    inventory["binding"]["deadbeef1234"] = {"enforcement": "x.py::f",
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


def test_a_fully_dispositioned_document_produces_no_findings(tmp_path):
    """Non-vacuity for every negative assertion above: they would all pass
    against a reconcile() that reported findings unconditionally."""
    doc = write_doc(tmp_path)
    (tmp_path / "gate.py").write_text("def verify_digest():\n    return True\n")
    (tmp_path / "test_gate.py").write_text("def test_digest():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"binding": {c.clause_id: complete_row() for c in clauses}}
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
    inventory = {"binding": {clauses[0].clause_id: row}}
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
    assert "id_collision" in kinds(reconcile(clauses, {"binding": {}}, tmp_path))


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
    inventory = {"binding": {clauses[0].clause_id: {
        "enforcement": "nonexistent.py::f", "test": "also_missing.py::t",
        "break_demonstrated": "yes"}}}
    assert "unresolved_enforcement" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_name_only_mentioned_in_a_comment_does_not_satisfy_a_citation(tmp_path):
    """Resolved from the AST, not by substring. A citation satisfied by a
    comment is the `endswith` failure from the vacuous-test catalogue."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "gate.py").write_text("# verify_digest is not defined here\n")
    inventory = {"binding": {clauses[0].clause_id: {
        "enforcement": "gate.py::verify_digest", "test": "gate.py::verify_digest",
        "break_demonstrated": "yes"}}}
    assert "unresolved_enforcement" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_mapping_without_break_evidence_is_a_finding(tmp_path):
    """"Gate cited, test cited" certifies spelling.

    The precedent is concrete: a test that grepped source for
    `halt_reasons.append(...)` stayed green when the branch guarding it was
    disabled by `if False:`. It would have passed on exactly the artifact
    that failed.
    """
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    inventory = {"binding": {clauses[0].clause_id: {
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
    inventory = {"binding": {clauses[0].clause_id: {
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
    inventory = {"binding": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f",
        "break_demonstrated": "yes", "trigger": "manual"}}}
    found = kinds(reconcile(clauses, inventory, tmp_path))
    assert "manual_trigger_only" in found
    assert "missing_trigger" not in found


def test_an_exemption_needs_a_reason(tmp_path):
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    inventory = {"binding": {}, "not_binding": {clauses[0].clause_id: {}}}
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
        f'[binding."{clause.clause_id}"]\n'
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
