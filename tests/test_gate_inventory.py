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
    findings = reconcile(clauses, {"gate": {}}, tmp_path)
    assert kinds(findings) == ["unmapped_clause", "unmapped_clause"]


def test_a_mapping_for_a_clause_that_no_longer_exists_is_a_finding(tmp_path):
    """The other direction. Without it, editing a requirement leaves a
    mapping that describes nothing and nothing reports it."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)
    inventory = {"gate": {c.clause_id: {"enforcement": "x.py::f",
                                        "test": "t.py::t",
                                        "break_demonstrated": "yes"}
                          for c in clauses}}
    inventory["gate"]["deadbeef1234"] = {"enforcement": "x.py::f",
                                         "test": "t.py::t",
                                         "break_demonstrated": "yes"}
    findings = reconcile(clauses, inventory, tmp_path)
    assert "orphaned_mapping" in kinds(findings)


def test_a_fully_mapped_document_produces_no_findings(tmp_path):
    """Non-vacuity for every negative assertion above: they would all pass
    against a reconcile() that reported findings unconditionally."""
    doc = write_doc(tmp_path)
    (tmp_path / "gate.py").write_text("def verify_digest():\n    return True\n")
    (tmp_path / "test_gate.py").write_text("def test_digest():\n    pass\n")
    clauses = derive_clauses([doc], tmp_path)
    inventory = {"gate": {c.clause_id: {"enforcement": "gate.py::verify_digest",
                                        "test": "test_gate.py::test_digest",
                                        "break_demonstrated": "disabled the "
                                                              "branch, test failed",
                                        "trigger": "ci-fast"}
                          for c in clauses}}
    assert reconcile(clauses, inventory, tmp_path) == []


# --- the mapped references must actually resolve ---------------------------

def test_an_enforcement_site_that_does_not_exist_is_a_finding(tmp_path):
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    inventory = {"gate": {clauses[0].clause_id: {
        "enforcement": "nonexistent.py::f", "test": "also_missing.py::t",
        "break_demonstrated": "yes"}}}
    assert "missing_enforcement" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_name_only_mentioned_in_a_comment_does_not_satisfy_a_citation(tmp_path):
    """Resolved from the AST, not by substring. A citation satisfied by a
    comment is the `endswith` failure from the vacuous-test catalogue."""
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "gate.py").write_text("# verify_digest is not defined here\n")
    inventory = {"gate": {clauses[0].clause_id: {
        "enforcement": "gate.py::verify_digest", "test": "gate.py::verify_digest",
        "break_demonstrated": "yes"}}}
    assert "missing_enforcement" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_mapping_without_break_evidence_is_a_finding(tmp_path):
    """"Gate cited, test cited" certifies spelling.

    The precedent is concrete: a test that grepped source for
    `halt_reasons.append(...)` stayed green when the branch guarding it was
    disabled by `if False:`. It would have passed on exactly the artifact
    that failed.
    """
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    inventory = {"gate": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f"}}}
    assert "unproven_test" in kinds(reconcile(clauses, inventory, tmp_path))


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
    inventory = {"gate": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f",
        "break_demonstrated": "yes"}}}
    assert "untriggered_test" in kinds(reconcile(clauses, inventory, tmp_path))


def test_a_manual_only_trigger_is_reported_but_not_treated_as_absent(tmp_path):
    """Some gates legitimately cost too much to run automatically.

    The point is that this is visible in the inventory rather than
    discovered when one fires late, so it is a distinct finding kind rather
    than either silence or a hard failure.
    """
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    (tmp_path / "g.py").write_text("def f():\n    pass\n")
    inventory = {"gate": {clauses[0].clause_id: {
        "enforcement": "g.py::f", "test": "g.py::f",
        "break_demonstrated": "yes", "trigger": "manual"}}}
    found = kinds(reconcile(clauses, inventory, tmp_path))
    assert "manual_trigger_only" in found
    assert "untriggered_test" not in found


def test_an_exemption_needs_a_reason(tmp_path):
    clauses = derive_clauses([write_doc(tmp_path)], tmp_path)[:1]
    inventory = {"gate": {}, "exempt": {clauses[0].clause_id: {}}}
    assert "unreasoned_exemption" in kinds(reconcile(clauses, inventory, tmp_path))


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
        f'[gate."{clause.clause_id}"]\nenforcement = "g.py::f"\n'
        f'test = "g.py::f"\nbreak_demonstrated = "disabled it, test failed"\n'
        f'trigger = "ci-fast"\n')
    exit_code = gate_inventory.main(
        ["--inventory", str(inventory), "--doc", str(doc),
         "--root", str(tmp_path)])
    assert exit_code == 0
