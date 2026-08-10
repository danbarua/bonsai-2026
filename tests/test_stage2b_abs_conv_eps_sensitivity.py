"""Tier 1/2 checks on Companion Protocol 2's sensitivity table
(`run_abs_conv_eps_sensitivity.py`): the encoder gate's verdict recomputed
from stored final-Delta arrays, no re-encoding, across
`eps in {1e-10, 1e-11, 1e-12, 1e-13}`.

Tier 1 (`summarize`, `verdict_is_invariant_at`) is pure and always runs.
Tier 2 (`test_the_real_diagnostic_pickle_produces_an_invariant_verdict`)
needs `results/encoder_gate_failure_diagnostic.pkl` -- reproducible via
`diagnose_encoder_gate_failure.py`, but gitignored and not committed -- and
skips cleanly when it is absent, per this project's two-tier convention.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE2B_DIR = REPO_ROOT / "experiments" / "stage2b_denoising"
sys.path.insert(0, str(STAGE2B_DIR))

import run_abs_conv_eps_sensitivity as sens  # noqa: E402
import stage2b_audit as audit  # noqa: E402
import stage2b_encoder_gate as gate  # noqa: E402


def _table_for(passed_by_eps_by_steps):
    """A synthetic table shaped like `sensitivity_table`'s real output,
    with only the one field `summarize`/`verdict_is_invariant_at` read."""
    return {steps: {eps: {"passed": passed} for eps, passed in by_eps.items()}
            for steps, by_eps in passed_by_eps_by_steps.items()}


def test_verdict_is_invariant_true_when_every_eps_agrees():
    table = _table_for({"1200": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True}})
    assert sens.verdict_is_invariant_at(table, 1200) is True


def test_verdict_is_invariant_false_when_any_eps_disagrees():
    table = _table_for({"1200": {"1e-10": True, "1e-11": True, "1e-12": False, "1e-13": True}})
    assert sens.verdict_is_invariant_at(table, 1200) is False


def test_summarize_does_not_trigger_the_halt_when_locked_steps_is_invariant():
    table = _table_for({
        "600": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True},
        "1200": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True},
    })
    summary = sens.summarize(table, locked_steps=1200)
    assert summary["invariant_at_locked_steps"] is True
    assert summary["halt_triggered"] is False


def test_summarize_triggers_the_halt_when_locked_steps_flips():
    """Break-confirmation: a verdict that flips at the LOCKED step count
    must trigger the halt, even if every OTHER step count stays invariant
    -- the consequence rule is scoped to the locked steps specifically."""
    table = _table_for({
        "600": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True},
        "1200": {"1e-10": True, "1e-11": True, "1e-12": False, "1e-13": True},
    })
    summary = sens.summarize(table, locked_steps=1200)
    assert summary["invariant_at_locked_steps"] is False
    assert summary["halt_triggered"] is True
    assert summary["per_step_count_invariance"]["600"] is True, (
        "a flip at 1200 must not be reported as a flip at 600 -- the two rows are "
        "independent")


def test_summarize_raises_when_the_locked_steps_row_is_absent():
    table = _table_for({"600": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True}})
    with pytest.raises(audit.AuditInputError, match="locked ENCODER_STEPS"):
        sens.summarize(table, locked_steps=1200)


def test_compute_table_calls_the_real_gate_unmodified():
    """`compute_table` must not reimplement `evaluate_rho_gate` -- confirmed
    by feeding it a case whose PASS/FAIL is known only via the real gate
    (the absolute-convergence escape) and checking the real gate's own
    field names come back untouched."""
    import numpy as np
    final_delta_by_steps = {1200: (np.zeros(10), np.zeros(10))}
    table = sens.compute_table(final_delta_by_steps)
    cell = table["1200"]["1e-12"]
    assert cell["passed"] is True
    assert cell["absolute_convergence"] is True
    assert set(cell) == set(gate.evaluate_rho_gate(np.zeros(10), np.zeros(10)))


@pytest.mark.skipif(not os.path.isfile(sens.DIAGNOSTIC_PKL),
                    reason="encoder_gate_failure_diagnostic.pkl not present locally "
                           "(gitignored, regenerable via diagnose_encoder_gate_failure.py)")
def test_the_real_diagnostic_pickle_produces_an_invariant_verdict():
    """Hand-verified interactively before this test existed (principle 20):
    every step count's verdict is invariant across the full swept eps
    range, so Protocol 2's halt condition does not fire at the locked
    ENCODER_STEPS=1200."""
    final_delta_by_steps, diagnostic = sens.load_final_deltas()
    table = sens.compute_table(final_delta_by_steps)
    summary = sens.summarize(table)
    assert summary["halt_triggered"] is False
    assert summary["invariant_at_locked_steps"] is True
    assert set(table) == {str(s) for s in diagnostic["step_counts"]}
    for steps in table:
        assert sens.verdict_is_invariant_at(table, int(steps)), (
            f"steps={steps} is not invariant across the swept eps range -- this contradicts "
            f"the hand-verified result and needs investigation, not a widened tolerance")
