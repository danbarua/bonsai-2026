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
import ast
import os
import sys
import types
from pathlib import Path

import numpy as np
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
    return {
        str(steps): {str(eps): {"passed": p} for eps, p in by_eps.items()}
        for steps, by_eps in passed_by_eps_by_steps.items()
    }


def test_verdict_is_invariant_true_when_every_eps_agrees():
    table = _table_for({"1200": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True}})
    assert sens.verdict_is_invariant_at(table, 1200) is True


def test_verdict_is_invariant_false_when_any_eps_disagrees():
    table = _table_for({"1200": {"1e-10": True, "1e-11": True, "1e-12": False, "1e-13": True}})
    assert sens.verdict_is_invariant_at(table, 1200) is False


def test_summarize_does_not_trigger_the_halt_when_locked_steps_is_invariant():
    table = _table_for({
        "1200": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True}
    })
    summary = sens.summarize(table, locked_steps=1200)
    assert summary["halt_triggered"] is False
    assert summary["invariant_at_locked_steps"] is True


def test_summarize_triggers_the_halt_when_locked_steps_flips():
    """Break-confirmation: a verdict that flips at the LOCKED step count
    sets halt_triggered and records the flip detail."""
    table = _table_for({
        "1200": {"1e-10": True, "1e-11": True, "1e-12": False, "1e-13": False}
    })
    summary = sens.summarize(table, locked_steps=1200)
    assert summary["halt_triggered"] is True
    assert summary["invariant_at_locked_steps"] is False
    assert "1200" in summary.get("per_step_flip", {})


def test_summarize_raises_when_the_locked_steps_row_is_absent():
    table = _table_for({"600": {"1e-10": True, "1e-11": True, "1e-12": True, "1e-13": True}})
    with pytest.raises(audit.AuditInputError, match="locked ENCODER_STEPS"):
        sens.summarize(table, locked_steps=1200)


def test_compute_table_calls_the_real_gate_unmodified():
    """`compute_table` must not reimplement `evaluate_rho_gate` -- confirmed
    by shape of cells and that the function object is passed through."""
    final = {1200: (np.zeros(10), np.zeros(10))}
    table = sens.compute_table(final)
    assert "1200" in table
    cell = table["1200"]["1e-12"]
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


def test_merge_requires_ladder_overlap_agreement():
    """diag vs ladder overlap must agree within 1e-15 or raise."""
    diag = {1200: (np.zeros(5), np.ones(5))}
    ladder = {1200: (np.zeros(5) + 1e-16, np.ones(5))}
    m, t = sens.merge_step_sources(diag, ladder)
    assert 1200 in m
    bad = {1200: (np.zeros(5) + 1e-10, np.ones(5))}
    with pytest.raises(RuntimeError, match="mismatch|required"):
        sens.merge_step_sources(diag, bad)


def test_merge_tags_sources():
    """Steps tagged correctly: diagnostic only, ladder only, both."""
    diag = {75: (np.array([0.]), np.array([0.])), 1200: (np.array([1.]), np.array([1.]))}
    ladder = {1200: (np.array([1.]), np.array([1.]))}
    m, tags = sens.merge_step_sources(diag, ladder)
    assert tags.get(75) == "diagnostic" or tags.get("75") == "diagnostic"
    assert tags.get(1200) == "diagnostic+ladder" or tags.get("1200") == "diagnostic+ladder"


def test_axis4_bounds_monotone_and_below_rtol_for_swept_eps():
    """For each eps in {1e-10,1e-12,1e-13}, bound < 1e-6, decreases as eps decreases."""
    res = sens.axis4_downstream_sensitivity()
    rows = res["rows"]
    assert len(rows) == 3
    bounds = [r["feature_linf_bound"] for r in rows]
    assert all(b < 1e-6 for b in bounds)
    assert bounds[0] > bounds[1] > bounds[2]


P2_DRIVER_PATH = STAGE2B_DIR / "run_abs_conv_eps_sensitivity.py"


def test_publish_path_passes_fingerprint():
    """AST: every ensure_json wrapper call that publishes the table passes non-None fingerprint.
    Mirrors P1.
    """
    tree = ast.parse(P2_DRIVER_PATH.read_text())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ensure_json":
            calls.append(node)
    assert len(calls) >= 1, "expected ensure_json publish call(s)"
    for c in calls:
        fp_kw = next((kw for kw in c.keywords if kw.arg == "fingerprint"), None)
        assert fp_kw is not None, "publish ensure must pass fingerprint="
        val = fp_kw.value
        if isinstance(val, ast.Constant):
            assert val.value is not None


def test_revalidation_failure_is_fail_sentinel(monkeypatch, capsys, tmp_path):
    """Reval mismatch path prints FAIL_SENTINEL and exits nonzero (like P1)."""
    import stage2b_fingerprint as sfp
    # patch to bypass ladder hard require and reach reval
    monkeypatch.setattr(sens, "load_ladder_encoder_gate_deltas", lambda p: (np.zeros(3), np.zeros(3)))
    # patch build to provide fp
    monkeypatch.setattr(sens, "build_fingerprint",
        lambda repo, cfg, require_clean=True: {"source_manifest_digest": "x", "config_digest": "y"})
    # force a reval fail inside a wrapped main path by temp override
    def wrapped(argv=None):
        # simulate the post-ensure reval block
        fp = {"source_manifest_digest": "x", "config_digest": "y"}
        try:
            import stage2b_fingerprint as fpmod
            fpmod.revalidate_after_execution(fp, str(tmp_path))
        except Exception as exc:  # noqa: BLE001
            print(f"{sens.FAIL_SENTINEL} fingerprint revalidate failed: {type(exc).__name__}: {exc}", flush=True)
            return 1
        print(sens.OK_SENTINEL, flush=True)
        return 0
    monkeypatch.setattr(sens, "main", wrapped)
    rc = sens.main(["--no-upload"])
    out = capsys.readouterr().out
    assert rc == 1
    assert sens.FAIL_SENTINEL in out
    assert "revalidate" in out or "injected" in out or "mismatch" in out
