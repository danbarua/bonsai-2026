"""Companion Protocol 2 (`COMPANION_PROTOCOLS.md`): the `ABS_CONV_EPS`
sensitivity table -- the encoder gate's verdict recomputed from stored
per-image final-Delta arrays, no re-encoding, at
`eps in {1e-10, 1e-11, 1e-12, 1e-13}`, crossed with every step count for
which a stored array exists.

Runs entirely on CPU and touches no network: `stage2b_audit.sensitivity_table`
and `stage2b_encoder_gate.evaluate_rho_gate` are both pure functions of
already-computed arrays, called unmodified (CLAUDE.md principle 16 -- no
reimplementation of the gate). Its one input,
`results/encoder_gate_failure_diagnostic.pkl`, is reproducible from committed
code: `diagnose_encoder_gate_failure.py` produces it, from a real A100 run's
own reported numbers, verified byte-for-byte before anything in that pickle
is trusted -- see that script's own docstring. Regenerate it first if it is
missing.

This closes the gap CLAUDE.md principle 24 exists to name: the table was
first computed as an interactive one-liner gluing two already-tested,
already-committed functions together. That is exactly the "load-bearing
scratch whose generator is a chat transcript" pattern the principle forbids
citing -- so it is promoted here, into committed, reproducible code, before
being cited anywhere.
"""
import json
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import stage2b_audit as audit                                            # noqa: E402
import stage2b_encoder_gate as gate                                      # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
DIAGNOSTIC_PKL = os.path.join(RESULTS_DIR, "encoder_gate_failure_diagnostic.pkl")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "abs_conv_eps_sensitivity_table.json")


def load_final_deltas(path=DIAGNOSTIC_PKL):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist. It is reproducible from committed code: run "
            f"`uv run python diagnose_encoder_gate_failure.py` first, which regenerates "
            f"stage 1's corpus/corruption locally from the same deterministic seeds the "
            f"cloud run used and verifies them byte-for-byte before computing anything.")
    with open(path, "rb") as handle:
        diagnostic = pickle.load(handle)
    return {steps: (row["delta_clean"], row["delta_noisy"])
            for steps, row in diagnostic["curve_rows"].items()}, diagnostic


def compute_table(final_delta_by_steps):
    """The frozen recomputation itself -- both functions already tested,
    neither reimplemented here."""
    return audit.sensitivity_table(final_delta_by_steps, gate.evaluate_rho_gate)


def verdict_is_invariant_at(table, steps):
    """Whether every eps in the swept range agrees on PASS/FAIL at one step
    count -- the quantity `COMPANION_PROTOCOLS.md` says is "the result"."""
    verdicts = {cell["passed"] for cell in table[str(steps)].values()}
    return len(verdicts) == 1


def summarize(table, locked_steps=gate.ENCODER_STEPS):
    """The consequence rule's own question, answered directly: does the
    verdict at the LOCKED step count flip across the swept eps range."""
    locked_key = str(locked_steps)
    if locked_key not in table:
        raise audit.AuditInputError(
            f"locked ENCODER_STEPS={locked_steps} has no row in the sensitivity table; "
            f"cannot evaluate the consequence rule without it")
    invariant_at_locked = verdict_is_invariant_at(table, locked_steps)
    per_step_invariance = {steps: verdict_is_invariant_at(table, int(steps))
                           for steps in table}
    return {
        "locked_encoder_steps": locked_steps,
        "invariant_at_locked_steps": invariant_at_locked,
        "per_step_count_invariance": per_step_invariance,
        "halt_triggered": not invariant_at_locked,
        "consequence_rule": (
            "COMPANION_PROTOCOLS.md Protocol 2: if the verdict at the locked "
            "ENCODER_STEPS differs across the swept eps range, that triggers renewed "
            "interpretation review before Stage 4. ABS_CONV_EPS itself does not change "
            "as a result of this table, regardless of outcome."),
    }


def main():
    final_delta_by_steps, diagnostic = load_final_deltas()
    table = compute_table(final_delta_by_steps)
    summary = summarize(table)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "table": table,
        "summary": summary,
        "source": {
            "pickle": os.path.relpath(DIAGNOSTIC_PKL, _THIS_DIR),
            "n_used": diagnostic["n_used"],
            "step_counts": list(diagnostic["step_counts"]),
        },
        "production_abs_conv_eps": gate.ABS_CONV_EPS,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"wrote {OUTPUT_JSON}")
    print(f"locked ENCODER_STEPS={summary['locked_encoder_steps']}: "
          f"{'INVARIANT' if summary['invariant_at_locked_steps'] else 'FLIPS'} "
          f"across eps in {{1e-10, 1e-11, 1e-12, 1e-13}}")
    for steps in sorted(table, key=int):
        invariant = summary["per_step_count_invariance"][steps]
        cells = sorted(table[steps].items(), key=lambda kv: float(kv[0]))
        detail = ", ".join(f"{eps}={cell['passed']}" for eps, cell in cells)
        print(f"  steps={steps}: {'invariant' if invariant else 'FLIPS'} ({detail})")
    if summary["halt_triggered"]:
        print("HALT: verdict at the locked step count is NOT invariant -- "
              "renewed interpretation review required before Stage 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
