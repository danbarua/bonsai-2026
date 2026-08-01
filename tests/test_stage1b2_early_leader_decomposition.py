"""
Verifies the early-leader decomposition documented in
experiments/stage1b2_structured_transformation/CONCENTRATION_REGIME_NOTE.md
Part 5 -- a load-bearing result (the early-leader mismatch found by plot10
is entirely a linear/tangent-dynamics phenomenon, not evidence of nonlinear
rerouting) that previously had no dedicated test, only a notebook/plot and
prose in the note. Exists to catch silent drift between the manuscript's
claimed numbers and what the analysis script actually produces if
`generate_frontier_visuals_data.py` or the underlying cached results ever
change.

Skip-if-missing (Tier 2 style, matching this project's convention): this
depends on gitignored cached results (stage1b2_results.pkl, Stage 1C's
per-seed result files, stage1b2_frontier_visuals_data.pkl) that aren't
committed. No new simulation -- calling `main()` here is pure re-analysis
of already-cached data, identical to running the script directly.
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE1B2_DIR = _REPO_ROOT / "experiments" / "stage1b2_structured_transformation"
sys.path.insert(0, str(_STAGE1B2_DIR))

_RESULTS_PATH = _STAGE1B2_DIR / "results" / "stage1b2_results.pkl"
_FRONTIER_DATA_PATH = _STAGE1B2_DIR / "results" / "stage1b2_frontier_visuals_data.pkl"
_STAGE1C_RESULTS_DIR = _REPO_ROOT / "experiments" / "stage1c_trajectory_generalization" / "results"

_REQUIRED = [_RESULTS_PATH, _FRONTIER_DATA_PATH] + [
    _STAGE1C_RESULTS_DIR / f"stage1c_results_seed{seed}.pkl"
    for seed in (3010, 3020, 3080, 3090)
]

# Exact expected breakdown, per CONCENTRATION_REGIME_NOTE.md Part 5 and the
# statistical review that requested this test.
EXPECTED_TOTAL_TRIALS = 87
EXPECTED_SKIPPED = 0
EXPECTED_BREAKDOWN_A_LINEAR_OVERTAKING = {
    3000: (0, 24),
    3010: (0, 21),
    3020: (0, 2),
    3080: (0, 5),
    3090: (35, 35),
}
EXPECTED_BREAKDOWN_B_NONLINEAR_MODIFICATION = {
    3000: (24, 24),
    3010: (21, 21),
    3020: (2, 2),
    3080: (5, 5),
    3090: (35, 35),
}


@pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="cached Stage 1B2/1C results not present locally (gitignored, not committed)",
)
def test_early_leader_decomposition_matches_documented_breakdown():
    from analyze_stage1b2_early_leader_decomposition import main

    rows, breakdown_a, breakdown_b = main()

    assert len(rows) == EXPECTED_TOTAL_TRIALS, (
        f"Expected {EXPECTED_TOTAL_TRIALS} concentrated trials, got {len(rows)} -- "
        "the underlying cached results or concentration threshold may have changed."
    )

    assert breakdown_a == EXPECTED_BREAKDOWN_A_LINEAR_OVERTAKING, (
        "Transition (a) (early tangent tau=0.95 -> final tangent tau=T, "
        "tests LINEAR overtaking) no longer matches CONCENTRATION_REGIME_NOTE.md "
        f"Part 5's documented per-seed breakdown. Got: {breakdown_a}"
    )

    assert breakdown_b == EXPECTED_BREAKDOWN_B_NONLINEAR_MODIFICATION, (
        "Transition (b) (final tangent tau=T -> final finite tau=T, tests "
        "NONLINEAR destination modification) no longer shows the documented "
        f"87/87 perfect match. Got: {breakdown_b}"
    )

    total_matches_b = sum(n_match for n_match, _ in breakdown_b.values())
    total_trials_b = sum(n for _, n in breakdown_b.values())
    assert total_matches_b == total_trials_b == EXPECTED_TOTAL_TRIALS, (
        "Transition (b) should be a perfect 87/87 match -- if this fails, "
        "the note's central claim (nonlinearity never changes endpoint argmax "
        "identity among concentrated trials) needs re-examining, not just this test."
    )
