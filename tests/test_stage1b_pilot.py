"""Verifies the quantitative claims in experiments/stage1b_pilot/FINDINGS.md.

Deliberately asymmetric: strict on the continuous dynamics (peak
amplification) and on the two taxonomy counts that are scientifically
load-bearing (0 different_equilibria, 0 non-convergent trials) -- these
reproduced identically in an independent re-run. Loose on
baseline_only_converged specifically, which FINDINGS.md's
"Reproducibility note" section documents as environment-sensitive right at
the FORCE_CONVERGED_THRESHOLD boundary (an independent re-run found 10
such trials against this document's reported 3, all sitting at 1.0-1.6x
the threshold) -- asserting an exact count there would make this test
flaky across machines/scipy versions for a reason that has nothing to do
with the dynamics themselves.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE1B_PILOT_DIR = REPO_ROOT / "experiments" / "stage1b_pilot"
CONSTRUCTIONS_PATH = (
    REPO_ROOT / "experiments" / "stage1b2_structured_transformation" / "results"
    / "class0_constructions.pkl"
)

sys.path.insert(0, str(STAGE1B_PILOT_DIR))
from stage1b_taxonomy import classify_one_trial  # noqa: E402

IC_SEEDS = [2000, 2001]
AMPLITUDES = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
SIGNS = [1, -1]

# FORCE_CONVERGED_THRESHOLD from stage1b_taxonomy.py -- the boundary
# baseline_only_converged classification sits on.
FORCE_CONVERGED_THRESHOLD = 1e-5

# This project's own previously-observed independent-reproduction count for
# baseline_only_converged (see FINDINGS.md's "Reproducibility note"),
# against FINDINGS.md's originally reported 3. Used only to set a generous,
# justified ceiling below -- not asserted as an exact target.
PREVIOUSLY_REPRODUCED_BASELINE_ONLY_CONVERGED = 10

# FINDINGS.md's "amplitude-response map" table: peak amplification across
# the amplitude grid, keyed by (ic_seed, node_label, sign).
REPORTED_PEAK_AMPLIFICATION = {
    (2000, "low", 1): [11.9, 11.9, 11.9, 12.1, 12.3, 7.9],
    (2000, "low", -1): [11.9, 11.9, 11.9, 11.7, 10.1, 7.9],
    (2000, "median", 1): [242.9, 162.8, 86.3, 34.5, 19.0, 15.8],
    (2000, "median", -1): [865.7, 2860.4, 1025.1, 266.8, 70.6, 18.1],
    (2000, "high", 1): [1.0, 1.0, 1.0, 1.0, 1.1, 1.3],
    (2000, "high", -1): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    (2001, "low", 1): [1.0, 1.0, 1.0, 1.0, 1.0, 6.6],
    (2001, "low", -1): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    (2001, "median", 1): [7.8, 7.6, 7.2, 6.5, 5.3, 3.5],
    (2001, "median", -1): [8.2, 8.4, 8.8, 9.6, 11.0, 20.6],
    (2001, "high", 1): [7.9, 7.6, 7.1, 6.1, 4.6, 2.8],
    (2001, "high", -1): [8.5, 8.9, 9.7, 11.3, 15.5, 36.7],
}


@pytest.fixture(scope="module")
def kmnist_class0_topology():
    # No stage-1b-specific topology cache exists in this checkout (the
    # original stage1a_all_classes.pkl doesn't exist here); this is the same
    # KMNIST class-0 T construction, structurally validated against
    # run_stage1b2.py's identical access pattern.
    if not CONSTRUCTIONS_PATH.exists():
        pytest.skip(f"KMNIST class-0 topology cache not found at {CONSTRUCTIONS_PATH}")
    with open(CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    return data["constructions"]["T"], data["n_active"]


@pytest.fixture(scope="module")
def stage1b_pilot_results(kmnist_class0_topology):
    """Runs the full 72-trial pilot grid (~13s/trial, ~15-20 min total)."""
    W, n = kmnist_class0_topology
    T_degree = W.sum(axis=1)
    order = np.argsort(T_degree)
    nodes = {
        "low": int(order[len(order) // 10]),
        "median": int(order[len(order) // 2]),
        "high": int(order[-len(order) // 10]),
    }

    results = {}
    for ic_seed in IC_SEEDS:
        theta0 = np.random.default_rng(ic_seed).uniform(0, 2 * np.pi, n)
        for node_label, node in nodes.items():
            for sign in SIGNS:
                for eps in AMPLITUDES:
                    key = (ic_seed, node_label, sign, eps)
                    results[key] = classify_one_trial(W, theta0, node, epsilon=sign * eps)
    return results


@pytest.mark.slow
def test_peak_amplification_matches_findings_table(stage1b_pilot_results):
    """Every cell of FINDINGS.md's 'amplitude-response map' table (the
    continuous dynamics) -- previously confirmed to reproduce to the
    reported decimal; toleranced here for FINDINGS.md's own 1-decimal
    rounding, not for genuine slack."""
    for (ic_seed, node_label, sign), reported_row in REPORTED_PEAK_AMPLIFICATION.items():
        for eps, reported in zip(AMPLITUDES, reported_row):
            key = (ic_seed, node_label, sign, eps)
            actual = stage1b_pilot_results[key]["peak_amplification"]
            assert actual == pytest.approx(reported, rel=0.02, abs=0.06), (
                f"{key}: got {actual}, FINDINGS.md reports {reported}"
            )


@pytest.mark.slow
def test_no_trial_reaches_a_distinct_equilibrium(stage1b_pilot_results):
    """FINDINGS.md: 'No trial produced two recovered but distinct
    phase-locked equilibria' -- 0 different_equilibria, identical in both
    the original run and this project's independent reproduction."""
    outcomes = [r["outcome"] for r in stage1b_pilot_results.values()]
    assert outcomes.count("different_equilibria") == 0


@pytest.mark.slow
def test_no_trial_fails_to_converge_on_both_sides(stage1b_pilot_results):
    """FINDINGS.md: 'No trial had both trajectories fail to converge' --
    0 no_equilibrium_recovered_within_horizon, also identical across both
    runs."""
    outcomes = [r["outcome"] for r in stage1b_pilot_results.values()]
    assert outcomes.count("no_equilibrium_recovered_within_horizon") == 0


@pytest.mark.slow
def test_baseline_only_converged_within_documented_fragile_range(stage1b_pilot_results):
    """Deliberately loose, unlike the two tests above: this outcome is
    environment-sensitive at the FORCE_CONVERGED_THRESHOLD boundary (see
    FINDINGS.md's 'Reproducibility note'). Asserts the count stays small
    relative to the 72-trial total -- generously above both FINDINGS.md's
    reported 3 and this project's previously-observed 10, not tuned to
    either -- and that every such trial is explainable by the documented
    boundary mechanism specifically, not some unrelated failure mode."""
    baseline_only = {
        key: r for key, r in stage1b_pilot_results.items()
        if r["outcome"] == "baseline_only_converged"
    }

    ceiling = 2 * PREVIOUSLY_REPRODUCED_BASELINE_ONLY_CONVERGED  # 20 of 72
    assert len(baseline_only) <= ceiling, (
        f"{len(baseline_only)} baseline_only_converged trials -- unexpectedly "
        f"large relative to FINDINGS.md's 3 and this project's previously "
        f"reproduced 10; may indicate a genuine regression, not the known "
        f"threshold artifact"
    )

    for key, r in baseline_only.items():
        ratio = r["pert_force_norm"] / FORCE_CONVERGED_THRESHOLD
        assert ratio < 2.0, (
            f"{key}: baseline_only_converged with pert_force_norm/threshold="
            f"{ratio:.2f}x -- outside the documented boundary-fragility range "
            f"(1.0-1.6x observed previously), may be a genuine new discrepancy"
        )
