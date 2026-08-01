"""
Tests for experiments/stage1c_trajectory_generalization/ (run_stage1c.py,
analyze_stage1c.py).

Two tiers, same convention as the rest of tests/:
1. Self-contained structural tests -- no external data required, always
   run. Covers the seed-list invariants (seed=3000 excluded from this
   script's own default and run-guard, included in analysis), that
   run_stage1c.py genuinely imports (not copies) Stage 1B2's core trial
   functions, and that checkpoint paths for Stage 1C's own trajectories
   can never collide with Stage 1B2's frozen results path.
2. An optional check against the real committed run artifacts -- skipped
   if not present locally (the full 10-trajectory run is not reproduced
   by a fresh checkout by default; see docs/PROJECT_MEMORY.md Part 4 on
   cold-clone reproducibility).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STAGE1B2_DIR = _REPO_ROOT / "experiments" / "stage1b2_structured_transformation"
_STAGE1C_DIR = _REPO_ROOT / "experiments" / "stage1c_trajectory_generalization"
sys.path.insert(0, str(_STAGE1B2_DIR))
sys.path.insert(0, str(_STAGE1C_DIR))

import stage1b2_core  # noqa: E402
import run_stage1c  # noqa: E402
import analyze_stage1c  # noqa: E402


# ---- Tier 1: self-contained structural tests ----

def test_new_baseline_seeds_excludes_3000_and_has_nine_entries():
    assert len(run_stage1c.NEW_BASELINE_SEEDS) == 9
    assert run_stage1c.STAGE1B2_REFERENCE_SEED not in run_stage1c.NEW_BASELINE_SEEDS
    assert run_stage1c.STAGE1B2_REFERENCE_SEED == 3000


def test_all_baseline_seeds_is_3000_plus_the_nine_new_ones():
    assert run_stage1c.ALL_BASELINE_SEEDS == [3000] + run_stage1c.NEW_BASELINE_SEEDS
    assert len(run_stage1c.ALL_BASELINE_SEEDS) == 10


def test_run_trajectory_refuses_to_regenerate_seed_3000():
    with pytest.raises(ValueError, match="frozen reference"):
        run_stage1c.run_trajectory(None, None, run_stage1c.STAGE1B2_REFERENCE_SEED)


def test_checkpoint_paths_never_collide_with_stage1b2s_own_results():
    stage1b2_results_path = str((_STAGE1B2_DIR / "results" / "stage1b2_results.pkl").resolve())
    for seed in run_stage1c.ALL_BASELINE_SEEDS:
        path = str(Path(run_stage1c.checkpoint_path(seed)).resolve())
        assert path != stage1b2_results_path
        assert "stage1c_trajectory_generalization" in path
        assert f"seed{seed}" in path


def test_analyze_stage1c_routes_seed_3000_to_stage1b2s_own_path_only():
    assert analyze_stage1c.STAGE1B2_REFERENCE_SEED == 3000
    resolved_ref_path = str(Path(analyze_stage1c.STAGE1B2_RESULTS_PATH).resolve())
    assert resolved_ref_path == str((_STAGE1B2_DIR / "results" / "stage1b2_results.pkl").resolve())
    # And it must differ from Stage 1C's own would-be checkpoint for the same seed number.
    own_checkpoint = str(Path(run_stage1c.checkpoint_path(3000)).resolve())
    assert resolved_ref_path != own_checkpoint


def test_imported_stage1b2_functions_are_the_actual_shared_implementations():
    """Confirms run_stage1c.py genuinely imports (not copies) Stage 1B2's
    core trial machinery -- same function objects, not look-alike
    reimplementations that could silently drift apart."""
    assert run_stage1c.run_one_trial is stage1b2_core.run_one_trial
    assert run_stage1c.get_degree_stratified_nodes is stage1b2_core.get_degree_stratified_nodes
    assert run_stage1c.generate_reference_baseline is stage1b2_core.generate_reference_baseline
    assert run_stage1c.generate_fixed_replica_directions is stage1b2_core.generate_fixed_replica_directions


def test_build_all_trial_specs_produces_432_specs_on_synthetic_topology():
    rng = np.random.default_rng(0)
    n = 30
    W = np.zeros((n, n))
    triu_i, triu_j = np.triu_indices(n, k=1)
    chosen = rng.choice(len(triu_i), size=60, replace=False)
    vals = rng.uniform(0.1, 1.0, size=60)
    W[triu_i[chosen], triu_j[chosen]] = vals
    W[triu_j[chosen], triu_i[chosen]] = vals

    specs, nodes = run_stage1c.build_all_trial_specs(W, n, baseline_seed=1, replica_direction_seed=2)
    # 4 t_p x 6 replicas x 3 nodes x 2 signs x 3 amplitudes
    assert len(specs) == 4 * 6 * 3 * 2 * 3 == 432
    assert set(nodes.keys()) == {"low", "median", "high"}


# ---- Tier 2: optional check against the real committed run ----

_RESULTS_DIR = _STAGE1C_DIR / "results"
_STAGE1B2_RESULTS = _STAGE1B2_DIR / "results" / "stage1b2_results.pkl"


def _new_trajectory_files_present():
    return all((_RESULTS_DIR / f"stage1c_results_seed{s}.pkl").exists()
               for s in run_stage1c.NEW_BASELINE_SEEDS)


@pytest.mark.skipif(not (_new_trajectory_files_present() and _STAGE1B2_RESULTS.exists()),
                     reason="Stage 1C trajectory results and/or Stage 1B2's frozen results "
                            "not present locally (full run not reproduced by default)")
def test_all_ten_trajectories_have_432_trials():
    for seed in run_stage1c.NEW_BASELINE_SEEDS:
        with open(_RESULTS_DIR / f"stage1c_results_seed{seed}.pkl", "rb") as f:
            results = pickle.load(f)
        assert len(results) == 432, seed

    with open(_STAGE1B2_RESULTS, "rb") as f:
        stage1b2_results = pickle.load(f)
    assert len(stage1b2_results) == 432


@pytest.mark.skipif(not (_STAGE1C_DIR / "results" / "stage1c_final_analysis.pkl").exists(),
                     reason="stage1c_final_analysis.pkl not present locally (run analyze_stage1c.py first)")
def test_final_analysis_covers_exactly_the_ten_all_baseline_seeds():
    with open(_STAGE1C_DIR / "results" / "stage1c_final_analysis.pkl", "rb") as f:
        per_trajectory = pickle.load(f)
    assert set(per_trajectory.keys()) == set(run_stage1c.ALL_BASELINE_SEEDS)
    for seed, result in per_trajectory.items():
        assert 0.0 <= result["p_value"] <= 1.0
