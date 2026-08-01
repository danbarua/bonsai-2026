"""
Stage 1C analysis: runs Stage 1B2's exact permutation-test machinery
(load_results_as_arrays, run_permutation_test, imported directly from
analyze_stage1b2.py, not copied) independently per baseline trajectory,
then aggregates Delta_map/p_MC across all trajectories to characterize
generalization.

seed=3000 is the one exception: rather than re-running it, its trial
results are read (read-only) directly from Stage 1B2's own already-
committed results/stage1b2_results.pkl -- keeping Stage 1B2 genuinely
frozen as the reference and avoiding a redundant ~3-4 minute re-run. Its
432 trials use the identical (t_p, replica, node_label, sign, amplitude)
key format run_stage1c.py's own checkpoints use, so no transformation is
needed. All other seeds (3010-3090) are Stage 1C's own freshly-run,
independently-cached trajectories.

Usage: python3 analyze_stage1c.py [seed1 [seed2 ...]]
       (defaults to all 10 trajectories in run_stage1c.py's ALL_BASELINE_SEEDS
       -- unlike run_stage1c.py itself, whose own default excludes seed=3000;
       this script is the one place that legitimately reads all 10, since it
       never re-simulates the seed=3000 leg)
"""
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "stage1b2_structured_transformation"))
from stage1b2_core import get_degree_stratified_nodes
from analyze_stage1b2 import load_results_as_arrays, run_permutation_test

from run_stage1c import ALL_BASELINE_SEEDS, checkpoint_path, CLASS0_CONSTRUCTIONS_PATH

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
FINAL_ANALYSIS_PATH = os.path.join(RESULTS_DIR, "stage1c_final_analysis.pkl")

STAGE1B2_REFERENCE_SEED = 3000
STAGE1B2_RESULTS_PATH = os.path.join(
    _THIS_DIR, "..", "stage1b2_structured_transformation", "results", "stage1b2_results.pkl")


def analyze_one_trajectory(baseline_seed, W, nodes):
    if baseline_seed == STAGE1B2_REFERENCE_SEED:
        path = STAGE1B2_RESULTS_PATH
        source = "Stage 1B2's own committed results (read-only, not re-run)"
    else:
        path = checkpoint_path(baseline_seed)
        source = "Stage 1C's own freshly-run results"
    with open(path, "rb") as f:
        results = pickle.load(f)
    print(f"\n{'='*60}\nTrajectory seed={baseline_seed}: {len(results)} trials loaded "
          f"(expect 432) from {source}\n{'='*60}")
    organized = load_results_as_arrays(results, nodes, time_key="event_aligned_q")
    return run_permutation_test(organized, nodes)


def main(seeds):
    with open(CLASS0_CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    W = data["constructions"]["T"]
    nodes = get_degree_stratified_nodes(W)

    per_trajectory = {}
    for baseline_seed in seeds:
        per_trajectory[baseline_seed] = analyze_one_trajectory(baseline_seed, W, nodes)

    with open(FINAL_ANALYSIS_PATH, "wb") as f:
        pickle.dump(per_trajectory, f)
    print(f"\nSaved per-trajectory analysis for {len(per_trajectory)} trajectories to {FINAL_ANALYSIS_PATH}")

    print(f"\n{'='*60}\nSummary across {len(per_trajectory)} trajectories\n{'='*60}")
    for seed, result in per_trajectory.items():
        print(f"seed={seed}: pooled Delta_map={result['pooled_delta_map']:.4f}, p_MC={result['p_value']:.5f}")

    return per_trajectory


if __name__ == "__main__":
    seeds_arg = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else ALL_BASELINE_SEEDS
    main(seeds_arg)
