"""
Stage 1C driver: does Stage 1B2's structured-internal-transformation
result generalize across independent baseline trajectories (different
seeds), not just the single seed=3000 trajectory Stage 1B2 established it
on?

Imports directly from stage1b2_structured_transformation/stage1b2_core.py
(get_degree_stratified_nodes, generate_reference_baseline,
generate_fixed_replica_directions, run_one_trial) rather than copying it --
same design (Option A: 3 nodes x 2 signs x 3 amplitudes x 4 t_p x 6
replicas = 432 trials), applied to 10 independent baseline trajectories:
seed=3000 (Stage 1B2's own reference, re-run here to give Stage 1C its own
self-contained, independent results cache -- NOT reusing Stage 1B2's
cached trial results) plus 9 new seeds (3010, 3020, ..., 3090), each
paired with a replica-direction seed of baseline+1 (matching Stage 1B2's
BASELINE_SEED=3000 / REPLICA_DIRECTION_SEED=3001 offset convention).

Does not read or write anything under
experiments/stage1b2_structured_transformation/ except a read-only load
of its results/class0_constructions.pkl (the KMNIST class-0 T topology --
same graph, since Stage 1C tests trajectory generalization on the SAME
topology, not a new one).

Checkpointed per trajectory (results/stage1c_results_seed<N>.pkl), so
individual trajectories can be timed, resumed, or re-run independently.

Usage: python3 run_stage1c.py <seed1> [<seed2> ...]
       (each argument a baseline seed, e.g. `python3 run_stage1c.py 3010`)
"""
import sys
import os
import pickle
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "stage1b2_structured_transformation"))
from stage1b2_core import (
    T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, NEARBY_SCALE,
    get_degree_stratified_nodes, generate_reference_baseline,
    generate_fixed_replica_directions, run_one_trial,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS0_CONSTRUCTIONS_PATH = os.path.join(
    _THIS_DIR, "..", "stage1b2_structured_transformation", "results", "class0_constructions.pkl")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
LOG_FILE = os.path.join(_THIS_DIR, "stage1c_progress.log")

ALL_BASELINE_SEEDS = [3000, 3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080, 3090]


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")
    print(msg)


def checkpoint_path(baseline_seed):
    return os.path.join(RESULTS_DIR, f"stage1c_results_seed{baseline_seed}.pkl")


def build_all_trial_specs(W, n, baseline_seed, replica_direction_seed):
    nodes = get_degree_stratified_nodes(W)
    ref_sol = generate_reference_baseline(W, baseline_seed, max(T_P_VALUES) + 2.5)
    replica_directions = generate_fixed_replica_directions(n, replica_direction_seed, N_REPLICAS)

    specs = []
    for t_p in T_P_VALUES:
        state_at_tp = ref_sol.sol(t_p)
        for r_idx, direction in enumerate(replica_directions):
            replica_state = (state_at_tp + NEARBY_SCALE * direction) % (2 * np.pi)
            for node_label, node in nodes.items():
                for sign in SIGNS:
                    for amplitude in AMPLITUDES:
                        specs.append({
                            "t_p": t_p, "replica": r_idx, "node_label": node_label,
                            "node": node, "sign": sign, "amplitude": amplitude,
                            "replica_state": replica_state,
                        })
    return specs, nodes


def _worker(args):
    W, spec = args
    key = (spec["t_p"], spec["replica"], spec["node_label"], spec["sign"], spec["amplitude"])
    try:
        result = run_one_trial(W, spec["replica_state"], spec["node"], spec["sign"], spec["amplitude"])
        return key, result, None
    except Exception as e:
        return key, None, str(e)


def run_trajectory(W, n, baseline_seed):
    replica_direction_seed = baseline_seed + 1
    out_path = checkpoint_path(baseline_seed)

    specs, nodes = build_all_trial_specs(W, n, baseline_seed, replica_direction_seed)
    log(f"[seed={baseline_seed}] built {len(specs)} trial specs, nodes={nodes}")

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        log(f"[seed={baseline_seed}] resuming: {len(results)} trials already done")
    else:
        results = {}

    remaining = [s for s in specs
                 if (s["t_p"], s["replica"], s["node_label"], s["sign"], s["amplitude"]) not in results]
    log(f"[seed={baseline_seed}] {len(remaining)} trials remaining")

    if not remaining:
        return results

    n_workers = max(1, mp.cpu_count() - 1)
    work_items = [(W, s) for s in remaining]
    t0 = time.time()
    completed = 0
    with mp.Pool(n_workers) as pool:
        for key, result, error in pool.imap_unordered(_worker, work_items):
            if error is not None:
                log(f"[seed={baseline_seed}] ERROR on {key}: {error}")
            else:
                results[key] = result
                completed += 1
                if completed % 50 == 0:
                    with open(out_path, "wb") as f:
                        pickle.dump(results, f)
                    elapsed = time.time() - t0
                    log(f"[seed={baseline_seed}] {len(results)}/{len(specs)} done, {elapsed:.1f}s elapsed")

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    elapsed = time.time() - t0
    log(f"[seed={baseline_seed}] COMPLETE: {len(results)}/{len(specs)} trials in {elapsed:.1f}s "
        f"({elapsed/60:.2f} min)")
    return results


def main(seeds):
    with open(CLASS0_CONSTRUCTIONS_PATH, "rb") as f:
        data = pickle.load(f)[0]
    W = data["constructions"]["T"]
    n = data["n_active"]

    for baseline_seed in seeds:
        t0 = time.time()
        run_trajectory(W, n, baseline_seed)
        log(f"=== trajectory seed={baseline_seed} total wall time: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    seeds_arg = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else ALL_BASELINE_SEEDS
    main(seeds_arg)
