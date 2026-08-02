"""
Stage 1D driver. Two independent legs, per DESIGN.md:

1. LATTICE (confirmatory, complete): the lattice construction is
   deterministic (no realization dimension), so it needs no piloting --
   run the full Stage 1B2/1C 432-trial design on it for the same 10
   matched trajectory seeds Stage 1C used for T (3000, 3010, ..., 3090),
   under fixed-coordinate intervention (T's own node indices, via
   nodes_T -- NOT lattice's own degree-stratified nodes, per DESIGN.md's
   "fixed graph coordinates" protocol).

2. PILOT (explicitly non-confirmatory): 3 graph realizations (seeds
   0, 1, 2) x the first 3 of Stage 1C's matched trajectory seeds
   (3000, 3010, 3020), for each of the three stochastic controls
   (rewired, hist_random, curr_random), fixed-coordinate intervention
   only. Its only purpose is estimating the crossed variance components
   needed to size the real confirmatory run -- see DESIGN.md's "Pilot
   vs. confirmatory" section and analyze_stage1d.py.

Imports run_one_trial / generate_reference_baseline /
generate_fixed_replica_directions directly from stage1b2_core.py, exactly
as run_stage1c.py does. build_all_trial_specs below differs from
run_stage1c.py's own version in exactly one way: it takes `nodes` as an
explicit argument instead of recomputing get_degree_stratified_nodes(W)
internally -- necessary because every construction here (lattice and all
three pilot families) is intervened on at T's own node indices, not its
own.

Checkpointed per (construction, trajectory seed), so individual
trajectories can be timed, resumed, or re-run independently, matching
Stage 1C's convention.

Usage:
    python3 run_stage1d.py time_one       # times ONE trajectory (lattice,
                                           # seed=3000) before committing
                                           # to the rest -- run this first.
    python3 run_stage1d.py lattice        # runs all 10 lattice trajectories
    python3 run_stage1d.py pilot          # runs all 27 pilot trajectories
    python3 run_stage1d.py all            # time_one, then lattice, then pilot
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
    generate_reference_baseline, generate_fixed_replica_directions, run_one_trial,
)

from build_stage1d_constructions import build_all, PILOT_REALIZATION_SEEDS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
LOG_FILE = os.path.join(_THIS_DIR, "stage1d_progress.log")

LATTICE_TRAJECTORY_SEEDS = [3000, 3010, 3020, 3030, 3040, 3050, 3060, 3070, 3080, 3090]
PILOT_TRAJECTORY_SEEDS = [3000, 3010, 3020]
PILOT_FAMILIES = ["rewired", "hist_random", "curr_random"]


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")
    print(msg)


def lattice_checkpoint_path(baseline_seed):
    return os.path.join(RESULTS_DIR, f"stage1d_lattice_results_seed{baseline_seed}.pkl")


def pilot_checkpoint_path(family, realization_seed, baseline_seed):
    return os.path.join(
        RESULTS_DIR, f"stage1d_pilot_{family}_r{realization_seed}_seed{baseline_seed}.pkl")


def build_all_trial_specs(W, n, baseline_seed, replica_direction_seed, nodes):
    """Identical to run_stage1c.py's build_all_trial_specs, except `nodes`
    is passed in explicitly rather than recomputed as
    get_degree_stratified_nodes(W) -- every Stage 1D construction is
    intervened on at T's own node indices (fixed-coordinate protocol),
    not its own."""
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
    return specs


def _worker(args):
    W, spec = args
    key = (spec["t_p"], spec["replica"], spec["node_label"], spec["sign"], spec["amplitude"])
    try:
        result = run_one_trial(W, spec["replica_state"], spec["node"], spec["sign"], spec["amplitude"])
        return key, result, None
    except Exception as e:
        return key, None, str(e)


def run_trajectory(W, n, baseline_seed, nodes, out_path, label):
    replica_direction_seed = baseline_seed + 1
    specs = build_all_trial_specs(W, n, baseline_seed, replica_direction_seed, nodes)
    log(f"[{label}] built {len(specs)} trial specs, nodes={nodes}")

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            results = pickle.load(f)
        log(f"[{label}] resuming: {len(results)} trials already done")
    else:
        results = {}

    remaining = [s for s in specs
                 if (s["t_p"], s["replica"], s["node_label"], s["sign"], s["amplitude"]) not in results]
    log(f"[{label}] {len(remaining)} trials remaining")

    if not remaining:
        return results, 0.0

    n_workers = max(1, mp.cpu_count() - 1)
    work_items = [(W, s) for s in remaining]
    t0 = time.time()
    completed = 0
    with mp.Pool(n_workers) as pool:
        for key, result, error in pool.imap_unordered(_worker, work_items):
            if error is not None:
                log(f"[{label}] ERROR on {key}: {error}")
            else:
                results[key] = result
                completed += 1
                if completed % 50 == 0:
                    with open(out_path, "wb") as f:
                        pickle.dump(results, f)
                    elapsed = time.time() - t0
                    log(f"[{label}] {len(results)}/{len(specs)} done, {elapsed:.1f}s elapsed")

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    elapsed = time.time() - t0
    log(f"[{label}] COMPLETE: {len(results)}/{len(specs)} trials in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return results, elapsed


def run_time_one(data):
    """Times ONE trajectory (lattice, seed=3000) before committing to the
    full 37, per this project's standing discipline (see CLAUDE.md /
    PROJECT_MEMORY.md)."""
    W_lattice, n, nodes_T = data["W_lattice"], data["n_active"], data["nodes_T"]
    out_path = lattice_checkpoint_path(3000)
    if os.path.exists(out_path):
        os.remove(out_path)  # force a clean timing run, not a resume
    _results, elapsed = run_trajectory(W_lattice, n, 3000, nodes_T, out_path, "TIMING: lattice seed=3000")
    log(f"=== TIMING RESULT: one trajectory took {elapsed:.1f}s ({elapsed/60:.2f} min). "
        f"Estimated total for 37 trajectories: {37*elapsed/60:.1f} min "
        f"(sequential; this repo runs trajectories one at a time, each internally "
        f"parallelized across {max(1, mp.cpu_count()-1)} workers) ===")
    return elapsed


def run_lattice(data, seeds=None):
    W_lattice, n, nodes_T = data["W_lattice"], data["n_active"], data["nodes_T"]
    seeds = seeds if seeds is not None else LATTICE_TRAJECTORY_SEEDS
    for baseline_seed in seeds:
        t0 = time.time()
        out_path = lattice_checkpoint_path(baseline_seed)
        run_trajectory(W_lattice, n, baseline_seed, nodes_T, out_path, f"lattice seed={baseline_seed}")
        log(f"=== lattice trajectory seed={baseline_seed} total wall time: {time.time()-t0:.1f}s ===")


def run_pilot(data, families=None, realization_seeds=None, trajectory_seeds=None):
    nodes_T = data["nodes_T"]
    n = data["n_active"]
    families = families if families is not None else PILOT_FAMILIES
    realization_seeds = realization_seeds if realization_seeds is not None else PILOT_REALIZATION_SEEDS
    trajectory_seeds = trajectory_seeds if trajectory_seeds is not None else PILOT_TRAJECTORY_SEEDS

    for family in families:
        for r_seed in realization_seeds:
            W_g_r = data["pilot_constructions"][family][r_seed]
            for baseline_seed in trajectory_seeds:
                t0 = time.time()
                out_path = pilot_checkpoint_path(family, r_seed, baseline_seed)
                label = f"pilot {family} r={r_seed} seed={baseline_seed}"
                run_trajectory(W_g_r, n, baseline_seed, nodes_T, out_path, label)
                log(f"=== {label} total wall time: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = build_all()

    if mode == "time_one":
        run_time_one(data)
    elif mode == "lattice":
        run_lattice(data)
    elif mode == "pilot":
        run_pilot(data)
    elif mode == "all":
        run_time_one(data)
        run_lattice(data)
        run_pilot(data)
    else:
        raise ValueError(f"unknown mode: {mode}")
