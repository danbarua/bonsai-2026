"""
Stage 1A re-verification driver, per DESIGN.md.

Computes AUC (perturbation-persistence, joint_tangent_matrix_response)
for: T and lattice (deterministic, one instance per class) and rewired /
hist_random / curr_random (25 seeds each per class) -- 10 * (2 + 3*25) =
770 instances total. Embarrassingly parallel across (class, construction,
seed) triples; checkpointed incrementally so progress survives
interruption.

Timing check performed before committing to the full run (per DESIGN.md's
"Computational scope" section): a single worst-case instance (class 9,
n_active=616, T construction) measured at ~2.0s serial. 770 instances at
that upper bound is ~1540s serial; with ~9 parallel workers on this
machine's 10 cores, comfortably a few minutes -- well within budget, no
staged commitment needed.
"""
import os
import pickle
import time
import multiprocessing as mp

import numpy as np

from reverification_core import (
    ALL_CLASSES_PATH, RESULTS_PATH, STOCHASTIC_CONTROLS, N_SEEDS,
    load_cached_deterministic_constructions, build_class_setup,
    build_stochastic_construction, compute_construction_auc,
)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1a_reverification_progress.log")

_WORKER_STATE = {}


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.ctime()}: {msg}\n")
    print(msg)


def _init_worker(class_data):
    global _WORKER_STATE
    _WORKER_STATE = class_data


def _compute_one(task):
    class_idx, construction, seed = task
    cd = _WORKER_STATE[class_idx]
    try:
        if construction == "T":
            W = cd["W_T"]
        elif construction == "lattice":
            W = cd["lattice"]
        else:
            W = build_stochastic_construction(
                construction, cd["W_T"], cd["ink_mask_active"], seed, cd["C"])
        auc = compute_construction_auc(W, cd["nodes"], cd["theta0"])
        return (class_idx, construction, seed), auc, None
    except Exception as e:
        return (class_idx, construction, seed), None, str(e)


def build_task_list():
    tasks = []
    for class_idx in range(10):
        tasks.append((class_idx, "T", None))
        tasks.append((class_idx, "lattice", None))
        for construction in STOCHASTIC_CONTROLS:
            for seed in range(N_SEEDS):
                tasks.append((class_idx, construction, seed))
    return tasks


def main():
    if not os.path.exists(ALL_CLASSES_PATH):
        raise FileNotFoundError(
            f"Prerequisite missing: {ALL_CLASSES_PATH} (build via "
            f"experiments/stage0_simulator_calibration/build_all_class_topologies.py)")

    all_classes_data = load_cached_deterministic_constructions()

    log("Building per-class setup (active indices, ink masks, node selection, theta0)...")
    class_data = {}
    for class_idx in range(10):
        t0 = time.time()
        setup = build_class_setup(class_idx, all_classes_data[class_idx]["constructions"]["T"])
        setup["lattice"] = all_classes_data[class_idx]["constructions"]["lattice"]
        class_data[class_idx] = setup
        log(f"  class {class_idx}: n_active={len(setup['theta0'])}, C={setup['C']:.4f}, "
            f"nodes={setup['nodes']}, {time.time() - t0:.2f}s")

    tasks = build_task_list()
    log(f"Built {len(tasks)} tasks (expect 770)")
    assert len(tasks) == 770

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        log(f"Resuming: {len(results)} instances already done")
    else:
        results = {}

    remaining = [t for t in tasks if t not in results]
    log(f"{len(remaining)} instances remaining")

    if not remaining:
        log("Nothing to do.")
        return

    n_workers = max(1, mp.cpu_count() - 1)
    log(f"Using {n_workers} worker processes (detected {mp.cpu_count()} cores)")

    t0 = time.time()
    completed = 0
    errors = 0
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(class_data,)) as pool:
        for key, auc, error in pool.imap_unordered(_compute_one, remaining):
            if error is not None:
                log(f"ERROR on {key}: {error}")
                errors += 1
            else:
                results[key] = auc
                completed += 1
                if completed % 25 == 0:
                    with open(RESULTS_PATH, "wb") as f:
                        pickle.dump(results, f)
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(remaining) - completed) / rate if rate > 0 else float("inf")
                    log(f"{len(results)}/770 done, {rate:.2f} instances/s, ETA {eta / 60:.1f} min")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    log(f"COMPLETE: {len(results)}/770 instances finished, {errors} errors")


if __name__ == "__main__":
    main()
