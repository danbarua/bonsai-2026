"""
Stage 1B.2 main driver. Auto-detects available CPU cores and distributes
the 432 independent finite-response trials across a multiprocessing pool.
Checkpoints incrementally so progress survives interruption.

Usage: python3 run_stage1b2.py
Expected runtime on an M1 Max (10 cores): roughly 4.8 hours / 8-9 effective
parallel workers ~ 35-45 minutes, depending on thermal/efficiency-core
behavior. Adjust N_WORKERS below if you want to leave headroom for other
work on the machine.
"""
import numpy as np
import pickle
import time
import os
import multiprocessing as mp
from stage1b2_core import (
    T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, NEARBY_SCALE,
    get_degree_stratified_nodes, generate_reference_baseline,
    generate_fixed_replica_directions, run_one_trial, rotation_projector
)

CHECKPOINT_FILE = 'results/stage1b2_results.pkl'
LOG_FILE = 'stage1b2_progress.log'
BASELINE_SEED = 3000
REPLICA_DIRECTION_SEED = 3001


def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(f'{time.ctime()}: {msg}\n')
    print(msg)


def build_all_trial_specs(W, n):
    """Builds the full list of 432 trial specifications: for each t_p,
    each of 6 replicas, each of 18 inputs (3 nodes x 2 signs x 3 amps)."""
    nodes = get_degree_stratified_nodes(W)
    ref_sol = generate_reference_baseline(W, BASELINE_SEED, max(T_P_VALUES) + 2.5)
    replica_directions = generate_fixed_replica_directions(n, REPLICA_DIRECTION_SEED, N_REPLICAS)

    specs = []
    for t_p in T_P_VALUES:
        state_at_tp = ref_sol.sol(t_p)
        for r_idx, direction in enumerate(replica_directions):
            replica_state = (state_at_tp + NEARBY_SCALE * direction) % (2 * np.pi)
            for node_label, node in nodes.items():
                for sign in SIGNS:
                    for amplitude in AMPLITUDES:
                        specs.append({
                            't_p': t_p, 'replica': r_idx, 'node_label': node_label,
                            'node': node, 'sign': sign, 'amplitude': amplitude,
                            'replica_state': replica_state,
                        })
    return specs, nodes


def _worker(args):
    W, spec = args
    key = (spec['t_p'], spec['replica'], spec['node_label'], spec['sign'], spec['amplitude'])
    try:
        result = run_one_trial(W, spec['replica_state'], spec['node'], spec['sign'], spec['amplitude'])
        return key, result, None
    except Exception as e:
        return key, None, str(e)


def main():
    with open('results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W = data['constructions']['T']
    n = data['n_active']

    specs, nodes = build_all_trial_specs(W, n)
    log(f'Built {len(specs)} trial specifications. Nodes: {nodes}')

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            results = pickle.load(f)
        log(f'Resuming: {len(results)} trials already done')
    else:
        results = {}

    remaining = [s for s in specs if (s['t_p'], s['replica'], s['node_label'], s['sign'], s['amplitude']) not in results]
    log(f'{len(remaining)} trials remaining')

    n_workers = max(1, mp.cpu_count() - 1)  # leave one core free
    log(f'Using {n_workers} worker processes (detected {mp.cpu_count()} cores)')

    work_items = [(W, s) for s in remaining]
    t0 = time.time()
    completed = 0
    with mp.Pool(n_workers) as pool:
        for key, result, error in pool.imap_unordered(_worker, work_items):
            if error is not None:
                log(f'ERROR on {key}: {error}')
            else:
                results[key] = result
                completed += 1
                if completed % 10 == 0:
                    with open(CHECKPOINT_FILE, 'wb') as f:
                        pickle.dump(results, f)
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(remaining) - completed) / rate if rate > 0 else float('inf')
                    log(f'{len(results)}/{len(specs)} done, {rate:.2f} trials/s, ETA {eta/60:.1f} min')

    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(results, f)
    log(f'COMPLETE: {len(results)}/{len(specs)} trials finished')


if __name__ == '__main__':
    main()
