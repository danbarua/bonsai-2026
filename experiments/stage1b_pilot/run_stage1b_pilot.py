"""
Stage 1B pilot batch runner: 1 class, 2 initial conditions, 3 nodes,
2 signs, 6 amplitudes = 72 trials, T only (capability-first design).
Runs as a long-lived background process; checkpoints after every trial
so progress can be inspected at any time without waiting for completion.
"""
import numpy as np
import pickle
import time
import sys
import os

sys.path.insert(0, '/home/claude/oscillator_field')
os.chdir('/home/claude/oscillator_field')
from stage1b_taxonomy import classify_one_trial

with open('stage1a_all_classes.pkl', 'rb') as f:
    all_data = pickle.load(f)

CLASS = 0
data = all_data[CLASS]
n = data['n_active']
W = data['constructions']['T']
T_degree = W.sum(axis=1)
order = np.argsort(T_degree)
nodes = {'low': int(order[len(order)//10]), 'median': int(order[len(order)//2]),
         'high': int(order[-len(order)//10])}
ic_seeds = [2000, 2001]  # 2 initial conditions, distinct from Stage 1A's seeds
amplitudes = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
signs = [1, -1]

CHECKPOINT_FILE = 'stage1b_pilot_results.pkl'
LOG_FILE = 'stage1b_pilot_progress.log'

# Resume from checkpoint if it exists
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'rb') as f:
        results = pickle.load(f)
else:
    results = {}

total_trials = len(ic_seeds) * len(nodes) * len(signs) * len(amplitudes)
completed = len(results)

with open(LOG_FILE, 'a') as log:
    log.write(f'=== Starting/resuming run at {time.ctime()}, {completed}/{total_trials} already done ===\n')
    log.flush()

for ic_seed in ic_seeds:
    theta0 = np.random.default_rng(ic_seed).uniform(0, 2*np.pi, n)
    for node_label, node in nodes.items():
        for sign in signs:
            for eps in amplitudes:
                key = (ic_seed, node_label, sign, eps)
                if key in results:
                    continue  # already done, skip (resume support)
                t0 = time.time()
                try:
                    r = classify_one_trial(W, theta0, node, epsilon=sign*eps)
                    r['elapsed'] = time.time() - t0
                    results[key] = r
                    with open(CHECKPOINT_FILE, 'wb') as f:
                        pickle.dump(results, f)
                    with open(LOG_FILE, 'a') as log:
                        log.write(f'{time.ctime()}: {key} -> {r["outcome"]} '
                                  f'(peak_amp={r["peak_amplification"]:.2f}, {r["elapsed"]:.1f}s) '
                                  f'[{len(results)}/{total_trials}]\n')
                        log.flush()
                except Exception as e:
                    with open(LOG_FILE, 'a') as log:
                        log.write(f'{time.ctime()}: {key} -> ERROR: {e}\n')
                        log.flush()

with open(LOG_FILE, 'a') as log:
    log.write(f'=== Run complete at {time.ctime()}, {len(results)}/{total_trials} done ===\n')
