"""
Stage 1B.2 analysis: computes W, B_node/sign/amplitude/balanced, and
Delta_map from completed trial results, then runs the CORRECTED
permutation test (independent per-replica label permutation -- the
prior "common relabeling across replicas" scheme was degenerate and is
NOT used here).

Permutation loop is parallelized across worker processes, each with its
own independent random stream (via numpy's SeedSequence.spawn, not a
shared or repeated seed) -- correctness of the permutation logic itself
is unchanged from the validated single-threaded version; only the
distribution of independent permutation draws across cores is new.

Run only after run_stage1b2.py has completed all 432 trials.
"""
import numpy as np
import pickle
import itertools
import multiprocessing as mp
from scipy.spatial.distance import jensenshannon
from stage1b2_core import T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, get_degree_stratified_nodes

N_PERMUTATIONS = 10000


def d_q(qa, qb):
    """Output-map distance: sqrt(JSD), a true metric (unlike raw JSD)."""
    if qa is None or qb is None:
        return None
    return float(jensenshannon(qa, qb))


def load_results_as_arrays(results, nodes, time_key='event_aligned_q'):
    """Reorganizes the flat results dict into per-(t_p) arrays of
    (replica, node, sign, amplitude) -> q vector, for pairwise distance
    computation."""
    node_labels = list(nodes.keys())
    organized = {}
    for t_p in T_P_VALUES:
        organized[t_p] = {}
        for r in range(N_REPLICAS):
            for node_label in node_labels:
                for sign in SIGNS:
                    for amp in AMPLITUDES:
                        key = (t_p, r, node_label, sign, amp)
                        if key in results:
                            organized[t_p][(r, node_label, sign, amp)] = results[key][time_key]
    return organized


def compute_W_B_deltamap(organized_tp, node_labels):
    """For one perturbation-time neighborhood: compute W (same-input,
    cross-replica), B_node/sign/amplitude (matched-factor, cross-replica),
    balanced B, and Delta_map = B - W. Pairs are formed ACROSS replicas,
    never within the same replica, and self-pairs are excluded."""
    inputs = list(itertools.product(node_labels, SIGNS, AMPLITUDES))
    replicas = list(range(N_REPLICAS))

    # W: same input, different replicas
    W_dists = []
    for inp in inputs:
        node_label, sign, amp = inp
        for r1, r2 in itertools.combinations(replicas, 2):
            q1 = organized_tp.get((r1, node_label, sign, amp))
            q2 = organized_tp.get((r2, node_label, sign, amp))
            d = d_q(q1, q2)
            if d is not None:
                W_dists.append(d)
    W = np.mean(W_dists) if W_dists else np.nan

    # B_node: differ only in node, matched sign & amplitude, cross-replica
    B_node_dists = []
    for sign, amp in itertools.product(SIGNS, AMPLITUDES):
        for n1, n2 in itertools.combinations(node_labels, 2):
            for r1, r2 in itertools.product(replicas, replicas):
                if r1 == r2:
                    continue
                q1 = organized_tp.get((r1, n1, sign, amp))
                q2 = organized_tp.get((r2, n2, sign, amp))
                d = d_q(q1, q2)
                if d is not None:
                    B_node_dists.append(d)
    B_node = np.mean(B_node_dists) if B_node_dists else np.nan

    # B_sign: differ only in sign, matched node & amplitude, cross-replica
    B_sign_dists = []
    for node_label, amp in itertools.product(node_labels, AMPLITUDES):
        for r1, r2 in itertools.product(replicas, replicas):
            if r1 == r2:
                continue
            q1 = organized_tp.get((r1, node_label, 1, amp))
            q2 = organized_tp.get((r2, node_label, -1, amp))
            d = d_q(q1, q2)
            if d is not None:
                B_sign_dists.append(d)
    B_sign = np.mean(B_sign_dists) if B_sign_dists else np.nan

    # B_amplitude: differ only in amplitude, matched node & sign, cross-replica
    B_amp_dists = []
    for node_label, sign in itertools.product(node_labels, SIGNS):
        for a1, a2 in itertools.combinations(AMPLITUDES, 2):
            for r1, r2 in itertools.product(replicas, replicas):
                if r1 == r2:
                    continue
                q1 = organized_tp.get((r1, node_label, sign, a1))
                q2 = organized_tp.get((r2, node_label, sign, a2))
                d = d_q(q1, q2)
                if d is not None:
                    B_amp_dists.append(d)
    B_amplitude = np.mean(B_amp_dists) if B_amp_dists else np.nan

    B_balanced = np.mean([B_node, B_sign, B_amplitude])
    delta_map = B_balanced - W
    return {'W': W, 'B_node': B_node, 'B_sign': B_sign, 'B_amplitude': B_amplitude,
            'B_balanced': B_balanced, 'delta_map': delta_map}


def permute_within_replica(organized_tp, node_labels, rng):
    """CORRECTED permutation: independently within each replica, randomly
    permute the 18 input labels across its 18 outputs, preserving the
    one-to-one assignment inside that replica."""
    inputs = list(itertools.product(node_labels, SIGNS, AMPLITUDES))
    permuted = {}
    for r in range(N_REPLICAS):
        actual_outputs = [organized_tp.get((r, *inp)) for inp in inputs]
        perm = rng.permutation(len(inputs))
        for i, inp in enumerate(inputs):
            permuted[(r, *inp)] = actual_outputs[perm[i]]
    return permuted


def _permutation_worker(args):
    """Runs `n_perms` permutation draws using an independent random
    stream (seeded via SeedSequence.spawn -- NOT the same seed reused
    across workers, which would produce correlated/duplicate draws and
    invalidate the Monte Carlo independence assumption)."""
    organized, node_labels, n_perms, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    deltamaps = []
    for _ in range(n_perms):
        perm_results_by_tp = []
        for t_p in T_P_VALUES:
            permuted_tp = permute_within_replica(organized[t_p], node_labels, rng)
            perm_results_by_tp.append(compute_W_B_deltamap(permuted_tp, node_labels)['delta_map'])
        deltamaps.append(np.mean(perm_results_by_tp))
    return deltamaps


def run_permutation_test(organized, nodes, seed=42, n_workers=None):
    node_labels = list(nodes.keys())

    observed = {t_p: compute_W_B_deltamap(organized[t_p], node_labels) for t_p in T_P_VALUES}
    pooled_observed_deltamap = np.mean([observed[t_p]['delta_map'] for t_p in T_P_VALUES])

    print('Observed results by perturbation time:')
    for t_p in T_P_VALUES:
        o = observed[t_p]
        print(f'  t_p={t_p}: W={o["W"]:.4f}, B_balanced={o["B_balanced"]:.4f}, '
              f'Delta_map={o["delta_map"]:.4f} (B_node={o["B_node"]:.4f}, '
              f'B_sign={o["B_sign"]:.4f}, B_amp={o["B_amplitude"]:.4f})')
    print(f'Pooled Delta_map: {pooled_observed_deltamap:.4f}')

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    print(f'\nRunning {N_PERMUTATIONS} permutations across {n_workers} worker processes...')

    # Independent, non-overlapping random streams per worker -- critical for
    # Monte Carlo validity. A shared or repeated seed across workers would
    # make their permutation draws correlated or identical.
    base_seed_seq = np.random.SeedSequence(seed)
    worker_seed_seqs = base_seed_seq.spawn(n_workers)

    perms_per_worker = [N_PERMUTATIONS // n_workers] * n_workers
    for i in range(N_PERMUTATIONS % n_workers):
        perms_per_worker[i] += 1

    work_items = [(organized, node_labels, perms_per_worker[i], worker_seed_seqs[i])
                  for i in range(n_workers)]

    perm_deltamaps = []
    with mp.Pool(n_workers) as pool:
        for worker_result in pool.imap_unordered(_permutation_worker, work_items):
            perm_deltamaps.extend(worker_result)
    perm_deltamaps = np.array(perm_deltamaps)
    assert len(perm_deltamaps) == N_PERMUTATIONS, f'expected {N_PERMUTATIONS}, got {len(perm_deltamaps)}'

    p_value = (1 + np.sum(perm_deltamaps >= pooled_observed_deltamap)) / (N_PERMUTATIONS + 1)
    print(f'\nOne-sided Monte Carlo p-value (H0: Delta_map <= 0): p = {p_value:.5f}')
    print(f'(minimum attainable p with {N_PERMUTATIONS} permutations: {1/(N_PERMUTATIONS+1):.5f})')

    return {'observed_by_tp': observed, 'pooled_delta_map': pooled_observed_deltamap,
            'permutation_distribution': perm_deltamaps, 'p_value': p_value}


if __name__ == '__main__':
    with open('results/stage1b2_results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W_matrix = data['constructions']['T']
    nodes = get_degree_stratified_nodes(W_matrix)

    print(f'Loaded {len(results)} trials (expect 432)')
    organized = load_results_as_arrays(results, nodes, time_key='event_aligned_q')
    final = run_permutation_test(organized, nodes)

    with open('results/stage1b2_final_analysis.pkl', 'wb') as f:
        pickle.dump(final, f)
    print('\nSaved to results/stage1b2_final_analysis.pkl')
