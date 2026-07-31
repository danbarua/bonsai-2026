"""
Generic balanced within-vs-between permutation test for a
node x sign x amplitude x replica x perturbation-time experimental
design (originally built for Stage 1B.2, extracted here because the
same design pattern -- N inputs varying along independent factors,
each repeated across nearby-state replicas -- is expected to recur in
later stages of the dynamics-as-computation lineage).

Uses independent per-replica label permutation under the null (the
only scheme that actually destroys input identity while preserving each
replica's own output geometry -- an earlier "common relabeling across
replicas" scheme was found to be degenerate: it left the test statistic
literally unchanged under every permutation, and was caught via a
synthetic-data unit test before being used for inference).

Permutation loop is parallelized across worker processes, each with its
own independent random stream (via numpy's SeedSequence.spawn, not a
shared or repeated seed) -- a shared/repeated seed across workers
produces correlated or duplicate draws and silently invalidates the
Monte Carlo independence assumption. This has been an actual bug in this
project before (a single-threaded version was used by oversight in an
early diagnostic script); always verify a new use of this module keeps
`n_workers` independent seed sequences.

This module makes NO assumptions about what "node", "sign", or
"amplitude" mean physically -- callers pass in the actual label lists
for their design. It only assumes: three named factors, each with a
node_labels x signs x amplitudes list of inputs, each input observed
across n_replicas nearby-state replicas, at one or more perturbation
times (t_p_values) which are pooled at the end.
"""
import numpy as np
import itertools
import multiprocessing as mp
from scipy.spatial.distance import jensenshannon


def d_q(qa, qb):
    """Output-map distance: sqrt(JSD), a true metric (unlike raw JSD)."""
    if qa is None or qb is None:
        return None
    return float(jensenshannon(qa, qb))


def compute_W_B_deltamap(organized_tp, node_labels, signs, amplitudes, n_replicas):
    """For one perturbation-time neighborhood: compute W (same-input,
    cross-replica), B_node/sign/amplitude (matched-factor, cross-replica),
    balanced B, and Delta_map = B - W. Pairs are formed ACROSS replicas,
    never within the same replica, and self-pairs are excluded."""
    inputs = list(itertools.product(node_labels, signs, amplitudes))
    replicas = list(range(n_replicas))

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
    for sign, amp in itertools.product(signs, amplitudes):
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
    for node_label, amp in itertools.product(node_labels, amplitudes):
        for r1, r2 in itertools.product(replicas, replicas):
            if r1 == r2:
                continue
            q1 = organized_tp.get((r1, node_label, signs[0], amp))
            q2 = organized_tp.get((r2, node_label, signs[-1], amp))
            d = d_q(q1, q2)
            if d is not None:
                B_sign_dists.append(d)
    B_sign = np.mean(B_sign_dists) if B_sign_dists else np.nan

    # B_amplitude: differ only in amplitude, matched node & sign, cross-replica
    B_amp_dists = []
    for node_label, sign in itertools.product(node_labels, signs):
        for a1, a2 in itertools.combinations(amplitudes, 2):
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


def permute_within_replica(organized_tp, node_labels, signs, amplitudes, rng, n_replicas):
    """CORRECTED permutation: independently within each replica, randomly
    permute the input labels across its outputs, preserving the
    one-to-one assignment inside that replica."""
    inputs = list(itertools.product(node_labels, signs, amplitudes))
    permuted = {}
    for r in range(n_replicas):
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
    organized, node_labels, signs, amplitudes, t_p_values, n_replicas, n_perms, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    deltamaps = []
    for _ in range(n_perms):
        perm_results_by_tp = []
        for t_p in t_p_values:
            permuted_tp = permute_within_replica(organized[t_p], node_labels, signs, amplitudes, rng, n_replicas)
            perm_results_by_tp.append(
                compute_W_B_deltamap(permuted_tp, node_labels, signs, amplitudes, n_replicas)['delta_map'])
        deltamaps.append(np.mean(perm_results_by_tp))
    return deltamaps


def run_permutation_test(organized, node_labels, signs, amplitudes, t_p_values, n_replicas,
                          n_permutations=10000, seed=42, n_workers=None, verbose=True):
    """organized: dict of t_p -> {(replica, node_label, sign, amplitude): q_vector}.
    Returns dict with observed per-t_p results, pooled observed Delta_map,
    the permutation-draw array, and the one-sided Monte Carlo p-value."""
    observed = {t_p: compute_W_B_deltamap(organized[t_p], node_labels, signs, amplitudes, n_replicas)
                for t_p in t_p_values}
    pooled_observed_deltamap = np.mean([observed[t_p]['delta_map'] for t_p in t_p_values])

    if verbose:
        print('Observed results by perturbation time:')
        for t_p in t_p_values:
            o = observed[t_p]
            print(f'  t_p={t_p}: W={o["W"]:.4f}, B_balanced={o["B_balanced"]:.4f}, '
                  f'Delta_map={o["delta_map"]:.4f} (B_node={o["B_node"]:.4f}, '
                  f'B_sign={o["B_sign"]:.4f}, B_amp={o["B_amplitude"]:.4f})')
        print(f'Pooled Delta_map: {pooled_observed_deltamap:.4f}')

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if verbose:
        print(f'\nRunning {n_permutations} permutations across {n_workers} worker processes...')

    # Independent, non-overlapping random streams per worker -- critical for
    # Monte Carlo validity. A shared or repeated seed across workers would
    # make their permutation draws correlated or identical.
    base_seed_seq = np.random.SeedSequence(seed)
    worker_seed_seqs = base_seed_seq.spawn(n_workers)

    perms_per_worker = [n_permutations // n_workers] * n_workers
    for i in range(n_permutations % n_workers):
        perms_per_worker[i] += 1

    work_items = [(organized, node_labels, signs, amplitudes, t_p_values, n_replicas,
                   perms_per_worker[i], worker_seed_seqs[i]) for i in range(n_workers)]

    perm_deltamaps = []
    with mp.Pool(n_workers) as pool:
        for worker_result in pool.imap_unordered(_permutation_worker, work_items):
            perm_deltamaps.extend(worker_result)
    perm_deltamaps = np.array(perm_deltamaps)
    assert len(perm_deltamaps) == n_permutations, f'expected {n_permutations}, got {len(perm_deltamaps)}'

    p_value = (1 + np.sum(perm_deltamaps >= pooled_observed_deltamap)) / (n_permutations + 1)
    if verbose:
        print(f'\nOne-sided Monte Carlo p-value (H0: Delta_map <= 0): p = {p_value:.5f}')
        print(f'(minimum attainable p with {n_permutations} permutations: {1/(n_permutations+1):.5f})')

    return {'observed_by_tp': observed, 'pooled_delta_map': pooled_observed_deltamap,
            'permutation_deltamaps': perm_deltamaps, 'p_value': p_value}
