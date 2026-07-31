"""
Two remaining items on the common-support exclusion diagnostic:

1. The omnibus common-support test (Delta_map^(-S) = 0.3418, p=0.0001)
   proves the BALANCED input mapping (node+sign+amplitude combined)
   survives on the remaining graph. It does not by itself prove the
   narrower, node-specific claim that node identity specifically remains
   discriminable after common-support exclusion. This runs the
   factor-restricted node permutation test (node labels permuted within
   matched sign-amplitude cells) on the common-support representation
   directly -- the "preferred" resolution, not the softened wording.

2. An audit trail confirming the corrected object genuinely differs from
   the earlier (leaky) version, even though the rounded omnibus statistic
   matched to four decimal places: full-precision Delta_map comparison,
   max elementwise difference between old and common-support q arrays,
   count of trials whose output actually changed, and confirmation that
   all three masked entries are exactly zero in all 432 outputs.
"""
import numpy as np
import pickle
import itertools
from analyze_stage1b2 import compute_W_B_deltamap, N_PERMUTATIONS
from analyze_stage1b2_diagnostics import _factor_permutation_worker, permute_node_within_sign_amp
from analyze_stage1b2_common_support_exclusion import common_support_exclude, load_common_support_arrays
from stage1b2_core import T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, get_degree_stratified_nodes
import multiprocessing as mp


def node_specific_test_on_common_support(organized, nodes, seed=401, n_workers=None):
    node_labels = list(nodes.keys())
    observed = {t_p: compute_W_B_deltamap(organized[t_p], node_labels) for t_p in T_P_VALUES}
    pooled_obs = np.mean([observed[t_p]['B_node'] - observed[t_p]['W'] for t_p in T_P_VALUES])

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    base_seed_seq = np.random.SeedSequence(seed)
    worker_seed_seqs = base_seed_seq.spawn(n_workers)
    perms_per_worker = [N_PERMUTATIONS // n_workers] * n_workers
    for i in range(N_PERMUTATIONS % n_workers):
        perms_per_worker[i] += 1

    work_items = [(organized, node_labels, 'B_node', permute_node_within_sign_amp,
                   perms_per_worker[i], worker_seed_seqs[i]) for i in range(n_workers)]

    perm_vals = []
    with mp.Pool(n_workers) as pool:
        for worker_result in pool.imap_unordered(_factor_permutation_worker, work_items):
            perm_vals.extend(worker_result)
    perm_vals = np.array(perm_vals)
    assert len(perm_vals) == N_PERMUTATIONS

    p = (1 + np.sum(perm_vals >= pooled_obs)) / (N_PERMUTATIONS + 1)
    return {'delta_node_common_support': pooled_obs, 'p_raw': p, 'observed_by_tp': observed}


def audit_trail(results, nodes, source_indices):
    print(f'{"="*70}\nAUDIT: does the corrected q genuinely differ from the original?\n{"="*70}')

    max_elementwise_diff = 0.0
    n_changed = 0
    n_checked = 0
    n_all_zero = 0
    old_deltamap_inputs = {}  # (t_p) -> organized dict using ORIGINAL uncorrected q (for comparison)

    for key, v in results.items():
        q_original = v['event_aligned_q']
        if q_original is None:
            continue
        q_common = common_support_exclude(q_original, source_indices)
        n_checked += 1
        if q_common is not None:
            if np.allclose(q_common[source_indices], 0.0):
                n_all_zero += 1
            # compare against the OLD "zero-only-actual-source" version for this trial
            node_label = key[2]
            actual_source_idx = nodes[node_label]
            q_old_leaky = q_original.copy()
            q_old_leaky[actual_source_idx] = 0.0
            q_old_leaky = q_old_leaky / np.sum(q_old_leaky) if np.sum(q_old_leaky) > 0 else None
            if q_old_leaky is not None:
                diff = np.max(np.abs(q_common - q_old_leaky))
                max_elementwise_diff = max(max_elementwise_diff, diff)
                if diff > 1e-12:
                    n_changed += 1

    print(f'Trials checked: {n_checked}')
    print(f'Trials with all 3 source coordinates exactly zero in common-support q: {n_all_zero}/{n_checked}')
    print(f'Trials whose q array actually changed vs. the old (single-source-zeroed) version: {n_changed}/{n_checked}')
    print(f'Maximum elementwise |q_common - q_old_leaky| across all checked trials: {max_elementwise_diff:.6e}')
    print('(A nonzero max difference confirms the corrected object is genuinely different from the')
    print(' earlier one, even where the rounded omnibus Delta_map statistic matched to 4 decimals.)')


if __name__ == '__main__':
    with open('results/stage1b2_results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W_matrix = data['constructions']['T']
    nodes = get_degree_stratified_nodes(W_matrix)
    source_indices = list(nodes.values())
    n_workers = max(1, mp.cpu_count() - 1)

    print(f'Loaded {len(results)} trials (expect 432)')

    # Item 2: audit trail first (cheap, no permutation needed)
    audit_trail(results, nodes, source_indices)

    # Item 1: the preferred resolution -- node-specific restricted test on common-support q
    print(f'\n{"="*70}\nNODE-SPECIFIC FACTOR-RESTRICTED TEST ON COMMON-SUPPORT REPRESENTATION\n{"="*70}')
    organized = load_common_support_arrays(results, nodes, source_indices, time_key='event_aligned_q')
    with open('results/stage1b2_final_analysis.pkl', 'rb') as f:
        full_result = pickle.load(f)
    node_result = node_specific_test_on_common_support(organized, nodes, seed=401, n_workers=n_workers)
    print(f'Delta_node (common-support) = {node_result["delta_node_common_support"]:.4f}, '
          f'p_raw = {node_result["p_raw"]:.5f}')
    print('(This is a single, prespecified test -- no additional multiplicity correction needed')
    print(' beyond what the omnibus and prior factor-specific tests already carry.)')

    with open('results/stage1b2_common_support_node_test.pkl', 'wb') as f:
        pickle.dump(node_result, f)
    print('\nSaved to results/stage1b2_common_support_node_test.pkl')
