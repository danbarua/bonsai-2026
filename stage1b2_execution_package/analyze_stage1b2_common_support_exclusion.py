"""
Corrected source-exclusion diagnostic, per review: zeroing only the
ACTUAL stimulated coordinate (as the previous fix did) leaks node
identity through the POSITION of the zero itself -- a low-node trial has
its zero at index 17, a median-node trial at 363, a high-node trial at
129, and JSD can trivially distinguish these patterns without any
genuine propagated response. This is a deterministic post-processing
correction on the already-saved event_aligned_q vectors (no
re-integration needed): zero ALL THREE candidate source coordinates
{i_low, i_median, i_high} in EVERY trial, regardless of which one was
actually stimulated, then renormalize over the common remaining support.
Every trial then shares exactly the same exclusion mask, so the output
cannot reveal which node was stimulated merely through which coordinate
is missing.

Mathematically equivalent to zeroing the raw displacement and
renormalizing from scratch: q_j = disp_j^2 / sum(disp^2), so zeroing
q_j for j in S and renormalizing over the rest gives the identical
result to zeroing disp_j for j in S first.
"""
import numpy as np
import pickle
import itertools
from analyze_stage1b2 import compute_W_B_deltamap, run_permutation_test
from stage1b2_core import T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, get_degree_stratified_nodes


def common_support_exclude(q, source_indices):
    """Zero ALL candidate source coordinates (not just the one actually
    stimulated in this trial), renormalize over the common remainder.
    Every trial gets the identical exclusion mask."""
    if q is None:
        return None
    q_masked = q.copy()
    q_masked[list(source_indices)] = 0.0
    total = np.sum(q_masked)
    if total < 1e-15:
        return None
    return q_masked / total


def load_common_support_arrays(results, nodes, source_indices, time_key='event_aligned_q'):
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
                            q = results[key][time_key]
                            organized[t_p][(r, node_label, sign, amp)] = common_support_exclude(q, source_indices)
    return organized


if __name__ == '__main__':
    with open('stage1b2_results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W_matrix = data['constructions']['T']
    nodes = get_degree_stratified_nodes(W_matrix)
    source_indices = list(nodes.values())  # {i_low, i_median, i_high} -- the fixed candidate set

    print(f'Loaded {len(results)} trials (expect 432)')
    print(f'Common exclusion set (same for every trial): {nodes} -> indices {source_indices}')

    organized = load_common_support_arrays(results, nodes, source_indices, time_key='event_aligned_q')

    # sanity check: confirm every trial's masked vector has exactly the same zero positions
    node_labels = list(nodes.keys())
    sample_keys = [(0, r, nl, s, a) for r in range(N_REPLICAS) for nl in node_labels
                   for s in SIGNS for a in AMPLITUDES]
    n_checked, n_correct_zeros = 0, 0
    for key in sample_keys:
        q = organized[0].get(key)
        if q is not None:
            n_checked += 1
            if np.allclose(q[source_indices], 0.0):
                n_correct_zeros += 1
    print(f'Sanity check: {n_correct_zeros}/{n_checked} trials have all 3 source coordinates exactly zero (should be {n_checked}/{n_checked})')

    print(f'\n{"="*70}\nCOMMON-SUPPORT EXCLUSION OMNIBUS TEST\n{"="*70}')
    result = run_permutation_test(organized, nodes, seed=301)

    with open('stage1b2_common_support_exclusion.pkl', 'wb') as f:
        pickle.dump(result, f)
    print('\nSaved to stage1b2_common_support_exclusion.pkl')
