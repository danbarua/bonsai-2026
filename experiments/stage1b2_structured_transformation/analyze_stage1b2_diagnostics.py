"""
Stage 1B.2 diagnostic decomposition, per review: the omnibus result
(p=0.0001, Delta_map=0.3505) establishes a structured internal MAPPING,
but does not by itself establish that the mapping reflects genuine
dynamical transformation rather than trivial source-node identity
retention. Four prespecified diagnostics, all using the already-
completed 432 trials -- no new experiment.

1. Remove the stimulated node, rerun omnibus test on the remainder.
2. Report source-node energy fraction at tau=0, tau*, tau=T.
3. Repeat the omnibus test on q_tangent (first-order propagation only).
4. Repeat the omnibus test on q_residual (finite-minus-tangent, the
   cleanest specifically-nonlinear object).

Plus: factor-specific restricted permutation tests (node/sign/amplitude),
Holm-corrected. All permutation loops are parallelized across worker
processes with independent random streams (SeedSequence.spawn) -- the
factor-specific tests were originally single-threaded (an oversight,
not a design choice) and have been rewritten to match the omnibus test's
parallelization pattern.
"""
import numpy as np
import pickle
import itertools
from analyze_stage1b2 import (
    load_results_as_arrays, compute_W_B_deltamap, permute_within_replica,
    run_permutation_test, N_PERMUTATIONS, d_q
)
from stage1b2_core import T_P_VALUES, AMPLITUDES, SIGNS, N_REPLICAS, get_degree_stratified_nodes
import multiprocessing as mp


def run_omnibus_on_field(results, nodes, time_key, label, n_workers=None, seed=42):
    print(f'\n{"="*70}\nOMNIBUS TEST on {label} (time_key={time_key})\n{"="*70}')
    organized = load_results_as_arrays(results, nodes, time_key=time_key)
    return run_permutation_test(organized, nodes, seed=seed, n_workers=n_workers)


def report_source_energy_fraction(results):
    print(f'\n{"="*70}\nSOURCE-NODE ENERGY FRACTION (identity retention vs. redistribution)\n{"="*70}')
    for field, label in [('initial_f_source', 'tau=0 (immediately after impulse)'),
                          ('event_aligned_f_source', 'tau=tau* (event-aligned)'),
                          ('fixed_time_f_source', 'tau=T (fixed-time)')]:
        vals = [v[field] for v in results.values() if v.get(field) is not None]
        vals = np.array(vals)
        print(f'  {label}: mean={vals.mean():.4f}, std={vals.std():.4f}, '
              f'median={np.median(vals):.4f}, n={len(vals)}')
    print('\n  Interpretation: if source fraction stays near 1.0 throughout, the')
    print('  mapping is largely identity retention. If it falls substantially')
    print('  from tau=0 while node-discrimination remains significant, that is')
    print('  much stronger evidence for genuine redistribution/routing.')


# ---- Factor-specific restricted permutations ----

def permute_node_within_sign_amp(organized_tp, node_labels, rng):
    """Permute NODE labels only, within each matched (sign, amplitude) cell,
    independently per replica -- preserves sign and amplitude assignment."""
    permuted = {}
    for r in range(N_REPLICAS):
        for sign, amp in itertools.product(SIGNS, AMPLITUDES):
            outputs = [organized_tp.get((r, nl, sign, amp)) for nl in node_labels]
            perm = rng.permutation(len(node_labels))
            for i, nl in enumerate(node_labels):
                permuted[(r, nl, sign, amp)] = outputs[perm[i]]
    return permuted


def permute_sign_within_node_amp(organized_tp, node_labels, rng):
    permuted = {}
    for r in range(N_REPLICAS):
        for nl, amp in itertools.product(node_labels, AMPLITUDES):
            outputs = [organized_tp.get((r, nl, s, amp)) for s in SIGNS]
            perm = rng.permutation(len(SIGNS))
            for i, s in enumerate(SIGNS):
                permuted[(r, nl, s, amp)] = outputs[perm[i]]
    return permuted


def permute_amplitude_within_node_sign(organized_tp, node_labels, rng):
    permuted = {}
    for r in range(N_REPLICAS):
        for nl, s in itertools.product(node_labels, SIGNS):
            outputs = [organized_tp.get((r, nl, s, a)) for a in AMPLITUDES]
            perm = rng.permutation(len(AMPLITUDES))
            for i, a in enumerate(AMPLITUDES):
                permuted[(r, nl, s, a)] = outputs[perm[i]]
    return permuted


def _factor_permutation_worker(args):
    """Runs `n_perms` factor-specific permutation draws using an
    independent random stream (SeedSequence-derived, not a shared seed
    across workers -- same correctness requirement as the omnibus test's
    worker)."""
    organized, node_labels, b_key, permute_fn, n_perms, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    vals = []
    for _ in range(n_perms):
        vals_by_tp = []
        for t_p in T_P_VALUES:
            permuted_tp = permute_fn(organized[t_p], node_labels, rng)
            r = compute_W_B_deltamap(permuted_tp, node_labels)
            vals_by_tp.append(r[b_key] - r['W'])
        vals.append(np.mean(vals_by_tp))
    return vals


def factor_specific_test(organized, nodes, factor_name, permute_fn, seed, n_workers=None):
    node_labels = list(nodes.keys())
    observed = {t_p: compute_W_B_deltamap(organized[t_p], node_labels) for t_p in T_P_VALUES}
    b_key = f'B_{factor_name}'
    pooled_obs = np.mean([observed[t_p][b_key] - observed[t_p]['W'] for t_p in T_P_VALUES])

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    base_seed_seq = np.random.SeedSequence(seed)
    worker_seed_seqs = base_seed_seq.spawn(n_workers)
    perms_per_worker = [N_PERMUTATIONS // n_workers] * n_workers
    for i in range(N_PERMUTATIONS % n_workers):
        perms_per_worker[i] += 1

    work_items = [(organized, node_labels, b_key, permute_fn, perms_per_worker[i], worker_seed_seqs[i])
                  for i in range(n_workers)]

    perm_vals = []
    with mp.Pool(n_workers) as pool:
        for worker_result in pool.imap_unordered(_factor_permutation_worker, work_items):
            perm_vals.extend(worker_result)
    perm_vals = np.array(perm_vals)
    assert len(perm_vals) == N_PERMUTATIONS, f'expected {N_PERMUTATIONS}, got {len(perm_vals)}'

    p = (1 + np.sum(perm_vals >= pooled_obs)) / (N_PERMUTATIONS + 1)
    return {'factor': factor_name, 'delta': pooled_obs, 'p_raw': p}


def holm_correction(p_values):
    """Holm-Bonferroni step-down correction -- more powerful than plain
    Bonferroni while still controlling family-wise error rate."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    prev_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * p_values[idx]
        adj = max(adj, prev_max)
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        prev_max = adj
    return adjusted


if __name__ == '__main__':
    with open('results/stage1b2_results.pkl', 'rb') as f:
        results = pickle.load(f)
    with open('results/class0_constructions.pkl', 'rb') as f:
        data = pickle.load(f)[0]
    W_matrix = data['constructions']['T']
    nodes = get_degree_stratified_nodes(W_matrix)
    n_workers = max(1, mp.cpu_count() - 1)

    print(f'Loaded {len(results)} trials (expect 432)')

    # Check 2: source energy fraction (descriptive, no permutation needed)
    report_source_energy_fraction(results)

    # Check 1: omnibus test excluding the stimulated node
    result_excl = run_omnibus_on_field(results, nodes, 'event_aligned_q_excl_node',
                                         'q EXCLUDING stimulated node', n_workers=n_workers, seed=101)

    # Check 3: omnibus test on tangent-only response
    result_tangent = run_omnibus_on_field(results, nodes, 'event_aligned_q_tangent',
                                            'TANGENT-ONLY response', n_workers=n_workers, seed=102)

    # Check 4: omnibus test on the nonlinear residual
    result_residual = run_omnibus_on_field(results, nodes, 'event_aligned_q_residual',
                                             'NONLINEAR RESIDUAL (finite - tangent)', n_workers=n_workers, seed=103)

    # Original, for comparison
    with open('results/stage1b2_final_analysis.pkl', 'rb') as f:
        result_original = pickle.load(f)

    print(f'\n{"="*70}\nEFFECT SIZE COMPARISON (finite vs tangent)\n{"="*70}')
    print(f'  Delta_map (original finite response): {result_original["pooled_delta_map"]:.4f}')
    print(f'  Delta_map (tangent-only):              {result_tangent["pooled_delta_map"]:.4f}')
    print(f'  Delta_finite - Delta_tangent:          {result_original["pooled_delta_map"] - result_tangent["pooled_delta_map"]:.4f}')
    print(f'  Delta_map (excl. stimulated node):     {result_excl["pooled_delta_map"]:.4f}')
    print(f'  Delta_map (nonlinear residual):        {result_residual["pooled_delta_map"]:.4f}')

    # Factor-specific tests, Holm-corrected -- NOW PARALLELIZED
    print(f'\n{"="*70}\nFACTOR-SPECIFIC TESTS (restricted permutations, Holm-corrected, parallelized)\n{"="*70}')
    organized = load_results_as_arrays(results, nodes, time_key='event_aligned_q')
    node_labels = list(nodes.keys())
    factor_results = [
        factor_specific_test(organized, nodes, 'node', permute_node_within_sign_amp, seed=201, n_workers=n_workers),
        factor_specific_test(organized, nodes, 'sign', permute_sign_within_node_amp, seed=202, n_workers=n_workers),
        factor_specific_test(organized, nodes, 'amplitude', permute_amplitude_within_node_sign, seed=203, n_workers=n_workers),
    ]
    p_raws = np.array([r['p_raw'] for r in factor_results])
    p_holm = holm_correction(p_raws)
    for r, p_adj in zip(factor_results, p_holm):
        print(f'  {r["factor"]:<12}: Delta={r["delta"]:.4f}, p_raw={r["p_raw"]:.5f}, p_holm={p_adj:.5f}')

    all_results = {
        'source_energy_fraction_reported': True,
        'excl_node': result_excl,
        'tangent_only': result_tangent,
        'residual': result_residual,
        'factor_specific': list(zip(factor_results, p_holm.tolist())),
    }
    with open('results/stage1b2_diagnostics.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print('\nSaved to results/stage1b2_diagnostics.pkl')
