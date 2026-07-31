"""
Degree-preserving edge rewiring via double-edge swaps: preserves each
node's exact degree and the exact multiset of edge weights, randomizes
only which specific pairs of nodes are connected to each other. A
stronger, more surgical control than matched-sparsity random topology
(which only matches aggregate edge count, not per-node degree).

This determines whether useful information lives in NODE PARTICIPATION
PATTERNS (which pixels tend to be highly-connected at all -- preserved by
this control) or in the SPECIFIC PAIRWISE ARRANGEMENT (which exact pixel
each connection goes to -- destroyed by this control).
"""
import numpy as np


def degree_preserving_rewire(topology, ink_mask, n_swaps_multiplier=10, seed=0):
    N = topology.shape[0]
    triu_i, triu_j = np.triu_indices(N, k=1)
    weights = topology[triu_i, triu_j]
    nonzero_mask = weights != 0
    edges_i = list(triu_i[nonzero_mask])
    edges_j = list(triu_j[nonzero_mask])
    edge_weights = list(weights[nonzero_mask])
    n_edges = len(edges_i)

    rng = np.random.default_rng(seed)
    edge_set = {(min(i, j), max(i, j)) for i, j in zip(edges_i, edges_j)}

    n_swaps_target = n_swaps_multiplier * n_edges
    successful_swaps = 0
    attempts = 0
    max_attempts = n_swaps_target * 20

    while successful_swaps < n_swaps_target and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = rng.choice(n_edges, size=2, replace=False)
        a, b = edges_i[idx1], edges_j[idx1]
        c, d = edges_i[idx2], edges_j[idx2]

        if len({a, b, c, d}) < 4:
            continue

        new_edge1 = (min(a, d), max(a, d))
        new_edge2 = (min(c, b), max(c, b))

        if new_edge1 == new_edge2 or new_edge1 in edge_set or new_edge2 in edge_set:
            continue
        # Preserve the background-background exclusion rule used everywhere
        # else in this pipeline -- otherwise this control would reintroduce
        # a different, already-diagnosed confound.
        i1, j1 = new_edge1
        i2, j2 = new_edge2
        if (~ink_mask[i1] and ~ink_mask[j1]) or (~ink_mask[i2] and ~ink_mask[j2]):
            continue

        old_edge1 = (min(a, b), max(a, b))
        old_edge2 = (min(c, d), max(c, d))
        edge_set.discard(old_edge1)
        edge_set.discard(old_edge2)
        edge_set.add(new_edge1)
        edge_set.add(new_edge2)

        edges_i[idx1], edges_j[idx1] = new_edge1
        edges_i[idx2], edges_j[idx2] = new_edge2
        successful_swaps += 1

    rewired = np.zeros((N, N))
    for i, j, w in zip(edges_i, edges_j, edge_weights):
        rewired[i, j] = w
        rewired[j, i] = w

    original_edges = {(min(i, j), max(i, j)) for i, j in
                       zip(*np.where(np.triu(topology, k=1) != 0))}
    fraction_retained = len(edge_set & original_edges) / len(original_edges)

    return rewired, {'successful_swaps': successful_swaps, 'attempts': attempts,
                      'fraction_original_edges_retained': fraction_retained}
