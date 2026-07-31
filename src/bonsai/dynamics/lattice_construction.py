"""
Regular-lattice graph control for oscillator dynamics: a 4-connectivity
pixel-grid adjacency, restricted to the same active-node set as a
learned topology, with uniform edge weight normalized so total edge
weight matches the learned topology's total.

Reverse-engineered from the historical cached artifact
(stage1a_all_classes.pkl / class0_constructions.pkl's 'lattice' key,
built at some point via inline code that was never saved as a script)
and verified byte-exact against it: the reconstructed matrix's sparsity
pattern matches the cached artifact's 1870 nonzero entries in identical
positions, and the total-edge-weight normalization reproduces the cached
uniform weight (1.048117588914926 for KMNIST class 0) to full
floating-point precision.
"""
import numpy as np


def build_lattice_topology(active_indices, total_weight_target, side=28):
    """
    Build a regular 4-connectivity pixel-lattice graph, restricted to
    active_indices (indices into a side x side grid, flattened row-major),
    with uniform edge weight normalized so the matrix's total edge weight
    (sum of all entries) equals total_weight_target.

    Parameters
    ----------
    active_indices : array-like of int
        Flattened (row-major) grid indices to include as active nodes.
        Node ordering in the returned matrix matches the order of
        active_indices (row/column i corresponds to active_indices[i]).
    total_weight_target : float
        The lattice's total edge weight (sum over the whole matrix) will
        be normalized to equal this value -- typically sum(T) for the
        learned topology T being controlled against, so the lattice
        carries the same total coupling "budget" distributed uniformly
        rather than concentrated unevenly like the learned topology.
    side : int
        Grid side length (28 for MNIST-family 28x28 images).

    Returns
    -------
    W : np.ndarray, shape (len(active_indices), len(active_indices))
        Symmetric weighted adjacency matrix.
    """
    active_indices = np.asarray(active_indices)
    n = len(active_indices)
    active_set = set(active_indices.tolist())
    index_map = {idx: i for i, idx in enumerate(active_indices.tolist())}

    W = np.zeros((n, n))
    for idx in active_indices.tolist():
        r, c = idx // side, idx % side
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < side and 0 <= nc < side:
                neighbor = nr * side + nc
                if neighbor in active_set:
                    i, j = index_map[idx], index_map[neighbor]
                    W[i, j] = 1.0
                    W[j, i] = 1.0

    n_edges_weighted_twice = np.count_nonzero(W)  # each edge counted twice (i,j) and (j,i)
    if n_edges_weighted_twice == 0:
        return W  # no edges possible (e.g. active_indices all isolated) -- return zero matrix
    uniform_weight = total_weight_target / n_edges_weighted_twice
    W = W * uniform_weight
    return W
