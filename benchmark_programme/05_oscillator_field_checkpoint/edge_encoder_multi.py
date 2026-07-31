"""
Efficient multi-cutoff edge/residual encoding: the eigendecomposition (the
expensive O(N^3) part) doesn't depend on cutoff_idx, only the final
low/high split does -- compute it once per image, extract multiple cutoffs
cheaply from the same decomposition, rather than repeating the expensive
part for every cutoff value tested.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph

H, W = 28, 28

def edge_signatures_multi_cutoff(image, cutoffs, k_coupling=1.0, sigma=0.1):
    """Returns {cutoff: edge_signal} for every cutoff in `cutoffs`, computing
    the expensive spike-time + graph + eigendecomposition steps only once."""
    spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
    graph = coincidence_graph(spike_times, sigma=sigma)
    laplacian = GraphLaplacian.from_adjacency(graph)
    decomposition = laplacian.spectral_decomposition()
    coeffs = decomposition.eigenvectors.T @ spike_times

    results = {}
    for cutoff in cutoffs:
        liberal_coeffs = coeffs.copy()
        liberal_coeffs[:cutoff] = 0
        liberal_signal = decomposition.eigenvectors @ liberal_coeffs
        results[cutoff] = liberal_signal
    return results

def edge_encode_multi_cutoff(X, cutoffs, k_coupling=1.0, sigma=0.1):
    """X: (N, 784). Returns {cutoff: (N, 784) feature array}."""
    N = X.shape[0]
    features = {c: np.zeros((N, H*W)) for c in cutoffs}
    for i in range(N):
        sigs = edge_signatures_multi_cutoff(X[i].reshape(H,W), cutoffs, k_coupling=k_coupling, sigma=sigma)
        for c in cutoffs:
            features[c][i] = sigs[c]
    return features
