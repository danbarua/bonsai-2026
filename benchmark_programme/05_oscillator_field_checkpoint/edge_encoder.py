"""
Edge/residual features: subtract the smooth (low-frequency) reconstruction
of a signal from the signal itself, using the graph's own eigenbasis --
what's left ("liberal"/high-frequency component, in graphs.py's terms) is
the part NOT explained by smooth, large-scale structure: local edges and
boundaries of change, not overall resonance or mode-amplitude.

Motivated by two things found earlier tonight: (1) fine, intermediate-
intensity edge pixels carry a disproportionate share of real discriminative
signal in MNIST (the full-2pi aliasing analysis), and (2) low-frequency
modes captured ~97% of energy for almost ANY signal on almost any graph --
meaning the residual (~3%) is where whatever ISN'T generic/shared structure
has to live.

Uses GraphLaplacian.filter_signal(), already built, never used until now.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph

H, W = 28, 28

def edge_signature(image, k_coupling=1.0, sigma=0.1, cutoff_idx=20):
    spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
    graph = coincidence_graph(spike_times, sigma=sigma)
    laplacian = GraphLaplacian.from_adjacency(graph)
    aligned, liberal = laplacian.filter_signal(spike_times, cutoff_idx=cutoff_idx)
    return liberal  # the residual/edge component

def edge_encode(X, k_coupling=1.0, sigma=0.1, cutoff_idx=20):
    N = X.shape[0]
    features = np.zeros((N, H*W))
    for i in range(N):
        features[i] = edge_signature(X[i].reshape(H,W), k_coupling=k_coupling, sigma=sigma, cutoff_idx=cutoff_idx)
    return features
