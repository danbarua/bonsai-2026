"""
Two diagnostic encoders to understand where the persistent gap to raw
pixels comes from in the edge-residual pipeline:

1. pixel_direct_edge_encode: skip the oscillator simulation and spike-time
   reduction entirely -- build the coincidence graph directly from raw
   pixel intensity differences, then apply the SAME edge-extraction. If
   this matches or beats the full oscillator-based pipeline, the oscillator
   step isn't adding value -- it's discarding information the graph trick
   doesn't actually need discarded.

2. aligned_plus_residual_encode: concatenate the low-frequency (aligned)
   and high-frequency (residual) components instead of using the residual
   alone. Tests whether discarding the low-frequency component outright
   (rather than just not needing it to DOMINATE) is costing real signal.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph

H, W = 28, 28


def pixel_direct_edge_encode(X, sigma=0.1, cutoff_idx=22):
    """Coincidence graph built directly from pixel intensity, no oscillator
    simulation or spike-time reduction at all."""
    N = X.shape[0]
    features = np.zeros((N, H * W))
    for i in range(N):
        pixel_signal = X[i]  # raw pixel intensities, (784,), no simulation
        graph = coincidence_graph(pixel_signal, sigma=sigma)
        laplacian = GraphLaplacian.from_adjacency(graph)
        decomposition = laplacian.spectral_decomposition()
        coeffs = decomposition.eigenvectors.T @ pixel_signal
        coeffs[:cutoff_idx] = 0
        features[i] = decomposition.eigenvectors @ coeffs
    return features


def aligned_plus_residual_encode(X, k_coupling=1.0, sigma=0.1, cutoff_idx=22):
    """Same oscillator + spike-time pipeline as edge_encoder.py, but
    concatenate BOTH aligned (low) and residual (high) components instead
    of discarding the aligned part entirely."""
    N = X.shape[0]
    features = np.zeros((N, 2 * H * W))
    for i in range(N):
        image = X[i].reshape(H, W)
        spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
        graph = coincidence_graph(spike_times, sigma=sigma)
        laplacian = GraphLaplacian.from_adjacency(graph)
        aligned, liberal = laplacian.filter_signal(spike_times, cutoff_idx=cutoff_idx)
        features[i] = np.concatenate([aligned, liberal])
    return features
