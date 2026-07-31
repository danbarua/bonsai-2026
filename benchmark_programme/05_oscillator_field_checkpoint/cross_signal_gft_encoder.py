"""
Test the circularity hypothesis: project a DIFFERENT signal (raw pixel
intensity) onto the spike-time-derived coincidence graph, rather than
re-projecting the same spike-time signal that built the graph. If self-GFT's
underperformance was due to circularity (graph built from a signal, then
re-projecting that same signal onto it), using a genuinely different signal
should avoid that specific redundancy.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph

H, W = 28, 28

def cross_signal_gft_signature(image, k_coupling=1.0, sigma=0.1):
    spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
    graph = coincidence_graph(spike_times, sigma=sigma)
    laplacian = GraphLaplacian.from_adjacency(graph)
    pixel_signal = image.flatten()  # DIFFERENT signal than the one that built the graph
    freq_signal = laplacian.apply_gft(pixel_signal)
    return freq_signal.amplitudes

def cross_signal_gft_encode(X, k_coupling=1.0, sigma=0.1):
    N = X.shape[0]
    features = np.zeros((N, H*W))
    for i in range(N):
        features[i] = cross_signal_gft_signature(X[i].reshape(H,W), k_coupling=k_coupling, sigma=sigma)
    return features
