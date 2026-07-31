"""
Self-referential GFT features: project each image's own spike-time signal
onto ITS OWN coincidence graph's eigenvectors -- not someone else's.

This sidesteps the confound found and confirmed three independent ways in
resonance_classifier.py / shape_resonance_encoder.py: comparing a signal's
fit against DIFFERENT reference graphs is dominated by which reference
graph's topology is generically "nicer" (closed-loop beats open-line,
unconditionally), independent of what's being reconstructed. Here there is
only ever one graph per image, and the signal is always projected onto its
own modes -- no cross-graph comparison exists for the confound to act on.

Feature: the GFT amplitude coefficients (not just the bare eigenvalues) --
how strongly does this image's own signal excite its own graph's low vs
high modes. Richer than eigenvalues alone (which only describe mode
structure, not how much the actual data excites each mode).
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph


def self_gft_signature(image: np.ndarray, k_coupling: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """Full pipeline for one image: spike times -> own coincidence graph ->
    own eigenvectors -> project own signal onto own eigenvectors -> GFT
    amplitude coefficients (sorted by the graph's own eigenvalue order)."""
    spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
    W = coincidence_graph(spike_times, sigma=sigma)
    laplacian = GraphLaplacian.from_adjacency(W)
    freq_signal = laplacian.apply_gft(spike_times)
    return freq_signal.amplitudes


def self_gft_encode(X: np.ndarray, k_coupling: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """X: (N, 784) raw pixel intensities in [0,1]. Returns (N, 784) GFT
    amplitude features, one row per image."""
    N = X.shape[0]
    H = W = 28
    features = np.zeros((N, H * W))
    for i in range(N):
        features[i] = self_gft_signature(X[i].reshape(H, W), k_coupling=k_coupling, sigma=sigma)
    return features
