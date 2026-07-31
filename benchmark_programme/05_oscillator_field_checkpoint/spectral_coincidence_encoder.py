"""
Temporal-coincidence graph encoding: a cheap approximation of an all-to-all
dynamical system, built as a post-hoc step on top of the already-cheap local
field simulation, rather than by actually running expensive all-to-all
dynamics.

Mechanism (directly inspired by Sakana AI's Continuous Thought Machines --
"neural synchronization employed as a direct latent representation... in the
timing of neural activity" -- and by the recollection that the 8x12
alphabet's ALL-TO-ALL weight matrix, a rich per-image object, looked
informative in a way the population-level shared local kernel structurally
cannot be):

1. Run LocalOscillatorField (cheap, local spatial coupling + closed-loop
   anchoring, as already built) to get first-spike-time per pixel.
2. Build a NEW graph where edge(i,j) = how close in TIME pixels i and j
   spiked, regardless of spatial distance -- relatedness by synchrony, not
   proximity. This is inherently per-image (like the rich weight matrix),
   but costs O(N^2) ONCE from already-computed spike times, not O(N^2) per
   simulation step the way genuine all-to-all dynamics would.
3. Treat this coincidence graph's Laplacian spectrum as the feature:
   eigenvalues are the graph's resonant frequencies, eigenvectors its normal
   modes -- literal, not metaphorical, given a graph of coupled oscillators.
   Reuses maths.graphs.GraphLaplacian, built and verified during the
   Bronski stability work earlier this project.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image


def coincidence_graph(spike_times: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """spike_times: (N,) normalized [0,1] first-spike-times, one per pixel.
    Returns (N, N) adjacency: Gaussian-kernel similarity in spike time,
    regardless of spatial position -- pure temporal-coincidence relatedness."""
    diff = spike_times[:, np.newaxis] - spike_times[np.newaxis, :]
    W = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def spectral_signature(image: np.ndarray, k_coupling: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """Full pipeline for one image: spike times -> coincidence graph ->
    GraphLaplacian -> sorted eigenvalue spectrum (the resonant-frequency
    profile). Fixed-length (H*W,) feature, directly comparable across
    images via the sorted eigenvalues (not eigenvectors, which have no
    canonical alignment/sign across different images)."""
    spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
    W = coincidence_graph(spike_times, sigma=sigma)
    laplacian = GraphLaplacian.from_adjacency(W)
    decomposition = laplacian.spectral_decomposition()
    return decomposition.eigenvalues


def spectral_encode(X: np.ndarray, k_coupling: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """X: (N, 784) raw pixel intensities in [0,1]. Returns (N, 784) spectral
    (eigenvalue) features, one row per image."""
    N = X.shape[0]
    H = W = 28
    features = np.zeros((N, H * W))
    for i in range(N):
        features[i] = spectral_signature(X[i].reshape(H, W), k_coupling=k_coupling, sigma=sigma)
    return features
