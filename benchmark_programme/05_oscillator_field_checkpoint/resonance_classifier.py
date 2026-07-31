"""
Resonance classifier: per-class reference coincidence graph (population-
level, unsupervised in learning mechanism -- no loss function, no backprop,
just averaging -- organized by label only to avoid the cross-class
cancellation problem found earlier for a single shared kernel), then
classify a new image by which class's normal modes its own spike-time
signal excites the most.

This is the "eigenvectors, not just eigenvalues" idea: rather than compare
sorted eigenvalue spectra across different per-image graphs (informative
but discards which MODE SHAPES matter), build one graph per class and use
maths.graphs.GraphLaplacian's existing GFT machinery (project_signal) to
measure how strongly a test image's own signal projects onto each class's
actual eigenvectors -- literal resonance, using code built and verified for
the Bronski stability work earlier this project.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph


def build_class_reference_graph(images: np.ndarray, k_coupling: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """images: (n, H, W) same-class images. Returns the averaged (H*W, H*W)
    coincidence adjacency matrix -- entrywise average across images, valid
    since every image shares the same pixel-index correspondence."""
    avg_graph = None
    for image in images:
        spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
        W = coincidence_graph(spike_times, sigma=sigma)
        avg_graph = W.copy() if avg_graph is None else avg_graph + W
    return avg_graph / len(images)


def resonance_score(test_signal: np.ndarray, class_eigenvectors: np.ndarray,
                     n_modes: int = 20) -> float:
    """Reconstruction error using this class's specific mode SHAPES (not just
    a low/high-frequency energy split, which turned out to measure generic
    signal smoothness rather than class-specific fit -- confirmed directly:
    the energy-ratio version collapsed to predicting the same class for
    every test image, since spike-time signals are broadly smooth regardless
    of source digit, and almost any reasonable graph's low modes capture
    ~97% of energy from any smooth signal, regardless of whether the mode
    SHAPES actually match this particular signal's spatial pattern).

    Returns NEGATIVE reconstruction error (so higher = better fit, consistent
    with resonance_score's use in classify_by_resonance, which picks the max).
    """
    basis = class_eigenvectors[:, :n_modes]
    coeffs = basis.T @ test_signal
    reconstructed = basis @ coeffs
    error = np.linalg.norm(test_signal - reconstructed)
    return -error


def classify_by_resonance(test_image: np.ndarray, class_eigenvector_bases: dict,
                           k_coupling: float = 1.0, n_modes: int = 20) -> tuple:
    """Returns (predicted_class, all_scores_dict)."""
    test_signal = _spike_time_for_image(test_image, k_coupling=k_coupling)
    scores = {}
    for cls, eigvecs in class_eigenvector_bases.items():
        scores[cls] = resonance_score(test_signal, eigvecs, n_modes=n_modes)
    predicted = max(scores, key=scores.get)
    return predicted, scores
