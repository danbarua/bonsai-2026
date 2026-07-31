"""
Shape-resonance features: instead of comparing per-class DATA-DERIVED
reference bases against each other (confounded -- found directly that some
classes' bases are just generically better at reconstructing ANY smooth
signal, unrelated to genuine class match, and this was robust to 6x more
reference data, ruling out "just needs more samples"), build a small set of
IDEALIZED, synthetic geometric reference shapes (a clean ring, a clean
vertical line, a clean horizontal line) and measure each real image's
reconstruction-fit against each one.

This gives a handful of interpretable scalar features per image ("how round
is this", "how vertical-line-like", "how horizontal-line-like") rather than
a single max-over-classes prediction -- sidesteps the "some bases are just
generically nice" confound entirely, since we're comparing against clean,
purpose-built references chosen for THEIR OWN geometric properties, not
noisy data averages. Fed into the existing, already-validated few-shot
classifier pipeline as just another (small, interpretable) encoding.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from spike_time_encoder import _spike_time_for_image
from spectral_coincidence_encoder import coincidence_graph
from resonance_classifier import resonance_score

H, W = 28, 28


def _ring_template():
    """Clean ring/loop shape -- like a '0'."""
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return ((r > 6) & (r < 11)).astype(np.float64)


def _vertical_line_template():
    """Clean vertical stroke -- like a '1'."""
    template = np.zeros((H, W))
    template[4:24, 13:16] = 1.0
    return template


def _horizontal_line_template():
    """Clean horizontal stroke."""
    template = np.zeros((H, W))
    template[13:16, 4:24] = 1.0
    return template


def _diagonal_template():
    """Clean diagonal stroke -- like part of a '7' or '4'."""
    template = np.zeros((H, W))
    for i in range(24):
        j = int(i * 0.8) + 3
        if 0 <= j < W - 2:
            template[i + 2, j:j + 2] = 1.0
    return template


TEMPLATES = {
    "ring": _ring_template,
    "vertical": _vertical_line_template,
    "horizontal": _horizontal_line_template,
    "diagonal": _diagonal_template,
}


def build_template_eigenbases(k_coupling: float = 1.0, sigma: float = 0.1) -> dict:
    """Build the eigenvector basis for each idealized template shape, ONCE
    (not per real image) -- these are fixed, reusable reference structures."""
    bases = {}
    for name, template_fn in TEMPLATES.items():
        image = template_fn()
        spike_times = _spike_time_for_image(image, k_coupling=k_coupling)
        W_graph = coincidence_graph(spike_times, sigma=sigma)
        laplacian = GraphLaplacian.from_adjacency(W_graph)
        bases[name] = laplacian.spectral_decomposition().eigenvectors
    return bases


def shape_resonance_features(image: np.ndarray, template_bases: dict,
                              k_coupling: float = 1.0, n_modes: int = 20) -> np.ndarray:
    """Returns a small (len(template_bases),) feature vector: this image's
    reconstruction-fit score against each idealized template shape."""
    signal = _spike_time_for_image(image, k_coupling=k_coupling)
    return np.array([resonance_score(signal, template_bases[name], n_modes=n_modes)
                      for name in TEMPLATES.keys()])


def shape_resonance_encode(X: np.ndarray, template_bases: dict,
                            k_coupling: float = 1.0, n_modes: int = 20) -> np.ndarray:
    """X: (N, 784) raw pixel intensities in [0,1]. Returns (N, len(templates))
    interpretable shape-resonance features."""
    N = X.shape[0]
    features = np.zeros((N, len(TEMPLATES)))
    for i in range(N):
        features[i] = shape_resonance_features(X[i].reshape(H, W), template_bases,
                                                  k_coupling=k_coupling, n_modes=n_modes)
    return features
