"""
Encoder function wiring LocalOscillatorField into the few_shot_harness.py /
classifier_sweep.py interface: encode_fn(X) -> X_encoded, where X is
(N, H*W) raw pixel intensities in [0,1] and X_encoded is (N, D).

Feature: the converged phase field, as [cos(phase), sin(phase)] concatenated
across all H*W sites -- same convention as the raw/cos-sin baselines already
established, so results are directly comparable in the harness/sweep.

Compute budget: convergence was measured to happen within a handful of
steps for smooth/realistic patterns (no hard-edged instability -- see
local_oscillator_field.py docstring), but STEPS is set with real margin
above that rather than the bare minimum observed, since MNIST's actual
digit shapes are more complex than the toy test patterns used to
characterize convergence speed.
"""
import numpy as np
from local_oscillator_field import LocalOscillatorField

H, W = 28, 28
STEPS = 100
ARC = np.pi
K_COUPLING = 1.0
K_BIAS = 1.0
PERTURBATION_STD = 0.01
PERTURBATION_SEED = 0  # fixed, not varied per-image -- see note below


def oscillator_field_encode(X: np.ndarray) -> np.ndarray:
    """X: (N, 784) raw pixel intensities in [0,1]. Returns (N, 1568) [cos,sin]
    features from each image's converged phase field.

    Note on PERTURBATION_SEED being fixed rather than varied per image: the
    perturbation is symmetry-breaking insurance against an edge-case
    instability (see LocalOscillatorField.initialize_at_target), not a
    meaningful source of information -- for realistic (smooth) images the
    converged state doesn't depend on it at all (verified: exact agreement
    across different perturbation seeds). Using the same fixed seed for
    every image keeps encoding fully deterministic without needing to trust
    that invariance perfectly for every possible input; if it were ever
    varied per-image, that would need its own justification.
    """
    N = X.shape[0]
    features = np.zeros((N, 2 * H * W))
    for i in range(N):
        image = X[i].reshape(H, W)
        field = LocalOscillatorField(H, W, dt=0.1, k_coupling=K_COUPLING, k_bias=K_BIAS)
        field.set_input(image, arc=ARC)
        field.initialize_at_target(perturbation_std=PERTURBATION_STD, seed=PERTURBATION_SEED)
        for _ in range(STEPS):
            field.step()
        phases = field.phases.flatten()
        features[i] = np.concatenate([np.cos(phases), np.sin(phases)])
    return features
