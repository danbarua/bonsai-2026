"""
The surgical ablation: topology built from the INITIAL encoded phase
(target_phase = image * arc, before any oscillator simulation) versus the
CONVERGED phase (after 150 steps of local coupling + closed-loop bias).
Everything else in the pipeline -- threshold, background exclusion,
calibration, hybrid head -- stays identical. This isolates whether the
dynamics themselves contribute anything, changing exactly one variable
rather than several at once (unlike a raw-pixel-correlation control, which
would also change the encoding scheme).
"""
import numpy as np
from developmental_pruning import full_pairwise_hebbian_stat

H, W = 28, 28
ARC = np.pi


def get_initial_encoded_phases(image, arc=ARC):
    """No simulation at all -- the literal input encoding used to
    initialize LocalOscillatorField, before field.step() is ever called."""
    target_phase = image.flatten() * arc
    return target_phase.reshape(H, W)


def population_developmental_stat_nodynamics(images, mu=1.0, alpha=1.0):
    """Identical structure to population_developmental_stat, with the
    converged-phase computation replaced by the initial-encoding-only
    version -- the one variable changed."""
    N = H * W
    accum = np.zeros((N, N))
    for image in images:
        phases = get_initial_encoded_phases(image)
        accum += full_pairwise_hebbian_stat(phases)
    mean_stat = accum / len(images)
    W_learned = mu * mean_stat / alpha
    np.fill_diagonal(W_learned, 0.0)
    return W_learned
