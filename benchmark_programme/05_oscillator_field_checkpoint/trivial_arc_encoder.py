"""
The trivial control for LocalOscillatorField: partial-arc cos/sin encoding
of the RAW pixel value, with NO simulation, no coupling, no dynamics at all.

This is the direct test of whether the oscillator field's dynamics add
anything beyond what the same [0,pi] arc mapping gives for free. If this
performs the same as oscillator_field_encode, the simulation isn't
contributing anything -- it would just be an expensive way to compute
something a single line of NumPy already computes.
"""
import numpy as np


def trivial_partial_arc_encode(X: np.ndarray, arc: float = np.pi) -> np.ndarray:
    """X: (N, 784) raw pixel intensities in [0,1]. Returns (N, 1568) [cos,sin]
    of the raw pixel value mapped to [0, arc] -- no simulation."""
    phase = X * arc
    return np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
