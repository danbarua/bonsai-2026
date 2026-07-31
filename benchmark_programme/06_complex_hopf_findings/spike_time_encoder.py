"""
First-spike-time readout for LocalOscillatorField -- genuinely untested
until now. Requires a nonzero shared omega (added to LocalOscillatorField
specifically for this) so oscillators keep advancing past a fixed point
rather than settling and stopping -- verified first in isolation (single
oscillator, no spatial coupling): once transients settle, first-spike-time
does encode a stable, input-dependent phase lag on top of an identical
period for every input, not chaos or a degenerate signal.

Two encoders here, mirroring the trivial-vs-simulated logic used throughout
this project:
- spike_time_encode: full model, spatial coupling included (k_coupling > 0).
- spike_time_encode_no_coupling: identical dynamics (omega + closed-loop
  bias) but k_coupling=0 -- isolates whether spatial coupling specifically
  adds anything to the spike-time signal over independent per-pixel
  oscillators, the same question asked (and answered "not much, small
  classifier-specific effect") for the phase-value readout earlier.
"""
import numpy as np
from local_oscillator_field import LocalOscillatorField

H, W = 28, 28
STEPS = 100
DT = 0.05
K_BIAS = 2.0
OMEGA = 1.0
ARC = np.pi


def _phase_gradient(H, W):
    """Deterministic spatial phase gradient across the grid, combining row
    and column position into a single reference phase per site -- restores
    the spatial-reference mechanism from the original notebook's
    `kappa * x_spatial_field` term, which we'd dropped when porting the
    encoder (reasoning that the classifier already sees position via array
    indexing -- true for the classifier, but irrelevant to the DYNAMICS:
    without this, every pixel starts at the literally identical phase,
    which is a genuine degeneracy, not just a simplification. Confirmed
    directly: with phases=0 everywhere, first-spike-time came out IDENTICAL
    (21.0) for target values 0.1, 0.5, and 0.9 -- the symmetry wasn't
    broken fast enough within the tracked window for "first crossing" to
    differ at all.
    """
    rows = np.linspace(0, 2 * np.pi, H, endpoint=False)
    cols = np.linspace(0, 2 * np.pi, W, endpoint=False)
    grid = (rows[:, np.newaxis] + cols[np.newaxis, :]) % (2 * np.pi)
    # Small offset avoids landing exactly on phase=0, which turned out to be
    # a coincidental degenerate starting angle for this specific omega/bias
    # combination (confirmed directly: target=0.1/0.5/0.9 all gave the
    # identical first-spike-time of 21 steps starting exactly at phase 0,
    # while every other tested starting phase showed clear, sensible
    # target-dependent variation).
    return (grid + 0.37) % (2 * np.pi)


def _spike_time_for_image(image, k_coupling, steps=STEPS):
    field = LocalOscillatorField(H, W, dt=DT, k_coupling=k_coupling, k_bias=K_BIAS,
                                  omega=OMEGA, seed=0)
    field.set_input(image, arc=ARC)
    field.phases = _phase_gradient(H, W)  # deterministic, but NOT degenerate
    spike_times = field.run_track_first_spike(steps)
    return spike_times.flatten() / steps  # normalize to [0, 1]


def spike_time_encode(X: np.ndarray, k_coupling: float = 1.0) -> np.ndarray:
    """X: (N, 784) raw pixel intensities in [0,1]. Returns (N, 784) normalized
    first-spike-times, WITH spatial coupling."""
    N = X.shape[0]
    features = np.zeros((N, H * W))
    for i in range(N):
        features[i] = _spike_time_for_image(X[i].reshape(H, W), k_coupling=k_coupling)
    return features


def spike_time_encode_no_coupling(X: np.ndarray) -> np.ndarray:
    """Same dynamics, k_coupling=0 -- the matched control isolating spatial
    coupling's contribution to the spike-time signal specifically."""
    return spike_time_encode(X, k_coupling=0.0)
