"""
Spike-train cross-correlation coincidence: the genuine version of the CTM-
inspired synchrony idea, of which the earlier single-scalar first-spike-
time-difference coincidence graph was a crude stand-in. Two pixels are
"coincident" here if their FULL spike trains overlap over the whole
simulation window, not just if two single summary numbers are close.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from local_oscillator_field import LocalOscillatorField
from spike_time_encoder import _phase_gradient

H, W = 28, 28
STEPS = 100
DT = 0.05
K_BIAS = 2.0
OMEGA = 1.0
ARC = np.pi


def get_spike_train(image, k_coupling=1.0, steps=STEPS):
    field = LocalOscillatorField(H, W, dt=DT, k_coupling=k_coupling, k_bias=K_BIAS,
                                  omega=OMEGA)
    field.set_input(image, arc=ARC)
    field.phases = _phase_gradient(H, W)
    return field.run_track_spike_train(steps)


def spike_train_coincidence_graph(spike_train, tolerance_steps=2):
    """spike_train: (T, H, W) boolean. Returns (H*W, H*W) coincidence matrix:
    dot product of each pixel's spike train against every other's, after
    softening each spike into a small +/- tolerance_steps window so
    near-simultaneous (not just exactly-simultaneous) firing counts."""
    T = spike_train.shape[0]
    flat = spike_train.reshape(T, -1).astype(np.float64)  # (T, N)
    if tolerance_steps > 0:
        kernel = np.ones(2 * tolerance_steps + 1)
        # Convolve each column (pixel's spike train) along the time axis.
        # Vectorized via FFT-based convolution across all columns at once.
        from scipy.signal import fftconvolve
        smoothed = fftconvolve(flat, kernel[:, np.newaxis], mode='same', axes=0)
    else:
        smoothed = flat
    coincidence = smoothed.T @ smoothed  # (N, N)
    np.fill_diagonal(coincidence, 0.0)
    return coincidence

def spike_train_aligned_residual_signature(image, k_coupling=1.0, tolerance_steps=2, cutoff_idx=22):
    spike_train = get_spike_train(image, k_coupling=k_coupling)
    graph = spike_train_coincidence_graph(spike_train, tolerance_steps=tolerance_steps)
    laplacian = GraphLaplacian.from_adjacency(graph)
    # Use the spike COUNT per pixel (a natural per-pixel signal derived from
    # the spike train) as the signal to filter, analogous to how spike-TIME
    # was used before -- but now the GRAPH itself is built from full
    # spike-train cross-correlation, not a single-scalar proxy.
    spike_counts = spike_train.sum(axis=0).flatten().astype(np.float64)
    aligned, liberal = laplacian.filter_signal(spike_counts, cutoff_idx=cutoff_idx)
    return np.concatenate([aligned, liberal])

def spike_train_encode(X, k_coupling=1.0, tolerance_steps=2, cutoff_idx=22):
    N = X.shape[0]
    features = np.zeros((N, 2*H*W))
    for i in range(N):
        features[i] = spike_train_aligned_residual_signature(
            X[i].reshape(H,W), k_coupling=k_coupling, tolerance_steps=tolerance_steps, cutoff_idx=cutoff_idx)
    return features
