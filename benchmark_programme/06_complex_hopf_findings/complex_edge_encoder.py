"""
Encoder for ComplexLocalOscillatorField: apply the same edge-residual
extraction (subtract low-frequency graph reconstruction, keep both aligned
and residual components -- the best-performing readout of the whole
session) to the complex field's output, using amplitude and phase as two
separate real-valued signals on the same coincidence graph.
"""
import numpy as np
from maths.graphs import GraphLaplacian
from complex_hopf_field import ComplexLocalOscillatorField
from spectral_coincidence_encoder import coincidence_graph

H, W = 28, 28
STEPS = 300
DT = 0.02


def complex_field_final_state(image, w_vertical=0.08, w_horizontal=0.08,
                               hopf_lambda=1.0, hopf_omega=1.0, k_bias=1.0, power=1.0):
    field = ComplexLocalOscillatorField(H, W, dt=DT, hopf_lambda=hopf_lambda, hopf_omega=hopf_omega,
                                         k_bias=k_bias, power=power,
                                         w_vertical=w_vertical, w_horizontal=w_horizontal, seed=0)
    field.set_input(image, arc=np.pi, amp_scale=1.0)
    for _ in range(STEPS):
        field.step()
    return field.z


def complex_edge_signature(image, w_vertical=0.08, w_horizontal=0.08, cutoff_idx=22, sigma=0.1):
    z = complex_field_final_state(image, w_vertical=w_vertical, w_horizontal=w_horizontal)
    amplitude = np.abs(z).flatten()
    phase = np.angle(z).flatten()

    # Build the coincidence graph from amplitude (the genuinely new signal
    # this model can provide that phase-only models couldn't)
    graph = coincidence_graph(amplitude, sigma=sigma)
    laplacian = GraphLaplacian.from_adjacency(graph)

    amp_aligned, amp_residual = laplacian.filter_signal(amplitude, cutoff_idx=cutoff_idx)
    phase_aligned, phase_residual = laplacian.filter_signal(phase, cutoff_idx=cutoff_idx)

    return np.concatenate([amp_aligned, amp_residual, phase_aligned, phase_residual])


def complex_edge_encode(X, w_vertical=0.08, w_horizontal=0.08, cutoff_idx=22, sigma=0.1):
    N = X.shape[0]
    features = np.zeros((N, 4 * H * W))
    for i in range(N):
        features[i] = complex_edge_signature(X[i].reshape(H, W), w_vertical=w_vertical,
                                               w_horizontal=w_horizontal, cutoff_idx=cutoff_idx, sigma=sigma)
    return features
