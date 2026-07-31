"""
Wire the learned (population-developmental, all-pairs) topology into an
actual running simulation and encoder, rather than just analyzing it
statically -- the real test of whether the long-range structure found in
developmental_pruning.py does anything, not just whether it looks
interesting.

Per-class learned topology: population_developmental_stat computed
separately per digit class (unsupervised WITHIN each class's population --
labels are used only to organize which images inform which class's
learned connections, same pattern as the earlier per-class Hebbian
coupling experiment, not to supervise the dynamics itself).
"""
import numpy as np
from local_oscillator_field import LocalOscillatorField
from developmental_pruning import population_developmental_stat

H, W = 28, 28


def build_class_topologies(X_train, y_train, classes, n_per_class=20, prune_threshold=0.9,
                            ink_threshold=0.15):
    """Returns {class: sparse_topology}. Background-background pairs are
    EXCLUDED entirely, not just pruned by magnitude -- confirmed directly
    (developmental_pruning.py analysis) that they survive a magnitude
    threshold ~88% of the time regardless of class, a trivial confound, not
    real structure. Only ink-ink and ink-background pairs are eligible to
    survive the prune_threshold."""
    topologies = {}
    for c in classes:
        idx = np.where(y_train == c)[0][:n_per_class]
        images = X_train[idx].astype(np.float64) / 255.0
        W_learned = population_developmental_stat(images)

        mean_intensity = images.mean(axis=0).flatten()
        ink_mask = mean_intensity > ink_threshold
        background_pair_mask = np.outer(~ink_mask, ~ink_mask)  # True where BOTH are background

        pruned = np.where(np.abs(W_learned) > prune_threshold, W_learned, 0.0)
        pruned[background_pair_mask] = 0.0  # explicit exclusion, not just magnitude pruning
        topologies[c] = pruned
    return topologies


class LearnedTopologyField:
    """Same intrinsic dynamics as LocalOscillatorField (closed-loop bias,
    partial-arc mapping), but coupling uses an arbitrary (N,N) learned
    weight matrix instead of the fixed 4-neighbor grid -- local coupling is
    a SPECIAL CASE of this (a sparse matrix with only adjacent-pixel
    entries), not a separate mechanism."""

    def __init__(self, topology, dt=0.1, k_bias=1.0):
        self.N = topology.shape[0]
        self.topology = topology  # (N,N), already pruned/sparse
        self.dt = dt
        self.k_bias = k_bias
        self.phases = None
        self.target_phase = None

    def set_input(self, image, arc=np.pi):
        self.target_phase = image.flatten() * arc

    def initialize_at_target(self, perturbation_std=0.01, seed=None):
        rng = np.random.default_rng(seed)
        self.phases = (self.target_phase + rng.normal(0, perturbation_std, self.target_phase.shape)) % (2*np.pi)

    def step(self):
        # coupling_i = sum_j topology[i,j] * sin(phase_j - phase_i)
        diff = self.phases[np.newaxis, :] - self.phases[:, np.newaxis]  # theta_j - theta_i
        coupling = np.sum(self.topology * np.sin(diff), axis=1)
        bias = np.sin(self.target_phase - self.phases)
        dtheta = coupling + self.k_bias * bias
        self.phases = (self.phases + self.dt * dtheta) % (2*np.pi)
        return self.phases

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.phases


def learned_topology_encode(X, topology, steps=150, dt=0.1, k_bias=1.0):
    N = X.shape[0]
    features = np.zeros((N, H*W*2))  # cos/sin, matching established convention
    for i in range(N):
        field = LearnedTopologyField(topology, dt=dt, k_bias=k_bias)
        field.set_input(X[i])
        field.initialize_at_target(seed=0)
        field.run(steps)
        features[i] = np.concatenate([np.cos(field.phases), np.sin(field.phases)])
    return features
