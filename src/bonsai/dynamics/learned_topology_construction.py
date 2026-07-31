"""
Constructs the learned topology T from a class's raw images: a per-class,
unsupervised, population-level pairwise Hebbian statistic (population-
developmental stat: mean cos(theta_i - theta_j) across each image's
locally-converged phase field), pruned to keep only strongly-correlated
ink-involving pixel pairs, then restricted to the active (non-isolated)
node set.

Ported from benchmark_programme/10_topology_as_representation_findings/
(local_oscillator_field.py's LocalOscillatorField, developmental_pruning.py's
population_developmental_stat, learned_topology_encoder.py's
build_class_topologies) -- that code is written for MNIST and produces a
784x784, unrestricted-node matrix; this module produces the
active-node-restricted format experiments/ uses (matching
class0_constructions.pkl's T), and is dataset-agnostic (works for any
28x28 IDX-loaded dataset, KMNIST included).

Deliberately narrower than the original LocalOscillatorField class: only
the closed-loop-anchored, no-natural-frequency convergence path is needed
here (the class's spike-timing methods, from a separate unused exploratory
direction, are not ported).

Confirmed byte-exact (max abs diff 2.22e-16, float64 machine epsilon)
against class0_constructions.pkl's cached KMNIST-class-0 T when built
from the first 200 class-0 training images (n_per_class=200, not
build_class_topologies' MNIST-oriented default of 20 -- the actual value
used historically, inferred from the recovered kmnist_class_topologies_200.pkl
handoff file's own name, not from any code comment).
"""
import numpy as np

H, W = 28, 28


def _local_converged_phases(image, steps=150, dt=0.1, k_coupling=1.0, k_bias=1.0,
                             perturbation_std=0.01, seed=0):
    """4-neighbor (von Neumann) local Kuramoto field with closed-loop
    input anchoring, run to convergence. image: (28, 28) array in [0, 1].
    Returns the converged (28, 28) phase field."""
    target_phase = image * np.pi  # partial-arc mapping, not a full 2*pi rotation

    rng = np.random.default_rng(seed)
    phases = (target_phase + rng.normal(0, perturbation_std, target_phase.shape)) % (2 * np.pi)

    for _ in range(steps):
        coupling = np.zeros_like(phases)
        coupling[1:, :] += np.sin(phases[:-1, :] - phases[1:, :])
        coupling[:-1, :] += np.sin(phases[1:, :] - phases[:-1, :])
        coupling[:, 1:] += np.sin(phases[:, :-1] - phases[:, 1:])
        coupling[:, :-1] += np.sin(phases[:, 1:] - phases[:, :-1])
        bias = np.sin(target_phase - phases)
        dtheta = k_coupling * coupling + k_bias * bias
        phases = (phases + dt * dtheta) % (2 * np.pi)

    return phases


def population_developmental_stat(images, steps=150, dt=0.1, k_coupling=1.0, k_bias=1.0):
    """Population-level, unsupervised, all-pairs Hebbian statistic: mean
    cos(theta_i - theta_j) across each image's locally-converged phase
    field. images: (n, 28, 28) array in [0, 1]. Returns (784, 784),
    zero diagonal."""
    n_pixels = H * W
    accum = np.zeros((n_pixels, n_pixels))
    for image in images:
        phases = _local_converged_phases(
            image, steps=steps, dt=dt, k_coupling=k_coupling, k_bias=k_bias).flatten()
        diff = phases[:, None] - phases[None, :]
        accum += np.cos(diff)
    W_learned = accum / len(images)
    np.fill_diagonal(W_learned, 0)
    return W_learned


def build_class_topology(images, prune_threshold=0.9, ink_threshold=0.15, **stat_kwargs):
    """Builds one class's learned topology from its images: the
    population-developmental stat, pruned to |W| > prune_threshold, with
    background-background pixel pairs excluded entirely regardless of
    magnitude (confirmed elsewhere in this project's history to survive
    magnitude pruning ~88% of the time as a trivial confound, not real
    structure), then restricted to the active (non-isolated) node set.

    Parameters
    ----------
    images : (n, 28, 28) array, values in [0, 1] (already normalized).

    Returns
    -------
    active_indices : (k,) array of int
        Flattened (row-major) 28x28-grid indices of the active nodes,
        in ascending order.
    W_active : (k, k) array
        Symmetric weighted adjacency matrix restricted to active_indices.
    """
    W_learned = population_developmental_stat(images, **stat_kwargs)

    mean_intensity = images.mean(axis=0).flatten()
    ink_mask = mean_intensity > ink_threshold
    background_pair_mask = np.outer(~ink_mask, ~ink_mask)

    pruned = np.where(np.abs(W_learned) > prune_threshold, W_learned, 0.0)
    pruned[background_pair_mask] = 0.0

    active_indices = np.where(np.any(pruned != 0, axis=1))[0]
    W_active = pruned[np.ix_(active_indices, active_indices)]
    return active_indices, W_active
