"""
Spike: population-level, all-pairs ("developmental") Hebbian statistic,
reusing the cheap LOCAL dynamics (already proven fast) to get each image's
converged phase field, then computing the FULL O(N^2) pairwise Bronski-
style Hebbian statistic (cos(theta_i - theta_j)) across many images --
testing whether a sparse set of LONG-RANGE (non-local) connections
"survives" pruning with real, non-background-trivial structure.
"""
import numpy as np
from local_oscillator_field import LocalOscillatorField

H, W = 28, 28

def get_local_converged_phases(image, steps=150, dt=0.1, k_coupling=1.0, k_bias=1.0):
    field = LocalOscillatorField(H, W, dt=dt, k_coupling=k_coupling, k_bias=k_bias)
    field.set_input(image, arc=np.pi)
    field.initialize_at_target(perturbation_std=0.01, seed=0)
    for _ in range(steps):
        field.step()
    return field.phases

def full_pairwise_hebbian_stat(phases):
    p = phases.flatten()
    diff = p[:, None] - p[None, :]
    return np.cos(diff)  # (N,N), Bronski-style ingredient: cos(theta_i - theta_j)

def population_developmental_stat(images, mu=1.0, alpha=1.0):
    """Accumulate the full pairwise stat across many images -- population
    level, unsupervised, no labels used."""
    N = H * W
    accum = np.zeros((N, N))
    for image in images:
        phases = get_local_converged_phases(image)
        accum += full_pairwise_hebbian_stat(phases)
    mean_stat = accum / len(images)
    W_learned = mu * mean_stat / alpha
    np.fill_diagonal(W_learned, 0)
    return W_learned


def compute_adaptation_baseline(images):
    """The 'common mode' / adapted baseline: population-mean converged
    phase per pixel, from a broad population -- the thing that's constant/
    unchanging across most images (dominated by background, which
    converges to similar phase regardless of digit) and should be
    desensitized to, exactly like sensory adaptation to an unchanging
    stimulus, or common-mode rejection in balanced audio."""
    all_phases = np.array([get_local_converged_phases(img).flatten() for img in images])
    return np.angle(np.mean(np.exp(1j * all_phases), axis=0))  # circular mean per pixel


def population_developmental_stat_adapted(images, baseline_phase, mu=1.0, alpha=1.0):
    """Same as population_developmental_stat, but correlating DEVIATIONS
    from the adaptation baseline, not raw phases -- common-mode rejected."""
    N = H * W
    accum = np.zeros((N, N))
    for image in images:
        phases_flat = get_local_converged_phases(image).flatten()
        deviation = np.angle(np.exp(1j * (phases_flat - baseline_phase)))  # wrapped residual
        diff = deviation[:, None] - deviation[None, :]
        accum += np.cos(diff)
    mean_stat = accum / len(images)
    W_learned = mu * mean_stat / alpha
    np.fill_diagonal(W_learned, 0)
    return W_learned
