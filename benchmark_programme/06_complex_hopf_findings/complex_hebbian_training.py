"""
Oja-normalized Hebbian population training for ComplexLocalOscillatorField's
local coupling weights -- adapting Bandyopadhyay et al. (2023)'s rule
(Delta W_jk = eta*(z_j * conj(z_k)^P - alpha*W_jk*|z_j|^2)) to Bonsai's
established population-level pattern (hebbian_local_field.py): shared,
direction-specific weights (vertical/horizontal), tied across every spatial
position, updated from unsupervised exposure to many images -- no labels,
no backprop, same learning paradigm as the rest of the project.

The Oja-style decay term (-alpha*W*|z|^2, scaling with the node's OWN
amplitude squared) is a more principled self-stabilization than the plain
constant decay used in hebbian_local_field.py's real-valued version --
here it falls out naturally from the complex-valued rule.
"""
import numpy as np
from complex_hopf_field import ComplexLocalOscillatorField

H, W = 28, 28


def _local_hebbian_stats(z: np.ndarray, power: float):
    """Raw Hebbian statistics for the current state: z_i * conj(z_j)^power
    for vertical and horizontal neighbor pairs, averaged over the whole
    grid, plus mean |z|^2 for the Oja normalization."""
    vertical_stat = np.mean(z[:-1, :] * np.conj(z[1:, :]) ** power)
    horizontal_stat = np.mean(z[:, :-1] * np.conj(z[:, 1:]) ** power)
    mean_sq_amp = np.mean(np.abs(z) ** 2)
    return vertical_stat, horizontal_stat, mean_sq_amp


def train_population_weights_oja(images: np.ndarray, steps_per_image: int = 300,
                                  eta: float = 1.0, alpha: float = 1.0,
                                  dt: float = 0.02, hopf_lambda: float = 1.0,
                                  hopf_omega: float = 1.0, k_bias: float = 1.0,
                                  power: float = 1.0, w_init: float = 0.08,
                                  arc: float = np.pi, seed: int = 0):
    """Unsupervised population-level Hebbian training (no labels used
    anywhere in this function). images: (N, H, W) in [0,1].
    Returns: (w_vertical, w_horizontal) as complex scalars."""
    w_v, w_h = complex(w_init), complex(w_init)  # start from the validated safe operating point
    vertical_accum, horizontal_accum, sq_amp_accum = [], [], []

    for i, image in enumerate(images):
        field = ComplexLocalOscillatorField(
            H, W, dt=dt, hopf_lambda=hopf_lambda, hopf_omega=hopf_omega,
            k_bias=k_bias, power=power, w_vertical=w_v, w_horizontal=w_h, seed=seed + i
        )
        field.set_input(image, arc=arc)
        for _ in range(steps_per_image):
            field.step()

        v_stat, h_stat, mean_sq_amp = _local_hebbian_stats(field.z, power)
        vertical_accum.append(v_stat)
        horizontal_accum.append(h_stat)
        sq_amp_accum.append(mean_sq_amp)

        # Oja-normalized running fixed-point update: W* = eta*<z_i*conj(z_j)^P> / (alpha*<|z_i|^2>)
        denom = alpha * np.mean(sq_amp_accum)
        w_v = eta * np.mean(vertical_accum) / denom
        w_h = eta * np.mean(horizontal_accum) / denom

    return w_v, w_h
