"""
Population-level Hebbian-adaptive local coupling for LocalOscillatorField.

Genuinely different from everything tested so far this session: coupling
strength is no longer a single fixed uniform scalar (k_coupling), and it's
no longer learned per-image (which couldn't plausibly escape the "basically
a blur kernel" ceiling found for fixed coupling -- not enough steps per
image for weights to move far from wherever they start, and no cross-image
information to shape them). Instead: two shared, direction-specific
weights (vertical-neighbor-pairs, horizontal-neighbor-pairs), tied across
every spatial position like a small fixed-shape convolution kernel, adapted
via the same Bronski-style Hebbian rule validated earlier this project
(dw/dt = mu*cos(delta_theta) - alpha*w), accumulated by exposing the field
to many UNLABELED training images before freezing the weights for
evaluation. This is the version that could plausibly learn genuine
cross-image structure (e.g. "vertically-adjacent pixels in real digit
strokes are typically more/less phase-correlated than horizontally-adjacent
ones"), not just react to one image at a time.

Few parameters (2 scalars) by design -- enough to capture real anisotropy
if it exists, constrained enough to generalize rather than memorize from
whatever finite set of unlabeled images gets used for the population phase.
"""
import numpy as np
from local_oscillator_field import LocalOscillatorField


class HebbianLocalField(LocalOscillatorField):
    """LocalOscillatorField with w_vertical/w_horizontal in place of a
    single uniform k_coupling, and a method to adapt them via the Bronski
    Hebbian rule during a population training phase."""

    def __init__(self, height, width, dt=0.1, w_vertical=1.0, w_horizontal=1.0,
                 k_bias=1.0, seed=None):
        # Deliberately NOT calling super().__init__ with k_coupling -- this
        # class replaces the single-scalar coupling entirely.
        self.H, self.W = height, width
        self.dt = dt
        self.w_vertical = w_vertical
        self.w_horizontal = w_horizontal
        self.k_bias = k_bias
        self.omega = 0.0
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0, 2 * np.pi, (height, width))
        self.target_phase = None

    def _neighbor_coupling(self):
        p = self.phases
        coupling = np.zeros_like(p)
        # Vertical neighbors (up/down)
        coupling[1:, :]  += self.w_vertical * np.sin(p[:-1, :] - p[1:, :])
        coupling[:-1, :] += self.w_vertical * np.sin(p[1:, :]  - p[:-1, :])
        # Horizontal neighbors (left/right)
        coupling[:, 1:]  += self.w_horizontal * np.sin(p[:, :-1] - p[:, 1:])
        coupling[:, :-1] += self.w_horizontal * np.sin(p[:, 1:]  - p[:, :-1])
        return coupling

    def step(self):
        coupling = self._neighbor_coupling()
        bias = np.sin(self.target_phase - self.phases) if self.target_phase is not None else 0.0
        dtheta = coupling + self.k_bias * bias  # w_vertical/w_horizontal already applied in coupling
        self.phases = (self.phases + self.dt * dtheta) % (2 * np.pi)
        return self.phases

    def accumulate_hebbian_statistics(self, steps=100):
        """Run the field on its currently-set input for `steps` steps, then
        return the mean cos(delta_theta) for vertical and horizontal
        neighbor pairs at the FINAL state -- the raw ingredient for the
        Hebbian update, matching the Bronski fixed-point form
        gamma* = mu*cos(delta_theta)/alpha."""
        for _ in range(steps):
            self.step()
        p = self.phases
        vertical_cos = np.mean(np.cos(p[:-1, :] - p[1:, :]))
        horizontal_cos = np.mean(np.cos(p[:, :-1] - p[:, 1:]))
        return vertical_cos, horizontal_cos


def train_population_weights(images, steps_per_image=100, mu=1.0, alpha=1.0,
                              k_bias=1.0, dt=0.1, arc=np.pi, seed=0):
    """Unsupervised population-level Hebbian training: expose w_vertical and
    w_horizontal to many images (no labels used anywhere here), updating
    via the Bronski rule accumulated across images (running mean of the
    per-image final-state cos(delta_theta), converted to the weight fixed
    point mu*cos/alpha) -- NOT gradient descent, no backprop, no loss
    function; a running local statistic, same learning paradigm as the
    rest of Bonsai.

    images: (N, H, W) array of raw pixel intensities in [0,1].
    Returns: (w_vertical, w_horizontal) -- the learned, frozen weights.
    """
    H, W = images.shape[1], images.shape[2]
    w_v, w_h = 1.0, 1.0  # start from the same uniform-coupling baseline
    vertical_cos_accum = []
    horizontal_cos_accum = []

    for i, image in enumerate(images):
        field = HebbianLocalField(H, W, dt=dt, w_vertical=w_v, w_horizontal=w_h,
                                   k_bias=k_bias, seed=seed + i)
        field.set_input(image, arc=arc)
        v_cos, h_cos = field.accumulate_hebbian_statistics(steps=steps_per_image)
        vertical_cos_accum.append(v_cos)
        horizontal_cos_accum.append(h_cos)
        # Running-mean update -- weights shift gradually as more images are
        # seen, rather than being fully re-fit from a single image.
        w_v = mu * np.mean(vertical_cos_accum) / alpha
        w_h = mu * np.mean(horizontal_cos_accum) / alpha

    return w_v, w_h
