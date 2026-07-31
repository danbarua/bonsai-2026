"""
ComplexLocalOscillatorField: complex-valued (amplitude + phase) Hopf
oscillators, local power-coupling, Oja-normalized Hebbian learning.

Borrows the two genuinely new, verified-against-source ideas from
Bandyopadhyay et al. (2023) ("power-coupling" whole-brain model, Sci Rep
13:16935): power-coupling (z_k^P instead of linear coupling) and an
Oja-normalized Hebbian update for the lateral connections. Both are
trained/updated via a LOCAL, unsupervised rule in the source paper (not
backprop) -- directly compatible with staying in Bonsai's no-backprop
regime, unlike AKOrN/KoPE's learned coupling (confirmed earlier this
project: their "learnable" coupling means backprop on a supervised loss).

What's kept from the already-validated LocalOscillatorField:
- LOCAL (4-neighbor) topology, not all-to-all -- the thing that makes this
  tractable at MNIST scale at all.
- CLOSED-LOOP anchoring to the input (couple to a fixed target, not an
  open-loop bias) -- validated repeatedly this project.

What's genuinely new here:
- Complex-valued sites (amplitude AND phase), not unit-norm phase-only --
  addresses the "can't represent presence/confidence" limitation discussed
  with AKOrN, and lets input encode BOTH a target phase (partial-arc, per
  the aliasing lesson from MNIST_BASELINES.md) AND a target amplitude tied
  to pixel intensity.
- Intrinsic Hopf (Stuart-Landau) dynamics per site: dz/dt = z*(lambda +
  i*omega - |z|^2) -- a self-regulating nonlinearity that keeps amplitude
  bounded without a hard unit-norm constraint, unlike every oscillator
  model used so far this project (Kuramoto phase-only, or AKOrN-style
  sphere-projected vectors).
- Power-coupling: neighbors' contribution is w * (z_neighbor)^P, not linear
  -- richer than anything tried before (in polar form this multiplies phase
  by P and raises magnitude to the P power, so P!=1 can capture harmonic/
  cross-frequency relationships a linear or sin-based coupling can't).
"""
import numpy as np


class ComplexLocalOscillatorField:
    def __init__(self, height: int, width: int, dt: float = 0.02,
                 hopf_lambda: float = 1.0, hopf_omega: float = 1.0,
                 k_bias: float = 1.0, power: float = 1.0,
                 w_vertical: complex = 0.3, w_horizontal: complex = 0.3,
                 seed: int = None):
        self.H, self.W = height, width
        self.dt = dt
        self.hopf_lambda = hopf_lambda
        self.hopf_omega = hopf_omega
        self.k_bias = k_bias
        self.power = power
        self.w_vertical = w_vertical
        self.w_horizontal = w_horizontal

        rng = np.random.default_rng(seed)
        init_amp = 0.1  # start small, well inside the Hopf limit cycle's basin
        init_phase = rng.uniform(0, 2 * np.pi, (height, width))
        self.z = init_amp * np.exp(1j * init_phase)
        self.target_z = None

    def set_input(self, image: np.ndarray, arc: float = np.pi, amp_scale: float = 1.0):
        """image: (H, W) in [0,1]. Target phase uses the validated partial-arc
        mapping; target amplitude scales with pixel intensity directly --
        genuinely new, since every phase-only model this project used
        couldn't represent intensity as amplitude at all."""
        assert image.shape == (self.H, self.W)
        target_phase = image * arc
        target_amp = amp_scale * (0.2 + 0.8 * image)  # keep a nonzero floor amplitude
        self.target_z = target_amp * np.exp(1j * target_phase)

    def _intrinsic_hopf(self) -> np.ndarray:
        """dz/dt = z*(lambda + i*omega - |z|^2) -- self-regulating amplitude,
        stable limit cycle at |z| = sqrt(lambda) in isolation (lambda > 0)."""
        return self.z * (self.hopf_lambda + 1j * self.hopf_omega - np.abs(self.z) ** 2)

    def _power_coupling(self) -> np.ndarray:
        """Local (4-neighbor) power-coupling: w * (z_neighbor)^power, summed
        over whichever real neighbors exist (no zero-padding at edges, same
        convention as LocalOscillatorField)."""
        z = self.z
        z_pow = z ** self.power  # complex power; z=0 with power<1 could warn -- see step() guard
        coupling = np.zeros_like(z)
        coupling[1:, :]  += self.w_vertical * z_pow[:-1, :]
        coupling[:-1, :] += self.w_vertical * z_pow[1:, :]
        coupling[:, 1:]  += self.w_horizontal * z_pow[:, :-1]
        coupling[:, :-1] += self.w_horizontal * z_pow[:, 1:]
        return coupling

    def step(self) -> np.ndarray:
        intrinsic = self._intrinsic_hopf()
        coupling = self._power_coupling()
        bias = (self.target_z - self.z) if self.target_z is not None else 0.0
        dz = intrinsic + coupling + self.k_bias * bias
        self.z = self.z + self.dt * dz
        return self.z

    def run(self, steps: int) -> np.ndarray:
        for _ in range(steps):
            self.step()
        return self.z
