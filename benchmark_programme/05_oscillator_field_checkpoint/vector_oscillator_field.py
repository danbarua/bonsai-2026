"""
VectorOscillatorField: extends LocalOscillatorField to D-dimensional
complex-vector oscillators per site, adding cross-channel mixing via a
skew-symmetric matrix -- borrowed from oscillator_field_dynamics.ipynb's
KuramotoVectorOscillatorField, ported to test one specific question: does
vector-valued/cross-channel structure add anything beyond what the scalar
LocalOscillatorField already does (which we measured directly: a small,
classifier-specific ~2-3pp denoising effect for KNN, nothing for
NearestCentroid).

What's kept from the already-validated scalar model, unchanged:
- Local (not all-to-all) spatial coupling, per channel independently.
- CLOSED-LOOP anchoring (bias = target - z), not the notebook's structurally
  similar but not identical `bias = c - z` -- same principle, validated
  three independent ways this project.
- Partial-arc [0, pi] phase mapping for pixel intensity, not the notebook's
  full [0, 2*pi] -- avoids the aliasing cost measured in MNIST_BASELINES.md.
- No-padding boundary handling (edge/corner sites simply have fewer real
  neighbors, no zero-padded phantom neighbors) -- deliberately NOT the
  notebook's F.pad(value=0) approach, which introduces an unintended
  boundary artifact (zero-padding is a real, if probably minor, difference
  from the notebook -- flagged so the cross-channel test isn't confounded
  by an unrelated boundary-handling difference).

What's genuinely new, from the notebook:
- D complex channels per site (default D=4, matching the notebook), not a
  single scalar phase.
- Cross-channel mixing: at each site, z_omega = omega @ z_site, where omega
  is a real, skew-symmetric (D,D) matrix (matching the notebook's
  omega_init - omega_init.T construction) -- a rotation generator mixing
  the D channels together, independent of spatial position.
- Spatial coupling uses (neighbor_sum - self), not sin(phase_diff) directly
  -- matching the notebook's actual formulation. Mathematically these
  coincide in effect once renormalized to the unit circle (confirmed:
  produces exactly zero phase change at synchrony, for any neighbor count,
  same as the scalar model's sin(0)=0 property).

Design choice worth being explicit about: a single grayscale pixel value
has to seed D channels somehow. Here, the target phase is IDENTICAL across
all D channels (same scalar mapped the same way to every channel) -- so
omega-mixing is the ONLY source of any cross-channel differentiation. This
isolates exactly the question being tested (does cross-channel mixing add
value), rather than also testing "does giving each channel a different
view of the input help," which would be a different, separate question.
"""
import numpy as np


class VectorOscillatorField:
    def __init__(self, height: int, width: int, dims: int = 4, dt: float = 0.1,
                 k_coupling: float = 1.0, k_omega: float = 0.5, k_bias: float = 1.0,
                 seed: int = None):
        self.H, self.W, self.D = height, width, dims
        self.dt = dt
        self.k_coupling = k_coupling
        self.k_omega = k_omega
        self.k_bias = k_bias

        rng = np.random.default_rng(seed)
        init_phases = rng.uniform(0, 2 * np.pi, (height, width, dims))
        self.z = np.exp(1j * init_phases)
        self.target_z = None

        # Real, skew-symmetric cross-channel mixing matrix, matching the
        # notebook's construction exactly. Fixed at init, not learned.
        omega_init = rng.random((dims, dims))
        self.omega = omega_init - omega_init.T

    def set_input(self, image: np.ndarray, arc: float = np.pi):
        assert image.shape == (self.H, self.W)
        target_phase = image * arc
        # Same scalar target replicated identically across all D channels --
        # see module docstring for why.
        target_phase_full = np.repeat(target_phase[:, :, np.newaxis], self.D, axis=2)
        self.target_z = np.exp(1j * target_phase_full)

    def initialize_at_target(self, perturbation_std: float = 0.01, seed: int = None):
        assert self.target_z is not None, "call set_input() first"
        rng = np.random.default_rng(seed)
        target_phase = np.angle(self.target_z)
        noisy_phase = target_phase + rng.normal(0, perturbation_std, target_phase.shape)
        self.z = np.exp(1j * noisy_phase)

    def _neighbor_sum(self) -> np.ndarray:
        """Per-channel spatial neighbor sum -- channel d only sums channel d
        of its spatial neighbors, no cross-channel mixing here (that's the
        separate omega term). No zero-padding: edge/corner sites sum only
        their real neighbors (see module docstring)."""
        z = self.z
        neighbor_sum = np.zeros_like(z)
        neighbor_sum[1:, :, :]  += z[:-1, :, :]
        neighbor_sum[:-1, :, :] += z[1:, :, :]
        neighbor_sum[:, 1:, :]  += z[:, :-1, :]
        neighbor_sum[:, :-1, :] += z[:, 1:, :]
        return neighbor_sum

    def step(self) -> np.ndarray:
        neighbor_sum = self._neighbor_sum()
        spatial_term = neighbor_sum - self.z
        omega_term = np.einsum('ij,hwj->hwi', self.omega, self.z)
        bias_term = (self.target_z - self.z) if self.target_z is not None else 0.0

        delta_z = (self.k_coupling * spatial_term
                   + self.k_omega * omega_term
                   + self.k_bias * bias_term)
        new_z = self.z + self.dt * delta_z
        self.z = new_z / (np.abs(new_z) + 1e-12)  # renormalize to unit circle per channel
        return self.z

    def run(self, steps: int) -> np.ndarray:
        for _ in range(steps):
            self.step()
        return self.z
