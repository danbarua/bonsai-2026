"""
Verification of HebbianKuramotoOperator against Bronski et al. (2017),
"The stability of fixed points for a Kuramoto model with Hebbian
interactions" (Chaos 27, 053110), arXiv:1611.09941.

Unlike tests/test_hebian_kuramoto.py (which checks coherence against ad hoc
thresholds and step counts), these tests check the model against the paper's
actual equations and its central theorem: that the stability of a Hebbian
fixed point equals the sign structure of a specific graph-Laplacian-shaped
matrix (see maths.graphs.GraphLaplacian.from_bronski_stability_matrix).

This test class was added after discovering that the coupling term in both
HebbianKuramotoOperator and models.hebbian.minimalist.update_hebbian_kuramoto
had its sign backwards relative to the paper (self-minus-other instead of
other-minus-self), which silently turned attractive coupling into repulsive
coupling for positive weights. That bug was invisible at exact synchrony
(sin(0)=0 regardless of sign) and was only caught because a test introduced
small frequency differences, breaking the degeneracy.
"""
import unittest
import numpy as np

from .. import maths
from ..models import HebbianKuramotoOperator, LayeredOscillatorState, update_hebbian_kuramoto


class TestMinimalistHebbianKuramoto(unittest.TestCase):
    """
    Direct verification of models.hebbian.minimalist.update_hebbian_kuramoto,
    which had zero test coverage before this. It's meant as a stripped-down
    reference implementation of the same Bronski et al. equations as
    HebbianKuramotoOperator, so it's checked against the same equations here.

    Note this function's `frequencies` parameter is used directly (no *2*pi
    scaling), unlike HebbianKuramotoOperator which multiplies by 2*pi -- the
    two are NOT drop-in equivalent unless frequencies are pre-scaled to match.
    This is a units inconsistency between the "reference" and "full"
    implementations worth resolving, not something this test papers over.
    """

    def test_matches_paper_equations_directly(self):
        rng = np.random.default_rng(3)
        n = 5
        phases = rng.uniform(0, 2 * np.pi, size=n)
        weights = rng.uniform(-1, 1, size=(n, n))
        np.fill_diagonal(weights, 0.0)
        frequencies = rng.uniform(-0.1, 0.1, size=n)
        dt, mu, alpha = 0.001, 0.2, 0.15

        new_phases, new_weights = update_hebbian_kuramoto(
            phases, weights, frequencies, dt=dt, learning_rate=mu, decay=alpha
        )

        # Paper eq. (2): dtheta_i/dt = omega_i + sum_j gamma_ij sin(theta_j - theta_i)
        expected_coupling = np.array([
            sum(weights[i, j] * np.sin(phases[j] - phases[i]) for j in range(n) if j != i)
            for i in range(n)
        ])
        expected_phases = (phases + dt * (frequencies + expected_coupling)) % (2 * np.pi)
        np.testing.assert_allclose(new_phases, expected_phases, atol=1e-10)

        # Paper eq. (2): dgamma_ij/dt = mu*cos(theta_i - theta_j) - alpha*gamma_ij
        cos_diffs = np.cos(phases[:, np.newaxis] - phases[np.newaxis, :])
        expected_weights = weights + dt * (mu * cos_diffs - alpha * weights)
        np.fill_diagonal(expected_weights, 0.0)  # self-coupling excluded, see fix in minimalist.py
        np.testing.assert_allclose(new_weights, expected_weights, atol=1e-10)

    def test_stable_fixed_point_is_stationary(self):
        """A known stable fixed point (fully synchronized, identical frequencies)
        should leave phases and weights essentially unchanged after a step."""
        n = 3
        phases = np.zeros(n)
        frequencies = np.zeros(n)
        alpha, mu = 0.3, 1.0
        weights = mu * np.cos(phases[:, np.newaxis] - phases[np.newaxis, :]) / alpha
        np.fill_diagonal(weights, 0.0)

        new_phases, new_weights = update_hebbian_kuramoto(
            phases, weights, frequencies, dt=0.01, learning_rate=mu, decay=alpha
        )
        np.testing.assert_allclose(new_phases, phases, atol=1e-10)
        np.testing.assert_allclose(new_weights, weights, atol=1e-10)


class TestHebbianKuramotoBronskiEquations(unittest.TestCase):
    """Direct checks that the update rule matches the paper's equations (2):

        dtheta_i/dt = omega_i + sum_j gamma_ij * sin(theta_j - theta_i)
        dgamma_ij/dt = mu * cos(theta_i - theta_j) - alpha * gamma_ij

    These are the kind of tests that would have caught the coupling-sign bug
    directly, rather than only observing its downstream effect on coherence.
    """

    def test_phase_update_matches_paper_coupling_term(self):
        """One Euler step's phase change should match dt * (paper's RHS)."""
        rng = np.random.default_rng(42)
        n = 5
        phases = rng.uniform(0, 2 * np.pi, size=(1, n))
        weights = rng.uniform(-1, 1, size=(n, n))
        np.fill_diagonal(weights, 0.0)
        frequencies = np.zeros((1, n))  # isolate the coupling term; no frequency contribution
        perturbations = np.zeros((1, n))

        state = LayeredOscillatorState(
            _phases=[phases], _frequencies=[frequencies], _perturbations=[perturbations],
            _layer_names=["t"], _layer_shapes=[(1, n)]
        )
        dt = 0.001  # small dt so the Euler step approximates the true derivative closely
        op = HebbianKuramotoOperator(init_weights=[weights.copy()], dt=dt, mu=0.0, alpha=0.0)
        new_state = op.apply(state)

        # Paper's RHS: sum_j gamma_ij * sin(theta_j - theta_i)
        theta = phases.flatten()
        expected_coupling = np.array([
            sum(weights[i, j] * np.sin(theta[j] - theta[i]) for j in range(n) if j != i)
            for i in range(n)
        ])
        actual_delta = (new_state.phases[0].flatten() - theta)
        # unwrap in case of wraparound (shouldn't happen at this dt, but be safe)
        actual_delta = (actual_delta + np.pi) % (2 * np.pi) - np.pi
        expected_delta = dt * expected_coupling

        np.testing.assert_allclose(actual_delta, expected_delta, atol=1e-6,
            err_msg="Phase update does not match Bronski et al. eq. (2): "
                    "sum_j gamma_ij * sin(theta_j - theta_i). "
                    "Check the sign convention of phase_diffs.")

    def test_weight_update_matches_paper_hebbian_rule(self):
        """One Euler step's weight change should match dt * (mu*cos(dtheta) - alpha*w)."""
        rng = np.random.default_rng(7)
        n = 4
        phases = rng.uniform(0, 2 * np.pi, size=(1, n))
        weights = rng.uniform(-1, 1, size=(n, n))
        np.fill_diagonal(weights, 0.0)
        frequencies = np.zeros((1, n))
        perturbations = np.zeros((1, n))

        state = LayeredOscillatorState(
            _phases=[phases], _frequencies=[frequencies], _perturbations=[perturbations],
            _layer_names=["t"], _layer_shapes=[(1, n)]
        )
        dt, mu, alpha = 0.01, 0.37, 0.13
        op = HebbianKuramotoOperator(init_weights=[weights.copy()], dt=dt, mu=mu, alpha=alpha)
        op.apply(state)

        theta = phases.flatten()
        cos_diffs = np.cos(theta[:, np.newaxis] - theta[np.newaxis, :])  # cos is even: order doesn't matter
        expected_weights = weights + dt * (mu * cos_diffs - alpha * weights)
        np.fill_diagonal(expected_weights, 0.0)

        np.testing.assert_allclose(op.weights[0], expected_weights, atol=1e-10,
            err_msg="Weight update does not match Bronski et al. eq. (2): "
                    "dgamma_ij/dt = mu*cos(theta_i - theta_j) - alpha*gamma_ij")


class TestHebbianKuramotoBronskiStability(unittest.TestCase):
    """
    Verifies actual simulated behavior against the paper's stability theorem
    (Theorem 2.3), using an analytically-derived pair of fixed points for two
    identical-topology oscillators with slightly detuned frequencies.

    For N=2, the classical (fixed-coupling 1/(2*alpha)) Kuramoto fixed points
    satisfy sin(theta*) = alpha * delta_omega, with two branches in [0, pi]:
    the one near 0 is classically stable, the one near pi is the classical
    saddle (unstable). Per the paper's Theorem 2.3, the Hebbian fixed points
    are at HALF these angles, with the same stability classification.
    """

    def setUp(self):
        self.alpha = 0.3
        self.mu = 1.0
        self.delta_omega = 0.5
        theta_near_zero = np.arcsin(self.alpha * self.delta_omega)
        theta_near_pi = np.pi - theta_near_zero
        self.phi_stable = theta_near_zero / 2
        self.phi_unstable = theta_near_pi / 2

    def _bronski_predicts_stable(self, phi_star: float) -> bool:
        phases = np.array([0.0, phi_star])
        gl = maths.GraphLaplacian.from_bronski_stability_matrix(phases, alpha=self.alpha)
        return gl.is_bronski_stable

    def _simulate_from_perturbation(self, phi_star: float, perturbation: float,
                                     steps: int = 500, dt: float = 0.01) -> float:
        """Run the real operator from a small perturbation and return the final relative phase."""
        phases = [np.array([[0.0, phi_star + perturbation]])]
        # code scales frequencies by 2*pi (cycles -> rad/time); divide out to
        # get the intended angular-frequency difference of self.delta_omega
        frequencies = [np.array([[0.0, self.delta_omega]]) / (2 * np.pi)]
        perturbations = [np.zeros((1, 2))]
        weights0 = self.mu * np.cos(np.array([[0.0, phi_star], [-phi_star, 0.0]])) / self.alpha
        np.fill_diagonal(weights0, 0.0)

        state = LayeredOscillatorState(
            _phases=phases, _frequencies=frequencies, _perturbations=perturbations,
            _layer_names=["t"], _layer_shapes=[(1, 2)]
        )
        op = HebbianKuramotoOperator(init_weights=[weights0], dt=dt, mu=self.mu, alpha=self.alpha)
        cs = state
        for _ in range(steps):
            cs = op.apply(cs)
        p = cs.phases[0].flatten()
        return (p[1] - p[0] + np.pi) % (2 * np.pi) - np.pi

    def test_bronski_predicts_correct_branches(self):
        """Sanity check the analytic setup itself: branches should be predicted
        stable / unstable respectively before checking simulated behavior."""
        self.assertTrue(self._bronski_predicts_stable(self.phi_stable),
            "Expected the near-zero branch to be predicted stable")
        self.assertFalse(self._bronski_predicts_stable(self.phi_unstable),
            "Expected the near-pi branch to be predicted unstable")

    def test_stable_branch_recovers_from_perturbation(self):
        """A fixed point Bronski predicts stable should return to itself after a small perturbation."""
        final_phi = self._simulate_from_perturbation(self.phi_stable, perturbation=0.05)
        self.assertAlmostEqual(final_phi, self.phi_stable, places=3,
            msg="Stable branch did not recover from a small perturbation")

    def test_unstable_branch_diverges_from_perturbation(self):
        """A fixed point Bronski predicts unstable should NOT return to itself -- the
        same small perturbation that the stable branch recovers from should grow."""
        final_phi = self._simulate_from_perturbation(self.phi_unstable, perturbation=0.05)
        drift = abs((final_phi - self.phi_unstable + np.pi) % (2 * np.pi) - np.pi)
        self.assertGreater(drift, 0.5,
            msg="Unstable branch should have drifted substantially from its fixed point, "
                "matching the Bronski stability prediction")

    def test_fully_synchronized_identical_oscillators_is_stable(self):
        """Baseline sanity case: N identical oscillators, all in phase, is always
        a stable fixed point for positive coupling (the trivial classical result)."""
        n = 4
        phases_arr = np.zeros(n)
        gl = maths.GraphLaplacian.from_bronski_stability_matrix(phases_arr, alpha=self.alpha)
        self.assertTrue(gl.is_bronski_stable)

        # Confirm against real simulated dynamics too, from a small perturbation.
        phases = [np.zeros((1, n)) ]
        phases[0][0, 0] += 0.05  # perturb one oscillator slightly
        frequencies = [np.zeros((1, n))]
        perturbations = [np.zeros((1, n))]
        weights0 = self.mu * np.cos(phases_arr[:, np.newaxis] - phases_arr[np.newaxis, :]) / self.alpha
        np.fill_diagonal(weights0, 0.0)
        state = LayeredOscillatorState(
            _phases=phases, _frequencies=frequencies, _perturbations=perturbations,
            _layer_names=["t"], _layer_shapes=[(1, n)]
        )
        op = HebbianKuramotoOperator(init_weights=[weights0], dt=0.01, mu=self.mu, alpha=self.alpha)
        cs = state
        for _ in range(200):
            cs = op.apply(cs)
        coherence = op.get_delta()["mean_coherence"]
        self.assertGreater(coherence, 0.99,
            msg="Fully synchronized state should recover high coherence after a small perturbation")


if __name__ == "__main__":
    unittest.main()
