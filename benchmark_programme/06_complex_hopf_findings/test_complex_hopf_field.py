"""
Verification test suite for ComplexLocalOscillatorField -- formalizing the
ad-hoc diagnostics run while building it into real, checked-in tests, same
discipline applied to the rest of Bonsai (test_hebbian_kuramoto_bronski.py
verified HebbianKuramotoOperator against Bronski et al.'s actual equations;
this verifies ComplexLocalOscillatorField against the Hopf/Stuart-Landau
theory it's built on, plus the coupling-strength tradeoff found empirically
while characterizing it).
"""
import unittest
import numpy as np
from complex_hopf_field import ComplexLocalOscillatorField


class TestIntrinsicHopfDynamics(unittest.TestCase):
    """Isolated single-oscillator checks -- no coupling possible on a 1x1
    grid, no input bias -- pure intrinsic Hopf (Stuart-Landau) dynamics
    against the theory: dz/dt = z*(lambda + i*omega - |z|^2) has a stable
    limit cycle at |z| = sqrt(lambda), with phase advancing at rate omega."""

    def test_amplitude_converges_to_sqrt_lambda(self):
        hopf_lambda = 1.0
        field = ComplexLocalOscillatorField(1, 1, dt=0.02, hopf_lambda=hopf_lambda,
                                             hopf_omega=1.0, seed=0)
        amplitudes = []
        for _ in range(500):
            field.step()
            amplitudes.append(np.abs(field.z[0, 0]))

        expected = np.sqrt(hopf_lambda)
        late_mean = np.mean(amplitudes[-50:])
        late_std = np.std(amplitudes[-50:])
        self.assertAlmostEqual(late_mean, expected, delta=0.01,
            msg="Isolated Hopf oscillator should settle to |z|=sqrt(lambda)")
        self.assertLess(late_std, 1e-4,
            msg="Amplitude should be essentially constant once on the limit cycle")

    def test_amplitude_converges_for_different_lambda(self):
        """Same check at a different lambda, confirming the sqrt(lambda)
        relationship isn't a coincidence of lambda=1. Tolerance is looser
        than the lambda=1 case: Euler discretization error is genuinely
        larger at smaller lambda (measured: ~0.5% relative error at
        lambda=1.0, ~4.4% at lambda=0.25) -- a real property of the
        integration, not a bug to paper over with a stricter delta."""
        for hopf_lambda in [0.25, 2.0]:
            field = ComplexLocalOscillatorField(1, 1, dt=0.02, hopf_lambda=hopf_lambda,
                                                 hopf_omega=1.0, seed=0)
            for _ in range(500):
                field.step()
            self.assertAlmostEqual(np.abs(field.z[0, 0]), np.sqrt(hopf_lambda), delta=0.03)

    def test_phase_rotates_at_natural_frequency(self):
        hopf_omega = 1.0
        field = ComplexLocalOscillatorField(1, 1, dt=0.02, hopf_lambda=1.0,
                                             hopf_omega=hopf_omega, seed=0)
        phases = []
        for _ in range(500):
            field.step()
            phases.append(np.angle(field.z[0, 0]))

        phases = np.unwrap(phases)
        late_phases = phases[300:]
        late_times = np.arange(300, 500) * field.dt
        measured_rate = np.polyfit(late_times, late_phases, 1)[0]
        self.assertAlmostEqual(measured_rate, hopf_omega, delta=0.01,
            msg="Phase should advance at rate omega once settled on the limit cycle")


class TestClosedLoopBiasTracksIntensity(unittest.TestCase):
    """Isolated (no coupling) check that the closed-loop bias term makes
    final amplitude track input intensity monotonically -- the whole
    motivation for using complex-valued (not phase-only) oscillators."""

    def test_amplitude_monotonic_in_isolation(self):
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
        final_amps = []
        for intensity in intensities:
            field = ComplexLocalOscillatorField(1, 1, dt=0.02, hopf_lambda=1.0,
                                                 hopf_omega=1.0, k_bias=1.0, power=1.0, seed=0)
            field.set_input(np.array([[intensity]]), arc=np.pi, amp_scale=1.0)
            for _ in range(300):
                field.step()
            final_amps.append(np.abs(field.z[0, 0]))

        for i in range(len(final_amps) - 1):
            self.assertLess(final_amps[i], final_amps[i + 1],
                msg=f"Amplitude should increase monotonically with intensity "
                    f"(failed between intensity={intensities[i]} and {intensities[i+1]})")


class TestCouplingStrengthAmplitudeFidelityTradeoff(unittest.TestCase):
    """The real, characterized tradeoff found while building this: local
    power-coupling, once neighbors interact, degrades and eventually
    INVERTS the amplitude-intensity correlation that holds perfectly in
    isolation -- a population-level effect (MNIST is ~80% background, whose
    numerical majority mutually reinforces via coupling, diluting/inverting
    the minority ink signal). This is a real design tradeoff, not a bug;
    this test protects the empirically-found safe operating range from
    silently regressing.
    """

    def setUp(self):
        from mnist_loader import load_idx_images
        X_train = load_idx_images("mnist_data/train-images.idx3-ubyte")
        self.image = X_train[0].astype(np.float64) / 255.0

    def _correlation_at_coupling_strength(self, w):
        field = ComplexLocalOscillatorField(28, 28, dt=0.02, hopf_lambda=1.0, hopf_omega=1.0,
                                             k_bias=1.0, power=1.0, w_vertical=w, w_horizontal=w, seed=0)
        field.set_input(self.image, arc=np.pi, amp_scale=1.0)
        for _ in range(300):
            field.step()
        final_amp = np.abs(field.z).flatten()
        pixel_flat = self.image.flatten()
        return np.corrcoef(pixel_flat, final_amp)[0, 1]

    def test_no_coupling_preserves_near_perfect_correlation(self):
        corr = self._correlation_at_coupling_strength(0.0)
        self.assertGreater(corr, 0.95,
            msg="With zero coupling, amplitude should track intensity almost perfectly (matches isolated case)")

    def test_safe_operating_range_preserves_strong_correlation(self):
        """w=0.08 is the parameter chosen for the encoder going forward --
        protect that choice landing in a genuinely safe range."""
        corr = self._correlation_at_coupling_strength(0.08)
        self.assertGreater(corr, 0.85,
            msg="Chosen coupling strength (0.08) should preserve strong amplitude-intensity correlation")

    def test_strong_coupling_inverts_correlation(self):
        """Confirms the tradeoff is real and monotonic -- protects against
        silently losing this characterization if the dynamics ever change."""
        corr = self._correlation_at_coupling_strength(0.3)
        self.assertLess(corr, 0.0,
            msg="Strong coupling (0.3) should invert the amplitude-intensity correlation, as characterized")

    def test_correlation_degrades_monotonically_with_coupling_strength(self):
        weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
        correlations = [self._correlation_at_coupling_strength(w) for w in weights]
        for i in range(len(correlations) - 1):
            self.assertGreaterEqual(correlations[i], correlations[i + 1] - 0.05,
                msg=f"Correlation should degrade roughly monotonically as coupling strength increases "
                    f"(w={weights[i]} -> w={weights[i+1]})")


class TestOjaHebbianTraining(unittest.TestCase):
    """Verifies the Oja-normalized population Hebbian training converges to
    a coupling strength meaningfully different from its starting point --
    and, critically, far outside the "safe" range characterized above
    (|w|~0.08 preserves amplitude-intensity correlation). This was an
    important, honest finding: the Hebbian rule's natural fixed point
    (|w|~1.0) violates that heuristic, and testing directly (not assuming)
    showed it's a BETTER representation for classification, at least for
    KNN -- see TestComplexEdgeEncoderClassification below."""

    def test_oja_training_converges_far_from_safe_range(self):
        from mnist_loader import load_idx_images
        from complex_hebbian_training import train_population_weights_oja

        X_train = load_idx_images("mnist_data/train-images.idx3-ubyte")
        images = X_train[:20].astype(np.float64) / 255.0  # unlabeled, no y_train used

        w_v, w_h = train_population_weights_oja(images, steps_per_image=300)

        # The "safe" range characterized above was |w| in roughly [0.05, 0.15].
        # Confirms this doesn't just happen to land there by coincidence.
        self.assertGreater(abs(w_v), 0.5,
            msg="Oja-trained w_vertical should converge far outside the safe range")
        self.assertGreater(abs(w_h), 0.5,
            msg="Oja-trained w_horizontal should converge far outside the safe range")


class TestComplexEdgeEncoderClassification(unittest.TestCase):
    """Small, fast smoke tests protecting the finding that actually held up
    at confident scale (200 test images, n=10 -- see
    bonsai_complex_hopf_findings.md for the full numbers). Deliberately
    does NOT assert a NearestCentroid comparison: that one REVERSED
    direction between the small-scale (n=5, 50 images) and confident-scale
    (n=10, 200 images) tests, so asserting either direction here would bake
    in an unreliable, scale-dependent result. The KNN finding held up and
    strengthened at scale, so it's the one worth protecting against
    regressions.
    """

    def setUp(self):
        from mnist_loader import load_idx_images, load_idx_labels
        from few_shot_harness import stratified_few_shot_sample
        X_train = load_idx_images("mnist_data/train-images.idx3-ubyte")
        y_train = load_idx_labels("mnist_data/train-labels.idx1-ubyte")
        X_test = load_idx_images("mnist_data/t10k-images.idx3-ubyte")
        y_test = load_idx_labels("mnist_data/t10k-labels.idx1-ubyte")
        X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
        X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64) / 255.0
        self.X_test_sub, self.y_test_sub = stratified_few_shot_sample(X_test_flat, y_test, 5, seed=999)
        self.X_train_flat, self.y_train = X_train_flat, y_train

    def test_oja_trained_knn_beats_safe_range_knn(self):
        """The one comparison confirmed robust at both small and confident
        scale (n=5: 0.48 vs 0.15; n=10/200-images: 0.595 vs 0.18) -- fast
        smoke-test version at small scale for regular test runs."""
        from complex_edge_encoder import complex_edge_encode
        from few_shot_harness import stratified_few_shot_sample
        from sklearn.neighbors import KNeighborsClassifier

        oja_w_v, oja_w_h = 1.0011 - 0.0063j, 0.9996 - 0.0025j
        X_sub, y_sub = stratified_few_shot_sample(self.X_train_flat, self.y_train, 5, seed=42)

        safe_features = complex_edge_encode(X_sub, w_vertical=0.08, w_horizontal=0.08)
        oja_features = complex_edge_encode(X_sub, w_vertical=oja_w_v, w_horizontal=oja_w_h)
        safe_test = complex_edge_encode(self.X_test_sub, w_vertical=0.08, w_horizontal=0.08)
        oja_test = complex_edge_encode(self.X_test_sub, w_vertical=oja_w_v, w_horizontal=oja_w_h)

        safe_clf = KNeighborsClassifier(n_neighbors=1).fit(safe_features, y_sub)
        oja_clf = KNeighborsClassifier(n_neighbors=1).fit(oja_features, y_sub)

        safe_acc = np.mean(safe_clf.predict(safe_test) == self.y_test_sub)
        oja_acc = np.mean(oja_clf.predict(oja_test) == self.y_test_sub)

        self.assertGreater(oja_acc, safe_acc,
            msg="Oja-trained coupling should beat the 'safe-range' coupling for KNN -- "
                "confirmed at both small and confident scale, see findings doc")



if __name__ == "__main__":
    unittest.main()
