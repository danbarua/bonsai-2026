import unittest
import numpy as np
from dataclasses import dataclass, field
from unittest import mock

from models import DeluxeHebbianKuramotoOperator

# --- Dummy LayeredOscillatorState for testing ---
@dataclass
class DummyLayeredOscillatorState:
    # Each layer has:
    #   phases: an NDArray of shape (H, W, oscillator_dim)
    #   frequencies: an NDArray of shape (H, W)
    #   amplitudes: an NDArray of shape (H, W)
    phases: list[np.ndarray]
    frequencies: list[np.ndarray]
    amplitudes: list[np.ndarray]
    num_layers: int = field(init=False)
    def __post_init__(self):
        self.num_layers = len(self.phases)
    def copy(self) -> 'DummyLayeredOscillatorState':
        return DummyLayeredOscillatorState(
            phases=[p.copy() for p in self.phases],
            frequencies=[f.copy() for f in self.frequencies],
            amplitudes=[a.copy() for a in self.amplitudes],
        )
    
# --- Import the operator to test ---
# from your_module import DeluxeHebbianKuramotoOperator
# For testing, we assume DeluxeHebbianKuramotoOperator is already defined in the context.
@unittest.skip("AKOrN Deluxe not currently being worked on")
class TestDeluxeHebbianKuramotoOperator(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility.
        np.random.seed(42)
        # Create a dummy layered oscillator state with one layer.
        oscillator_dim = 4
        grid_shape = (16, 16)
        # For each layer, create a phases array of shape (16, 16, oscillator_dim)
        phases = [np.random.rand(*grid_shape, oscillator_dim) * 2 * np.pi]
        # Frequencies of shape (16, 16)
        frequencies = [np.random.uniform(0.1, 1.0, size=grid_shape)]
        # Amplitudes (all ones)
        amplitudes = [np.ones(grid_shape)]
        self.state = DummyLayeredOscillatorState(phases, frequencies, amplitudes)
        # Create an instance of the operator with default parameters.
        self.operator:DeluxeHebbianKuramotoOperator = DeluxeHebbianKuramotoOperator(
            dt=0.1,
            alpha=0.1,
            mu=0.01,
            oscillator_dim=oscillator_dim,
            grid_size=grid_shape,
            weight_symmetry=True
        )
        # Override discover_patterns with a no-op for testing.
        self.operator.discover_patterns = lambda: None

    def test_initialize_connectome(self):
        # Check that coupling matrix is correctly shaped.
        coupling = self.operator.initialize_connectome_inspired_coupling()
        expected_shape = self.operator.grid_size + self.operator.grid_size
        self.assertEqual(coupling.shape, expected_shape,
                         "Coupling matrix shape mismatch.")
        
        # Check that hub structure modifies some connections.
        coupling_hub = self.operator.add_hub_structure(coupling.copy())
        self.assertFalse(np.allclose(coupling, coupling_hub),
                         "Hub structure did not alter coupling matrix.")
        
    def test_detect_harmonic_relationships(self):
        # Set frequencies to a constant so that all ratios are 1.0.
        self.operator.freq = np.full(self.operator.grid_size, 1.0)
        resonance = self.operator.detect_harmonic_relationships()

        # For constant frequencies, the ratio is 1.0, so resonance for target 1.0
        # should be high. Expect the resonance matrix to be nonzero.
        self.assertTrue(np.mean(resonance) > 0.5,
                        "Resonance detection failed for constant frequencies.")
        
    def test_update_coupling_from_resonance(self):
        # Set W to zeros and run update_coupling_from_resonance.
        self.operator.W = np.zeros((self.operator.oscillator_dim, self.operator.oscillator_dim))
        self.operator.update_coupling_from_resonance()

        # After update, W should be nonzero and decayed.
        self.assertTrue(np.any(self.operator.W > 0),
                        "W was not updated from resonance.")
        self.assertTrue(np.all(self.operator.W <= 1),
                        "W values exceed expected bounds.")
        
    def test_phase_distance(self):
        # Create two phase arrays with a known difference.
        phase1 = np.full((16, 16, 4), np.pi / 2)
        phase2 = np.full((16, 16, 4), np.pi / 4)
        distance = self.operator.phase_distance(phase1, phase2)

        # The absolute difference is pi/4 everywhere, so mean should be pi/4.
        self.assertAlmostEqual(distance, np.pi/4, places=4,
                               msg="Phase distance calculation is incorrect.")
        
    def test_phase_distance_edge_cases(self):
        # Test phase distance with values near 0 and 2*pi
        phase1 = np.array([0.01, 6.27]) # close to 0 and 2*pi
        phase2 = np.array([0.02, 6.26])
        distance = self.operator.phase_distance(phase1, phase2)
        self.assertAlmostEqual(distance, 0.01, places=4,
                               msg="Phase distance calculation is incorrect for edge cases.")
        
    def test_detect_stable_representations(self):
        # Populate phase_history with nearly identical phases.
        stable_phase = np.full((16, 16, 4), np.pi/3)
        self.operator.phase_history.clear()
        for _ in range(self.operator.config["STABILITY_WINDOW"]):
            self.operator.phase_history.append(stable_phase)
        stable = self.operator.detect_stable_representations()
        self.assertTrue(stable, "Stable representation not detected with identical phases.")

    def test_project_to_tangent_space(self):
        # Create a dummy state vector and an update vector.
        state_vectors = np.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0]])
        update_vectors = np.array([[0.5, 0.5, 0.0, 0.0],
                                   [0.0, 0.5, 0.5, 0.0]])
        projected = self.operator.project_to_tangent_space(state_vectors, update_vectors)

        # For each row, the projected update should be orthogonal to the state vector.
        dot_products = np.sum(state_vectors * projected, axis=1)
        self.assertTrue(np.allclose(dot_products, 0, atol=1e-6),
                        
                        "Projected update is not orthogonal to state vectors.")
        
    def test_project_to_tangent_space_non_unit(self):
        # Test projection with non-unit state vectors.
        state_vectors = np.array([[2.0, 0.0, 0.0, 0.0],
                                  [0.0, 3.0, 0.0, 0.0]])
        update_vectors = np.array([[0.5, 0.5, 0.0, 0.0],
                                   [0.0, 0.5, 0.5, 0.0]])
        projected = self.operator.project_to_tangent_space(state_vectors, update_vectors)

        # For each row, the projected update should be orthogonal to the state vector.
        dot_products = np.sum(state_vectors * projected, axis=1)
        self.assertTrue(np.allclose(dot_products, 0, atol=1e-6),
                        "Projected update is not orthogonal to non-unit state vectors.")

#    @mock.patch.object(DeluxeHebbianKuramotoOperator, 'discover_patterns')
    def test_apply_operator(self): #, mock_discover_patterns):
        # Test the full apply() method on a dummy layered state.
        new_state = self.operator.apply(self.state)

        # Check that last_delta has been updated.
        self.assertIn("coherence", self.operator.last_delta,
                      "Operator did not update last_delta with coherence.")
        self.assertIn("stability", self.operator.last_delta,
                      "Operator did not update last_delta with stability metrics.")
        
        # Check that new_state is a valid copy (phases shape unchanged).
        self.assertEqual(new_state.phases[0].shape, self.state.phases[0].shape,
                         "State phase shape changed unexpectedly after apply().")
        
        # Check that coherence is in a reasonable range [0, 1]
        coherence = self.operator.last_delta.get("mean_coherence", 0)
        self.assertTrue(0 <= coherence <= 1,
                        "Mean coherence out of bounds.")
        
        # Check that discover_patterns was called.
        #mock_discover_patterns.assert_called_once()

    def test_make_skew_symmetric(self):
        # Create a simple 3D matrix (simulate batch of matrices)
        matrix = np.array([[[1, 2], [3, 4]],
                           [[5, 6], [7, 8]]])
        skew = self.operator.make_skew_symmetric(matrix)

        # Check for skew-symmetry: A^T should equal -A
        for i in range(matrix.shape[0]):
            self.assertTrue(np.allclose(skew[i].T, -skew[i], atol=1e-6),
                            "Matrix is not skew-symmetric.")
            
    def test_weight_asymmetry(self):
        # Create an operator with weight_symmetry=False
        asymmetric_operator = DeluxeHebbianKuramotoOperator(
            dt=0.1,
            alpha=0.1,
            mu=0.01,
            oscillator_dim=4,
            grid_size=(16, 16),
            weight_symmetry=False
        )
        asymmetric_operator.discover_patterns = lambda: None # Mock discover_patterns

        # Apply the operator to the state
        asymmetric_operator.apply(self.state)

        # Check that the weights are not symmetric
        self.assertFalse(np.allclose(asymmetric_operator.weights[0], asymmetric_operator.weights[0].T),
                        "Weights are symmetric when asymmetry is expected.")
        
    def test_different_grid_size(self):
        # Create an operator with a different grid size
        grid_shape = (8, 8)
        different_grid_operator = DeluxeHebbianKuramotoOperator(
            dt=0.1,
            alpha=0.1,
            mu=0.01,
            oscillator_dim=4,
            grid_size=grid_shape,
            weight_symmetry=True
        )
        different_grid_operator.discover_patterns = lambda: None # Mock discover_patterns

        # Create a dummy layered oscillator state with the new grid size
        phases = [np.random.rand(*grid_shape, self.operator.oscillator_dim) * 2 * np.pi]
        frequencies = [np.random.uniform(0.1, 1.0, size=grid_shape)]
        amplitudes = [np.ones(grid_shape)]
        different_grid_state = DummyLayeredOscillatorState(phases, frequencies, amplitudes)

        # Apply the operator to the state
        different_grid_operator.apply(different_grid_state)

        # Check that the phases shape is correct
        self.assertEqual(different_grid_state.phases[0].shape, (*grid_shape, self.operator.oscillator_dim),
                        "Phases shape is incorrect for different grid size.")

    def test_numerical_stability(self):
        # Test with very small frequency values
        small_freq_state = self.state.copy()
        small_freq_state.frequencies[0] = np.ones_like(small_freq_state.frequencies[0]) * 1e-10

        # This should run without numerical errors
        try:
            new_state = self.operator.apply(small_freq_state)

            # Check that values are finite
            self.assertTrue(np.all(np.isfinite(new_state.phases[0])),
                           "Output contains non-finite values with small inputs")
        except Exception as e:
            self.fail(f"Operator failed with small frequency values: {e}")

    def test_convergence(self):
        # Run many steps and check if coherence increases or stabilizes
        coherence_values = []
        current_state = self.state.copy()

        for _ in range(10):  # Run for 10 steps
            current_state = self.operator.apply(current_state)
            coherence_values.append(self.operator.last_delta["mean_coherence"])

        # Check if coherence has improved or stabilized
        self.assertGreaterEqual(coherence_values[-1], coherence_values[0] * 0.9,
                              "Coherence did not improve or stabilize over time")

    def test_compute_energy(self):
        # Calculate energy for a simple state
        energy = self.operator.compute_energy(self.state)

        # Energy should be a scalar
        self.assertTrue(np.isscalar(energy), "Energy should be a scalar value")

        # Create a highly coherent state
        coherent_state = self.state.copy()
        coherent_phase = np.full_like(coherent_state.phases[0], 0.5)
        coherent_state.phases[0] = coherent_phase

        # Create a random state
        random_state = self.state.copy()
        random_state.phases[0] = np.random.rand(*random_state.phases[0].shape)

        # Coherent state should have lower energy
        coherent_energy = self.operator.compute_energy(coherent_state)
        random_energy = self.operator.compute_energy(random_state)

        self.assertLess(coherent_energy, random_energy,
                       "Coherent state does not have lower energy than random state")

    def test_pattern_memory(self):
        # Create a distinct pattern
        pattern = np.zeros(self.operator.grid_size + (self.operator.oscillator_dim,))
        pattern[4:8, 4:8, :] = 1.0  # Create a square pattern

        # Store pattern
        pattern_idx = self.operator.store_new_pattern(pattern)

        # Retrieve pattern
        retrieved_pattern = self.operator.pattern_memory[pattern_idx]['representation']

        # Pattern should be accurately stored
        self.assertTrue(np.allclose(pattern, retrieved_pattern),
                       "Pattern was not accurately stored in memory")

        # Label the pattern
        self.operator.associate_label(pattern_idx, "square")

        # Find pattern by label
        found_idx = self.operator.find_pattern_by_label("square")

        self.assertEqual(pattern_idx, found_idx,
                        "Could not retrieve pattern by label")

    #@unittest.parameterize([
    #     {"grid_size": (8, 8), "oscillator_dim": 4},
    #     {"grid_size": (16, 16), "oscillator_dim": 2},
    #     {"grid_size": (32, 32), "oscillator_dim": 8}
    # ])
    def test_different_configurations(self, grid_size=(16,16), oscillator_dim=64):
        # Create an operator with the specified configuration
        operator = DeluxeHebbianKuramotoOperator(
            dt=0.1,
            alpha=0.1,
            mu=0.01,
            oscillator_dim=oscillator_dim,
            grid_size=grid_size,
            weight_symmetry=True
        )
        operator.discover_patterns = lambda: None

        # Create a dummy state with matching dimensions
        phases = [np.random.rand(*grid_size, oscillator_dim) * 2 * np.pi]
        frequencies = [np.random.uniform(0.1, 1.0, size=grid_size)]
        amplitudes = [np.ones(grid_size)]
        state = DummyLayeredOscillatorState(phases, frequencies, amplitudes)

        # Apply the operator
        new_state = operator.apply(state)

        # Check that the result has the expected shape
        self.assertEqual(new_state.phases[0].shape, (*grid_size, oscillator_dim))

if __name__ == '__main__':
    unittest.main()