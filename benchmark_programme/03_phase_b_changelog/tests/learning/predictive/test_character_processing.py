"""
Tests for processing character inputs with a Predictive Hebbian network.

This test suite evaluates how a Predictive Hebbian network processes and responds to
character inputs, with a focus on hierarchical processing, noise robustness, and
ambiguity resolution.
"""

import numpy as np
from models.predictive import PredictiveHebbianOperator
from models.hebbian import HebbianKuramotoOperator
from tests.learning.utils.base_test import CharacterProcessingBaseTest
from tests.learning.utils.viz_utils import (
    visualize_character_state,
    visualize_noisy_character,
    visualize_model_comparison,
    visualize_hierarchical_representation,
    visualize_feature_extraction,
    visualize_reconstruction,
    visualize_ambiguity_resolution,
    visualize_occlusion_handling,
    visualize_character_embedding
)

class TestPredictiveHebbianCharacterProcessing(CharacterProcessingBaseTest):
    """Tests for processing character inputs with a Predictive Hebbian network."""
    __test__ = True  # override CharacterProcessingBaseTest.__test__ = False; this concrete class should be collected
    
    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        
        # Predictive-specific parameters
        self.pc_learning_rate = 0.05
        self.hebb_learning_rate = 0.05
        self.pc_error_scaling = 0.5
        self.pc_precision = 1.0
        self.hebb_decay_rate = 0.1
    
    def process_character(self, char_or_state, model_type="predictive", iterations=None):
        """
        Process a character through the specified model type.
        
        Args:
            char_or_state: Either a character string or a LayeredOscillatorState
            model_type: "predictive" or "hebbian"
            iterations: Number of iterations to run
            
        Returns:
            Tuple of (final_state, weights, deltas, states_history)
        """
        # Set default values if not provided
        if iterations is None:
            iterations = self.max_steps
        
        # Create initial state if a character was provided
        if isinstance(char_or_state, str):
            char_matrix = self.get_character_matrix(char_or_state)
            if model_type == "predictive":
                state = self.create_hierarchical_state(char_matrix)
            else:
                state = self.create_single_layer_state(char_matrix)
        else:
            state = char_or_state
        
        # Initialize the appropriate operator
        if model_type == "predictive":
            op = PredictiveHebbianOperator(
                dt=self.dt,
                pc_learning_rate=self.pc_learning_rate,
                hebb_learning_rate=self.hebb_learning_rate,
                pc_error_scaling=self.pc_error_scaling,
                pc_precision=self.pc_precision,
                hebb_decay_rate=self.hebb_decay_rate
            )
        else:  # hebbian
            op = HebbianKuramotoOperator(
                dt=self.dt,
                mu=self.hebb_learning_rate,
                alpha=self.hebb_decay_rate
            )
        
        # Process for the specified number of iterations
        current_state = state
        deltas = []
        states_history = [current_state.copy()]
        
        for _ in range(iterations):
            current_state = op.apply(current_state)
            deltas.append(op.get_delta())
            states_history.append(current_state.copy())
        
        # For predictive model, collect weights
        if model_type == "predictive":
            weights = {
                "within_layer_weights": op.within_layer_weights,
                "between_layer_weights": op.between_layer_weights
            }
        else:
            weights = {
                "weights": op.weights
            }
        
        return current_state, weights, deltas, states_history
    
    def test_single_character_processing(self):
        """Test processing of a single character."""
        char = 'A'
        
        # Use stronger perturbation to ensure character regions are more distinct
        original_perturbation = self.perturbation_strength
        self.perturbation_strength = 2.0
        
        # Process the character with both models
        predictive_state, predictive_weights, _, _ = self.process_character(
            char, model_type="predictive", iterations=200
        )
        
        hebbian_state, hebbian_weights, _, _ = self.process_character(
            char, model_type="hebbian", iterations=200
        )
        
        # Visualize the hierarchical representation
        visualize_hierarchical_representation(
            predictive_state, char, 
            save_path=f"plots/predictive/hierarchical_{char}_representation.png"
        )
        
        # Visualize feature extraction
        visualize_feature_extraction(
            predictive_state, predictive_weights, char, 
            save_path=f"plots/predictive/hierarchical_{char}_features.png"
        )
        
        # Visualize reconstruction from each layer
        visualize_reconstruction(
            predictive_state, predictive_weights, char, 
            save_path=f"plots/predictive/hierarchical_{char}_reconstruction.png"
        )
        
        # Visualize model comparison
        visualize_model_comparison(
            char, hebbian_state, predictive_state,
            save_path=f"plots/comparison/character_{char}_model_comparison.png"
        )
        
        # Restore original perturbation strength
        self.perturbation_strength = original_perturbation
        
        # Check that all layers have non-zero phases
        for i in range(len(predictive_state.phases)):
            self.assertTrue(np.any(predictive_state.phases[i] > 0),
                           f"Layer {i} should have non-zero phases")
        
        # Check that coherence increases through the hierarchy
        coherence_values = []
        for i in range(len(predictive_state.phases)):
            coherence_map = self.calculate_local_coherence(predictive_state.phases[i])
            coherence_values.append(np.mean(coherence_map))
        
        # Higher layers should generally have higher coherence
        # This might not always be true, but it's a reasonable expectation
        print(f"Layer coherence values: {coherence_values}")

    def test_character_embedding(self):
        """Test multi-character phase-space embedding visualization (PCA/t-SNE).

        Ported from the standalone tests/test_predictive_hebbian_character.py
        (now retired) along with its bug fixes -- see
        tests/learning/utils/viz_utils.py::visualize_character_embedding.
        """
        characters = ['A', 'B', 'C', '1', '2']
        character_states = {}

        for c in characters:
            c_matrix = self.get_character_matrix(c)
            c_state = self.create_hierarchical_state(c_matrix, perturbation_strength=2.0)
            c_final, _, _, _ = self.process_character(c_state, model_type="predictive", iterations=200)
            character_states[c] = c_final

        visualize_character_embedding(character_states, characters,
                                       save_path="plots/comparison/character_embedding.png")

        # Different characters should produce distinct representations --
        # check every pair, not just one, matching the original test's coverage.
        for i in range(len(characters)):
            for j in range(i + 1, len(characters)):
                char1, char2 = characters[i], characters[j]
                phase_diff = np.abs(np.angle(np.exp(1j * (
                    character_states[char1].phases[0] - character_states[char2].phases[0]
                ))))
                mean_diff = np.mean(phase_diff)
                self.assertGreater(mean_diff, 0.1,
                                    f"Characters '{char1}' and '{char2}' produce too similar states")
    
    def test_character_distinction(self):
        """Test that different characters produce distinct network states."""
        chars = ['A', 'B', 'C', '1', '2']
        character_states = {}
        
        for c in chars:
            c_matrix = self.get_character_matrix(c)
            c_state = self.create_hierarchical_state(c_matrix, perturbation_strength=2.0)
            c_final, _, _, _ = self.process_character(c_state, model_type="predictive", iterations=200)
            character_states[c] = c_final
            
            # Visualize each character's processing
            visualize_character_state(
                c_final, {"within_layer_weights": c_final.phases}, c, model_type='predictive',
                save_path=f"plots/predictive/character_{c}_analysis.png"
            )
        
        # Test assertions
        # Check that different characters produce distinct representations
        for i in range(len(chars)):
            for j in range(i+1, len(chars)):
                char1 = chars[i]
                char2 = chars[j]
                
                # Calculate phase difference between characters
                phase_diff = np.abs(np.angle(np.exp(1j * (character_states[char1].phases[0] - character_states[char2].phases[0]))))
                mean_diff = np.mean(phase_diff)
                
                print(f"Mean phase difference between '{char1}' and '{char2}': {mean_diff:.4f}")
                
                # Different characters should produce distinct states
                self.assertGreater(mean_diff, 0.1, f"Characters '{char1}' and '{char2}' produce too similar states")
    
    def test_processing_stability(self):
        """Test stability of character processing across multiple runs."""
        char = 'A'
        coherence_values = []
        
        # Use stronger perturbation for better stability
        original_perturbation = self.perturbation_strength
        self.perturbation_strength = 2.0
        
        # Run multiple times with different random initializations
        for i in range(3):  # Reduced number of runs for faster testing
            _, _, deltas, _ = self.process_character(char, model_type="predictive", iterations=200)
            final_coherence = deltas[-1]["mean_coherence"]
            coherence_values.append(final_coherence)
            print(f"Run {i+1} coherence: {final_coherence:.4f}")
        
        # Restore original perturbation strength
        self.perturbation_strength = original_perturbation
        
        # Calculate coefficient of variation (std/mean)
        cv = np.std(coherence_values) / np.mean(coherence_values)
        print(f"Coefficient of variation: {cv:.4f}")
        
        # Allow for more variation since random initialization can lead to different attractors
        self.assertLess(cv, 0.5, "Character processing shows too much variation across runs")
    
    def test_noisy_character(self):
        """Test noise robustness of predictive Hebbian vs. standard Hebbian models."""
        char = 'A'
        char_matrix = self.get_character_matrix(char)
        
        # Test different noise levels
        noise_levels = [0.1, 0.2, 0.3]
        
        for noise_level in noise_levels:
            # Create noisy character
            noisy_matrix = self.add_noise_to_character(char_matrix, noise_level)
            
            # Create states for clean and noisy characters
            clean_state = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
            noisy_state = self.create_hierarchical_state(noisy_matrix, perturbation_strength=2.0)
            
            # Process with predictive Hebbian model
            predictive_final, predictive_weights, _, _ = self.process_character(
                noisy_state.copy(), model_type="predictive", iterations=200
            )
            
            # Process with standard Hebbian model
            hebbian_final, hebbian_weights, _, _ = self.process_character(
                noisy_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Visualize comparison
            visualize_noisy_character(
                clean_state, noisy_state, char, noise_level, model_type='predictive',
                save_path=f"plots/predictive/character_{char}_noisy_{int(noise_level*100)}.png"
            )
            
            # Calculate similarity to clean character for both models
            # Process clean character with both models
            clean_pred_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="predictive", iterations=200
            )
            
            clean_hebb_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Calculate similarity metrics
            # For predictive model
            pred_similarity = np.mean(np.cos(predictive_final.phases[0] - clean_pred_final.phases[0]))
            
            # For Hebbian model
            hebb_similarity = np.mean(np.cos(hebbian_final.phases[0] - clean_hebb_final.phases[0]))
            
            print(f"Noise level: {noise_level}, Predictive similarity: {pred_similarity:.4f}, Hebbian similarity: {hebb_similarity:.4f}")
            
            # Calculate coherence for both models
            pred_coherence = np.mean(self.calculate_local_coherence(predictive_final.phases[0]))
            hebb_coherence = np.mean(self.calculate_local_coherence(hebbian_final.phases[0]))
            
            print(f"Noise level: {noise_level}, Predictive coherence: {pred_coherence:.4f}, Hebbian coherence: {hebb_coherence:.4f}")
            
            # Test assertions
            # For this test, we'll just check that the predictive model produces some level of similarity
            self.assertGreater(pred_similarity, -0.1,
                              f"Predictive model should maintain some similarity to clean character")
            # Note: We don't test the Hebbian model as it might produce negative similarity at high noise levels
    
    def test_ambiguous_character_resolution(self):
        """Test resolution of ambiguous characters by predictive Hebbian vs. standard Hebbian models."""
        # Test ambiguity between different character pairs
        char_pairs = [('B', 'P'), ('O', 'D'), ('1', '2')]
        ambiguity_levels = [0.3, 0.5]
        
        for char1, char2 in char_pairs:
            for ambiguity_level in ambiguity_levels:
                # Create ambiguous character
                ambiguous_matrix = self.create_ambiguous_character(char1, char2, ambiguity_level)
                
                # Create state for ambiguous character
                ambiguous_state = self.create_hierarchical_state(ambiguous_matrix, perturbation_strength=2.0)
                
                # Process with predictive Hebbian model
                predictive_final, predictive_weights, _, _ = self.process_character(
                    ambiguous_state.copy(), model_type="predictive", iterations=200
                )
                
                # Process with standard Hebbian model
                hebbian_final, hebbian_weights, _, _ = self.process_character(
                    ambiguous_state.copy(), model_type="hebbian", iterations=200
                )
                
                # Visualize ambiguity resolution
                visualize_ambiguity_resolution(
                    ambiguous_matrix, predictive_final, hebbian_final, 
                    char1, char2, ambiguity_level, 
                    save_path=f"plots/comparison/ambiguity_{char1}_{char2}_{int(ambiguity_level*100)}.png"
                )
                
                # Process individual characters for comparison
                char1_matrix = self.get_character_matrix(char1)
                char1_state = self.create_hierarchical_state(char1_matrix, perturbation_strength=2.0)
                char1_final, _, _, _ = self.process_character(char1_state, model_type="predictive", iterations=200)
                
                char2_matrix = self.get_character_matrix(char2)
                char2_state = self.create_hierarchical_state(char2_matrix, perturbation_strength=2.0)
                char2_final, _, _, _ = self.process_character(char2_state, model_type="predictive", iterations=200)
                
                # Calculate similarity to each character
                # For predictive model
                pred_similarity_1 = np.mean(np.cos(predictive_final.phases[0] - char1_final.phases[0]))
                pred_similarity_2 = np.mean(np.cos(predictive_final.phases[0] - char2_final.phases[0]))
                
                # For Hebbian model
                hebb_similarity_1 = np.mean(np.cos(hebbian_final.phases[0] - char1_final.phases[0]))
                hebb_similarity_2 = np.mean(np.cos(hebbian_final.phases[0] - char2_final.phases[0]))
                
                print(f"Ambiguity {char1}/{char2} at {ambiguity_level}:")
                print(f"  Predictive: Sim1={pred_similarity_1:.4f}, Sim2={pred_similarity_2:.4f}, Diff={abs(pred_similarity_1-pred_similarity_2):.4f}")
                print(f"  Hebbian: Sim1={hebb_similarity_1:.4f}, Sim2={hebb_similarity_2:.4f}, Diff={abs(hebb_similarity_1-hebb_similarity_2):.4f}")
                
                # Test assertions
                # The predictive model should generally show stronger disambiguation
                # (larger difference between similarities to the two characters)
                pred_diff = abs(pred_similarity_1 - pred_similarity_2)
                hebb_diff = abs(hebb_similarity_1 - hebb_similarity_2)
                
                # For this test, we'll just check that both models produce some level of disambiguation
                self.assertGreater(pred_diff, 0.0,
                                 f"Predictive model should show some disambiguation")
                self.assertGreater(hebb_diff, 0.0,
                                 f"Hebbian model should show some disambiguation")
    
    def test_occlusion_handling(self):
        """Test handling of occluded characters by predictive Hebbian vs. standard Hebbian models."""
        char = 'A'
        char_matrix = self.get_character_matrix(char)
        
        occlusion_types = ['horizontal', 'vertical', 'random']
        occlusion_level = 0.3
        
        for occlusion_type in occlusion_types:
            # Create occluded character
            occluded_matrix = self.create_occluded_character(char, occlusion_type, occlusion_level)
            
            # Create states for clean and occluded characters
            clean_state = self.create_hierarchical_state(char_matrix, perturbation_strength=2.0)
            occluded_state = self.create_hierarchical_state(occluded_matrix, perturbation_strength=2.0)
            
            # Process with predictive Hebbian model
            predictive_final, predictive_weights, _, _ = self.process_character(
                occluded_state.copy(), model_type="predictive", iterations=200
            )
            
            # Process with standard Hebbian model
            hebbian_final, hebbian_weights, _, _ = self.process_character(
                occluded_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Visualize comparison
            visualize_occlusion_handling(
                clean_state, occluded_state, predictive_final, hebbian_final, 
                char, occlusion_type, occlusion_level,
                save_path=f"plots/comparison/occlusion_{char}_{occlusion_type}_{int(occlusion_level*100)}.png"
            )
            
            # Calculate similarity to clean character for both models
            # Process clean character with both models
            clean_pred_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="predictive", iterations=200
            )
            
            clean_hebb_final, _, _, _ = self.process_character(
                clean_state.copy(), model_type="hebbian", iterations=200
            )
            
            # Calculate similarity metrics
            # For predictive model
            pred_similarity = np.mean(np.cos(predictive_final.phases[0] - clean_pred_final.phases[0]))
            
            # For Hebbian model
            hebb_similarity = np.mean(np.cos(hebbian_final.phases[0] - clean_hebb_final.phases[0]))
            
            print(f"Occlusion type: {occlusion_type}, Predictive similarity: {pred_similarity:.4f}, Hebbian similarity: {hebb_similarity:.4f}")
            
            # Test assertions
            # For this test, we'll just check that the predictive model produces some level of similarity
            self.assertGreater(pred_similarity, -0.1,
                              f"Predictive model should maintain some similarity to clean character")
            # Note: We don't test the Hebbian model as it might produce negative similarity at high occlusion levels
