"""
Base test class for learning tests.

This module provides a base test class with common functionality for
testing oscillator-based learning models.
"""

import unittest
import numpy as np
from .character_utils import (
    get_character_matrix, 
    add_noise_to_character, 
    create_ambiguous_character, 
    create_occluded_character,
    create_single_layer_state,
    create_hierarchical_state,
    calculate_local_coherence
)

class CharacterProcessingBaseTest(unittest.TestCase):
    """
    Base class for character processing tests with common functionality.

    NOT meant to be run directly -- its test_* methods are abstract stubs
    that raise NotImplementedError, meant to be overridden by concrete
    subclasses. `__test__ = False` tells pytest not to collect this class
    itself (only unittest.TestCase subclasses that don't override this back
    to True get collected); without it, pytest ran this base class's own
    stub methods as if they were real tests, and they always failed --
    12 permanent, meaningless failures across the files that import this,
    regardless of whether the actual models under test were correct.

    Concrete subclasses must set `__test__ = True` to be collected again,
    since this attribute is inherited via the class's MRO otherwise.
    """
    __test__ = False
    
    def setUp(self):
        """Set up common test parameters"""
        # Default parameters for character processing
        self.dt = 0.01
        self.perturbation_strength = 1.0
        self.max_steps = 1000
        self.convergence_threshold = 1e-4
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def get_character_matrix(self, char):
        """Return an 8x12 binary matrix representing the given character."""
        return get_character_matrix(char)
    
    def add_noise_to_character(self, char_matrix, noise_level=0.1):
        """Add random noise to a character matrix."""
        return add_noise_to_character(char_matrix, noise_level)
    
    def create_ambiguous_character(self, char1, char2, ambiguity_level=0.5):
        """Create an ambiguous character by blending two characters."""
        return create_ambiguous_character(char1, char2, ambiguity_level)
    
    def create_occluded_character(self, char, occlusion_type='horizontal', occlusion_level=0.3):
        """Create a partially occluded character."""
        return create_occluded_character(char, occlusion_type, occlusion_level)
    
    def create_single_layer_state(self, char_matrix, perturbation_strength=None):
        """Create a single-layer oscillator state from a character matrix."""
        if perturbation_strength is None:
            perturbation_strength = self.perturbation_strength
        return create_single_layer_state(char_matrix, perturbation_strength)
    
    def create_hierarchical_state(self, input_matrix, layer_shapes=None, perturbation_strength=None):
        """Create a hierarchical LayeredOscillatorState from an input matrix."""
        if perturbation_strength is None:
            perturbation_strength = self.perturbation_strength
        return create_hierarchical_state(input_matrix, layer_shapes, perturbation_strength)
    
    def calculate_local_coherence(self, phase_data):
        """Calculate local phase coherence map."""
        return calculate_local_coherence(phase_data)
    
    def process_character(self, *args, **kwargs):
        """
        Process a character through the model.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_character")
    
    def test_single_character_processing(self):
        """
        Test processing of a single character.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement test_single_character_processing")
    
    def test_character_distinction(self):
        """
        Test that different characters produce distinct network states.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement test_character_distinction")
    
    def test_processing_stability(self):
        """
        Test stability of character processing across multiple runs.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement test_processing_stability")
    
    def test_noisy_character(self):
        """
        Test processing of a noisy character.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement test_noisy_character")
