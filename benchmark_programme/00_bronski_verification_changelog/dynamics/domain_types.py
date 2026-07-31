from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, NewType, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ..maths import FrequencyHz

# Create semantic newtypes instead of just type aliases
Radians = NewType('Radians', float)
TimeStep = NewType('TimeStep', float)
LearningRate = NewType('LearningRate', float)
OscillatorCount = NewType('OscillatorCount', int)

# Use annotated types for additional information and constraints
PositiveFloat = Annotated[float, "Value must be > 0"]
UnitInterval = Annotated[float, "Value must be between 0 and 1"]
PhaseAngle = Annotated[Radians, "Value should be in [0, 2π)"]

# Create semantic data structures
@dataclass(frozen=True)
class OscillatorParameters:
    """Parameters for oscillator dynamics"""
    natural_frequency: FrequencyHz  # in Hz
    coupling_strength: PositiveFloat
    decay_rate: PositiveFloat
    
@dataclass
class OscillatorState:
    """The current state of oscillators in the network"""
    phases: NDArray[np.float64]  # Shape: (N,)
    amplitudes: NDArray[np.float64]  # Shape: (N,)
    
    def __post_init__(self):
        assert self.phases.shape == self.amplitudes.shape, "Phases and amplitudes must have same shape"
    
    @property
    def oscillator_count(self) -> OscillatorCount:
        return OscillatorCount(len(self.phases))

# Abstract base class for oscillator dynamics
class OscillatorDynamics(ABC):
    @abstractmethod
    def update(self, state: OscillatorState, dt: TimeStep) -> OscillatorState:
        """Update oscillator state based on dynamics"""
        pass
    
    @abstractmethod
    def apply_hebbian_learning(self, 
                            state: OscillatorState, 
                            weights: NDArray[np.float64], 
                            learning_rate: LearningRate) -> NDArray[np.float64]:
        """Apply Hebbian learning to update weights"""
        pass

# Implementation example
class KuramotoOscillators(OscillatorDynamics):
    """
    Kuramoto model of coupled oscillators.
    
    This model implements phase-coupled oscillators where each oscillator's
    phase evolution depends on its natural frequency and the influence of
    other oscillators according to the coupling weights.
    
    Key equations:
    dθ_i/dt = ω_i + K/N * Σ_j w_ij * sin(θ_j - θ_i)
    
    References:
    - Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. 
    """
    
    def __init__(self, 
                natural_frequencies: NDArray[np.float64],
                coupling_strength: PositiveFloat = 1.0) -> None:
        
        self.natural_frequencies = natural_frequencies
        self.coupling_strength = coupling_strength
        self._oscillator_count = OscillatorCount(len(natural_frequencies))
        
        # Initialize weights to ensure zero self-coupling (diagonal is zero)
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights with zero diagonal to prevent self-coupling"""
        self._weights = np.ones((self._oscillator_count, self._oscillator_count))
        np.fill_diagonal(self._weights, 0.0)  # Explicit zero diagonal
    
    @property
    def weights(self) -> NDArray[np.float64]:
        """Get the current weight matrix (read-only view)"""
        return self._weights.copy()
    
    def update(self, state: OscillatorState, dt: TimeStep) -> OscillatorState:
        """
        Update oscillator phases based on Kuramoto dynamics.
        
        Args:
            state: Current oscillator state
            dt: Time step for integration
            
        Returns:
            Updated oscillator state
        """
        assert state.oscillator_count == self._oscillator_count, "State size must match oscillator count"
        
        # Calculate phase differences matrix
        phase_diffs = state.phases[:, np.newaxis] - state.phases
        
        # Calculate coupling term
        coupling = self.coupling_strength * np.mean(
            self._weights * np.sin(phase_diffs), axis=1
        )
        
        # Update phases
        new_phases = (state.phases + dt * (self.natural_frequencies + coupling)) % (2 * np.pi)
        
        # Keep same amplitudes for basic Kuramoto model
        return OscillatorState(phases=new_phases, amplitudes=state.amplitudes.copy())
    
    def apply_hebbian_learning(self, state: OscillatorState, 
                              learning_rate: LearningRate = LearningRate(0.01)) -> None:
        """
        Apply Hebbian learning rule to update coupling weights.
        
        The rule strengthens connections between oscillators with similar phases
        and weakens connections between oscillators with dissimilar phases.
        
        Args:
            state: Current oscillator state
            learning_rate: Controls the rate of weight updates
        """
        # Phase similarity matrix (closer to 1 when phases are similar)
        phase_diffs = state.phases[:, np.newaxis] - state.phases
        phase_similarity = np.cos(phase_diffs)  # 1 when phases identical, -1 when opposite
        
        # Update weights based on phase similarity (Hebbian update)
        dw = learning_rate * phase_similarity
        
        # Apply update
        self._weights += dw
        
        # Ensure weights stay in reasonable range
        self._weights = np.clip(self._weights, 0, 1)
        
        # Maintain zero diagonal (no self-coupling)
        np.fill_diagonal(self._weights, 0.0)

# Example usage in test
def test_zero_diagonal():
    """Test that oscillators don't couple with themselves (zero diagonal in weights)"""
    # Setup
    n_oscillators = 10
    natural_frequencies = np.random.uniform(0.5, 2.0, n_oscillators)
    model = KuramotoOscillators(natural_frequencies)
    
    # Verify diagonal is zero
    weights = model.weights
    diagonal_elements = np.diag(weights)
    
    # Assert all diagonal elements are zero
    assert np.all(diagonal_elements == 0), "Diagonal elements should be zero to prevent self-coupling"

if (__name__ == "__main__"):
    test_zero_diagonal()