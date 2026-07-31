from dataclasses import dataclass, field
from typing import Any, Generic, Optional, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from .domain_types import TimeStep


# --- Mock LayeredOscillatorState for testing ---
@dataclass
class LayeredOscillatorState:
    """Test version of LayeredOscillatorState"""
    _phases: list[NDArray[np.float64]]
    _frequencies: list[NDArray[np.float64]]
    _perturbations: list[NDArray[np.float64]]
    _layer_names: list[str]
    _layer_shapes: list[tuple[int, ...]]
    
    @property
    def phases(self) -> list[NDArray[np.float64]]:
        return self._phases
    
    @property
    def frequencies(self) -> list[NDArray[np.float64]]:
        return self._frequencies
        
    @property
    def perturbations(self) -> list[NDArray[np.float64]]:
        return self._perturbations
    
    @property
    def layer_names(self) -> list[str]:
        return self._layer_names
    
    @property
    def layer_shapes(self) -> list[tuple[int, ...]]:
        return self._layer_shapes
    
    @property
    def num_layers(self) -> int:
        return len(self._phases)
    
    def copy(self) -> 'LayeredOscillatorState':
        return LayeredOscillatorState(
            _phases=[phase.copy() for phase in self._phases],
            _frequencies=[freq.copy() for freq in self._frequencies],
            _perturbations=[pert.copy() for pert in self._perturbations],
            _layer_names=self._layer_names.copy(),
            _layer_shapes=self._layer_shapes.copy()
        )

# --- Protocol and Implementation Classes ---
S = TypeVar('S')

class StateMutation(Protocol, Generic[S]):
    def apply(self, state: S, dt:TimeStep = 0.01) -> S: ...
    def get_delta(self) -> dict[str, Any]: ...
