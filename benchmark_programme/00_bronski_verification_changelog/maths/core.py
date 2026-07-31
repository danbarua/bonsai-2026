import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Generic, List, NewType, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray


# Base dimension types to avoid confusion
class Dimension(Enum):
    TIME = auto()
    FREQUENCY = auto()
    SPACE = auto()
    GRAPH = auto()

# Type variable for shape parameters
N = TypeVar('N', bound=int)  # Number of nodes/data points

# === Frequency Types ===

@dataclass(frozen=True)
class FrequencyHz:
    """Frequency in Hertz (cycles per second)"""
    value: float
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Frequency must be non-negative, got {self.value}")
    
    def to_rads(self) -> 'FrequencyRads':
        """Convert frequency from Hz to radians per second"""
        return FrequencyRads(self.value * 2 * math.pi)
    
    def __mul__(self, other: float) -> 'FrequencyHz':
        return FrequencyHz(self.value * other)
    
    def __add__(self, other: 'FrequencyHz') -> 'FrequencyHz':
        return FrequencyHz(self.value + other.value)

@dataclass(frozen=True)
class FrequencyRads:
    """Frequency in radians per second"""
    value: float
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError(f"Frequency must be non-negative, got {self.value}")
    
    def to_hz(self) -> FrequencyHz:
        """Convert frequency from radians per second to Hz"""
        return FrequencyHz(self.value / (2 * math.pi))
    
    def __mul__(self, other: float) -> 'FrequencyRads':
        return FrequencyRads(self.value * other)
    
    def __add__(self, other: 'FrequencyRads') -> 'FrequencyRads':
        return FrequencyRads(self.value + other.value)

@dataclass(frozen=True)
class FrequencyBand:
    """A range of frequencies, like EEG bands (theta, alpha, beta, etc.)"""
    name: str
    min_freq: FrequencyHz
    max_freq: FrequencyHz
    
    def __post_init__(self):
        if self.min_freq.value >= self.max_freq.value:
            raise ValueError("Minimum frequency must be less than maximum frequency")
    
    def contains(self, freq: FrequencyHz) -> bool:
        """Check if the given frequency is within this band"""
        return self.min_freq.value <= freq.value <= self.max_freq.value
    
    @property
    def center_frequency(self) -> FrequencyHz:
        """Get the center frequency of the band"""
        return FrequencyHz((self.min_freq.value + self.max_freq.value) / 2)

# Common EEG frequency bands
DELTA_BAND = FrequencyBand("Delta", FrequencyHz(0.5), FrequencyHz(4))
THETA_BAND = FrequencyBand("Theta", FrequencyHz(4), FrequencyHz(8))
ALPHA_BAND = FrequencyBand("Alpha", FrequencyHz(8), FrequencyHz(13))
BETA_BAND = FrequencyBand("Beta", FrequencyHz(13), FrequencyHz(30))
GAMMA_BAND = FrequencyBand("Gamma", FrequencyHz(30), FrequencyHz(100))

# === Phase Types ===

@dataclass(frozen=True)
class Phase:
    """Angular phase in radians, constrained to [0, 2π)"""
    value: float
    
    def __post_init__(self):
        # Normalize to [0, 2π)
        object.__setattr__(self, 'value', self.value % (2 * math.pi))
    
    def __add__(self, other: 'Phase') -> 'Phase':
        return Phase(self.value + other.value)
    
    def __sub__(self, other: 'Phase') -> 'Phase':
        return Phase(self.value - other.value)
    
    def circular_distance(self, other: 'Phase') -> float:
        """Compute the shortest angular distance between two phases"""
        diff = abs((self.value - other.value) % (2 * math.pi))
        return min(diff, 2 * math.pi - diff)
    
    @staticmethod
    def from_complex(z: complex) -> 'Phase':
        """Create phase from a complex number"""
        return Phase(math.atan2(z.imag, z.real) % (2 * math.pi))
    
    def to_complex(self) -> complex:
        """Convert to complex number on the unit circle"""
        return complex(math.cos(self.value), math.sin(self.value))

@dataclass
class PhaseVector(Generic[N]):
    """A vector of phases for N oscillators"""
    values: NDArray[np.float64]
    
    def __post_init__(self):
        # Normalize all phases to [0, 2π)
        self.values = self.values % (2 * math.pi)
    
    def __array__(self, dtype=None):
        """NumPy array interface"""
        return np.asarray(self.values, dtype=dtype)
    
    def __getitem__(self, idx) -> Phase:
        """Access individual phases"""
        return Phase(self.values[idx])
    
    def __len__(self) -> int:
        return len(self.values)
    
    @property
    def synchronization_order_parameter(self) -> complex:
        """
        Compute the Kuramoto order parameter r*exp(i*ψ)
        
        Returns a complex number where:
        - Magnitude (r) indicates synchronization (0 = no sync, 1 = perfect sync)
        - Angle (ψ) indicates the mean phase
        """
        return np.mean(np.exp(1j * self.values))
    
    @property
    def synchronization_degree(self) -> float:
        """The degree of synchronization (0 to 1)"""
        return abs(self.synchronization_order_parameter)
    
    @property
    def mean_phase(self) -> Phase:
        """The mean phase of all oscillators"""
        return Phase.from_complex(self.synchronization_order_parameter)