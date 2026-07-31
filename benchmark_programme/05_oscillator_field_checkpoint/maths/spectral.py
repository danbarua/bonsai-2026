# === Spectral Decomposition Types ===
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Generic, List, NewType, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from maths import Dimension, FrequencyBand, FrequencyHz

N = TypeVar('N')

@dataclass
class FrequencyDomainSignal(Generic[N]):
    """
    Representation of a signal in the frequency domain after FFT
    
    Attributes:
        frequencies: The frequency values corresponding to each component
        amplitudes: Magnitude of each frequency component
        phases: Phase angle of each frequency component
        complex_values: Complex FFT output (alternative to amplitude+phase representation)
        dimension: Domain of the original signal (TIME, SPACE, GRAPH)
    """
    frequencies: NDArray[np.float64]  # Shape: (N//2+1,) for real FFT
    amplitudes: NDArray[np.float64]   # Shape: (N//2+1,)
    phases: NDArray[np.float64]       # Shape: (N//2+1,)
    complex_values: NDArray[np.complex128]  # Shape: (N//2+1,) or (N,) depending on FFT type
    dimension: Dimension
    sampling_rate: Optional[float] = None  # Only applicable for TIME dimension
    
    @classmethod
    def from_time_signal(cls, signal: NDArray[np.float64], sampling_rate: float) -> 'FrequencyDomainSignal':
        """Create frequency domain representation from a time domain signal"""
        # Compute FFT
        complex_fft = np.fft.rfft(signal)
        
        # Get frequency values
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
        
        # Compute amplitude and phase
        amplitudes = np.abs(complex_fft)
        phases = np.angle(complex_fft)
        
        return cls(
            frequencies=freqs,
            amplitudes=amplitudes,
            phases=phases,
            complex_values=complex_fft,
            dimension=Dimension.TIME,
            sampling_rate=sampling_rate
        )
    
    @classmethod
    def from_graph_signal(cls, signal: NDArray[np.float64], 
                         laplacian_eigvals: NDArray[np.float64],
                         laplacian_eigvecs: NDArray[np.float64]) -> 'FrequencyDomainSignal':
        """Create frequency domain representation from a graph signal using GFT"""
        # Graph Fourier Transform: project signal onto eigenvectors
        gft_coeffs = laplacian_eigvecs.T @ signal
        
        return cls(
            frequencies=laplacian_eigvals,  # Graph frequencies correspond to eigenvalues
            amplitudes=np.abs(gft_coeffs),
            phases=np.angle(gft_coeffs),
            complex_values=gft_coeffs,
            dimension=Dimension.GRAPH
        )
    
    def band_energy(self, band: FrequencyBand) -> float:
        """Compute energy in the given frequency band"""
        if self.dimension != Dimension.TIME:
            raise ValueError(f"Band energy only applicable for time domain signals, not {self.dimension}")
            
        mask = (self.frequencies >= band.min_freq.value) & (self.frequencies <= band.max_freq.value)
        return np.sum(self.amplitudes[mask]**2)
    
    def dominant_frequency(self) -> FrequencyHz:
        """Find the dominant frequency (highest amplitude)"""
        max_idx = np.argmax(self.amplitudes)
        return FrequencyHz(self.frequencies[max_idx])

@dataclass
class SpectralDecomposition(Generic[N]):
    """
    Complete spectral decomposition of a system, including eigenvectors and eigenvalues
    
    This represents a decomposition where a system is broken down into its fundamental
    modes characterized by eigenvalues (frequencies) and eigenvectors (mode shapes).
    """
    eigenvalues: NDArray[np.float64]   # Shape: (N,) - sorted in ascending order
    eigenvectors: NDArray[np.float64]  # Shape: (N, N) - columns are eigenvectors  
    dimension: Dimension
    
    def __post_init__(self):
        n_evals = len(self.eigenvalues)
        n_rows, n_cols = self.eigenvectors.shape
        
        if n_rows != n_cols or n_cols != n_evals:
            raise ValueError(f"Shape mismatch: {n_evals} eigenvalues but eigenvectors shape is {self.eigenvectors.shape}")
    
    @property
    def spectral_gap(self) -> float:
        """
        The spectral gap (difference between first non-zero and zero eigenvalues)
        
        In graph theory, larger spectral gaps indicate better connectivity.
        """
        # Find first non-zero eigenvalue (allowing for numerical precision issues)
        non_zero_idx = np.where(np.abs(self.eigenvalues) > 1e-10)[0][0]
        return self.eigenvalues[non_zero_idx]
    
    def project_signal(self, signal: NDArray[np.float64]) -> FrequencyDomainSignal:
        """Project a signal onto the eigenbasis"""
        if len(signal) != len(self.eigenvalues):
            raise ValueError(f"Signal length {len(signal)} doesn't match decomposition size {len(self.eigenvalues)}")
        
        # Project signal onto eigenvectors
        coefficients = self.eigenvectors.T @ signal
        
        return FrequencyDomainSignal(
            frequencies=self.eigenvalues,
            amplitudes=np.abs(coefficients),
            phases=np.angle(coefficients.astype(complex)),
            complex_values=coefficients.astype(complex),
            dimension=self.dimension
        )
    
    def reconstruct_signal(self, coefficients: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Reconstruct a signal from its spectral coefficients"""
        return np.real(self.eigenvectors @ coefficients)
    
    def filter_signal(self, signal: NDArray[np.float64], cutoff_idx: int) -> NDArray[np.float64]:
        """
        Filter a signal by keeping only components up to cutoff_idx
        
        This implements spectral filtering where:
        - Lower indices = "aligned" components (smooth, global patterns)
        - Higher indices = "liberal" components (detailed, local patterns)
        """
        # Project signal onto eigenbasis
        coeffs = self.eigenvectors.T @ signal
        
        # Apply filter (zero out high-frequency components)
        filtered_coeffs = coeffs.copy()
        filtered_coeffs[cutoff_idx:] = 0
        
        # Reconstruct filtered signal
        return self.eigenvectors @ filtered_coeffs