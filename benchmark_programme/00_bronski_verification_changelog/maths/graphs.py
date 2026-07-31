# === Graph Laplacian Types ===
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Generic, List, NewType, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from .core import Dimension
from .spectral import FrequencyDomainSignal, SpectralDecomposition

N = TypeVar('N')

@dataclass
class GraphLaplacian(Generic[N]):
    """
    Graph Laplacian matrix representation with spectral methods.

    The Laplacian matrix L = D - A where:
    - D is the degree matrix (diagonal matrix with node degrees)
    - A is the adjacency matrix

    Properties (unnormalized Laplacian only -- see is_normalized note below):
    - Symmetric for undirected graphs
    - Rows sum to zero (by construction: L[i,i] = sum_j A[i,j], L[i,j] = -A[i,j],
      so row i sums to sum_j A[i,j] - sum_j A[i,j] = 0 -- true for ANY square
      symmetric A, including one with negative entries; see below)
    - Positive semi-definite when A has non-negative entries
    - Has at least one eigenvalue of 0 (for connected graphs, exactly one is 0)
    - Number of zero eigenvalues equals number of connected components

    Relevance to Bronski et al. (2017), "The stability of fixed points for a
    Kuramoto model with Hebbian interactions" (Chaos 27, 053110):

    That paper proves that the *stability* of a fixed point of the Hebbian-
    Kuramoto system (coupled phase + adaptive-weight ODEs) reduces to the sign
    structure of a single N x N matrix, obtained from the system's Jacobian via
    a Schur complement (Haynsworth's theorem):

        A - B C^-1 B^T

    where A is the classical-Kuramoto Jacobian, B is a weighted incidence
    matrix, and C = -alpha * I. The paper shows this reduces to exactly the
    form of a graph Laplacian:

        (A - B C^-1 B^T)_ij = { -sum_k cos(2(theta_i - theta_k)) / alpha   i == j
                               {  cos(2(theta_i - theta_j)) / alpha        i != j

    i.e. a Laplacian D - K where the "adjacency" K has (possibly negative,
    since cosine can be negative) edge weights kappa_ij = cos(2(theta_i -
    theta_j)) / alpha. The number of NEGATIVE eigenvalues of this matrix is
    exactly the dimension of the unstable manifold of the corresponding
    Hebbian-Kuramoto fixed point (Theorem 2.3 in the paper) -- and by the
    paper's central result, this is the *same* as the number of negative
    eigenvalues of the analogous *classical* (fixed-weight) Kuramoto Jacobian,
    despite the adaptive-weight system having far higher dimension.

    So: to check whether a candidate Hebbian-Kuramoto fixed point is stable,
    build this matrix as a GraphLaplacian via `from_adjacency(kappa)` (passing
    the *signed* kappa_ij matrix as the "adjacency" -- from_adjacency handles
    negative entries fine, since D - A sums to zero regardless of the sign of
    A's entries, which is just algebra, not a positivity requirement), then
    check `spectral_decomposition().eigenvalues`: any negative eigenvalue means
    an unstable direction; all non-negative (with exactly one zero, from the
    rotational symmetry of the all-in-phase direction) means a stable fixed
    point.
    """
    matrix: NDArray[np.float64]  # Shape: (N, N)
    is_normalized: bool = False  # True for the symmetric-normalized Laplacian D^-1/2 L D^-1/2
    
    def __post_init__(self):
        # Verify matrix is square
        n_rows, n_cols = self.matrix.shape
        if n_rows != n_cols:
            raise ValueError(f"Laplacian must be square, got shape {self.matrix.shape}")
        
        # Verify Laplacian properties
        if not np.allclose(self.matrix, self.matrix.T):
            raise ValueError("Laplacian must be symmetric")
        
        # Row sums should be approximately zero for the *unnormalized* Laplacian
        # (L = D - A). This does NOT hold for the symmetric-normalized Laplacian
        # (D^-1/2 L D^-1/2) in general -- only for regular graphs -- so the check
        # only applies when is_normalized is False.
        if not self.is_normalized:
            row_sums = np.abs(np.sum(self.matrix, axis=1))
            if not np.allclose(row_sums, 0, atol=1e-10):
                raise ValueError("Laplacian matrix rows must sum to zero")
    
    @classmethod
    def from_adjacency(cls, adjacency: NDArray[np.float64]) -> 'GraphLaplacian':
        """
        Create Laplacian from adjacency matrix.

        Works for *signed* adjacency matrices too (negative entries), not just
        the non-negative-weight case typical of graph theory -- L = D - A sums
        to zero per row regardless of the sign of A's entries, since D is
        defined as the row-sum of A. This is what makes it usable directly for
        the Bronski et al. Hebbian-Kuramoto stability matrix (see class
        docstring), whose "edge weights" kappa_ij = cos(2*dtheta_ij)/alpha can
        be negative.
        """
        # Compute degree matrix (diagonal matrix with row sums of adjacency)
        degrees = np.sum(adjacency, axis=1)
        degree_matrix = np.diag(degrees)
        
        # L = D - A
        laplacian = degree_matrix - adjacency
        
        return cls(matrix=laplacian, is_normalized=False)
    
    @classmethod
    def from_adjacency_normalized(cls, adjacency: NDArray[np.float64]) -> 'GraphLaplacian':
        """Create normalized Laplacian from adjacency matrix"""
        # Compute degree matrix and its inverse square root
        degrees = np.sum(adjacency, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
        
        # Standard Laplacian
        degree_matrix = np.diag(degrees)
        laplacian = degree_matrix - adjacency
        
        # Normalized Laplacian: L_norm = D^(-1/2) L D^(-1/2)
        normalized_laplacian = d_inv_sqrt @ laplacian @ d_inv_sqrt
        
        return cls(matrix=normalized_laplacian, is_normalized=True)

    @classmethod
    def from_bronski_stability_matrix(cls, phases: NDArray[np.float64], alpha: float) -> 'GraphLaplacian':
        """
        Build the Bronski et al. (2017) Hebbian-Kuramoto stability matrix for
        a phase-locked fixed point (all-to-all topology), as a GraphLaplacian.

        Per the paper (see class docstring), the linearization of the
        Hebbian-Kuramoto system at a fixed point reduces via a Schur
        complement to a graph-Laplacian-shaped matrix with signed edge
        weights kappa_ij = cos(2*(theta_i - theta_j)) / alpha. The number of
        negative eigenvalues of that matrix equals the dimension of the
        fixed point's unstable manifold (Theorem 2.3): if the resulting
        GraphLaplacian's spectral_decomposition() has any negative
        eigenvalue, the fixed point at `phases` is unstable; otherwise (all
        eigenvalues >= 0, with exactly one ~0 from rotational symmetry) it's
        stable.

        Args:
            phases: 1D array of phase angles theta_i at the candidate fixed point.
            alpha: the Hebbian weight-decay rate (must be > 0).

        Returns:
            GraphLaplacian representing A - B C^-1 B^T from the paper.
        """
        phase_diffs = phases[:, np.newaxis] - phases[np.newaxis, :]  # theta_i - theta_j
        kappa = np.cos(2.0 * phase_diffs) / alpha
        np.fill_diagonal(kappa, 0.0)  # self-coupling excluded, as for any adjacency matrix
        return cls.from_adjacency(kappa)
    
    def spectral_decomposition(self) -> SpectralDecomposition:
        """Compute the spectral decomposition (eigenvalues and eigenvectors)"""
        # For symmetric matrices, eigenvalues are real
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        
        # Sort by eigenvalues (smallest first, typically 0 is first)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            dimension=Dimension.GRAPH
        )
    
    @property
    def connected_components(self) -> int:
        """Determine number of connected components in the graph"""
        # Number of zero eigenvalues equals number of connected components
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return np.sum(np.abs(eigenvalues) < 1e-10)

    def unstable_dimension(self, atol: float = 1e-10) -> int:
        """
        Number of strictly negative eigenvalues.

        When this GraphLaplacian was built via `from_bronski_stability_matrix`,
        this is exactly the dimension of the unstable manifold of the
        corresponding Hebbian-Kuramoto fixed point (Bronski et al. Theorem 2.3).
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return int(np.sum(eigenvalues < -atol))

    @property
    def is_bronski_stable(self) -> bool:
        """
        True if a fixed point represented by this stability matrix (built via
        `from_bronski_stability_matrix`) is stable per Bronski et al.: no
        negative eigenvalues (the one zero eigenvalue from the model's
        rotational symmetry -- shifting all phases by a constant -- is
        expected and does not itself indicate instability).
        """
        return self.unstable_dimension() == 0
    
    def apply_gft(self, signal: NDArray[np.float64]) -> FrequencyDomainSignal:
        """Apply Graph Fourier Transform to a signal"""
        # Get spectral decomposition
        decomposition = self.spectral_decomposition()
        
        # Project signal onto eigenvectors
        return decomposition.project_signal(signal)
    
    def filter_signal(self, signal: NDArray[np.float64], cutoff_idx: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Filter a signal into aligned (low-frequency) and liberal (high-frequency) components
        
        Args:
            signal: Graph signal to filter
            cutoff_idx: Index to split aligned and liberal components
            
        Returns:
            Tuple of (aligned_component, liberal_component)
        """
        # Get spectral decomposition
        decomposition = self.spectral_decomposition()
        
        # Project signal
        coeffs = decomposition.eigenvectors.T @ signal
        
        # Create aligned and liberal masks
        aligned_mask = np.zeros_like(coeffs)
        aligned_mask[:cutoff_idx] = 1
        
        liberal_mask = np.zeros_like(coeffs)
        liberal_mask[cutoff_idx:] = 1
        
        # Create components
        aligned_coeffs = coeffs * aligned_mask
        liberal_coeffs = coeffs * liberal_mask
        
        # Reconstruct components
        aligned_signal = decomposition.eigenvectors @ aligned_coeffs
        liberal_signal = decomposition.eigenvectors @ liberal_coeffs
        
        return aligned_signal, liberal_signal
