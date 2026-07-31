"""
Capacity Experiment III: class-conditioned low-frequency spectral
projection readout.

s_c(x) = sum_{k in first 5 non-trivial} w_k |<x_active, u_{c,k}>|^2

using the symmetric normalized Laplacian, uniform weighting (w_k=1).

Two subtleties a naive implementation would miss:
1. Most pixels are isolated (zero degree) in these sparse topologies --
   D^(-1/2) is undefined there. Restrict the Laplacian to the active
   (non-isolated) node subgraph only.
2. The active subgraph itself typically has MANY connected components
   (confirmed earlier: hundreds for MNIST/Kuzushiji-MNIST), each
   contributing its own trivial zero eigenvalue -- "exclude the constant
   eigenvector" means excluding one eigenvalue per component, not just
   one overall. This is computed explicitly via connected_components,
   not assumed.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def build_spectral_basis(topology, n_eigenvectors=5):
    """Returns (active_node_indices, eigenvectors) -- eigenvectors are
    defined only over the active nodes, in the same order as
    active_node_indices."""
    adj = np.abs(topology) > 0
    degrees = adj.sum(axis=1)
    active = np.where(degrees > 0)[0]
    n_active = len(active)

    W_active = topology[np.ix_(active, active)]
    W_active = np.abs(W_active)  # all weights confirmed positive in this project already
    d_active = W_active.sum(axis=1)

    n_components, _ = connected_components(sparse.csr_matrix(W_active > 0), directed=False)

    d_inv_sqrt = np.where(d_active > 0, 1.0 / np.sqrt(d_active), 0.0)
    L_sym = np.eye(n_active) - (d_inv_sqrt[:, None] * W_active * d_inv_sqrt[None, :])

    eigvals, eigvecs = np.linalg.eigh(L_sym)  # ascending order
    # Skip exactly n_components trivial (near-zero) eigenvalues, one per
    # connected component, then take the next n_eigenvectors
    start = n_components
    end = start + n_eigenvectors
    if end > n_active:
        end = n_active  # degenerate case: not enough active nodes/structure
    selected = eigvecs[:, start:end]
    return active, selected, eigvals[start:end], n_components


def spectral_score(phases_flat, active_indices, eigenvectors):
    """|<x, u_k>|^2 summed over the provided eigenvectors, uniform weight."""
    x_active = phases_flat[active_indices]
    projections = eigenvectors.T @ x_active  # shape (n_eigenvectors,)
    return np.sum(projections ** 2)
