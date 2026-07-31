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


def spectral_score_normalized_and_energy(phases_flat, active_indices, eigenvectors, epsilon=1e-8):
    """Returns (normalized_spectral_score, active_support_energy) for one
    class. Normalized score divides the raw projection energy by the
    total input energy on the active subgraph, isolating whether the
    signal is about how energy is DISTRIBUTED within the low-frequency
    subspace, versus simply how much of the image overlaps the class's
    active support at all (retained separately as its own control)."""
    x_active = phases_flat[active_indices]
    projections = eigenvectors.T @ x_active
    raw_energy = np.sum(projections ** 2)
    active_support_energy = np.sum(x_active ** 2)
    normalized = raw_energy / (active_support_energy + epsilon)
    return normalized, active_support_energy


def build_random_orthonormal_basis(active_indices, n_vectors, seed):
    """Same support (active_indices), same dimensionality, same class
    conditioning as the learned spectral basis -- but a random orthonormal
    subspace instead of the learned low-frequency eigenvectors. Isolates
    whether spectral's advantage is about active-support restriction and
    class conditioning generally, or the specific learned eigenstructure."""
    rng = np.random.default_rng(seed)
    n_active = len(active_indices)
    random_matrix = rng.standard_normal((n_active, n_vectors))
    Q, _ = np.linalg.qr(random_matrix)
    return Q[:, :n_vectors]
