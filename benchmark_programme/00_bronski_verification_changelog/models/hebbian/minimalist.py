from beartype import beartype
import numpy as np
from numpy.typing import NDArray

@beartype
def update_hebbian_kuramoto(phases: NDArray[np.float64], 
                            weights: NDArray[np.float64], 
                            frequencies: NDArray[np.float64], 
                            dt: float=0.1, 
                            learning_rate: float =0.01, 
                            decay: float=0.1) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Simple update function for Kuramoto oscillators with Hebbian plasticity.

    Implements Bronski et al. (2017), "The stability of fixed points for a
    Kuramoto model with Hebbian interactions" (Chaos 27, 053110), equation (2):

        dtheta_i/dt = omega_i + sum_j gamma_ij * sin(theta_j - theta_i)
        dgamma_ij/dt = mu * cos(theta_i - theta_j) - alpha * gamma_ij

    Note the coupling term's argument order: sin(theta_j - theta_i), i.e.
    "other minus self" -- this is the standard *attractive* Kuramoto coupling
    (for gamma_ij > 0, oscillator i is pulled toward oscillator j's phase).
    Getting this backwards (self minus other) silently flips the sign of the
    coupling term, turning positive weights into *repulsive* coupling instead
    -- this was an actual bug here previously, only visible once oscillators
    have any real phase spread to act on (it's invisible at exact synchrony,
    since sin(0) = 0 either way).
    
    Parameters:
    - phases: Array of oscillator phases (radians)
    - weights: Coupling weight matrix
    - frequencies: Natural frequencies of oscillators
    - dt: Time step
    - learning_rate: Hebbian learning rate (mu)
    - decay: Weight decay rate (alpha)
    
    Returns:
    - new_phases: Updated phases
    - new_weights: Updated weights
    """
    n_oscillators = len(phases)
    
    # phase_diffs[i, j] = theta_j - theta_i ("other minus self"), matching the
    # paper's sin(theta_j - theta_i) coupling term exactly.
    phase_diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    
    # Kuramoto phase update: dtheta_i/dt = omega_i + sum_j gamma_ij sin(theta_j - theta_i)
    coupling:NDArray[np.float64] = np.sum(weights * np.sin(phase_diffs), axis=1)
    phase_update:NDArray[np.float64] = frequencies + coupling
    new_phases:NDArray[np.float64] = (phases + dt * phase_update) % (2 * np.pi)
    
    # Hebbian weight update: dgamma_ij/dt = mu * cos(theta_i - theta_j) - alpha * gamma_ij
    # cos is even, so cos(theta_i - theta_j) == cos(theta_j - theta_i) == cos(phase_diffs);
    # the sign convention chosen for phase_diffs above doesn't affect this term.
    weight_update:NDArray[np.float64] = learning_rate * np.cos(phase_diffs) - decay * weights
    new_weights:NDArray[np.float64] = weights + dt * weight_update
    # Zero the diagonal: an oscillator shouldn't accumulate a spurious "self-coupling"
    # weight. Without this, cos(theta_i - theta_i) = cos(0) = 1 drives the diagonal
    # toward learning_rate/decay over time even though oscillators are only meant
    # to couple to *other* oscillators. HebbianKuramotoOperator guards against this
    # explicitly; this reference implementation previously didn't.
    np.fill_diagonal(new_weights, 0.0)
    
    return new_phases, new_weights
