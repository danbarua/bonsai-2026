"""
Stage 1B: finite-amplitude nonlinear response classification.
Operationalizes the outcome taxonomy, tangent-departure diagnostics, and
equilibrium classification specified in pre-registration, before any
result is examined.
"""
import numpy as np
from scipy.integrate import solve_ivp
from graph_oscillator_field import rotation_projector, force_jacobian, find_equilibrium_lbfgs, GraphOscillatorField

# Prespecified horizons (time units)
T1_PRIMARY = 2.5      # matches Stage 1A's window, for comparability
T2_EXTENSION = 25.0    # 10x T1, using fast adaptive integration
PERSISTENCE_START_FRAC = 0.8  # final persistence interval = last 20% of T2
DECAY_S_THRESHOLD = 0.05       # normalized: 95% attenuation relative to imposed perturbation
DECAY_RMS_THRESHOLD = 0.01     # absolute: RMS phase separation, rad -- practical reunion
FORCE_CONVERGED_THRESHOLD = 1e-5   # relaxed from 1e-6 after direct diagnostic: L-BFGS
# reports genuine convergence (gtol satisfied, more iterations don't improve further) at
# force~1.5e-6 for this system's small-spectral-gap graphs -- 1e-6 was stricter than the
# optimizer's own achievable precision here, not evidence of a real non-equilibrium
DEDUP_RESIDUAL_THRESHOLD = 0.05    # matches Stage 0's multistability dedup rule


def run_baseline_and_perturbed(W, theta0, perturb_node, epsilon, t_span, k_coupling=1.0,
                                 rtol=1e-6, atol=1e-8, max_step=0.05, t_eval=None):
    """Integrates paired baseline and perturbed trajectories over t_span,
    returns full theta(t) for both (not just a scalar summary) -- needed
    for equilibrium classification and spatial-redistribution analysis."""
    n = len(theta0)
    P = rotation_projector(n)

    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return k_coupling * np.sum(W * np.sin(diff), axis=1)

    sol_base = solve_ivp(rhs, t_span, theta0, method='RK45', rtol=rtol, atol=atol,
                          t_eval=t_eval, max_step=max_step)
    e = np.zeros(n); e[perturb_node] = 1.0
    direction = P @ e
    direction = direction / np.linalg.norm(direction)
    theta0_pert = theta0 + epsilon * direction
    sol_pert = solve_ivp(rhs, t_span, theta0_pert, method='RK45', rtol=rtol, atol=atol,
                          t_eval=t_eval, max_step=max_step)
    return sol_base, sol_pert, P


def gauge_corrected_separation(theta_a, theta_b, P):
    """Returns (D_raw, S_would_need_D0_separately) -- raw squared gauge-
    corrected separation only; normalization done by caller with D(0)."""
    shift = np.angle(np.mean(np.exp(1j * (theta_a - theta_b))))
    diff = np.angle(np.exp(1j * (theta_a - theta_b - shift)))
    return np.sum((P @ diff) ** 2)


def rms_phase_separation(theta_a, theta_b):
    """Absolute RMS circular phase separation, no gauge projection --
    for deciding practical reunion, not just relative attenuation."""
    shift = np.angle(np.mean(np.exp(1j * (theta_a - theta_b))))
    diff = np.angle(np.exp(1j * (theta_a - theta_b - shift)))
    return np.sqrt(np.mean(diff ** 2))


def classify_terminal_state(W, theta_terminal, k_coupling=1.0):
    """Refines a near-terminal trajectory state to its nearest equilibrium
    via L-BFGS warm-started from that state, then checks force and
    stability (Jacobian eigenvalues) before calling it a genuine
    equilibrium -- not just a low-force snapshot."""
    from scipy.optimize import minimize
    n = len(theta_terminal)
    pin_node = 0

    def potential_and_grad(theta_free):
        theta = np.zeros(n)
        free_idx = [i for i in range(n) if i != pin_node]
        theta[free_idx] = theta_free
        theta[pin_node] = theta_terminal[pin_node]
        diff = theta[None, :] - theta[:, None]
        V = -k_coupling * 0.5 * np.sum(W * np.cos(diff))
        grad_full = k_coupling * np.sum(W * np.sin(theta[:, None] - theta[None, :]), axis=1)
        return V, grad_full[free_idx]

    free_idx = [i for i in range(n) if i != pin_node]
    result = minimize(potential_and_grad, theta_terminal[free_idx], jac=True, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
    theta_refined = theta_terminal.copy()
    theta_refined[free_idx] = result.x

    field = GraphOscillatorField(W, dt=0.05, k_coupling=k_coupling)
    force = field._coupling_force(theta_refined)
    force_norm = np.max(np.abs(force))
    converged = force_norm < FORCE_CONVERGED_THRESHOLD

    stable = None
    if converged:
        J = -force_jacobian(W, theta_refined, k_coupling=k_coupling)  # stability convention: ddelta/dt=-J*delta
        eigvals = np.linalg.eigvalsh(J)
        n_near_zero = np.sum(np.abs(eigvals) < 1e-4)
        n_negative = np.sum(eigvals < -1e-6)
        stable = (n_negative == 0)

    return {'theta_refined': theta_refined, 'force_norm': force_norm,
            'converged': converged, 'stable': stable}


def same_attractor(theta_a, theta_b):
    """Deduplication test matching Stage 0's multistability audit:
    residual after best rotational alignment below threshold."""
    shift = np.angle(np.mean(np.exp(1j * (theta_a - theta_b))))
    residual = np.angle(np.exp(1j * (theta_a - theta_b - shift)))
    return np.mean(np.abs(residual)) < DEDUP_RESIDUAL_THRESHOLD


def tangent_departure_diagnostics(theta0_base, theta_pert_t, theta_base_t, delta_t, epsilon, P):
    """E_eps(t): amplitude departure from tangent prediction.
    C_eps(t): directional cosine similarity between actual and predicted
    displacement. Distinguishes amplitude departure from directional
    reorganization -- a scalar S(t) comparison alone conflates these."""
    shift = np.angle(np.mean(np.exp(1j * (theta_pert_t - theta_base_t))))
    actual_disp = P @ np.angle(np.exp(1j * (theta_pert_t - theta_base_t - shift)))
    predicted_disp = epsilon * (P @ delta_t)

    eta = 1e-10
    E = np.linalg.norm(actual_disp - predicted_disp) / (abs(epsilon) * np.linalg.norm(P @ delta_t) + eta)
    denom = np.linalg.norm(actual_disp) * np.linalg.norm(predicted_disp)
    C = np.dot(actual_disp, predicted_disp) / denom if denom > eta else np.nan
    return E, C


def spatial_concentration(theta_pert_t, theta_base_t, P):
    """How concentrated (vs. diffuse) the perturbation-induced separation
    is across nodes -- inverse participation ratio of the squared,
    gauge-corrected per-node separation. Near 1: concentrated on a few
    nodes. Near n: diffuse across the whole graph."""
    shift = np.angle(np.mean(np.exp(1j * (theta_pert_t - theta_base_t))))
    diff = P @ np.angle(np.exp(1j * (theta_pert_t - theta_base_t - shift)))
    sq = diff ** 2
    total = np.sum(sq)
    if total < 1e-15:
        return np.nan
    p = sq / total
    ipr = 1.0 / np.sum(p ** 2)
    return ipr


def classify_one_trial(W, theta0, perturb_node, epsilon, delta_ref=None, k_coupling=1.0):
    """Full Stage 1B classification for one (graph, IC, node, epsilon)
    trial. FIXED: baseline and perturbed trajectories are each integrated
    as ONE CONTINUOUS solve from t=0 to T1+T2, not split into a primary
    solve followed by a separate restarted extension solve -- the
    two-stage version was found to introduce a small but classification-
    relevant discrepancy at the restart seam, since the adaptive solver
    resets its own step-size history there, and near this system's very
    flat, slow-converging regions that discontinuity was large enough to
    flip an equilibrium classification right at the threshold boundary."""
    n = len(theta0)
    T_total = T1_PRIMARY + T2_EXTENSION

    # combined evaluation grid: dense over the primary window (for S(t),
    # peak amplification, tangent departure), sparse over the persistence
    # window (last 20% of T2, i.e. [T1 + 0.8*T2, T1+T2])
    t_primary = np.linspace(0, T1_PRIMARY, 51)
    persist_start = T1_PRIMARY + PERSISTENCE_START_FRAC * T2_EXTENSION
    t_persist = np.linspace(persist_start, T_total, 6)
    t_eval = np.concatenate([t_primary, t_persist[1:]])  # avoid duplicating the boundary point

    sol_base, sol_pert, P = run_baseline_and_perturbed(W, theta0, perturb_node, epsilon,
                                                         t_span=(0, T_total), t_eval=t_eval)

    n_primary = len(t_primary)
    D0 = gauge_corrected_separation(sol_pert.y[:, 0], sol_base.y[:, 0], P)
    S_primary = np.array([gauge_corrected_separation(sol_pert.y[:, i], sol_base.y[:, i], P) / D0
                           for i in range(n_primary)])
    peak_amplification = np.max(S_primary)

    S_persist = np.array([gauge_corrected_separation(sol_pert.y[:, i], sol_base.y[:, i], P) / D0
                           for i in range(n_primary, len(t_eval))])
    rms_persist = np.array([rms_phase_separation(sol_pert.y[:, i], sol_base.y[:, i])
                             for i in range(n_primary, len(t_eval))])
    decayed = np.all(S_persist < DECAY_S_THRESHOLD) and np.all(rms_persist < DECAY_RMS_THRESHOLD)

    base_class = classify_terminal_state(W, sol_base.y[:, -1], k_coupling=k_coupling)
    pert_class = classify_terminal_state(W, sol_pert.y[:, -1], k_coupling=k_coupling)

    if base_class['converged'] and pert_class['converged']:
        if same_attractor(base_class['theta_refined'], pert_class['theta_refined']):
            outcome = 'decayed_to_same' if decayed else 'persistent_transient_same_attractor'
        else:
            outcome = 'different_equilibria'
    elif base_class['converged'] and not pert_class['converged']:
        outcome = 'baseline_only_converged'
    elif pert_class['converged'] and not base_class['converged']:
        outcome = 'perturbed_only_converged'
    else:
        outcome = 'no_equilibrium_recovered_within_horizon'

    spatial_conc = spatial_concentration(sol_pert.y[:, n_primary-1], sol_base.y[:, n_primary-1], P)

    return {
        'outcome': outcome,
        'peak_amplification': peak_amplification,
        'S_final_primary': S_primary[-1],
        'decayed': decayed,
        'spatial_concentration_ipr': spatial_conc,
        'base_force_norm': base_class['force_norm'],
        'pert_force_norm': pert_class['force_norm'],
        'base_stable': base_class['stable'],
        'pert_stable': pert_class['stable'],
    }
