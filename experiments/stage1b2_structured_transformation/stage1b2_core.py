"""
Stage 1B.2 core trial computation, implementing the fully locked design
from bonsai_stage1b_pilot_findings.md:

- Option A: 3 nodes (low/median/high weighted-degree in T) x 2 signs x
  3 amplitudes (0.025 tangent, 0.2 intermediate, 0.8 nonlinear) = 18 inputs
- 4 perturbation times t_p in {0, 0.833, 1.667, 2.5} along ONE baseline
  trajectory (seed=3000, class 0, T topology)
- 6 fixed nearby-state replicas per t_p (scale=0.1, verified locally at
  every t_p in the calibration step)
- Response horizon T=2.5, measured as tau (elapsed time since t_p), i.e.
  each perturbation is observed over absolute baseline time [t_p, t_p+T]
- Event-aligned tau*_eps = argmax E(tau), protected by E_min=1e-4
- Fixed-time endpoint at tau=T as a robustness check
- q(tau) (normalized nodewise energy) and r(tau) (signed direction,
  unit-normalized) both computed and retained
- J_tan(tau) = JSD(q_finite(tau), q_tangent(tau)) kept separate from
  d_q(a,b) = sqrt(JSD(q_a,q_b)), the output-map distance
- Raw residual norm ||z_eps(tau)|| retained alongside the normalized
  q_residual, so the residual mapping's absolute materiality (not just
  its normalized geometric shape) can be reported and checked against
  the numerical-validity and nonlinear-departure thresholds.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import jensenshannon

# ---- Locked design constants ----
T_HORIZON = 2.5
T_P_VALUES = [0, 0.833, 1.667, 2.5]
AMPLITUDES = [0.025, 0.2, 0.8]  # tangent, intermediate, nonlinear
SIGNS = [1, -1]
N_REPLICAS = 6
NEARBY_SCALE = 0.1
Q_NORM_THRESHOLD = 1e-6
E_MIN = 1e-4
TOP_K_FRACTION = 0.05
RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05


def rotation_projector(n):
    return np.eye(n) - np.ones((n, n)) / n


def force_jacobian(W, theta, k_coupling=1.0):
    diff = theta[None, :] - theta[:, None]
    off_diag = k_coupling * W * np.cos(diff)
    DF = off_diag.copy()
    np.fill_diagonal(DF, -np.sum(off_diag, axis=1))
    return DF


def get_degree_stratified_nodes(W):
    degree = W.sum(axis=1)
    order = np.argsort(degree)
    n = len(order)
    return {
        'low': int(order[n // 10]),
        'median': int(order[n // 2]),
        'high': int(order[-n // 10]),
    }


def generate_reference_baseline(W, seed, t_max):
    n = W.shape[0]
    theta0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)

    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return np.sum(W * np.sin(diff), axis=1)

    sol = solve_ivp(rhs, (0, t_max), theta0, method='RK45', rtol=RTOL, atol=ATOL,
                     max_step=MAX_STEP, dense_output=True)
    return sol


def generate_fixed_replica_directions(n, seed, n_replicas=N_REPLICAS):
    P = rotation_projector(n)
    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(n_replicas):
        d = rng.normal(0, 1, n)
        d = P @ d
        d = d / np.linalg.norm(d)
        directions.append(d)
    return directions


def normalized_energy(x):
    """q(t): normalized nodewise energy. Returns None if below q-norm threshold."""
    norm = np.linalg.norm(x)
    if norm < Q_NORM_THRESHOLD:
        return None
    return x ** 2 / np.sum(x ** 2)


def signed_direction(x):
    """r(t): unit-normalized signed displacement. Returns None if below threshold."""
    norm = np.linalg.norm(x)
    if norm < Q_NORM_THRESHOLD:
        return None
    return x / norm


def run_one_trial(W, replica_state, node, sign, amplitude, k_coupling=1.0):
    """Runs the joint baseline+tangent+finite-perturbed computation from
    one nearby-replica state, for one (node, sign, amplitude) input.
    Returns a dict of all diagnostics needed for the Stage 1B.2 analysis."""
    n = len(replica_state)
    P = rotation_projector(n)
    epsilon = sign * amplitude

    delta0 = np.zeros(n)
    delta0[node] = 1.0
    delta0 = P @ delta0
    delta0 = delta0 / np.linalg.norm(delta0)

    y0 = np.concatenate([replica_state, delta0])

    def rhs_tan(t, y):
        theta = y[:n]
        delta = y[n:]
        diff = theta[None, :] - theta[:, None]
        dtheta = k_coupling * np.sum(W * np.sin(diff), axis=1)
        DF = force_jacobian(W, theta, k_coupling=k_coupling)
        return np.concatenate([dtheta, DF @ delta])

    t_eval = np.linspace(0, T_HORIZON, 51)
    sol_tan = solve_ivp(rhs_tan, (0, T_HORIZON), y0, method='RK45', rtol=RTOL, atol=ATOL,
                         t_eval=t_eval, max_step=MAX_STEP)
    theta_base_tau = sol_tan.y[:n, :]
    delta_tau = sol_tan.y[n:, :]

    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return k_coupling * np.sum(W * np.sin(diff), axis=1)

    theta0_pert = replica_state + epsilon * delta0
    sol_pert = solve_ivp(rhs, (0, T_HORIZON), theta0_pert, method='RK45', rtol=RTOL, atol=ATOL,
                          t_eval=t_eval, max_step=MAX_STEP)
    theta_pert_tau = sol_pert.y

    eta = 1e-10
    E_list, C_list = [], []
    for i in range(len(t_eval)):
        shift = np.angle(np.mean(np.exp(1j * (theta_pert_tau[:, i] - theta_base_tau[:, i]))))
        actual_disp = P @ np.angle(np.exp(1j * (theta_pert_tau[:, i] - theta_base_tau[:, i] - shift)))
        predicted_disp = epsilon * (P @ delta_tau[:, i])
        E = np.linalg.norm(actual_disp - predicted_disp) / (abs(epsilon) * np.linalg.norm(P @ delta_tau[:, i]) + eta)
        denom = np.linalg.norm(actual_disp) * np.linalg.norm(predicted_disp)
        C = np.dot(actual_disp, predicted_disp) / denom if denom > eta else np.nan
        E_list.append(E)
        C_list.append(C)
    E_arr, C_arr = np.array(E_list), np.array(C_list)

    # Event-aligned time, protected by E_min
    tau_star_idx = int(np.argmax(E_arr))
    tau_star = t_eval[tau_star_idx]
    event_aligned_valid = E_arr[tau_star_idx] >= E_MIN

    def source_energy_fraction(disp):
        """f_source(tau): fraction of total displacement energy sitting
        at the directly-perturbed node -- diagnostic for identity
        retention vs. genuine redistribution elsewhere in the graph."""
        total = np.sum(disp ** 2)
        if total < 1e-15:
            return None
        return float(disp[node] ** 2 / total)

    def q_excluding_node(disp, exclude_idx):
        """q^(-i): normalized energy distribution over all nodes EXCEPT
        the stimulated one -- tests whether input identity predicts
        response structure elsewhere, not just at the input site.

        CRITICAL: the source coordinate is ZEROED, not deleted, so every
        trial's output vector remains defined over the same full,
        globally-aligned node coordinate system regardless of which node
        was stimulated. Physically deleting the coordinate (e.g. via
        boolean masking that shortens the vector) would misalign node
        identity across trials that stimulated different nodes -- index j
        in one shortened vector would not correspond to the same graph
        node as index j in another, and JSD/d_q comparisons between them
        would be comparing different node identities, not the same
        node's response under different inputs."""
        sub = disp.copy()
        sub[exclude_idx] = 0.0
        norm = np.linalg.norm(sub)
        if norm < Q_NORM_THRESHOLD:
            return None
        return sub ** 2 / np.sum(sub ** 2)

    def get_outputs_at(idx):
        shift = np.angle(np.mean(np.exp(1j * (theta_pert_tau[:, idx] - theta_base_tau[:, idx]))))
        actual_disp = P @ np.angle(np.exp(1j * (theta_pert_tau[:, idx] - theta_base_tau[:, idx] - shift)))
        tangent_disp = epsilon * (P @ delta_tau[:, idx])
        residual_disp = actual_disp - tangent_disp  # z_eps(tau): finite-minus-tangent, the cleanest nonlinear object
        residual_norm = float(np.linalg.norm(residual_disp))  # ABSOLUTE materiality, not just normalized shape

        q_finite = normalized_energy(actual_disp)
        q_tangent = normalized_energy(tangent_disp)
        q_residual = normalized_energy(residual_disp)
        q_finite_excl = q_excluding_node(actual_disp, node)
        r_finite = signed_direction(actual_disp)
        f_source = source_energy_fraction(actual_disp)

        J_tan = None
        if q_finite is not None and q_tangent is not None:
            J_tan = float(jensenshannon(q_finite, q_tangent) ** 2)

        return {'q': q_finite, 'q_tangent': q_tangent, 'q_residual': q_residual,
                'q_excl_node': q_finite_excl, 'r': r_finite, 'J_tan': J_tan,
                'f_source': f_source, 'residual_norm': residual_norm}

    empty = {'q': None, 'q_tangent': None, 'q_residual': None, 'q_excl_node': None,
             'r': None, 'J_tan': None, 'f_source': None, 'residual_norm': None}
    event_aligned = get_outputs_at(tau_star_idx) if event_aligned_valid else empty
    fixed_time = get_outputs_at(len(t_eval) - 1)  # tau = T
    initial = get_outputs_at(0)  # tau = 0, for f_source baseline

    return {
        'tau_star': tau_star,
        'E_at_tau_star': float(E_arr[tau_star_idx]),
        'event_aligned_valid': bool(event_aligned_valid),
        'event_aligned_q': event_aligned['q'],
        'event_aligned_r': event_aligned['r'],
        'event_aligned_J_tan': event_aligned['J_tan'],
        'event_aligned_q_tangent': event_aligned['q_tangent'],
        'event_aligned_q_residual': event_aligned['q_residual'],
        'event_aligned_q_excl_node': event_aligned['q_excl_node'],
        'event_aligned_f_source': event_aligned['f_source'],
        'event_aligned_residual_norm': event_aligned['residual_norm'],
        'fixed_time_q': fixed_time['q'],
        'fixed_time_r': fixed_time['r'],
        'fixed_time_J_tan': fixed_time['J_tan'],
        'fixed_time_q_tangent': fixed_time['q_tangent'],
        'fixed_time_q_residual': fixed_time['q_residual'],
        'fixed_time_q_excl_node': fixed_time['q_excl_node'],
        'fixed_time_f_source': fixed_time['f_source'],
        'fixed_time_residual_norm': fixed_time['residual_norm'],
        'initial_f_source': initial['f_source'],
        'peak_C': float(np.nanmin(C_arr)),  # most negative = strongest directional reversal
    }
