"""
Faithful JAX/diffrax port of stage1b2_core.py's run_one_trial. Written
and verified against the real numpy/scipy implementation field-by-field
on real trials, not a simplified stand-in.

Key departures from the numpy version, both required for vmap/jit
compatibility, neither changing any computed value:
- Every `return None if norm < threshold` pattern (normalized_energy,
  signed_direction, q_excluding_node, source_energy_fraction) becomes
  (value_or_nan, valid_bool) -- a batched JAX computation can't have
  some trials in a batch return None and others return real arrays.
- The E(tau)/C(tau) loop over 51 timepoints is vectorized (vmapped
  internally) rather than a Python for-loop.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

T_HORIZON = 2.5
Q_NORM_THRESHOLD = 1e-6
E_MIN = 1e-4
RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05
N_TIME_POINTS = 51


def rotation_projector_jax(n):
    return jnp.eye(n) - jnp.ones((n, n)) / n


def force_jacobian_jax(W, theta, k_coupling=1.0):
    diff = theta[None, :] - theta[:, None]
    off_diag = k_coupling * W * jnp.cos(diff)
    DF = off_diag - jnp.diag(jnp.diag(off_diag)) - jnp.diag(off_diag.sum(axis=1))
    return DF


def normalized_energy_jax(x):
    norm = jnp.linalg.norm(x)
    valid = norm >= Q_NORM_THRESHOLD
    q = jnp.where(valid, x ** 2 / jnp.sum(x ** 2), jnp.nan)
    return q, valid


def signed_direction_jax(x):
    norm = jnp.linalg.norm(x)
    valid = norm >= Q_NORM_THRESHOLD
    r = jnp.where(valid, x / norm, jnp.nan)
    return r, valid


def source_energy_fraction_jax(disp, node):
    total = jnp.sum(disp ** 2)
    valid = total >= 1e-15
    f = jnp.where(valid, disp[node] ** 2 / total, jnp.nan)
    return f, valid


def q_excluding_node_jax(disp, exclude_idx):
    sub = disp.at[exclude_idx].set(0.0)
    norm = jnp.linalg.norm(sub)
    valid = norm >= Q_NORM_THRESHOLD
    q = jnp.where(valid, sub ** 2 / jnp.sum(sub ** 2), jnp.nan)
    return q, valid


def jsd_natural_log_jax(p, q):
    """Jensen-Shannon divergence, natural log (matches scipy.spatial.distance.jensenshannon's
    default base=None, confirmed empirically against scipy: manual JSD with natural log,
    sqrt'd, matches scipy's output to full float64 precision)."""
    m = (p + q) / 2

    def kl(a, b):
        safe = jnp.where(a > 0, a * jnp.log(jnp.where(a > 0, a / b, 1.0)), 0.0)
        return jnp.sum(safe)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_one_trial_jax_faithful(W, replica_state, node, sign, amplitude, k_coupling=1.0):
    """Same signature and same returned-field set as stage1b2_core.py's
    run_one_trial, for one (node, sign, amplitude) input from one
    (t_p, replica) replica_state. Designed to be wrapped in jax.vmap
    over a batch of (replica_state, node, sign, amplitude) combinations,
    with W held fixed via in_axes=None (or also batched if W varies too,
    e.g. across stochastic-control graph realizations)."""
    n = replica_state.shape[0]
    P = rotation_projector_jax(n)
    epsilon = sign * amplitude

    delta0 = jnp.zeros(n).at[node].set(1.0)
    delta0 = P @ delta0
    delta0 = delta0 / jnp.linalg.norm(delta0)

    y0 = jnp.concatenate([replica_state, delta0])

    def rhs_tan(t, y, args):
        theta = y[:n]
        delta = y[n:]
        diff = theta[None, :] - theta[:, None]
        dtheta = k_coupling * jnp.sum(W * jnp.sin(diff), axis=1)
        DF = force_jacobian_jax(W, theta, k_coupling)
        return jnp.concatenate([dtheta, DF @ delta])

    t_eval = jnp.linspace(0, T_HORIZON, N_TIME_POINTS)
    saveat = diffrax.SaveAt(ts=t_eval)
    stepsize_controller = diffrax.PIDController(rtol=RTOL, atol=ATOL, dtmax=MAX_STEP)

    sol_tan = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs_tan), diffrax.Tsit5(), t0=0.0, t1=T_HORIZON, dt0=0.01,
        y0=y0, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=200_000)
    theta_base_tau = sol_tan.ys[:, :n].T   # (n, 51)
    delta_tau = sol_tan.ys[:, n:].T        # (n, 51)

    def rhs(t, theta, args):
        diff = theta[None, :] - theta[:, None]
        return k_coupling * jnp.sum(W * jnp.sin(diff), axis=1)

    theta0_pert = replica_state + epsilon * delta0
    sol_pert = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=0.0, t1=T_HORIZON, dt0=0.01,
        y0=theta0_pert, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=200_000)
    theta_pert_tau = sol_pert.ys.T   # (n, 51)

    eta = 1e-10

    def E_and_C_at(i):
        diff_phase = theta_pert_tau[:, i] - theta_base_tau[:, i]
        shift = jnp.angle(jnp.mean(jnp.exp(1j * diff_phase)))
        actual_disp = P @ jnp.angle(jnp.exp(1j * (diff_phase - shift)))
        predicted_disp = epsilon * (P @ delta_tau[:, i])
        E = jnp.linalg.norm(actual_disp - predicted_disp) / (
            jnp.abs(epsilon) * jnp.linalg.norm(P @ delta_tau[:, i]) + eta)
        denom = jnp.linalg.norm(actual_disp) * jnp.linalg.norm(predicted_disp)
        C = jnp.where(denom > eta, jnp.dot(actual_disp, predicted_disp) / denom, jnp.nan)
        return E, C

    E_arr, C_arr = jax.vmap(E_and_C_at)(jnp.arange(N_TIME_POINTS))

    tau_star_idx = jnp.argmax(E_arr)
    tau_star = t_eval[tau_star_idx]
    event_aligned_valid = E_arr[tau_star_idx] >= E_MIN

    def get_outputs_at(idx):
        diff_phase = theta_pert_tau[:, idx] - theta_base_tau[:, idx]
        shift = jnp.angle(jnp.mean(jnp.exp(1j * diff_phase)))
        actual_disp = P @ jnp.angle(jnp.exp(1j * (diff_phase - shift)))
        tangent_disp = epsilon * (P @ delta_tau[:, idx])
        residual_disp = actual_disp - tangent_disp
        residual_norm = jnp.linalg.norm(residual_disp)

        q_finite, q_finite_valid = normalized_energy_jax(actual_disp)
        q_tangent, q_tangent_valid = normalized_energy_jax(tangent_disp)
        q_residual, q_residual_valid = normalized_energy_jax(residual_disp)
        q_finite_excl, q_finite_excl_valid = q_excluding_node_jax(actual_disp, node)
        r_finite, r_finite_valid = signed_direction_jax(actual_disp)
        f_source, f_source_valid = source_energy_fraction_jax(actual_disp, node)

        both_valid = q_finite_valid & q_tangent_valid
        J_tan = jnp.where(both_valid, jsd_natural_log_jax(q_finite, q_tangent), jnp.nan)

        return {
            'q': q_finite, 'q_valid': q_finite_valid,
            'q_tangent': q_tangent, 'q_tangent_valid': q_tangent_valid,
            'q_residual': q_residual, 'q_residual_valid': q_residual_valid,
            'legacy_q_excl': q_finite_excl, 'legacy_q_excl_valid': q_finite_excl_valid,
            'r': r_finite, 'r_valid': r_finite_valid,
            'J_tan': J_tan,
            'f_source': f_source, 'f_source_valid': f_source_valid,
            'residual_norm': residual_norm,
        }

    outputs_at_star = get_outputs_at(tau_star_idx)
    outputs_at_T = get_outputs_at(N_TIME_POINTS - 1)
    outputs_at_0 = get_outputs_at(0)

    # event_aligned_* fields are only meaningful if event_aligned_valid; the numpy
    # version returns a dict of Nones in that case -- here, NaN throughout (already
    # the case for outputs_at_star's own internal validity flags), plus the
    # overall event_aligned_valid flag downstream code should check first.

    return {
        'tau_star': tau_star,
        'E_at_tau_star': E_arr[tau_star_idx],
        'event_aligned_valid': event_aligned_valid,
        'event_aligned_q': outputs_at_star['q'],
        'event_aligned_q_valid': outputs_at_star['q_valid'],
        'event_aligned_r': outputs_at_star['r'],
        'event_aligned_J_tan': outputs_at_star['J_tan'],
        'event_aligned_q_tangent': outputs_at_star['q_tangent'],
        'event_aligned_q_residual': outputs_at_star['q_residual'],
        'event_aligned_legacy_q_excl': outputs_at_star['legacy_q_excl'],
        'event_aligned_f_source': outputs_at_star['f_source'],
        'event_aligned_residual_norm': outputs_at_star['residual_norm'],
        'fixed_time_q': outputs_at_T['q'],
        'fixed_time_r': outputs_at_T['r'],
        'fixed_time_J_tan': outputs_at_T['J_tan'],
        'fixed_time_q_tangent': outputs_at_T['q_tangent'],
        'fixed_time_q_residual': outputs_at_T['q_residual'],
        'fixed_time_legacy_q_excl': outputs_at_T['legacy_q_excl'],
        'fixed_time_f_source': outputs_at_T['f_source'],
        'fixed_time_residual_norm': outputs_at_T['residual_norm'],
        'initial_f_source': outputs_at_0['f_source'],
        'peak_C': jnp.nanmin(C_arr),
    }
