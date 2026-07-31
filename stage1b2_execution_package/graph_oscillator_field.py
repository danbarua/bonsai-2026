"""
General weighted-graph coupled-oscillator simulator. Unlike
LocalOscillatorField (fixed 4-neighbor pixel grid, always the same
regardless of class), this accepts an arbitrary weighted adjacency
matrix as the coupling structure -- the learned topology, or any control
graph, has never been used as a coupling structure for dynamics before
this. Built specifically for Stage 0/Stage 1 graph-dynamics
characterization, not for image encoding (no input-anchoring bias term;
pure coupling dynamics only).
"""
import numpy as np


class GraphOscillatorField:
    def __init__(self, W, dt=0.05, k_coupling=1.0, seed=None):
        """W: (n,n) symmetric non-negative weighted adjacency matrix over
        the ACTIVE node set only (isolated nodes excluded before
        construction, matching the convention established throughout this
        project's spectral work)."""
        self.n = W.shape[0]
        self.W = W
        self.dt = dt
        self.k_coupling = k_coupling
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0, 2 * np.pi, self.n)

    def set_phases(self, phases):
        self.phases = phases.copy()

    def _coupling_force(self, phases):
        # dtheta_i/dt = k_coupling * sum_j W_ij sin(theta_j - theta_i)
        diff = phases[None, :] - phases[:, None]  # diff[i,j] = theta_j - theta_i
        return self.k_coupling * np.sum(self.W * np.sin(diff), axis=1)

    def step(self):
        dtheta = self._coupling_force(self.phases)
        self.phases = (self.phases + self.dt * dtheta) % (2 * np.pi)
        return self.phases

    def run(self, steps, record_every=1):
        history = []
        for i in range(steps):
            self.step()
            if i % record_every == 0:
                history.append(self.phases.copy())
        return history

    def find_equilibrium(self, steps=2000, seed=0, convergence_tol=1e-8, check_every=50):
        """Run from a random initial condition until the coupling force
        norm drops below convergence_tol, or steps is reached. Returns
        (equilibrium_phases, converged_bool, steps_taken)."""
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0, 2 * np.pi, self.n)
        for t in range(steps):
            force = self._coupling_force(self.phases)
            self.phases = (self.phases + self.dt * force) % (2 * np.pi)
            if (t + 1) % check_every == 0:
                if np.max(np.abs(force)) < convergence_tol:
                    return self.phases.copy(), True, t + 1
        force = self._coupling_force(self.phases)
        return self.phases.copy(), np.max(np.abs(force)) < convergence_tol, steps

    def jacobian_at(self, phases):
        """J_ij = -d(dtheta_i/dt)/d(theta_j) at the given phase
        configuration, for the linearized dynamics ddelta/dt = -J delta
        near equilibrium."""
        diff = phases[None, :] - phases[:, None]  # diff[i,j] = theta_j - theta_i
        off_diag = -self.k_coupling * self.W * np.cos(diff)
        J = off_diag.copy()
        np.fill_diagonal(J, -np.sum(off_diag, axis=1))
        return J


def find_equilibrium_lbfgs(W, k_coupling=1.0, seed=0, pin_node=0):
    """Fast equilibrium finding via L-BFGS on the Kuramoto potential
    V(theta) = -k_coupling * sum_{i<j} W_ij cos(theta_i - theta_j),
    with one node's phase pinned to remove the global rotational zero
    mode. Much faster than explicit Euler for graphs with a small
    spectral gap, where Euler needs many steps to resolve the slow mode."""
    from scipy.optimize import minimize
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, 2 * np.pi, n)
    free_idx = [i for i in range(n) if i != pin_node]

    def potential_and_grad(theta_free):
        theta = np.zeros(n)
        theta[free_idx] = theta_free
        theta[pin_node] = 0.0
        diff = theta[None, :] - theta[:, None]
        V = -k_coupling * 0.5 * np.sum(W * np.cos(diff))
        # gradient: dV/dtheta_i = k_coupling * sum_j W_ij sin(theta_i - theta_j)
        grad_full = k_coupling * np.sum(W * np.sin(theta[:, None] - theta[None, :]), axis=1)
        return V, grad_full[free_idx]

    result = minimize(potential_and_grad, theta0[free_idx], jac=True, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
    theta_final = np.zeros(n)
    theta_final[free_idx] = result.x
    theta_final[pin_node] = 0.0
    return theta_final % (2 * np.pi), result


def gauge_corrected_distance(theta_a, theta_b):
    """D = mean squared circular distance between theta_a and theta_b,
    minimized over a global phase offset -- removes the neutral global
    rotation mode so genuine relative-phase separation is measured, not
    harmless collective rotation."""
    shift = np.angle(np.mean(np.exp(1j * (theta_a - theta_b))))
    diff = np.angle(np.exp(1j * (theta_a - theta_b - shift)))
    return np.mean(diff ** 2)


def paired_trajectory_response(W, theta0, perturb_node, epsilon, steps, dt=0.05, k_coupling=1.0, record_every=1):
    """Runs baseline (from theta0) and perturbed (from theta0 with
    perturb_node displaced by epsilon) trajectories in lockstep from the
    SAME initial condition, returns the gauge-corrected distance D(t) at
    each recorded step. Sidesteps any need to identify 'the' equilibrium
    -- valid regardless of how many attractors the graph supports."""
    base = GraphOscillatorField(W, dt=dt, k_coupling=k_coupling)
    base.set_phases(theta0)
    pert = GraphOscillatorField(W, dt=dt, k_coupling=k_coupling)
    theta0_pert = theta0.copy()
    theta0_pert[perturb_node] = (theta0_pert[perturb_node] + epsilon) % (2 * np.pi)
    pert.set_phases(theta0_pert)

    D_history = [gauge_corrected_distance(pert.phases, base.phases)]
    for i in range(steps):
        base.step()
        pert.step()
        if (i + 1) % record_every == 0:
            D_history.append(gauge_corrected_distance(pert.phases, base.phases))
    return np.array(D_history)


def force_jacobian(W, theta, k_coupling=1.0):
    """DF_i/dtheta_k, the Jacobian of the coupling FORCE itself (not the
    stability Jacobian used in jacobian_at, which has an extra sign flip
    for -J*delta around a fixed point). This is what the tangent/
    variational equation ddelta/dt = DF(theta(t)) delta(t) needs, valid
    along any trajectory, not just at equilibrium.
    DF_ik = k_coupling * W_ik * cos(theta_k - theta_i), k != i
    DF_ii = -sum_{k!=i} DF_ik
    """
    diff = theta[None, :] - theta[:, None]  # diff[i,k] = theta_k - theta_i
    off_diag = k_coupling * W * np.cos(diff)
    DF = off_diag.copy()
    np.fill_diagonal(DF, -np.sum(off_diag, axis=1))
    return DF


def rotation_projector(n):
    """P = I - (1/n) * ones @ ones^T -- removes the global rotation mode."""
    return np.eye(n) - np.ones((n, n)) / n


def tangent_linear_response(W, theta0, perturb_node, steps, dt=0.05, k_coupling=1.0, record_every=1):
    """Integrates the nonlinear baseline trajectory theta(t) and the
    tangent vector delta(t) simultaneously: ddelta/dt = DF(theta(t))
    delta(t), starting from a unit-magnitude local perturbation at
    perturb_node, projected to remove the global rotation mode. Returns
    S(t) = ||P delta(t)||^2 / ||P delta(0)||^2 at each recorded step --
    exact infinitesimal response, no epsilon to choose."""
    n = len(theta0)
    P = rotation_projector(n)

    theta = theta0.copy()
    delta0 = np.zeros(n)
    delta0[perturb_node] = 1.0
    delta0 = P @ delta0  # e_i - (1/n)*1, i.e. project out global mode
    delta0 = delta0 / np.linalg.norm(delta0)  # unit magnitude
    delta = delta0.copy()

    norm0 = np.linalg.norm(P @ delta0) ** 2
    S_history = [1.0]  # S(0) = 1 by construction
    field = GraphOscillatorField(W, dt=dt, k_coupling=k_coupling)
    field.set_phases(theta)

    for i in range(steps):
        DF = force_jacobian(W, field.phases, k_coupling=k_coupling)
        # advance both simultaneously (theta first using its own dynamics,
        # delta using DF evaluated at the phases BEFORE this step -- matches
        # a simple explicit Euler co-integration)
        delta = delta + dt * (DF @ delta)
        field.step()
        if (i + 1) % record_every == 0:
            S = np.linalg.norm(P @ delta) ** 2 / norm0
            S_history.append(S)
    return np.array(S_history), delta


def tangent_linear_response_adaptive(W, theta0, perturb_node, t_span, k_coupling=1.0,
                                       rtol=1e-8, atol=1e-10, t_eval=None):
    """Same tangent-linear response as tangent_linear_response, but
    co-integrates theta(t) and delta(t) as one combined system using
    scipy's adaptive-step solver (RK45 with tight error tolerances)
    instead of a hand-picked fixed Euler step. This removes the need to
    guess a stable dt: the solver's own error control determines step
    size locally, and rtol/atol give a directly interpretable accuracy
    guarantee instead of an unverified stability assumption.

    Phases are left UNWRAPPED during integration (mod 2pi only applied
    for output/interpretation) since the dynamics depend only on phase
    DIFFERENCES, which are unaffected by wrapping -- this avoids any
    discontinuity that would violate the smoothness solve_ivp assumes.
    """
    from scipy.integrate import solve_ivp
    n = len(theta0)
    P = rotation_projector(n)

    delta0 = np.zeros(n)
    delta0[perturb_node] = 1.0
    delta0 = P @ delta0
    delta0 = delta0 / np.linalg.norm(delta0)

    y0 = np.concatenate([theta0, delta0])

    def rhs(t, y):
        theta = y[:n]
        delta = y[n:]
        diff = theta[None, :] - theta[:, None]
        dtheta = k_coupling * np.sum(W * np.sin(diff), axis=1)
        off_diag = k_coupling * W * np.cos(diff)
        DF = off_diag.copy()
        np.fill_diagonal(DF, -np.sum(off_diag, axis=1))
        ddelta = DF @ delta
        return np.concatenate([dtheta, ddelta])

    sol = solve_ivp(rhs, t_span, y0, method='RK45', rtol=rtol, atol=atol, t_eval=t_eval, dense_output=False)
    thetas = sol.y[:n, :].T
    deltas = sol.y[n:, :].T
    norm0 = np.linalg.norm(P @ delta0) ** 2
    S = np.array([np.linalg.norm(P @ d) ** 2 / norm0 for d in deltas])
    return sol.t, S, sol.success, sol.message


def joint_tangent_matrix_response(W, theta0, perturb_nodes, t_span, k_coupling=1.0,
                                    rtol=1e-6, atol=1e-8, t_eval=None, max_step=0.05,
                                    method='RK45'):
    """Joint integration of the baseline trajectory theta(t) and a TANGENT
    MATRIX Delta(t) in R^(n x k) for k perturbed nodes simultaneously --
    one baseline solve carries all k tangent columns, rather than
    resolving the baseline separately per node. Uses dense-output
    evaluation at a fixed t_eval grid (scipy's t_eval already does this
    via interpolation regardless of the adaptive solver's internal step
    choices) and a prespecified max_step so transient growth cannot be
    undersampled between observation points. method is exposed so an
    independent solver family (e.g. DOP853) can be substituted for
    cross-validation without duplicating this function."""
    from scipy.integrate import solve_ivp
    n = len(theta0)
    k = len(perturb_nodes)
    P = rotation_projector(n)

    Delta0 = np.zeros((n, k))
    for col, node in enumerate(perturb_nodes):
        e = np.zeros(n)
        e[node] = 1.0
        v = P @ e
        Delta0[:, col] = v / np.linalg.norm(v)

    y0 = np.concatenate([theta0, Delta0.flatten()])

    def rhs(t, y):
        theta = y[:n]
        Delta = y[n:].reshape(n, k)
        diff = theta[None, :] - theta[:, None]
        dtheta = k_coupling * np.sum(W * np.sin(diff), axis=1)
        off_diag = k_coupling * W * np.cos(diff)
        DF = off_diag.copy()
        np.fill_diagonal(DF, -np.sum(off_diag, axis=1))
        dDelta = DF @ Delta
        return np.concatenate([dtheta, dDelta.flatten()])

    sol = solve_ivp(rhs, t_span, y0, method=method, rtol=rtol, atol=atol,
                     t_eval=t_eval, max_step=max_step, dense_output=False)
    thetas = sol.y[:n, :].T
    Deltas = sol.y[n:, :].T.reshape(-1, n, k)

    norm0 = np.array([np.linalg.norm(P @ Delta0[:, col]) ** 2 for col in range(k)])
    S = np.zeros((len(sol.t), k))
    for ti in range(len(sol.t)):
        for col in range(k):
            S[ti, col] = np.linalg.norm(P @ Deltas[ti, :, col]) ** 2 / norm0[col]
    return sol.t, S, sol.success, sol.message


def adaptive_finite_difference_response(W, theta0, perturb_node, epsilon, t_span, k_coupling=1.0,
                                          rtol=1e-6, atol=1e-8, t_eval=None, max_step=0.05):
    """Finite-impulse baseline-vs-perturbed comparison, using the SAME
    adaptive solver machinery as the tangent-linear response (not the
    old fixed-step Euler), for revalidating agreement under the actual
    numerical method that will generate the Stage 1A results."""
    from scipy.integrate import solve_ivp
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

    diff_over_eps = (sol_pert.y - sol_base.y) / epsilon  # (n, T)
    projected = P @ diff_over_eps
    norm0 = np.linalg.norm(projected[:, 0]) ** 2
    S = np.array([np.linalg.norm(projected[:, ti]) ** 2 / norm0 for ti in range(projected.shape[1])])
    return sol_base.t, S
