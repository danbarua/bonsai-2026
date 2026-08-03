"""
JAX/diffrax port of stage2a_core.py's evolve_on_graph -- plain
unperturbed graph evolution only (no tangent system, no perturbation),
since Stage 2A never needed that half of Stage 1D's
run_one_trial_jax_faithful.py. Reuses that file's already-verified
`rhs` (the plain-evolution half, lines 115-117 there) directly:

    dtheta/dt = k_coupling * sum_j W_ij sin(theta_j - theta_i)

Encoding (_local_converged_phases) stays on CPU/numpy -- timing showed
it is ~80x cheaper than evolution (4.6 ms/image vs 365.6 ms/image for
one topology), so porting it would not materially change throughput.
Only the graph-evolution ODE solve is ported.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

T_HORIZON = 2.5
RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05


def evolve_on_graph_jax(theta0, W, k_coupling=1.0):
    """Same contract as stage2a_core.evolve_on_graph, minus the CPU
    recovery-policy retries (a batched/jitted GPU computation can't
    branch per-trial the way the numpy retry loop does -- see
    verify_evolve_on_graph_jax.py for how this is checked against real
    data before being trusted). Returns (theta_T, success_bool) --
    success is checked via diffrax's own solver-status result, not
    assumed; a batched/jitted computation can't raise per-trial, so
    the caller must gate on this exactly like the numpy path's
    diag['failed']."""
    n = theta0.shape[0]

    def rhs(t, theta, args):
        diff = theta[None, :] - theta[:, None]
        return k_coupling * jnp.sum(W * jnp.sin(diff), axis=1)

    stepsize_controller = diffrax.PIDController(rtol=RTOL, atol=ATOL, dtmax=MAX_STEP)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Tsit5(), t0=0.0, t1=T_HORIZON, dt0=0.01,
        y0=theta0, saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller, max_steps=200_000)
    theta_T = sol.ys[0] % (2 * jnp.pi)
    success = diffrax.is_successful(sol.result)
    return theta_T, success


batched_evolve_on_graph_jax = jax.jit(jax.vmap(evolve_on_graph_jax, in_axes=(0, None)))
