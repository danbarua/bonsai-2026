"""
Stage 2A shared pipeline code: encoding, active-node restriction, graph
evolution, and the locked reference-node gauge -- implementing
DESIGN.md's locked pipeline exactly, reused across every feasibility
stage and the eventual confirmatory run rather than reimplemented per
stage.

    theta_0^784 = _local_converged_phases(x)          # encode
    theta_0^505 = theta_0^784[active_indices]          # restrict
    theta_T^505 = evolve_on_graph(theta_0^505, W, T_HORIZON)

Feature representation (DESIGN.md, "Feature representation"):
reference-node gauge is primary (theta_ref = T's median-degree node,
active index 363), circular-mean is secondary/robustness. The reference
node's own two circular features are trivially constant
(cos(0)=1, sin(0)=0) for every image -- dropped deterministically,
giving an effective feature dimension of 1008, not 1010.
"""
import os
import sys
import warnings

import numpy as np
from scipy.integrate import solve_ivp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1b2_structured_transformation"))

from build_stage1d_constructions import build_and_verify_T  # noqa: E402
from stage1b2_core import get_degree_stratified_nodes  # noqa: E402
from bonsai.dynamics.learned_topology_construction import _local_converged_phases  # noqa: E402

T_HORIZON = 2.5
RTOL, ATOL, MAX_STEP = 1e-6, 1e-8, 0.05
ENCODER_SEED = 0  # locked primary encoder seed (DESIGN.md, "Encoding")

# ---- Recovery policy for a failed graph-evolution solve (DESIGN.md's
# fourth-review correction: NOT "retry with tighter tolerance", which
# demands more of the solver -- smaller MAX_STEP first, then a further-
# reduced MAX_STEP (a pragmatic stand-in for "increased max_steps
# allowance": forcing a smaller step mechanically forces the solver to
# take more, smaller steps), then a prespecified alternative solver
# (Radau, for stiffness). ----
RECOVERY_STEPS = [
    {"method": "RK45", "max_step": MAX_STEP},          # primary
    {"method": "RK45", "max_step": MAX_STEP / 5},       # smaller MAX_STEP
    {"method": "RK45", "max_step": MAX_STEP / 25},      # more, smaller steps
    {"method": "Radau", "max_step": MAX_STEP},          # alternative solver
]


def load_T():
    """Returns (active_indices, W_T, ink_mask_active, nodes_T). W_T and
    active_indices are verified byte-exact against class0_constructions.pkl's
    cached T inside build_and_verify_T() itself -- not reimplemented here."""
    active_indices, W_T, ink_mask_active, _cached = build_and_verify_T()
    nodes_T = get_degree_stratified_nodes(W_T)
    return active_indices, W_T, ink_mask_active, nodes_T


def encode_and_restrict(image_01, active_indices, seed=ENCODER_SEED):
    """image_01: (28, 28) array in [0, 1]. Returns theta_0^505 restricted
    to active_indices, via the unmodified, already-established
    `_local_converged_phases` encoder (150 steps, dt=0.1, k_coupling=1.0,
    k_bias=1.0, perturbation_std=0.01, all defaults, per DESIGN.md's
    'reuse the established convention' choice)."""
    theta_0_784 = _local_converged_phases(image_01, seed=seed).flatten()
    return theta_0_784[active_indices]


def evolve_on_graph(theta0, W, t_horizon=T_HORIZON):
    """Evolves theta0 under dtheta/dt = sum_j W_ij sin(theta_j - theta_i)
    (unperturbed, no tangent system -- plain graph evolution of an
    encoded state) to t_horizon, with the locked recovery policy on
    failure. Returns (theta_T, diagnostics) where diagnostics records
    which recovery step succeeded (0 = primary, no recovery needed)."""
    def rhs(t, theta):
        diff = theta[None, :] - theta[:, None]
        return np.sum(W * np.sin(diff), axis=1)

    last_message = None
    for step_idx, params in enumerate(RECOVERY_STEPS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(rhs, (0, t_horizon), theta0, method=params["method"],
                             rtol=RTOL, atol=ATOL, max_step=params["max_step"])
        if sol.success:
            theta_T = sol.y[:, -1] % (2 * np.pi)
            return theta_T, {"recovery_step": step_idx, "method": params["method"],
                              "solver_message": sol.message}
        last_message = sol.message

    # All recovery steps exhausted: a genuine, disclosed solver failure --
    # not silently absorbed (DESIGN.md's "zero silent solver failures").
    return None, {"recovery_step": None, "method": None, "solver_message": last_message,
                   "failed": True}


def order_parameter(theta):
    """R(theta) = |mean(exp(i*theta))| -- required dynamical diagnostic,
    recorded for every state, never used to change the gauge (DESIGN.md)."""
    return float(np.abs(np.mean(np.exp(1j * theta))))


def reference_node_features(theta, ref_idx):
    """Locked primary gauge: h(theta) = [cos(theta_i - theta_ref),
    sin(theta_i - theta_ref) for all i], with the two trivially-constant
    columns for ref_idx itself dropped deterministically (DESIGN.md's
    fourth-review correction). Returns a (2*n - 2,) vector.

    The dropped pair is asserted to be exactly (1.0, 0.0) before removal
    -- this is the property tests/test_stage2a_core.py checks directly."""
    shifted = theta - theta[ref_idx]
    cos_part = np.cos(shifted)
    sin_part = np.sin(shifted)
    assert abs(cos_part[ref_idx] - 1.0) < 1e-12
    assert abs(sin_part[ref_idx] - 0.0) < 1e-12
    cos_part = np.delete(cos_part, ref_idx)
    sin_part = np.delete(sin_part, ref_idx)
    return np.concatenate([cos_part, sin_part])


def circular_mean_features(theta):
    """Secondary robustness gauge: circular-mean centering (DESIGN.md)."""
    mu = np.angle(np.mean(np.exp(1j * theta)))
    shifted = theta - mu
    return np.concatenate([np.cos(shifted), np.sin(shifted)])
