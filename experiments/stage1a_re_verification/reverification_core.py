"""
Shared construction + AUC computation for the Stage 1A re-verification
described in DESIGN.md. Reuses T and lattice (both deterministic) from
build_all_class_topologies.py's cached output, and builds the three
stochastic controls (degree-preserving rewiring, historical half-edge
random coupling-budget normalized, current edge-count-matched random)
fresh at 25 seeds each, per class -- these are not cached anywhere at
more than one seed. Adds no new construction algorithm of its own.
"""
import os
import pickle

import numpy as np

from bonsai.data.mnist_loader import load_mnist
from bonsai.dynamics.learned_topology_construction import build_class_topology
from bonsai.dynamics.degree_preserving_rewiring import degree_preserving_rewire
from bonsai.dynamics.matched_sparsity_ablation import generate_matched_sparsity_topology
from bonsai.dynamics.historical_matched_sparsity_random import (
    generate_historical_matched_sparsity_random, rescale_to_common_budget,
)
from bonsai.dynamics.graph_oscillator_field import joint_tangent_matrix_response

N_PER_CLASS = 200          # matches build_all_class_topologies.py / the historically
                            # recovered hyperparameter for T
INK_THRESHOLD = 0.15
PRUNE_THRESHOLD = 0.9
REWIRE_SWAPS_MULTIPLIER = 10
N_SEEDS = 25                # seeds 0..24, per DESIGN.md
THETA0_SEED_BASE = 4000     # theta0 seed = THETA0_SEED_BASE + class_idx: one fixed initial
                            # phase vector per class, shared across every construction and
                            # seed for that class (matching Stage 1A's original "one shared
                            # initial phase vector per class" design). This specific seed is
                            # a new, documented choice for this re-verification -- the
                            # original run's initial conditions were never recorded (see
                            # stage1a_infinitesimal_response/FINDINGS.md's "Honest
                            # limitations").
T_SPAN = (0, 2.5)
T_EVAL = np.linspace(0, 2.5, 51)

STOCHASTIC_CONTROLS = ("rewired", "hist_random", "curr_random")
ALL_CONSTRUCTIONS = ("T", "lattice") + STOCHASTIC_CONTROLS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
KMNIST_DIR = os.path.join(REPO_ROOT, "datasets", "kmnist")
ALL_CLASSES_PATH = os.path.join(
    REPO_ROOT, "experiments", "stage0_simulator_calibration", "results",
    "stage1a_all_classes.pkl")
RESULTS_PATH = os.path.join(_THIS_DIR, "results", "stage1a_reverification_results.pkl")


def load_cached_deterministic_constructions():
    """T and lattice (both deterministic), plus n_active, for all 10
    classes -- from build_all_class_topologies.py's committed output."""
    with open(ALL_CLASSES_PATH, "rb") as f:
        return pickle.load(f)


def get_degree_stratified_nodes(W):
    """Same low/median/high weighted-degree node selection as
    stage1b2_core.py's get_degree_stratified_nodes -- reimplemented here
    (not imported) since that module lives in an unrelated experiment
    folder and this is a two-line function; behavior is identical."""
    degree = W.sum(axis=1)
    order = np.argsort(degree)
    n = len(order)
    return {
        "low": int(order[n // 10]),
        "median": int(order[n // 2]),
        "high": int(order[-n // 10]),
    }


def build_class_setup(class_idx, cached_T):
    """Rebuilds active_indices and the ink mask over active nodes for one
    class from the raw images -- neither is stored in
    stage1a_all_classes.pkl (which keeps only the resulting weight
    matrices). Also re-derives T from scratch and asserts it matches the
    cached artifact byte-exact, as an integrity check on this rebuild
    rather than a trusted assumption.

    Returns dict: W_T, lattice (filled in by caller), ink_mask_active,
    nodes (degree-stratified, computed from T), theta0, C (T's own mean
    weighted degree, the common coupling-budget target for this class).
    """
    X_train, y_train, _, _ = load_mnist(KMNIST_DIR, gz=False)
    idx = np.where(y_train == class_idx)[0][:N_PER_CLASS]
    images = X_train[idx].astype(np.float64) / 255.0

    active_indices, W_T_rebuilt = build_class_topology(
        images, prune_threshold=PRUNE_THRESHOLD, ink_threshold=INK_THRESHOLD)
    if not np.array_equal(W_T_rebuilt, cached_T):
        raise RuntimeError(
            f"class {class_idx}: rebuilt T does not byte-exact match "
            f"stage1a_all_classes.pkl's cached T -- rebuild is not "
            f"reproducing the cached construction as expected.")

    mean_intensity = images.mean(axis=0).flatten()
    ink_mask_active = (mean_intensity > INK_THRESHOLD)[active_indices]

    nodes = get_degree_stratified_nodes(cached_T)
    theta0 = np.random.default_rng(THETA0_SEED_BASE + class_idx).uniform(
        0, 2 * np.pi, len(active_indices))
    C = float(cached_T.sum(axis=1).mean())

    return {
        "W_T": cached_T,
        "ink_mask_active": ink_mask_active,
        "nodes": nodes,
        "theta0": theta0,
        "C": C,
    }


def build_stochastic_construction(name, W_T, ink_mask_active, seed, target_c):
    """Builds one raw stochastic control at one seed, then rescales it to
    the class's common coupling budget (a no-op, up to float64 rounding,
    for rewired and curr_random, whose mean weighted degree already
    equals target_c by construction -- see rescale_to_common_budget's
    docstring)."""
    if name == "rewired":
        raw, _info = degree_preserving_rewire(
            W_T, ink_mask_active, n_swaps_multiplier=REWIRE_SWAPS_MULTIPLIER, seed=seed)
    elif name == "hist_random":
        raw = generate_historical_matched_sparsity_random(W_T, ink_mask_active, seed=seed)
    elif name == "curr_random":
        raw = generate_matched_sparsity_topology(W_T, ink_mask_active, seed=seed)
    else:
        raise ValueError(f"unknown stochastic construction {name!r}")
    return rescale_to_common_budget(raw, target_c)


def compute_construction_auc(W, nodes, theta0):
    """Class-level AUC for one construction instance: mean of the three
    nodewise AUCs (low/median/high-degree-in-T nodes, the same node
    identities used for every construction of this class), trapezoidal
    over the fixed 51-point t in [0, 2.5] grid -- matching
    stage1a_infinitesimal_response/FINDINGS.md's methodology
    (joint_tangent_matrix_response, max_step=0.05, rtol=1e-6, atol=1e-8,
    RK45)."""
    node_list = [nodes["low"], nodes["median"], nodes["high"]]
    t, S, ok, msg = joint_tangent_matrix_response(
        W, theta0, node_list, t_span=T_SPAN, t_eval=T_EVAL, max_step=0.05,
        rtol=1e-6, atol=1e-8, method="RK45")
    if not ok:
        raise RuntimeError(f"solve_ivp failed: {msg}")
    aucs = [np.trapezoid(S[:, i], t) for i in range(len(node_list))]
    return float(np.mean(aucs))
