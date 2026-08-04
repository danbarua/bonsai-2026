"""
Builds every graph construction Stage 1D needs and caches them to
results/stage1d_constructions.pkl:

- T, ink_mask_active, active_indices: reconstructed from scratch (first
  200 KMNIST class-0 training images, learned_topology_construction's
  default prune_threshold=0.9/ink_threshold=0.15), NOT loaded from
  class0_constructions.pkl -- this is because building the three
  stochastic controls' pilot realizations requires ink_mask_active,
  which class0_constructions.pkl does not cache. Verified below
  (build_and_verify_T) to reproduce class0_constructions.pkl's own T
  byte-exact (max abs diff 2.22e-16, float64 machine epsilon), the same
  guarantee tests/test_construction_driver.py's Tier-2 test already
  establishes -- so this reconstruction is equivalent to the cached T
  Stage 1B2/1C already used, not a new topology.
- lattice: loaded directly from class0_constructions.pkl (already
  verified byte-exact against the historical cached artifact; no reason
  to rebuild it).
- rewired / hist_random / curr_random: three fresh graph realizations
  each (seeds 0, 1, 2 -- chosen here since the design doc pins
  trajectory seeds but deliberately leaves graph-realization seeds
  unpinned), needed only for the non-confirmatory 3x3 pilot.

nodes_T = get_degree_stratified_nodes(T) is computed once and reused as
the fixed-coordinate intervention node set for every construction in
Stage 1D (lattice and all pilot realizations alike) -- this is the
"fixed graph coordinates" protocol from DESIGN.md: perturb the same
three T-defined node indices regardless of which role they play in
another construction's own degree distribution.
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "stage1b2_structured_transformation"))
from stage1b2_core import get_degree_stratified_nodes

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KMNIST_DIR = os.path.join(_THIS_DIR, "..", "..", "datasets", "kmnist")
CLASS0_CONSTRUCTIONS_PATH = os.path.join(
    _THIS_DIR, "..", "stage1b2_structured_transformation", "results", "class0_constructions.pkl")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
CACHE_PATH = os.path.join(RESULTS_DIR, "stage1d_constructions.pkl")

PILOT_REALIZATION_SEEDS = [0, 1, 2]  # graph-realization seeds; NOT pinned by DESIGN.md, chosen here


def build_and_verify_T(require_historical_verification=True):
    """Reconstructs active_indices, T, ink_mask_active from the raw KMNIST
    class-0 images. The construction itself never depended on
    class0_constructions.pkl -- only the verification step below did.

    Default (require_historical_verification=True, unchanged from
    before this parameter existed): asserts the result matches
    class0_constructions.pkl's cached T byte-exact, so this
    reconstruction is provably the same T Stage 1B2/1C/1D already used,
    not a new topology. Every caller from before this parameter was
    added gets exactly this behavior, still the default -- if the cache
    is absent, this still raises (now with a clearer message; previously
    a bare FileNotFoundError from the failed open()), not a silent
    behavior change.

    require_historical_verification=False (opt-in; added for Stage 2A's
    full-raw-data-regeneration path -- see stage2a_dynamics_
    classification/README.md's "Artifact replay vs. full raw-data
    regeneration"): if class0_constructions.pkl is present, verifies
    against it exactly as above (still checked when available, not
    skipped just because it's optional); if absent, reconstructs T from
    scratch and skips the comparison rather than raising. Returns
    cached=None in that branch since there is no cache to return."""
    X_train, y_train, _, _ = load_mnist(KMNIST_DIR, gz=False)
    idx = np.where(y_train == 0)[0][:200]
    images = X_train[idx].astype(np.float64) / 255.0
    active_indices, W_T = build_class_topology(images)

    cached = None
    if os.path.exists(CLASS0_CONSTRUCTIONS_PATH):
        with open(CLASS0_CONSTRUCTIONS_PATH, "rb") as f:
            cached = pickle.load(f)[0]
        assert len(active_indices) == cached["n_active"]
        max_diff = np.max(np.abs(W_T - cached["constructions"]["T"]))
        assert max_diff < 1e-9, f"reconstructed T does not match cached artifact (max diff {max_diff})"
    elif require_historical_verification:
        raise FileNotFoundError(
            f"{CLASS0_CONSTRUCTIONS_PATH} not present locally, and "
            f"require_historical_verification=True (the default). Either "
            f"provide that historical artifact, or explicitly call "
            f"build_and_verify_T(require_historical_verification=False) to "
            f"reconstruct T from scratch without the byte-exact check "
            f"against it.")
    else:
        print(f"NOTE: {CLASS0_CONSTRUCTIONS_PATH} not present locally -- "
              f"reconstructed T from scratch WITHOUT verifying it matches "
              f"the historical cached artifact byte-exact "
              f"(require_historical_verification=False). The construction "
              f"itself is identical either way; only this verification "
              f"step was skipped.")

    mean_intensity = images.mean(axis=0).flatten()
    ink_mask_active = (mean_intensity > 0.15)[active_indices]
    return active_indices, W_T, ink_mask_active, cached


def build_pilot_realization(family, seed, W_T, ink_mask_active):
    """One graph realization of one stochastic-control family."""
    if family == "rewired":
        W, _info = degree_preserving_rewire(W_T, ink_mask_active, seed=seed)
        return W
    elif family == "hist_random":
        target = W_T.sum(axis=1).mean()
        raw = generate_historical_matched_sparsity_random(W_T, ink_mask_active, seed=seed)
        return rescale_to_common_budget(raw, target)
    elif family == "curr_random":
        return generate_matched_sparsity_topology(W_T, ink_mask_active, seed=seed)
    else:
        raise ValueError(f"unknown family: {family}")


def build_all(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    active_indices, W_T, ink_mask_active, cached = build_and_verify_T()
    nodes_T = get_degree_stratified_nodes(W_T)
    W_lattice = cached["constructions"]["lattice"]

    pilot_constructions = {}
    for family in ["rewired", "hist_random", "curr_random"]:
        pilot_constructions[family] = {}
        for seed in PILOT_REALIZATION_SEEDS:
            pilot_constructions[family][seed] = build_pilot_realization(
                family, seed, W_T, ink_mask_active)

    data = {
        "active_indices": active_indices,
        "W_T": W_T,
        "ink_mask_active": ink_mask_active,
        "nodes_T": nodes_T,
        "n_active": len(active_indices),
        "W_lattice": W_lattice,
        "pilot_constructions": pilot_constructions,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    data = build_all(force=True)
    print(f"n_active={data['n_active']}, nodes_T={data['nodes_T']}")
    print(f"lattice: shape={data['W_lattice'].shape}, "
          f"total weight={np.sum(data['W_lattice']):.6f} (T total={np.sum(data['W_T']):.6f})")
    for family, realizations in data["pilot_constructions"].items():
        for seed, W in realizations.items():
            print(f"{family} seed={seed}: n_edges={np.count_nonzero(np.triu(W, 1))}, "
                  f"mean_weighted_degree={W.sum(axis=1).mean():.6f}")
