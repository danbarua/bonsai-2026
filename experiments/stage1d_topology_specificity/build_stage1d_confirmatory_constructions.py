"""
Builds the 25-realization-per-family confirmatory graph set for Stage 1D's
locked (R=25, K=3) design (DESIGN.md, "Locked confirmatory-run allocation").

Fresh seeds throughout, continuing past the pilot's own range (rewired and
curr_random used seeds 0-2; hist_random used 0-4 after its variance
follow-up) -- this is an explicit choice, not something DESIGN.md or
PILOT_RESULTS.md pins: keeping confirmatory and pilot draws disjoint avoids
any non-independence concern between the two runs. All three families
start candidate seeds at 5.

- rewired, curr_random: 25 realizations, seeds 5..29 directly -- neither
  family showed full fixed-coordinate degeneracy in the pilot (rewired
  provably cannot, by construction; curr_random showed one mild,
  non-fatal instance), so no pre-screening/replacement-draw protocol is
  applied to them, per DESIGN.md.
- hist_random: DESIGN.md's locked pre-screening protocol. For each
  candidate seed (starting at 5, incrementing), compute the weighted
  degree of nodes_T's three fixed coordinates BEFORE any simulation; if
  any is zero (isolated), reject without simulating and record the
  rejected seed + its degrees; otherwise accept. Continue drawing
  candidates until 25 evaluable realizations are obtained.

Reuses build_pilot_realization() and build_and_verify_T() from
build_stage1d_constructions.py unchanged -- no reimplementation of the
construction recipes.
"""
import os
import pickle

import numpy as np

from build_stage1d_constructions import build_and_verify_T, build_pilot_realization

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
CACHE_PATH = os.path.join(RESULTS_DIR, "stage1d_confirmatory_constructions.pkl")

N_REALIZATIONS = 25
CANDIDATE_SEED_START = 5  # continues past the pilot's own seed range (max used: 4, hist_random)


def build_fixed_family(family, W_T, ink_mask_active):
    """rewired / curr_random: no pre-screening needed -- straight seeds
    5..5+N_REALIZATIONS-1."""
    realizations = {}
    for i in range(N_REALIZATIONS):
        seed = CANDIDATE_SEED_START + i
        realizations[seed] = build_pilot_realization(family, seed, W_T, ink_mask_active)
    return realizations, list(realizations.keys()), []


def build_hist_random_with_prescreening(W_T, ink_mask_active, nodes_T):
    """DESIGN.md's locked protocol: pre-screen every candidate before
    simulating anything, reject isolated-node draws, redraw until 25
    evaluable realizations. Every rejected candidate's seed and degrees
    are recorded -- required disclosed output, not discarded."""
    accepted = {}
    accepted_seeds = []
    rejected_log = []  # list of dicts: seed, degrees, isolated_labels

    seed = CANDIDATE_SEED_START
    while len(accepted) < N_REALIZATIONS:
        W = build_pilot_realization("hist_random", seed, W_T, ink_mask_active)
        deg = W.sum(axis=1)
        degrees = {label: float(deg[idx]) for label, idx in nodes_T.items()}
        isolated = [label for label, idx in nodes_T.items() if deg[idx] < 1e-9]

        if isolated:
            rejected_log.append({"seed": seed, "degrees": degrees, "isolated_labels": isolated})
        else:
            accepted[seed] = W
            accepted_seeds.append(seed)
        seed += 1

    return accepted, accepted_seeds, rejected_log


def build_all(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    active_indices, W_T, ink_mask_active, cached = build_and_verify_T()
    import sys
    sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1b2_structured_transformation"))
    from stage1b2_core import get_degree_stratified_nodes
    nodes_T = get_degree_stratified_nodes(W_T)

    rewired_realizations, rewired_seeds, _ = build_fixed_family("rewired", W_T, ink_mask_active)
    curr_random_realizations, curr_random_seeds, _ = build_fixed_family("curr_random", W_T, ink_mask_active)
    hist_random_realizations, hist_random_seeds, hist_random_rejected = \
        build_hist_random_with_prescreening(W_T, ink_mask_active, nodes_T)

    data = {
        "active_indices": active_indices,
        "W_T": W_T,
        "ink_mask_active": ink_mask_active,
        "nodes_T": nodes_T,
        "n_active": len(active_indices),
        "constructions": {
            "rewired": rewired_realizations,
            "curr_random": curr_random_realizations,
            "hist_random": hist_random_realizations,
        },
        "accepted_seeds": {
            "rewired": rewired_seeds,
            "curr_random": curr_random_seeds,
            "hist_random": hist_random_seeds,
        },
        "hist_random_rejected_candidates": hist_random_rejected,
        "candidate_seed_start": CANDIDATE_SEED_START,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    data = build_all(force=True)
    print(f"n_active={data['n_active']}, nodes_T={data['nodes_T']}")
    for family, seeds in data["accepted_seeds"].items():
        print(f"{family}: {len(seeds)} realizations, seeds={seeds}")
    rejected = data["hist_random_rejected_candidates"]
    n_drawn = len(data["accepted_seeds"]["hist_random"]) + len(rejected)
    print(f"\nhist_random pre-screening: {len(rejected)} rejected out of {n_drawn} candidates drawn "
          f"({100*len(rejected)/n_drawn:.1f}% rejection rate)")
    for r in rejected:
        print(f"  REJECTED seed={r['seed']}: degrees={r['degrees']}, isolated={r['isolated_labels']}")
