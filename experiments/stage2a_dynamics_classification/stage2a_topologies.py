"""
Stage 2A's confirmatory-expansion graphs (DESIGN.md, "Confirmatory
expansion: graph-specific, not family-general, unless scaled up"):
T, lattice, canonical rewired (seed=0), canonical current-random
(seed=0) -- all four using T's active_indices/nodes_T, all reused
verified-correct constructions, none reimplemented here.

Locked seeds, disclosed rationale (DESIGN.md): rewired and curr_random
both seed=0, reused from Stage 1D's own pilot artifacts rather than
drawn fresh -- named before any Stage 2A result exists, so "canonical"
cannot become outcome-selected.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))

from build_stage1d_constructions import build_and_verify_T  # noqa: E402
from bonsai.dynamics.degree_preserving_rewiring import degree_preserving_rewire  # noqa: E402
from bonsai.dynamics.matched_sparsity_ablation import generate_matched_sparsity_topology  # noqa: E402

CANONICAL_SEED = 0


def build_all_topologies():
    """Returns (active_indices, ink_mask_active, nodes_T, topologies) where
    topologies = {'T': W_T, 'lattice': W_lattice, 'rewired': W_rewired,
    'curr_random': W_curr_random}, all sharing T's 505-node active
    support and ordering (DESIGN.md's locked requirement)."""
    active_indices, W_T, ink_mask_active, cached = build_and_verify_T()
    W_lattice = cached["constructions"]["lattice"]
    W_rewired, _info = degree_preserving_rewire(W_T, ink_mask_active, seed=CANONICAL_SEED)
    W_curr_random = generate_matched_sparsity_topology(W_T, ink_mask_active, seed=CANONICAL_SEED)

    import sys as _sys
    _sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1b2_structured_transformation"))
    from stage1b2_core import get_degree_stratified_nodes
    nodes_T = get_degree_stratified_nodes(W_T)

    topologies = {"T": W_T, "lattice": W_lattice, "rewired": W_rewired, "curr_random": W_curr_random}
    return active_indices, ink_mask_active, nodes_T, topologies


if __name__ == "__main__":
    import numpy as np
    active_indices, ink_mask_active, nodes_T, topologies = build_all_topologies()
    print(f"n_active={len(active_indices)}, nodes_T={nodes_T}")
    for name, W in topologies.items():
        n_edges = np.count_nonzero(np.triu(W, 1))
        deg = W.sum(axis=1)
        print(f"{name}: n_edges={n_edges}, total_weight={W.sum():.4f}, "
              f"mean_deg={deg.mean():.4f}, min_deg={deg.min():.4f}, max_deg={deg.max():.4f}")
