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

**Amended by external review, closing README.md's "Artifact replay vs.
full raw-data regeneration" gap, option (2)**: every construction is
now built from scratch from raw KMNIST, with no hard dependency on the
gitignored, cross-stage class0_constructions.pkl. Previously, `lattice`
was read directly from that cache (`cached["constructions"]["lattice"]`)
rather than reconstructed, and `T`'s own reconstruction hard-required
the cache to be present just to verify against (raising if absent, even
though the construction itself never needed it). Both fixed:
- `T` now calls `build_and_verify_T(require_historical_verification=
  False)` -- reconstructs identically either way; verifies against the
  cache opportunistically if present, skips the check (not the
  construction) if not, rather than raising.
- `lattice` now calls `build_lattice_topology` directly (the same
  function `tests/test_lattice_construction.py`'s own Tier-2 test
  already verifies byte-exact against a historical artifact), instead
  of reading a cached value.

Verified before committing, not assumed: with class0_constructions.pkl
present locally, this new from-scratch path was compared directly
against the prior cache-reading path's output -- byte-identical for
both `T` and `lattice` (see the commit message for the exact
comparison). `rewired`/`curr_random` were already fully
cache-independent (derived from freshly-reconstructed `W_T`/
`ink_mask_active`, never from the cache) -- unchanged here.
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "stage1d_topology_specificity"))

from build_stage1d_constructions import build_and_verify_T  # noqa: E402
from bonsai.dynamics.degree_preserving_rewiring import degree_preserving_rewire  # noqa: E402
from bonsai.dynamics.matched_sparsity_ablation import generate_matched_sparsity_topology  # noqa: E402
from bonsai.dynamics.lattice_construction import build_lattice_topology  # noqa: E402

CANONICAL_SEED = 0


def build_all_topologies():
    """Returns (active_indices, ink_mask_active, nodes_T, topologies) where
    topologies = {'T': W_T, 'lattice': W_lattice, 'rewired': W_rewired,
    'curr_random': W_curr_random}, all sharing T's 505-node active
    support and ordering (DESIGN.md's locked requirement). Every
    construction is built from scratch from raw KMNIST -- see module
    docstring for what changed and why."""
    active_indices, W_T, ink_mask_active, _cached = build_and_verify_T(
        require_historical_verification=False)
    W_lattice = build_lattice_topology(active_indices, total_weight_target=np.sum(W_T), side=28)
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
