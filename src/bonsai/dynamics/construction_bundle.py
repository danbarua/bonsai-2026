"""
Ties the four matched graph constructions (learned topology T,
degree-preserving rewiring, current edge-count-matched random, regular
lattice) into one per-class bundle, matching class0_constructions.pkl's
format: {'constructions': {'T', 'rewired', 'random', 'lattice'},
'n_active'}.

Class 0 only, for now (scope explicitly agreed) -- this function is
dataset/class-agnostic (any per-class image batch), but has only been
run and verified against class 0.

'random' here uses matched_sparsity_ablation.py's **current
edge-count-matched random** (same edge count as T, T's own values
redistributed onto new positions, no rescaling) -- confirmed NOT to
reproduce the historical cached artifact under any of 10 candidate seeds
swept (0-9), and the mismatch is structural (edge count, value scale),
not a seed problem: the cached class0_constructions.pkl's 'random' has
roughly half T's edge count (1090 vs 2102) and per-edge values scaled up
so its mean weighted degree is bit-identical to T's own
(3.8811482995463593). A separate, structurally-verified reconstruction
of that historical construction -- **historical half-edge random,
coupling-budget normalized** -- lives in
historical_matched_sparsity_random.py, not wired into this bundle by
default; callers wanting the historically-faithful control should build
it directly from that module instead of this bundle's 'random' key.

'rewired', by contrast, DOES reproduce exactly -- at seed=1 (confirmed
byte-exact for class 0). This is therefore no longer an undocumented
seed for class 0 specifically, though the function below still defaults
to seed=0 as a generic, class-agnostic choice (callers reproducing the
historical class-0 artifact should pass rewired_seed=1 explicitly, as
tests/test_construction_driver.py's Tier-2 check does).
"""
import numpy as np

from bonsai.dynamics.learned_topology_construction import build_class_topology
from bonsai.dynamics.degree_preserving_rewiring import degree_preserving_rewire
from bonsai.dynamics.matched_sparsity_ablation import generate_matched_sparsity_topology
from bonsai.dynamics.lattice_construction import build_lattice_topology


def build_class_construction_bundle(images, ink_threshold=0.15, prune_threshold=0.9,
                                     n_swaps_multiplier=10, rewired_seed=0, random_seed=0,
                                     side=28):
    """Builds all four matched constructions for one class's images.

    Parameters
    ----------
    images : (n, 28, 28) array, values in [0, 1] (already normalized).

    Returns
    -------
    dict : {'constructions': {'T', 'rewired', 'random', 'lattice'},
            'n_active': int}
    """
    active_indices, W_T = build_class_topology(
        images, prune_threshold=prune_threshold, ink_threshold=ink_threshold)

    mean_intensity = images.mean(axis=0).flatten()
    ink_mask_active = (mean_intensity > ink_threshold)[active_indices]

    W_rewired, _rewire_info = degree_preserving_rewire(
        W_T, ink_mask_active, n_swaps_multiplier=n_swaps_multiplier, seed=rewired_seed)
    W_random = generate_matched_sparsity_topology(W_T, ink_mask_active, seed=random_seed)
    W_lattice = build_lattice_topology(active_indices, total_weight_target=np.sum(W_T), side=side)

    return {
        'constructions': {'T': W_T, 'rewired': W_rewired, 'random': W_random, 'lattice': W_lattice},
        'n_active': len(active_indices),
    }
