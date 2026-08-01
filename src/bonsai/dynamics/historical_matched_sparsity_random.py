"""
Label for unambiguous reference in reports: **historical half-edge
random, coupling-budget normalized** -- an independently-placed, sparser
random support (~half the real topology's edge count), values drawn from
the real topology's own weight pool, then rescaled so its mean weighted
degree matches the real topology's exactly. Distinct from
matched_sparsity_ablation.py's **current edge-count-matched random**
(same edge count as the real topology, no rescaling). Do not call both
"matched-sparsity random" without qualification -- that's an actively
different null model, not a naming detail.

Reconstruction of the historical "matched-sparsity random" construction
used to build class0_constructions.pkl's cached 'random' -- NOT the same
algorithm as the current matched_sparsity_ablation.py (preserved
unchanged; see its own docstring, which describes a deliberately
different design: same edge count as the real topology, values resampled
onto new positions, no final rescaling). Git history shows only one
commit for that file (the tarball decant), with no evidence of a step
having been dropped -- its docstring reads as an intentional, different
design, not a partial port of this one.

Recovered from tarballs/random_control_handoff/ (DESIGN_DOC_EXCERPT.md,
kmnist_c0_controls.npz [pre-normalization], kmnist_c0_controls_normalized.npz
[post-normalization]) plus direct inspection of both files.

CONFIRMED, not assumed:
- Final rescaling: A_tilde = A * (C / mean_weighted_degree(A)), where
  C = mean_weighted_degree(real_topology) -- verified byte-exact
  (max abs diff 0.0) against the real historical raw-to-normalized pair
  for all four constructions (T, rewired, random, lattice), not just
  random. Applied here to 'random' specifically, since it's the only one
  of the four whose raw mean weighted degree differs from C (T and
  rewired hit C by construction already; lattice's build_lattice_topology
  already targets C directly via total_weight_target).
- Support is independently sampled from the eligible (non-background-
  background) pool, not derived from T's specific edge positions: the
  historical raw artifact's overlap with T's own edges (4 of 552 unique
  edges) matches almost exactly what pure uniform-random sampling from
  the ~120,000-pair eligible pool would predict by chance
  (552 * 1051/120000 ~ 4.83), not some structured relationship.
- Edge weight values are drawn from the real topology's own nonzero
  value pool: 100% of the historical raw artifact's 552 unique nonzero
  values are found exactly in T's own value set.

NOT CONFIRMED, reported openly rather than concealed:
- The exact edge-count rule. The two known historical realizations (552
  unique edges in the raw npz, 545 in the final cached pkl -- different
  runs, evidenced by their different counts despite an identical
  post-rescaling mean weighted degree) are both close to, but do not
  exactly match, any of the simple deterministic candidates tested:
  floor(|E_T|/2)=525, round/ceil(|E_T|/2)=526, a fixed fraction of the
  ~120,000-pair eligible pool, or a count tied to the number of "ink"
  active nodes (384). The spread between the two counts is consistent
  with a stochastic (e.g. Bernoulli-style) count-generating process
  rather than a fixed formula, but this isn't confirmed either -- two
  data points cannot distinguish between these possibilities. This
  module defaults n_edges to round(|E_T| / 2) as the best-available
  approximation, NOT a proven exact rule -- callers reproducing a
  specific known artifact should pass its exact recorded count
  explicitly (as tests/test_historical_random_construction.py's Tier-2
  test does, using the exact recorded counts 552 and 545).
- The exact seed and RNG call order. Swept seeds 0-199 across three
  plausible call-order variants (shuffle-then-choose, choose-then-
  shuffle, choose-values-directly-from-pool) against the raw npz's
  'random' construction using its exact known edge count (552) -- no
  exact or near match found in any. The historical .npz/.pkl artifacts
  are therefore NOT reproduced byte-exact by this module; only
  structural equivalence (support independence, value-pool origin, and
  post-rescaling mean weighted degree) is established.
"""
import numpy as np


def generate_historical_matched_sparsity_random(real_topology, ink_mask, seed, n_edges=None):
    """real_topology: (N,N) the actual learned topology, used to get the
    eligible pool, the value pool, and (if n_edges is None) the default
    ~half-count target. ink_mask: (N,) boolean, which nodes are 'ink'.
    n_edges: exact edge count to place; defaults to round(|E_T|/2), the
    best-available (not proven exact) approximation of the historical
    rule -- see module docstring."""
    N = real_topology.shape[0]
    rng = np.random.default_rng(seed)

    triu_i, triu_j = np.triu_indices(N, k=1)
    eligible = ~(~ink_mask[triu_i] & ~ink_mask[triu_j])
    eligible_i, eligible_j = triu_i[eligible], triu_j[eligible]

    real_values = real_topology[triu_i, triu_j]
    nonzero_mask = real_values != 0
    values_pool = real_values[nonzero_mask].copy()

    if n_edges is None:
        n_edges = round(int(nonzero_mask.sum()) / 2)

    rng.shuffle(values_pool)
    values_to_place = values_pool[:n_edges]

    chosen = rng.choice(len(eligible_i), size=n_edges, replace=False)
    chosen_i, chosen_j = eligible_i[chosen], eligible_j[chosen]

    random_topo = np.zeros((N, N))
    random_topo[chosen_i, chosen_j] = values_to_place
    random_topo[chosen_j, chosen_i] = values_to_place
    return random_topo


def rescale_to_common_budget(A, target_mean_weighted_degree):
    """A_tilde = A * (C / mean_weighted_degree(A)) -- the confirmed
    normalization formula applied identically to all four matched graph
    constructions in the original design (see module docstring). A no-op
    (up to float64 rounding) for constructions whose mean weighted degree
    already equals the target by construction (T defines its own target;
    degree-preserving rewiring and build_lattice_topology's
    total_weight_target both already hit it)."""
    mean_weighted_degree = A.sum(axis=1).mean()
    return A * (target_mean_weighted_degree / mean_weighted_degree)
