"""
Label for unambiguous reference in reports: **current edge-count-matched
random** -- same edge count as the real topology, no final coupling-budget
rescaling. Distinct from `historical_matched_sparsity_random.py`'s
**historical half-edge random, coupling-budget normalized**, which
reconstructs a different, earlier design (see that module's docstring).
Do not call both "matched-sparsity random" without qualification --
that's an actively different null model, not a naming detail.

Matched-sparsity random topology ablation: for each class, generate a
random topology with the SAME edge count, drawn from the SAME eligible
candidate pool (ink-involving pairs only, background-background excluded,
matching the frozen configuration), and assigned VALUES resampled from the
real topology's own value distribution (preserving sign/magnitude
distribution exactly -- only WHICH pairs get connected is randomized).

Per review: generate multiple random seeds and report a distribution, not
a single draw.
"""
import numpy as np


def generate_matched_sparsity_topology(real_topology, ink_mask, seed):
    """real_topology: (N,N) the actual learned topology, used only to get
    the target edge count and the pool of values to redistribute.
    ink_mask: (N,) boolean, which pixels are 'ink' for this class."""
    N = real_topology.shape[0]
    rng = np.random.default_rng(seed)

    # Eligible candidate pairs: NOT both-background (matches the frozen
    # configuration's exclusion rule exactly), upper triangle only
    triu_i, triu_j = np.triu_indices(N, k=1)
    eligible = ~(~ink_mask[triu_i] & ~ink_mask[triu_j])  # NOT (bg AND bg)
    eligible_i, eligible_j = triu_i[eligible], triu_j[eligible]

    # Real edge count and real values, to redistribute onto random positions
    real_values = real_topology[triu_i, triu_j]
    nonzero_mask = real_values != 0
    n_edges = nonzero_mask.sum()
    values_to_place = real_values[nonzero_mask].copy()
    rng.shuffle(values_to_place)  # randomize which value goes where too

    # Randomly select n_edges positions from the eligible pool
    chosen = rng.choice(len(eligible_i), size=n_edges, replace=False)
    chosen_i, chosen_j = eligible_i[chosen], eligible_j[chosen]

    random_topo = np.zeros((N, N))
    random_topo[chosen_i, chosen_j] = values_to_place
    random_topo[chosen_j, chosen_i] = values_to_place  # symmetric
    return random_topo
