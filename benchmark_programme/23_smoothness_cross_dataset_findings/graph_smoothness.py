"""
Capacity Experiment II: graph smoothness (x^T L x) against each class's
Laplacian -- a qualitatively different graph functional than the
pairwise-correlation-matching used for the original 20D readout. Where
the 20D score asks "does this image's own pairwise correlation structure
agree with the class's diagnostic connections," smoothness asks "does
this image's own per-pixel signal vary smoothly or roughly across the
class's learned edges" -- genuinely different information, not another
view of the same pairwise-correlation summary.

All topology edge weights were confirmed positive (0.9 to ~0.999,
consistent with a magnitude-threshold on same-phase correlations), so the
standard graph Laplacian quadratic form is well-defined without needing a
signed-Laplacian variant.
"""
import numpy as np
from developmental_pruning import get_local_converged_phases


def graph_smoothness(phases_flat, topology):
    """x^T L x = 0.5 * sum_{i,j} W_ij (x_i - x_j)^2, computed using only
    the sparse nonzero entries of the topology (efficient for graphs with
    a few hundred to tens of thousands of edges out of 784*784 possible
    pairs)."""
    rows, cols = np.nonzero(topology)
    weights = topology[rows, cols]
    diffs = phases_flat[rows] - phases_flat[cols]
    return 0.5 * np.sum(weights * diffs ** 2)


def per_image_phases_and_smoothness(image, topologies, steps=150):
    """Single dynamics run, reused for both the phase signal (smoothness)
    and (by the caller, separately) the pairwise correlation matrix if
    needed -- avoids running the oscillator simulation twice per image."""
    phases = get_local_converged_phases(image, steps=steps).flatten()
    smoothness_scores = [graph_smoothness(phases, topologies[c]) for c in range(10)]
    return phases, smoothness_scores
