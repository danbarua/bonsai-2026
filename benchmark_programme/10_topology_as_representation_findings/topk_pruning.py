"""
Top-K adaptive pruning: keep the K strongest ink-involving connections per
class (by |correlation|), rather than a fixed absolute magnitude threshold.
Directly addresses a concern raised earlier -- different classes have
different intrinsic correlation-score distributions (confirmed via the
calibration baselines), so a fixed absolute threshold selects a different
NUMBER of connections per class in a way that isn't obviously fair, on top
of whatever the z-score calibration corrects for downstream.
"""
import numpy as np


def build_topology_topk(raw_stat, mean_intensity, k, ink_threshold=0.15):
    ink_mask = mean_intensity > ink_threshold
    bg_pair_mask = np.outer(~ink_mask, ~ink_mask)
    stat = raw_stat.copy()
    stat[bg_pair_mask] = 0.0  # exclude background-background entirely, same as before

    N = stat.shape[0]
    triu_i, triu_j = np.triu_indices(N, k=1)
    values = stat[triu_i, triu_j]
    abs_values = np.abs(values)

    if k >= len(values):
        keep_mask = np.ones_like(abs_values, dtype=bool)
    else:
        threshold = np.partition(abs_values, -k)[-k]
        keep_mask = abs_values >= threshold

    pruned = np.zeros_like(stat)
    kept_i, kept_j = triu_i[keep_mask], triu_j[keep_mask]
    pruned[kept_i, kept_j] = stat[kept_i, kept_j]
    pruned[kept_j, kept_i] = stat[kept_i, kept_j]  # symmetric
    return pruned
