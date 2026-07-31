"""
Class-independent global ink statistics, matched to E's exact
dimensionality (10D) for a fair comparison. Tests whether E's value is
genuinely about class-specific support alignment, or just generic
stroke density/ink amount any class-agnostic statistic would capture.
"""
import numpy as np


def global_ink_stats_10d(image_flat):
    """10 class-agnostic scalars: total sum, total squared energy, and
    ink-pixel counts at 8 evenly-spaced thresholds. None of these
    reference any class-specific topology or support."""
    total_sum = np.sum(image_flat)
    total_sq_energy = np.sum(image_flat ** 2)
    thresholds = np.linspace(0.1, 0.8, 8)
    counts = [np.sum(image_flat > t) for t in thresholds]
    return np.array([total_sum, total_sq_energy] + counts, dtype=np.float64)
