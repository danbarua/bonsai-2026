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


def spatial_structure_stats_10d(image_flat, side=28):
    """10 class-agnostic, spatially-structured statistics -- strictly
    harder to beat than aggregate ink amount, since it captures HOW
    energy is spatially distributed without referencing any class's
    topology or support: quadrant energies, center of mass, second
    moments, and total energy for scale reference."""
    img = image_flat.reshape(side, side)
    half = side // 2
    quad_tl = np.sum(img[:half, :half] ** 2)
    quad_tr = np.sum(img[:half, half:] ** 2)
    quad_bl = np.sum(img[half:, :half] ** 2)
    quad_br = np.sum(img[half:, half:] ** 2)

    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    total = np.sum(img) + 1e-8
    row_com = np.sum(rows * img) / total
    col_com = np.sum(cols * img) / total
    row_var = np.sum(((rows - row_com) ** 2) * img) / total
    col_var = np.sum(((cols - col_com) ** 2) * img) / total
    cov = np.sum((rows - row_com) * (cols - col_com) * img) / total

    total_energy = np.sum(img ** 2)

    return np.array([quad_tl, quad_tr, quad_bl, quad_br,
                      row_com, col_com, row_var, col_var, cov,
                      total_energy], dtype=np.float64)
