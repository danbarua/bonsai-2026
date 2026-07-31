"""
Direct measures of statistical dependence between feature blocks, rather
than inferring relatedness from downstream classifier performance.
CCA captures linear shared structure; distance correlation captures both
linear and nonlinear dependence in a single scalar, with dCor=0 iff the
blocks are (linearly and nonlinearly) independent.
"""
import numpy as np
from sklearn.cross_decomposition import CCA


def cca_canonical_correlations(X, Y, n_components=None):
    """Returns the canonical correlations between X and Y, largest first."""
    if n_components is None:
        n_components = min(X.shape[1], Y.shape[1])
    cca = CCA(n_components=n_components, max_iter=2000)
    X_c, Y_c = cca.fit_transform(X, Y)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    return np.array(correlations)


def distance_correlation(X, Y, subsample=None, seed=0):
    """Szekely-Rizzo distance correlation. O(n^2) -- subsample for large n."""
    if subsample is not None and X.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=subsample, replace=False)
        X, Y = X[idx], Y[idx]

    def double_centered_distances(Z):
        # pairwise Euclidean distances
        n = Z.shape[0]
        sq = np.sum(Z ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand_mean = D.mean()
        return D - row_mean - col_mean + grand_mean

    n = X.shape[0]
    A = double_centered_distances(X)
    B = double_centered_distances(Y)
    dcov2 = np.sum(A * B) / (n * n)
    dvarX2 = np.sum(A * A) / (n * n)
    dvarY2 = np.sum(B * B) / (n * n)
    denom = np.sqrt(max(dvarX2 * dvarY2, 0))
    return np.sqrt(max(dcov2, 0) / denom) if denom > 0 else 0.0
