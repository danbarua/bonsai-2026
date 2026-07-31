"""
Mantel permutation test and MLP baseline comparison for the topology-
overlap / confusion-matrix correspondence found on Fashion-MNIST.
Reproduces the numbers in bonsai_fashion_mnist_findings.md's "The headline
finding has changed" section.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


def jaccard_normalize(overlap_matrix):
    """Normalize raw overlap counts by the union of each pair's own edge
    counts -- controls for the confound that larger/denser classes
    trivially overlap more with everything."""
    n = overlap_matrix.shape[0]
    edge_counts = np.diag(overlap_matrix)
    normalized = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                union = edge_counts[i] + edge_counts[j] - overlap_matrix[i, j]
                normalized[i, j] = overlap_matrix[i, j] / union if union > 0 else 0
    return normalized


def symmetrize_confusion(confusion_matrix):
    sym = confusion_matrix + confusion_matrix.T
    np.fill_diagonal(sym, 0)
    return sym


def mantel_test(mat1, mat2, n_perm=10000, seed=0):
    """Permutation-based test for correlation between two matrices sharing
    the same set of objects (here, classes) -- properly accounts for the
    non-independence of entries within each matrix, unlike a naive
    Spearman p-value treating all pairs as independent samples."""
    n = mat1.shape[0]
    iu = np.triu_indices(n, k=1)
    observed_rho, _ = spearmanr(mat1[iu], mat2[iu])
    rng = np.random.default_rng(seed)
    null_rhos = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        permuted = mat2[np.ix_(perm, perm)]
        rho, _ = spearmanr(mat1[iu], permuted[iu])
        null_rhos[i] = rho
    p_value = np.mean(np.abs(null_rhos) >= np.abs(observed_rho))
    return observed_rho, p_value


def train_mlp_baseline(X_train, y_train, X_test, y_test, hidden_layer_sizes=(128, 64), seed=0):
    """Fully-supervised, backpropagation-trained baseline with no
    architectural relationship to graph-based topology matching --
    deliberately chosen to be a cleaner comparison than a CNN would be,
    since it introduces no translation-equivariance prior that could
    itself explain convergent confusion patterns."""
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=300,
                         random_state=seed, early_stopping=True)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    cm = confusion_matrix(y_test, predictions, labels=range(10))
    return mlp, accuracy, cm
