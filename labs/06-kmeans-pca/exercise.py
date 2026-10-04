# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 06: k-means (Lloyd's algorithm + k-means++) and PCA via the SVD."""
import numpy as np


def assign_clusters(X, centers):
    """Index of the nearest center (squared Euclidean) for every row. Ties: lowest index."""
    raise NotImplementedError("your code here")


def kmeans(X, init_centers, n_iter=100):
    """Lloyd's algorithm from the given centers. Repeat: assign, then move each center to the mean of its points.
    An empty cluster keeps its previous center. Stop early when assignments stop changing.
    Return (centers, labels, inertia), where inertia is the sum of squared distances to the assigned center."""
    raise NotImplementedError("your code here")


def kmeans_plus_plus_init(X, k, rng):
    """k-means++: the first center is a uniformly random row; each next center is a row drawn with probability
    proportional to D(x)^2, its squared distance to the nearest center chosen so far. Return a (k, d) array."""
    raise NotImplementedError("your code here")


def pca_svd(X, n_components):
    """PCA by the SVD of the centered data Xc = U S V^T.
    Return (mean, components, explained_variance_ratio, scores):
      mean (d,), components = first n_components rows of V^T (k, d),
      explained_variance_ratio (k,) = S^2 / sum(S^2) for the kept components,
      scores = Xc @ components.T  (n, k).
    Sign convention (to make the answer unique): flip each component so that its largest-|value| entry is positive."""
    raise NotImplementedError("your code here")


def reconstruct(mean, components, scores):
    """Map scores back to the original space: mean + scores @ components."""
    raise NotImplementedError("your code here")


def n_components_for_variance(X, threshold):
    """Smallest k whose cumulative explained variance ratio is >= threshold."""
    raise NotImplementedError("your code here")
