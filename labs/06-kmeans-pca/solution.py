"""Lab 06: k-means (Lloyd's algorithm + k-means++) and PCA via the SVD."""
import numpy as np


def assign_clusters(X, centers):
    """Index of the nearest center (squared Euclidean) for every row. Ties: lowest index."""
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1)


def kmeans(X, init_centers, n_iter=100):
    """Lloyd's algorithm from the given centers. Repeat: assign, then move each center to the mean of its points.
    An empty cluster keeps its previous center. Stop early when assignments stop changing.
    Return (centers, labels, inertia), where inertia is the sum of squared distances to the assigned center."""
    centers = np.array(init_centers, dtype=float)
    labels = None
    for _ in range(n_iter):
        new = assign_clusters(X, centers)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(len(centers)):
            if np.any(labels == j):
                centers[j] = X[labels == j].mean(axis=0)
    labels = assign_clusters(X, centers)
    inertia = ((X - centers[labels]) ** 2).sum()
    return centers, labels, inertia


def kmeans_plus_plus_init(X, k, rng):
    """k-means++: the first center is a uniformly random row; each next center is a row drawn with probability
    proportional to D(x)^2, its squared distance to the nearest center chosen so far. Return a (k, d) array."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = ((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1).min(axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


def pca_svd(X, n_components):
    """PCA by the SVD of the centered data Xc = U S V^T.
    Return (mean, components, explained_variance_ratio, scores):
      mean (d,), components = first n_components rows of V^T (k, d),
      explained_variance_ratio (k,) = S^2 / sum(S^2) for the kept components,
      scores = Xc @ components.T  (n, k).
    Sign convention (to make the answer unique): flip each component so that its largest-|value| entry is positive."""
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = Vt[:n_components]
    signs = np.sign(comps[np.arange(n_components), np.abs(comps).argmax(axis=1)])
    comps = comps * signs[:, None]
    evr = (S ** 2 / (S ** 2).sum())[:n_components]
    return mean, comps, evr, (X - mean) @ comps.T


def reconstruct(mean, components, scores):
    """Map scores back to the original space: mean + scores @ components."""
    return mean + scores @ components


def n_components_for_variance(X, threshold):
    """Smallest k whose cumulative explained variance ratio is >= threshold."""
    S = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    return int(np.searchsorted(cum, threshold - 1e-12) + 1)
