import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [6, 0], [0, 6]], float)
    return np.concatenate([c + rng.normal(size=(50, 2)) for c in centers])


def test_kmeans_matches_sklearn_from_same_init(impl, blobs):
    init = blobs[[0, 50, 100]] + 0.5
    c, labels, inertia = impl.kmeans(blobs, init)
    ref = KMeans(n_clusters=3, init=init, n_init=1, algorithm="lloyd", tol=0).fit(blobs)
    np.testing.assert_allclose(c, ref.cluster_centers_, atol=1e-8)
    np.testing.assert_array_equal(labels, ref.labels_)
    assert np.isclose(inertia, ref.inertia_)


def test_kmeans_empty_cluster_keeps_center(impl, blobs):
    init = np.r_[blobs[[0, 50, 100]], [[100.0, 100.0]]]
    c, labels, _ = impl.kmeans(blobs, init)
    np.testing.assert_allclose(c[3], [100.0, 100.0])
    assert not np.any(labels == 3)


def test_kmeans_plus_plus(impl, blobs):
    c = impl.kmeans_plus_plus_init(blobs, 3, np.random.default_rng(0))
    assert c.shape == (3, 2)
    assert all(any(np.array_equal(ci, x) for x in blobs) for ci in c)       # centers are data points
    blob_centers = np.array([[0, 0], [6, 0], [0, 6]], float)
    hits = 0
    for s in range(20):                                                       # spread out: usually one per blob
        c = impl.kmeans_plus_plus_init(blobs, 3, np.random.default_rng(s))
        hits += len(set(impl.assign_clusters(c, blob_centers).tolist())) == 3
    assert hits >= 15


def test_pca_matches_sklearn(impl):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5)) @ rng.normal(size=(5, 5))
    mean, comps, evr, scores = impl.pca_svd(X, 3)
    ref = PCA(n_components=3).fit(X)
    np.testing.assert_allclose(mean, ref.mean_)
    np.testing.assert_allclose(np.abs(comps), np.abs(ref.components_), atol=1e-8)
    np.testing.assert_allclose(evr, ref.explained_variance_ratio_)
    assert np.all(comps[np.arange(3), np.abs(comps).argmax(1)] > 0)
    np.testing.assert_allclose(scores.T @ scores / 99, np.diag(ref.explained_variance_), atol=1e-8)


def test_reconstruction_and_variance_threshold(impl):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2)) @ rng.normal(size=(2, 6)) + 0.01 * rng.normal(size=(80, 6))    # ~rank 2
    mean, comps, _, scores = impl.pca_svd(X, 2)
    assert np.abs(impl.reconstruct(mean, comps, scores) - X).max() < 0.1
    mean6, comps6, _, scores6 = impl.pca_svd(X, 6)
    np.testing.assert_allclose(impl.reconstruct(mean6, comps6, scores6), X, atol=1e-10)
    assert impl.n_components_for_variance(X, 0.99) == 2
    assert impl.n_components_for_variance(X, 1.0) == 6
