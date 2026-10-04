# Lab 06: k-means & PCA via the SVD

[← Labs](../README.md) · Lesson: [CORE-06 Unsupervised learning](../../lessons/core-ml/06-unsupervised-learning.md) · Notes: [CORE-06 notes](../../notes/core-ml/06-unsupervised-learning.md)

**Time** ≈ 1.5 h · **You'll practise:** alternating optimization (assign ↔ update), D²-sampling, and PCA as "SVD of the centered data" with a sign convention.

| Function | Checked against |
|---|---|
| `assign_clusters`, `kmeans` | `sklearn.cluster.KMeans(init=..., n_init=1, algorithm="lloyd")` from the same start |
| `kmeans_plus_plus_init` | centers are data points, and usually land in different blobs |
| `pca_svd` | `sklearn.decomposition.PCA` (up to sign), and uncorrelated scores |
| `reconstruct`, `n_components_for_variance` | exact reconstruction with all components; rank-2 data needs 2 |

```bash
pytest labs/06-kmeans-pca
```

**Bonus:** k-means minimizes inertia, which always decreases as k grows. So why can't you choose k by minimizing it? Run your `kmeans` for k = 1…8 and plot inertia,
then plot the silhouette score (`sklearn.metrics.silhouette_score`). Do the two agree on k = 3?
