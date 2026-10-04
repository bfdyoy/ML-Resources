# CORE-06 notes: Unsupervised Learning

[← Lesson CORE-06](../../lessons/core-ml/06-unsupervised-learning.md) · [All notes](../README.md) · [← CORE-05 notes](05-trees-and-ensembles.md) · Next: [CORE-07 notes →](07-feature-engineering-pipelines-leakage.md)

> **Reading time** ≈ 55 min. **You need:** [MATH-01 notes](../math/01-linear-algebra.md) Block C (eigenvectors, SVD) and [MATH-03 notes](../math/03-probability-statistics.md) §A4 (the Gaussian).

---

## Where we are

Until now every example had a label. Most real data doesn't. Unsupervised learning asks what structure is in $X$ by itself. Two families of questions:

- **Are there a few directions that capture most of the variation?** That's dimensionality reduction: PCA, t-SNE, UMAP.
- **Are there natural groups?** That's clustering: k-means, Gaussian mixtures, DBSCAN, hierarchical.

There's no label to check against, so **judgement and domain sense** matter more here than anywhere else in Path 1.

---

## 1. PCA, two equivalent views

MATH-01 §C3 showed that the direction of maximum projected variance is the top eigenvector of the covariance matrix. The other view is just as useful:
**PCA finds the $k$-dimensional subspace that loses the least information when you project onto it.**

Project each centered point onto the top $k$ components and reconstruct it: $\hat x = V_k V_k^\top x$. The total squared reconstruction error equals the variance you dropped:

```math
\frac{1}{N-1}\sum_i \lVert x_i - V_k V_k^\top x_i\rVert^2 = \sum_{j > k}\lambda_j .
```

Since the total variance $\sum_j \lambda_j$ is fixed, **maximizing kept variance = minimizing reconstruction error**. One problem, two descriptions.

**Choosing $k$:** plot the explained-variance ratio $\lambda_j/\sum\lambda$ (a **scree plot**) and look for an elbow, or keep enough components for 90–95% of the variance.
For a downstream model, treat $k$ as a hyperparameter and tune it by CV.

**Standardize first?** Usually yes. PCA maximizes *variance*, so a feature measured in large units dominates just because of its scale.
Don't standardize when every feature shares the same meaningful unit (pixel intensities, or the same sensor at different times). There, the relative variances *are* the information.

---

## 2. k-means

### 2.1 The objective

Find $K$ centres $\mu_1..\mu_K$ and an assignment $c(i)$ of each point to a centre, to minimize the **inertia** (the within-cluster sum of squares):

```math
J = \sum_{i=1}^N \lVert x_i - \mu_{c(i)}\rVert^2 .
```

### 2.2 Lloyd's algorithm, and why it always converges

Alternate two steps:

1. **Assign:** with the centres fixed, put each point with its nearest centre. This can only lower $J$, since each term gets its smallest possible value.
2. **Update:** with the assignments fixed, move each centre to the mean of its points. The mean minimizes the sum of squared distances (set the gradient $\sum (\mu - x_i)$ to 0), so $J$ can only go down again.

$J$ never increases, and there are finitely many possible assignments, so the algorithm stops. But it stops at a **local** minimum that depends on the starting centres.
**k-means++** spreads out the initial centres (each new one is chosen with probability proportional to its squared distance from the nearest existing centre), and scikit-learn reruns everything `n_init` times and keeps the best.

### 2.3 What k-means assumes, so where it fails

Squared Euclidean distance to a centre implies **round (spherical) clusters of similar size and spread**. It fails on:

- elongated, curved, or nested shapes (two moons, concentric rings);
- clusters of very different sizes or densities;
- unscaled features. The distance is dominated by the feature with the largest numbers: "income in dollars" swamps "age in years" (the lesson's debug question).

### 2.4 Choosing $K$

- **Elbow:** inertia always falls as $K$ grows. Look for where it stops falling quickly.
- **Silhouette:** for point $i$, let $a$ = its mean distance to its own cluster and $b$ = its mean distance to the nearest other cluster. Then
  $s_i = \frac{b - a}{\max(a, b)} \in [-1, 1]$. Average it over all points and pick the $K$ with the highest mean. Values near 1 mean tight, well-separated clusters.
- **Domain sense:** a segmentation with 4 actionable groups beats 11 statistically "optimal" groups nobody can use.

---

## 3. Gaussian mixture models: soft k-means

A GMM says each point comes from one of $K$ Gaussians: pick a component $k$ with probability $\pi_k$, then draw $x \sim \mathcal N(\mu_k, \Sigma_k)$. Fit it with **EM (expectation–maximization)**, which mirrors Lloyd's two steps:

- **E-step (soft assign):** compute the **responsibility** of each component for each point, using Bayes' rule:

```math
r_{ik} = \frac{\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal N(x_i\mid\mu_j,\Sigma_j)} .
```

- **M-step (weighted update):** $\mu_k$ = the responsibility-weighted mean, $\Sigma_k$ = the weighted covariance, and $\pi_k$ = the average responsibility.

Each EM iteration never decreases the likelihood. Compared with k-means:

- assignments are **probabilities** (a point can be 70/30 between two clusters);
- clusters can be **elliptical** and of different sizes;
- k-means is the limit where every covariance is $\sigma^2 I$ with $\sigma \to 0$: responsibilities become 0/1, and EM becomes Lloyd's algorithm.

Choose $K$ (and the covariance type) with BIC. Because a GMM is a density, it also gives anomaly scores: low $p(x)$ means unusual (EL-06).

---

## 4. Density- and hierarchy-based clustering

- **DBSCAN:** a point is a *core* point if at least `min_samples` points lie within distance `eps` of it. Clusters are connected chains of core points (plus their border points), and everything else is **noise**.
  It finds clusters of **any shape**, doesn't need $K$, and labels outliers. It fails when clusters have **very different densities** (a single `eps` can't suit all of them) and in high dimensions, where distances concentrate (CORE-09).
  HDBSCAN relaxes the single-`eps` limitation.
- **Hierarchical (agglomerative):** start with every point as its own cluster and repeatedly merge the closest pair. The **linkage** defines "closest": *single* (nearest points; it chains), *complete* (farthest points; it gives compact clusters), *average*, or *Ward* (smallest increase in within-cluster variance, similar in spirit to k-means).
  The resulting **dendrogram** shows the clustering at every scale, and you cut it at the height you want.

| Method | Shape | Needs $K$ | Handles noise | Fails when |
|---|---|---|---|---|
| k-means | Round, similar size | Yes | No | Non-convex shapes, unscaled features |
| GMM | Ellipses | Yes | Via low density | Very non-Gaussian shapes |
| DBSCAN | Any | No | Yes | Clusters of very different densities |
| Agglomerative (Ward) | Roughly round | Cut height | No | Large $N$ (memory $O(N^2)$) |

---

## 5. t-SNE and UMAP: maps for looking, not for measuring

Both build a 2-D map that **preserves neighbourhoods**. Points that are close in the original space should be close on the map.

- **t-SNE** turns distances into neighbour probabilities (Gaussian in the original space, heavy-tailed Student-t on the map), then minimizes the KL divergence between them. The heavy tail lets clusters spread apart on the map.
  The **perplexity** parameter is roughly "how many neighbours each point considers". UMAP is faster, with similar goals.
- **What you can read:** which points are neighbours, and whether there are separated groups.
- **What you can't:** cluster **sizes** (t-SNE expands dense clusters and contracts sparse ones), **distances between** clusters (only local structure is optimized), and any single run's exact shape (change the perplexity or seed and it changes).
  Never cluster *on* a t-SNE map, and never feed its coordinates to a model as features. [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) shows all of these traps interactively.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(0)

# PCA: reconstruction error = sum of dropped eigenvalues
X = rng.normal(size=(400, 5)) @ rng.normal(size=(5, 5))
Xc = X - X.mean(0)
pca = PCA(n_components=2).fit(Xc)
recon = pca.inverse_transform(pca.transform(Xc))
err = np.sum((Xc - recon) ** 2) / (len(Xc) - 1)
lam = np.sort(np.linalg.eigvalsh(np.cov(Xc.T)))[::-1]
print(f"reconstruction error {err:.4f} = dropped eigenvalues {lam[2:].sum():.4f}")

# k-means from scratch (Lloyd) vs sklearn
Xb, yb = make_blobs(n_samples=600, centers=3, random_state=42)
mu = Xb[rng.choice(len(Xb), 3, replace=False)]
for it in range(100):
    c = np.argmin(((Xb[:, None, :] - mu[None]) ** 2).sum(-1), axis=1)
    new = np.array([Xb[c == k].mean(0) for k in range(3)])
    if np.allclose(new, mu): break
    mu = new
J = sum(((Xb[c == k] - mu[k]) ** 2).sum() for k in range(3))
print(f"Lloyd converged in {it} iterations, inertia {J:.1f}; sklearn inertia {KMeans(3, n_init=10, random_state=0).fit(Xb).inertia_:.1f}")

# Choose K by silhouette
for k in [2, 3, 4, 5]:
    print(f"K={k} silhouette={silhouette_score(Xb, KMeans(k, n_init=10, random_state=0).fit_predict(Xb)):.3f}")
```

```python
# Shapes: k-means vs DBSCAN on two moons (ARI = agreement with the truth; 1 is perfect)
Xm, ym = make_moons(500, noise=0.05, random_state=0)
print("moons  ARI k-means:", round(adjusted_rand_score(ym, KMeans(2, n_init=10, random_state=0).fit_predict(Xm)), 2),
      " DBSCAN:", round(adjusted_rand_score(ym, DBSCAN(eps=0.2).fit_predict(Xm)), 2))
# ...and the opposite: two tight clusters next to one diffuse cluster. No single eps fits all three.
Xd, yd = make_blobs(n_samples=[300, 300, 300], centers=[[0, 0], [1.5, 0], [7, 0]], cluster_std=[0.2, 0.2, 1.5], random_state=0)
print("3 blobs of different density, GMM ARI:", round(adjusted_rand_score(yd, GaussianMixture(3, n_init=10, random_state=0).fit_predict(Xd)), 2))
for eps in [0.2, 0.5, 0.8]:
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Xd)
    print(f"  DBSCAN eps={eps}: {len(set(lab)) - (-1 in lab)} clusters found, "
          f"{(lab[yd == 2] == -1).mean():.0%} of the diffuse cluster called noise, ARI {adjusted_rand_score(yd, lab):.2f}")
# (eps=0.2 scores a high ARI only because "noise" coincides with the diffuse cluster: DBSCAN has declared a whole cluster to be outliers.)

# GMM soft assignments
Xo, _ = make_blobs(n_samples=400, centers=[[0, 0], [2.5, 0]], cluster_std=1.0, random_state=0)   # overlapping pair
gmm = GaussianMixture(2, random_state=0).fit(Xo)
R = gmm.predict_proba(Xo)
unsure = np.argsort(R.max(1))[:3]                    # the 3 points the model is least sure about
print("responsibilities of the 3 most ambiguous points:\n", R[unsure].round(2))

# Unscaled features: income dominates the clustering
age = rng.normal(40, 12, 600); income = rng.normal(50_000, 15_000, 600)
group = (age > 40).astype(int)                       # real structure is in age
Z = np.c_[age, income]
for name, data in [("raw", Z), ("standardized", StandardScaler().fit_transform(Z))]:
    lab = KMeans(2, n_init=10, random_state=0).fit_predict(data)
    print(f"{name:12s} corr(cluster, age)={abs(np.corrcoef(lab, age)[0,1]):.2f}  corr(cluster, income)={abs(np.corrcoef(lab, income)[0,1]):.2f}")
```

---

## Pitfalls & misconceptions

- **Forgetting to scale** before k-means or PCA.
- **"The elbow is obvious."** Often it isn't. Combine it with silhouette scores and with what the clusters are for.
- **Reading t-SNE cluster sizes or the gaps between clusters.**
- **Clustering always returns clusters,** even on uniform noise. Check stability: rerun with different seeds or subsamples and compare the results (ARI).
- **Fitting PCA on all the data before a train/test split.** It's a fitted preprocessing step, so it belongs in the pipeline (CORE-01 §5).

## Cheat sheet

| Item | Formula |
|---|---|
| PCA reconstruction error | $\sum_{j>k}\lambda_j$ |
| k-means objective | $\sum_i \lVert x_i - \mu_{c(i)}\rVert^2$ |
| Silhouette | $(b - a)/\max(a, b)$ |
| GMM responsibility | $\pi_k \mathcal N_k(x) / \sum_j \pi_j \mathcal N_j(x)$ |
| DBSCAN core point | at least `min_samples` neighbours within `eps` |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why standardize before PCA, and when not to?</summary>

PCA maximizes variance, so features in large units dominate. Standardize when the units differ. Don't when all features share a meaningful common scale (pixels, the same measurement over time) and their variances carry the information.
</details>

<details>
<summary>2. What does it mean that PC1 is an eigenvector of the covariance matrix?</summary>

The projected variance in a unit direction $u$ is $u^\top S u$. Maximizing it subject to $\lVert u\rVert = 1$ gives $Su = \lambda u$ (Lagrange, MATH-02 §B7), and the best choice is the top eigenvector, whose eigenvalue is the variance captured.
</details>

<details>
<summary>3. Where k-means fails and DBSCAN succeeds, and vice versa.</summary>

Two moons or concentric rings: DBSCAN wins. Clusters of very different density (two tight ones beside a diffuse one): a small `eps` calls the diffuse cluster noise, and a large one merges the tight pair. A GMM recovers all three. The demo shows both cases.
</details>

<details>
<summary>4. How is a GMM a soft k-means?</summary>

The E-step gives each point a probability of belonging to each cluster instead of a hard label, and the M-step takes weighted means and covariances. With equal spherical covariances shrinking to 0, the responsibilities become hard, and EM reduces to Lloyd's algorithm.
</details>

<details>
<summary>5. Why not interpret distances between clusters in t-SNE?</summary>

t-SNE only optimizes the preservation of local neighbourhoods. The heavy-tailed map distribution pushes clusters apart by arbitrary amounts, so global distances (and cluster sizes) aren't meaningful and change with the perplexity and seed.
</details>

<details>
<summary>6. k-means split customers purely by income.</summary>

The features weren't scaled. Income's numeric range (tens of thousands) dominates the Euclidean distance. Standardize, or use domain-appropriate transforms, and reconsider which features belong in the clustering at all.
</details>

## Where this leads

Next: [CORE-07 notes](07-feature-engineering-pipelines-leakage.md). Every model so far takes features as given. Next we *make* features, and learn the most
expensive bug in applied ML, leakage, well enough to catch it.
