# CORE-06: Unsupervised Learning: Clustering & Dimensionality Reduction

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~6 h | L2 | CORE-04 · math: [MATH-01](../math/01-linear-algebra.md) (eigenvectors) |

## Why this matters
Most data has no labels. Clustering, PCA, and embeddings visualized with t-SNE/UMAP are how you *explore* data,
compress it, find segments, and build features. You'll also meet them again inside deep learning, where
autoencoders and embeddings are learned versions of the same ideas.

## Learning goals
By the end you can:
- Explain PCA as finding directions of maximum variance, and read a scree / explained-variance plot.
- Run and compare k-means, hierarchical clustering, DBSCAN, and Gaussian mixtures, and know each one's failure modes.
- Choose the number of clusters with the elbow method, silhouette score, and domain sense.
- Use t-SNE and UMAP for visualization, and avoid the classic misreadings (cluster sizes, distances between clusters).

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/) (Setosa) | The whole page. Rotate the 3D example. | 20 min |
| 2 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 12 "Unsupervised Learning" | §12.1–12.2 (PCA), §12.4 (k-means and hierarchical clustering, practical issues). Skim §12.3 (missing values / matrix completion). | 2 h |
| 3 | **Read + Build** | [Géron, *Hands-On ML*, Ch. 7 "Dimensionality Reduction" + Ch. 8 "Unsupervised Learning"](https://github.com/ageron/handson-mlp) | Run `07_dimensionality_reduction.ipynb` and `08_unsupervised_learning.ipynb`. Focus on k-means limits, DBSCAN, and Gaussian mixtures. | 2.5 h |
| 4 | **Intuition** | [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) (Distill) then [Understanding UMAP](https://pair-code.github.io/understanding-umap/) (Google PAIR) | Both. Play with the perplexity and n_neighbors sliders. | 45 min |

## Check your understanding
1. Why must you standardize features before PCA (usually)? When would you *not*?
2. What does it mean that the first principal component is an eigenvector of the covariance matrix?
3. Give a dataset shape where k-means fails and DBSCAN succeeds. And one where the opposite is true.
4. How is a Gaussian mixture model a "soft" version of k-means?
5. Why shouldn't you interpret distances *between* clusters in a t-SNE plot?
6. *(debug)* Your k-means clusters split customers purely by income, ignoring every other feature. What went wrong?

## Mini-project
**Task:** Segment customers or documents. Reduce with PCA, cluster with two methods, visualize with UMAP, and
*describe each cluster in plain words* using feature averages.
**Dataset:** Mall Customers or Online Retail (UCI), or 20 Newsgroups (`sklearn.datasets.fetch_20newsgroups`) with TF-IDF.
**Deliverable:** A notebook plus a "segment profile" table a business person could read.

## Go deeper
- [MML book](https://mml-book.github.io/) Ch. 10 "Dimensionality Reduction with PCA" and Ch. 11 "Density Estimation with Gaussian Mixture Models": the math behind both.
- [UDL](https://udlbook.github.io/udlbook/) Ch. 14 "Unsupervised Learning": a bridge to generative models (GEN-04).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 02: Classical ML](../../toolbox/02-classical-ml.md), for every concept in this lesson, with alternatives.
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 13–15.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
