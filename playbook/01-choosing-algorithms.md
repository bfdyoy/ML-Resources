# Playbook 1: Choosing algorithms, A vs B

[← Playbook](README.md) · Next: [Tabular tricks](02-tabular-tricks.md)

> "Which one should I use?" is the most common question after learning several methods. Each section below is a decision guide:
> **pick A when…, pick B when…, and the quick experiment that settles it** for your data. In the spirit of
> [*Machine Learning Yearning*](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf) and [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml),
> the guides favour *the simplest thing that could work, measured*. **You need:** Path 1 through CORE-05. Later sections refer to DL and GEN lessons.

---

## 1. Tabular data: linear model vs tree ensemble vs kNN vs neural net

| Pick | When |
|---|---|
| **Linear / logistic regression** (with good features) | Few rows, a mostly additive signal, extrapolation needed, coefficients must be explained, or a strong baseline is needed in a minute |
| **Gradient-boosted trees** (LightGBM, XGBoost, CatBoost, `HistGradientBoosting*`) | The default for medium and large tables: interactions, mixed types, missing values, monotone constraints, little preprocessing |
| **kNN** | A smooth target in a low-dimensional, well-scaled space, or "similar cases" is itself the product |
| **Neural nets** ([EL-10](../lessons/electives/10-tabular-deep-learning.md)) | Very large data, rich categorical embeddings, multimodal inputs (text or images next to the table), or one model trained end to end with others |

**The quick experiment:** run all three classical families with the same CV. Each one wins in a different "world":

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 1500
X = rng.uniform(-2, 2, size=(n, 5))
worlds = {
    "additive & linear": X @ np.array([1.0, -2.0, 0.5, 0.0, 0.0]),
    "interactions & steps": np.where(X[:, 0] > 0, 2.0, -1.0) * (X[:, 1] > 0.5) + X[:, 2] * X[:, 3],
    "smooth & oblique (all 5 dims)": np.sin(X.sum(axis=1)),
}
models = {
    "ridge": make_pipeline(StandardScaler(), RidgeCV()),
    "GBM": HistGradientBoostingRegressor(random_state=0),
    "kNN (k=15)": make_pipeline(StandardScaler(), KNeighborsRegressor(15)),
}
for wname, signal in worlds.items():
    y = signal + rng.normal(scale=0.3, size=n)
    scores = {m: cross_val_score(est, X, y, cv=5, scoring="r2").mean() for m, est in models.items()}
    print(f"{wname:34s}" + "  ".join(f"{m}: {s:.2f}" for m, s in scores.items()))
```

In the **additive** world, ridge is best (R² **0.99**), with the GBM just behind (0.98). With **interactions and steps**, the GBM wins clearly (**0.92**, against 0.10 for ridge).
In the **smooth, oblique** world, a wave along the diagonal of all five features, kNN wins (**0.52** against 0.16 for the GBM). A tree can only split one axis at a time, so it
approximates a diagonal wave with a coarse staircase, while kNN simply averages nearby points in every direction.
On real data you don't know which world you're in. That's why you run the comparison.

## 2. Random forest vs gradient boosting

| Random forest | Gradient boosting |
|---|---|
| Hard to get badly wrong: few important hyperparameters, little tuning | Usually more accurate *after tuning* (learning rate, depth/leaves, regularization, early stopping) |
| Averages deep, independent trees, which reduces variance | Adds shallow trees sequentially, each one correcting the last, which reduces bias |
| Free out-of-bag validation; trivially parallel | Native missing values, categorical support, monotone constraints, custom losses |
| A good first model and a robust baseline | The workhorse for competitions and production tabular ML |

**Settle it:** tune both with the same budget ([CORE-12](../lessons/core-ml/12-hyperparameter-optimization.md)). If the GBM's gain is within the CV noise, the forest is the cheaper and more robust choice.

## 3. Clustering: k-means vs GMM vs DBSCAN/HDBSCAN vs hierarchical

| Pick | When |
|---|---|
| **k-means** | Roughly round clusters of similar size, a k you know (or can choose), large data |
| **Gaussian mixture** | Elliptical or overlapping clusters, soft (probabilistic) membership wanted |
| **DBSCAN / HDBSCAN** | Arbitrary shapes, noise points that should be left unclustered, k unknown |
| **Agglomerative (hierarchical)** | A dendrogram is useful (taxonomies), small to medium data, custom distances |

```python
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

Xm, ym = make_moons(600, noise=0.06, random_state=0)
for name, est in [("k-means (k=2)", KMeans(2, n_init=10, random_state=0)), ("DBSCAN (eps=0.2)", DBSCAN(eps=0.2))]:
    print(f"{name:17s} adjusted Rand index vs the true moons: {adjusted_rand_score(ym, est.fit_predict(Xm)):.2f}")
```

On two interleaved half-moons, k-means cuts straight through both (ARI **0.26**), because it can only draw a straight boundary between two centroids. DBSCAN follows the density and recovers them perfectly (ARI **1.00**).
**Settle it:** there's no label to score against in real use, so check the stability of the clusters across seeds and subsamples, and whether a domain expert can *name* them ([CORE-06](../lessons/core-ml/06-unsupervised-learning.md)).

## 4. Prompting vs RAG vs fine-tuning

| Pick | When | Not when |
|---|---|---|
| **Prompting** (+ few-shot examples) | Always first: hours to try, and it sets the baseline | The knowledge isn't in the model, or the prompt keeps growing |
| **RAG** | The model needs **knowledge**: private, fresh, or large documents; answers must cite sources | The problem is the *form* or *skill* of the answer, not missing facts |
| **Fine-tuning (LoRA)** | The model needs a **behaviour**: a format, style, domain language, a narrow skill, or a smaller and cheaper model for one task | You mainly need facts that change (fine-tuning bakes in a snapshot, and invents the rest confidently) |
| **RAG + fine-tuning** | Both: domain behaviour *and* fresh knowledge | Before you've measured each one separately |

**Rule of thumb:** *knowledge → RAG, behaviour → fine-tune, and prompting first for both.*
**Settle it:** the same eval set ([GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md)) for each option, with the failure analysis split into "retrieval missed the fact" vs "the model had the fact but answered badly".
That split tells you which lever to pull ([GEN-02](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md)).

## 5. Dimensionality reduction: PCA vs t-SNE/UMAP vs autoencoder vs random projection

| Pick | For |
|---|---|
| **PCA** | Preprocessing, denoising, compression, and *quantitative* use: it's linear, invertible, and fast |
| **t-SNE / UMAP** | **Visualization only.** Distances between clusters and cluster sizes in the plot are not meaningful |
| **Autoencoder** | Nonlinear compression of images, audio, or large data, when you can afford training |
| **Random projection** | Instant, data-independent compression that preserves distances ([outside the box §7](05-outside-the-box.md#7-random-projections-are-almost-free-dimensionality-reduction)) |

**The trap:** clustering in a t-SNE/UMAP embedding and reporting the clusters as findings. Cluster in the original space (or in PCA space), and use UMAP only to *look*.

## 6. Classification metrics: ROC-AUC vs PR-AUC vs log loss vs F1

| Use | When |
|---|---|
| **ROC-AUC** | Ranking quality when the classes are reasonably balanced, or when you care about the whole ranking |
| **PR-AUC (average precision)** | Rare positives where only the top of the ranking matters (fraud, search, screening) |
| **Log loss / Brier score** | The probabilities themselves are used (pricing, expected cost, combining with other models) |
| **F1 / precision@k / cost** | A thresholded decision: prefer the actual **expected cost** with a chosen threshold ([Lab 04](../labs/04-logistic-regression-metrics/README.md)) |

```python
from sklearn.metrics import average_precision_score, roc_auc_score

n_neg, n_pos = 100_000, 100                                  # 0.1% positives
y_true = np.r_[np.zeros(n_neg), np.ones(n_pos)]
for name, shift in [("model A", 3.0), ("model B", 4.0)]:
    s = np.r_[rng.normal(size=n_neg), rng.normal(loc=shift, size=n_pos)]
    print(f"{name}: ROC-AUC {roc_auc_score(y_true, s):.3f}   PR-AUC {average_precision_score(y_true, s):.3f}")
```

With 0.1% positives, both models look excellent on ROC-AUC (**0.980** vs **0.998**). PR-AUC shows the difference that matters for alerts: **0.270** vs **0.757**.
Model A's top-ranked cases are mostly false alarms. ROC-AUC barely notices, because 100,000 negatives make even many false positives a small *rate*.

## 7. Optimizers: SGD + momentum vs AdamW

| SGD + momentum (+ schedule) | AdamW |
|---|---|
| Classic CNN image classification at scale; often generalizes a little better *when tuned* | Transformers, LLMs, NLP, GANs, most fine-tuning: it's the default |
| Sensitive to the LR; needs a schedule | Works from a reasonable default (lr ≈ 1e-3 to 3e-4) with warmup |
| Less memory (one buffer per parameter) | Two buffers per parameter (matters for big models: [DL-07](../lessons/deep-learning/07-performance-gpus-mixed-precision.md)) |

**Settle it:** start with AdamW. Switch to SGD only for well-trodden vision recipes, where it's the known winner.
Use AdamW, not Adam with L2: decoupled weight decay is what you want ([DL-03](../lessons/deep-learning/03-training-deep-networks.md)).

## 8. Vision: train from scratch vs fine-tune vs linear probe, and CNN vs ViT

| Labels | Do |
|---|---|
| ~0–50 per class | Zero-shot CLIP/SigLIP, or kNN / a linear probe on frozen DINOv2/CLIP features ([CV-03](../lessons/vision/03-clip-vision-language-models.md)) |
| ~50–5,000 per class | **Fine-tune a pretrained model** (a ConvNeXt/ResNet or a ViT) with discriminative LRs |
| Huge, or a very unusual domain (medical, satellite, microscopy) | Self-supervised pretraining on your own unlabeled data ([CV-02](../lessons/vision/02-vision-transformers-self-supervised.md)), then fine-tune |

**CNN vs ViT:** with pretraining, both are excellent. From scratch on small data, CNNs win, because convolution's built-in assumptions (locality, translation equivariance) replace data.
ViTs scale better with data and pretraining, and they are the native input format for multimodal models.

## 9. Search: BM25 vs dense embeddings vs hybrid

| BM25 | Dense embeddings | Hybrid (RRF) + reranker |
|---|---|---|
| Exact terms: IDs, names, codes, rare words; zero training; explainable | Paraphrases, synonyms, cross-lingual queries, natural-language questions | The default for RAG: it rarely loses to either one alone |

**Settle it:** recall@10 and NDCG@10 on ≥ 30 labeled queries ([Lab 11](../labs/11-retrieval-metrics/README.md), [GEN-05](../lessons/llms-genai/05-retrieval-engineering.md)).

## 10. Forecasting: classical vs GBM with lags vs deep models

| Pick | When |
|---|---|
| **Seasonal naive / ETS / ARIMA** | One or a few series, short history, strong seasonality; also as the *baseline you must beat* ([outside the box §1](05-outside-the-box.md#1-the-no-model-baselines)) |
| **GBM with lag and calendar features** | Many related series (stores × products), external regressors, a forecast needed per row |
| **Deep / foundation models** | Thousands of series, long contexts, zero-shot needs. Benchmark them against the two rows above ([EL-01](../lessons/electives/01-time-series-forecasting.md)) |

Always validate with **rolling-origin** splits, never random K-fold.

## 11. Anomaly detection: which detector?

| Pick | When |
|---|---|
| **Robust statistics** (z-score with median/MAD, IQR) | One variable at a time; explainable alerts |
| **Isolation forest** | Many features, mixed scales; fast; a solid default |
| **LOF / kNN distance** | Anomalies are *locally* unusual (normal globally, odd for their neighbourhood) |
| **Autoencoder reconstruction error** | Images, signals, high-dimensional data with structure |
| **Supervised GBM** | You have even a few hundred labeled anomalies: it usually beats every unsupervised detector ([EL-06](../lessons/electives/06-anomaly-detection.md)) |

## 12. Hyperparameter search: grid vs random vs Bayesian vs successive halving

| Pick | When |
|---|---|
| **Grid** | 1–2 hyperparameters, a cheap model, or a final plot over one parameter |
| **Random** | A solid default for 3+ dimensions: it covers the important dimensions better than a grid for the same budget |
| **Bayesian (TPE, Optuna)** | Expensive trials, a moderate number of dimensions; add **pruning** to stop bad trials early |
| **Successive halving / Hyperband** | Many configurations, where a partial training run predicts the final rank |

Whatever you choose, keep a hold-out the search never sees ([CORE-12](../lessons/core-ml/12-hyperparameter-optimization.md)).

## 13. Imbalanced classes: class weights vs resampling vs moving the threshold

| Pick | Why |
|---|---|
| **Do nothing to the data, move the threshold** | Ranking metrics (AUC) don't need rebalancing; the decision threshold comes from costs ([Lab 04](../labs/04-logistic-regression-metrics/README.md)) |
| **Class weights** | Make the loss care about the rare class; easy in every library |
| **Undersampling the majority** | Training is too slow on huge data; correct the probabilities afterwards |
| **SMOTE-style oversampling** | Rarely helps strong models such as GBMs; try it last, and measure |

Resampling and class weights *distort the predicted probabilities*. Recalibrate if the probabilities are used ([CORE-11](../lessons/core-ml/11-uncertainty-calibration-conformal.md)).

## 14. Explaining a model: coefficients vs permutation importance vs SHAP vs PDP/ALE

| Question | Tool |
|---|---|
| "How does the prediction change with x, holding the rest fixed?" (a linear model) | Coefficients, on standardized features |
| "Which features does the model rely on overall?" | Permutation importance, on held-out data |
| "Why *this* prediction?" | SHAP values (local), and their summary plot (global) |
| "What's the shape of the effect of x?" | PDP, or **ALE** when features are correlated |

None of these are **causal**. "The model uses x" is not "changing x changes the outcome" ([EL-05](../lessons/electives/05-causal-inference-uplift.md)).

---

## Cheat sheet

| Decision | Default | Switch when |
|---|---|---|
| Tabular model | GBM | Few rows or extrapolation → linear; similarity is the product → kNN |
| Clustering | k-means | Odd shapes or noise → HDBSCAN; soft membership → GMM |
| LLM adaptation | Prompting | Missing knowledge → RAG; missing behaviour → LoRA |
| Rare-positive metric | PR-AUC + expected cost | Balanced classes → ROC-AUC; probabilities matter → log loss |
| Optimizer | AdamW + warmup + cosine | Known CNN recipe → SGD + momentum |
| Vision with few labels | Frozen features + linear probe | More labels → fine-tune |
| Search | Hybrid + rerank | Pure-identifier lookups → BM25 alone |
