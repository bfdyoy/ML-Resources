# EL-10: Deep Learning for Tabular Data

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~5 h | L2 | CORE-05, CORE-07, DL-03 |

## Why this matters
On tabular data, gradient-boosted trees remain the default, and knowing *why* saves you from wasted effort. Still, neural nets have real uses:
entity embeddings for high-cardinality categoricals, multimodal rows (tables plus text or images), and in-context "tabular foundation models" like
TabPFN on small datasets. This elective shows you how to make an honest comparison.

## Learning goals
By the end you can:
- Explain the inductive-bias reasons trees beat neural nets on typical tabular data (irregular functions, uninformative features, rotation invariance).
- Build a neural net for tables with entity embeddings for categoricals and proper numeric preprocessing.
- Use a tabular foundation model (TabPFN) for small datasets, and know its limits.
- Run a fair benchmark: the same splits, equal tuning budgets, multiple seeds, and significance.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | The whole paper (benchmark design and the inductive-bias experiments) | 1.5 h |
| 2 | **Read + Build** | [fastbook](https://github.com/fastai/fastbook) | Ch. 9 "Tabular Modeling Deep Dive" (random forests, entity embeddings, neural nets) | 2 h |
| 3 | **Read** | [TabPFN](https://arxiv.org/abs/2207.01848) | §1–3 (prior-fitted networks, in-context learning on tables) | 45 min |
| 4 | **Build** | [tabular-benchmark](https://github.com/LeoGrin/tabular-benchmark) | Reproduce one comparison on 2–3 datasets | 1 h |

## Check your understanding
1. Name two inductive biases that help trees on tabular data but hurt MLPs.
2. What do entity embeddings learn for a categorical variable, and how can you reuse them in a GBM?
3. Why does feature scaling matter for neural nets but not for trees?
4. When would you pick TabPFN, and when would it fail (dataset size, number of features)?
5. What makes a tabular benchmark "fair"?
6. *(debug)* Your MLP beats XGBoost by 3% on one random split. What do you check before you believe it?

## Mini-project
**Task:** On 3 tabular datasets, compare a tuned GBM, an MLP with entity embeddings, and TabPFN (where the size allows), over 5 seeds with equal tuning budgets.
**Deliverable:** A table of mean ± std per dataset, and a conclusion you'd defend in a code review.

## Go deeper
- [CatBoost](https://arxiv.org/abs/1706.09516): a strong GBM for categoricals.
- [EL-01 Time Series](01-time-series-forecasting.md): where neural and foundation models are more competitive.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (tabular deep learning) · [Toolbox 02: Classical ML](../../toolbox/02-classical-ml.md).
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md) (tabular & boosting section).
- **Implement it yourself:** an entity-embedding MLP in PyTorch. Feed its embeddings into a GBM and measure the gain.
- **Drills:** more in [exercises/](../../exercises/README.md).
