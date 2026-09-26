# CORE-10: Data-Centric ML: Label Quality, Imbalance & Learning with Few Labels

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~7 h | L2 | CORE-03, CORE-07 |

## Why this matters
In real projects, improving the **data** usually beats improving the model. Label errors, class imbalance, outliers, and
too few labels are the everyday problems of applied ML. This lesson turns "clean the data" from a chore into a systematic,
measurable discipline.

## Learning goals
By the end you can:
- Find likely label errors with confident learning, and estimate how much they hurt a model.
- Handle class imbalance deliberately: reweighting vs resampling vs threshold tuning, and know when SMOTE doesn't help.
- Measure label quality (inter-annotator agreement) and write good labeling guidelines.
- Use semi-supervised learning (self-training, consistency regularization) when labels are scarce.
- Run an active-learning loop to decide *which* examples to label next.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [MIT Introduction to Data-Centric AI](https://dcai.csail.mit.edu/) | Lecture "Data-Centric AI vs. Model-Centric AI" (notes) | 30 min |
| 2 | **Read + Build** | [MIT DCAI](https://dcai.csail.mit.edu/) + [dcai-lab](https://github.com/dcai-course/dcai-lab) | Lectures "Label Errors and Confident Learning" and "Class Imbalance, Outliers, and Distribution Shift", each with its lab | 3 h |
| 3 | **Read** | [imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) + [sklearn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) | Both pages | 45 min |
| 4 | **Read** | [Lilian Weng: Thinking about High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) | The whole post: rater agreement, guidelines, and spotting bad labels | 45 min |
| 5 | **Read** | [Lilian Weng: Semi-supervised learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/) → [Active learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/) | Focus on self-training / pseudo-labels, consistency regularization, and uncertainty sampling | 1.5 h |

**Notes for the learner:** The Lilian Weng posts are research surveys. Read the intro and the first method family in each
properly, and skim the rest. The goal is to know what tools exist, not to memorize every paper.

## Check your understanding
1. What does confident learning estimate, and why does it need *out-of-sample* predicted probabilities?
2. You have 1% positives. Compare class weights, random undersampling, SMOTE, and threshold tuning. Which do you try first, and why?
3. Why can oversampling *before* the train/test split be disastrous?
4. What does Cohen's κ measure that raw agreement doesn't?
5. When does self-training make a model *worse*?
6. Uncertainty sampling picks the most uncertain points. What failure mode does that have, and how does diversity sampling help?
7. *(debug)* Fixing 3% of the labels in your training set made test accuracy *drop*. What's the most likely explanation?

## Mini-project
**Task:** Take a dataset, inject 10% label noise, use [cleanlab](https://github.com/cleanlab/cleanlab) to find it, and measure how much fixing it recovers.
Then simulate an active-learning loop (start with 50 labels, add 25 per round by uncertainty) and plot the learning curve against random sampling.
**Dataset:** `sklearn.datasets.load_digits` or a text dataset such as 20 Newsgroups.
**Deliverable:** Precision/recall of the noise detection, and a learning-curve plot (active vs random).

## Go deeper
- [MIT DCAI](https://dcai.csail.mit.edu/): lectures "Dataset Creation and Curation" and "Data Curation for LLMs".
- [FixMatch](https://arxiv.org/abs/2001.07685): a simple, strong semi-supervised baseline.
- [SMOTE](https://arxiv.org/abs/1106.1813): the original paper, so you know exactly what it does.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) · [Toolbox 02: Classical ML](../../toolbox/02-classical-ml.md) (semi-supervised and active learning).
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md) (SMOTE) · [Interpretability, uncertainty & responsible ML](../../papers/10-interpretability-uncertainty-responsible.md) (datasheets).
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 6–7 (metrics and stratified splits). Then write a 20-line uncertainty-sampling loop.
- **Drills:** [dcai-lab](https://github.com/dcai-course/dcai-lab) · more in [exercises/](../../exercises/README.md).
