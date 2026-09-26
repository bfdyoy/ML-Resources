# CORE-07: Feature Engineering, Pipelines & Data Leakage

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~6 h | L2 | CORE-05 |

## Why this matters
On real tabular problems, better features usually beat better algorithms. And the most common *silent* failure in
applied ML is **leakage**: information from the future or from the test set slipping into training. This lesson
makes both of these deliberate skills rather than accidents.

## Learning goals
By the end you can:
- Encode categorical variables (one-hot, ordinal, target encoding done safely) and handle missing values on purpose.
- Create useful features: interactions, ratios, date parts, aggregations, and text/TF-IDF basics.
- Spot the main kinds of leakage (target leakage, train/test contamination, temporal leakage) and prevent them with pipelines.
- Do feature selection without leaking (inside CV).

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) | The whole page: inconsistent preprocessing, data leakage, randomness | 30 min |
| 2 | **Build** | [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | Lessons "Missing Values", "Categorical Variables", "Pipelines", "Cross-Validation", "Data Leakage" | 2 h |
| 3 | **Build** | [Kaggle Learn: Feature Engineering](https://www.kaggle.com/learn/feature-engineering) | All lessons: mutual information, creating features, clustering as features, PCA, target encoding | 2 h |
| 4 | **Read** | [Feature Engineering and Selection](https://bookdown.org/max/FES) (Kuhn & Johnson) | Chapter "Encoding Categorical Predictors" and the chapter on feature-selection overfitting. The code is in R. Read for the ideas and examples. | 1.5 h |

**Notes for the learner:** After this lesson, go back to your CORE-01 and CORE-05 mini-projects and try to find a leak
in them. It's the best exercise there is.

## Check your understanding
1. Why is target encoding dangerous, and how do you do it safely?
2. A feature called `days_since_account_closed` has huge importance in your churn model. Why should you be suspicious?
3. Why must feature selection happen *inside* each CV fold?
4. When is "missingness" itself a useful feature?
5. How would you encode a categorical feature with 50,000 levels?
6. *(debug)* A random-split CV score is 0.92, and a time-based split gives 0.64. What does that tell you?

## Mini-project
**Task:** Take a messy dataset, build a single `Pipeline` with a `ColumnTransformer`, add 5 engineered features, and
show (with CV) which of them actually helped. Then plant a deliberate leak and show how badly it inflates the score.
**Dataset:** Kaggle House Prices (from the Intermediate ML course), or the Ames Housing data on OpenML.
**Deliverable:** A notebook with an ablation table (feature, CV delta).

## Go deeper
- [Inria scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/): its preprocessing and pipeline modules, as a second pass.
- [Géron, *Hands-On ML*, Ch. 2](https://github.com/ageron/handson-mlp): revisit its custom transformers section.
