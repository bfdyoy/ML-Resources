# CORE-08: Interpreting Models & Responsible ML

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~6 h | L2 | CORE-05, CORE-07 |

## Why this matters
Stakeholders ask "why did the model say that?", regulators ask "is it fair?", and debugging often starts with
"what is the model actually using?" Interpretation tools answer all three questions, but only if you know their
assumptions and failure modes.

## Learning goals
By the end you can:
- Separate *global* explanations (feature importance, PDP) from *local* ones (SHAP, LIME), and say when to use each.
- Explain why impurity-based importance is biased, and use permutation importance instead.
- Read SHAP summary and dependence plots correctly, including their limits (correlated features, and correlation vs causation).
- Name common fairness criteria (demographic parity, equalized odds) and audit a model for group disparities.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Interpreting Models & Responsible ML](../../notes/core-ml/08-interpretability-and-responsible-ml.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (Molnar) | The intro chapters on interpretability and its taxonomy, then the chapters on **Permutation Feature Importance** and **Partial Dependence Plot** | 1.5 h |
| 2 | **Read** | [Molnar: SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) | The whole chapter, including the limitations | 1 h |
| 3 | **Build** | `shap` + scikit-learn on your CORE-05 GBM | Compute permutation importance, PDPs (`sklearn.inspection`), and a SHAP summary plot. Compare what each one says. | 1.5 h |
| 4 | **Read + Build** | [Google MLCC: Fairness](https://developers.google.com/machine-learning/crash-course/fairness) | The whole module, plus its programming exercise | 1.5 h |

## Check your understanding
1. Why can impurity-based importance rank a random ID column highly?
2. What does a partial dependence plot assume about feature correlations, and what happens when that assumption is false?
3. What's the difference between "this feature has a high SHAP value" and "this feature *causes* the outcome"?
4. Explain demographic parity vs equalized odds. Why can't you usually satisfy both?
5. *(debug)* SHAP says `zip_code` is your loan model's top feature. What concerns does that raise, and what do you check?

## Mini-project
**Task:** Write a one-page "model explanation report" for your CORE-05 model: the top global drivers (with the method and
its caveats), three individual predictions explained, and a group-fairness check on one sensitive attribute.
**Dataset:** UCI Adult (includes sex/race attributes, a classic fairness benchmark).
**Deliverable:** A markdown report plus a notebook.

## Go deeper
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course): the "Production ML systems" module, which pairs well with PROD-01.
- [UDL](https://udlbook.github.io/udlbook/) Ch. 21 "Deep Learning and Ethics": a thoughtful, broad view.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md), for every concept in this lesson, with alternatives.
- **Papers:** [Interpretability, uncertainty & responsible ML](../../papers/10-interpretability-uncertainty-responsible.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rung 16.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
- **Playbook:** [choosing an explanation method](../../playbook/01-choosing-algorithms.md#14-explaining-a-model-coefficients-vs-permutation-importance-vs-shap-vs-pdpale) · [error models for slice discovery](../../playbook/05-outside-the-box.md#5-train-a-model-on-your-models-errors).
