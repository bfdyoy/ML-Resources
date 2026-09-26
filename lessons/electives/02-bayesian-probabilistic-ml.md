# EL-02: Bayesian & Probabilistic ML

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~10 h | L2 | CORE-03, [MATH-03](../math/03-probability-statistics.md) |

## Why this matters
Point predictions hide uncertainty. Bayesian thinking gives you calibrated uncertainty, principled ways to use prior
knowledge, and better decisions from small data. Those matter in A/B testing, medicine, science, and anywhere
being wrong is expensive.

## Learning goals
By the end you can:
- Explain priors, likelihoods, and posteriors, and do a Bayesian update by hand for simple conjugate cases.
- Fit Bayesian models with a probabilistic programming library (PyMC), and read the posterior.
- Explain MCMC at an intuitive level, and check convergence (trace plots, R-hat).
- Build a hierarchical (multilevel) model, and explain partial pooling.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read + Build** | [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) | Ch. 1 (the philosophy of Bayesian inference), Ch. 2 (more PyMC), Ch. 3 (opening the MCMC black box). Use the PyMC notebooks. | 4 h |
| 2 | **Watch + Read** | [Statistical Rethinking 2026](https://github.com/rmcelreath/stat_rethinking_2026) (McElreath) | The first 4–5 lectures (Bayesian workflow, the garden of forking data, …) and the matching problem sets | 5 h |
| 3 | **Build** | PyMC | Rebuild one Statistical Rethinking example in PyMC, as a hierarchical model | 1 h |

**Notes for the learner:** Statistical Rethinking is lecture-based, but it's the rare lecture series that behaves like
a book: slow, careful, and full of worked examples. The code in the lectures is R/Stan. Community PyMC ports exist.

## Check your understanding
1. What's the difference between a confidence interval and a credible interval?
2. What does a hierarchical model do for a group with very few observations?
3. Why do we need MCMC? Why can't we just compute the posterior directly?
4. *(debug)* Your MCMC chains don't mix (R-hat = 1.4). What would you try?

## Mini-project
**Task:** A Bayesian A/B test. Estimate the posterior of the conversion-rate difference between two variants, compute
P(B > A), and make a decision with an explicit loss function.
**Deliverable:** A notebook with posterior plots and a recommendation.

## Go deeper
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/) (Murphy) + [pyprobml](https://github.com/probml/pyprobml): the full probabilistic view of ML.
- [MML book](https://mml-book.github.io/) Ch. 6 "Probability and Distributions" and Ch. 8 "When Models Meet Data"
