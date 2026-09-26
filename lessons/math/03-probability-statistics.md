# MATH-03: Probability & Statistics for ML (just-in-time)

| Track | Time | Level | Used by |
|---|---|---|---|
| Math | ~9 h (in pieces) | L1→L2 | CORE-03, CORE-04, CORE-11, GEN-04, EL-02, EL-03, EL-05 |

## Block A: Probability basics, Bayes, and distributions (for CORE-03)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [Seeing Theory](https://seeing-theory.brown.edu/) (Brown) | "Basic Probability", "Compound Probability", "Probability Distributions", "Bayesian Inference" | 1.5 h |
| **Read** | [MML book](https://mml-book.github.io/) ([PDF](https://mml-book.github.io/book/mml-book.pdf)) | Ch. 6 "Probability and Distributions": sum/product rule and Bayes (§6.3), summary statistics (§6.4), **the Gaussian (§6.5)** | 1.5 h |

## Block B: Estimation and likelihood (for CORE-03 logistic regression, DL-01 loss functions)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [Seeing Theory](https://seeing-theory.brown.edu/) | "Frequentist Inference" and "Regression Analysis" | 45 min |
| **Read** | [MML book](https://mml-book.github.io/) | Ch. 8 "When Models Meet Data": empirical risk minimization (§8.2), **parameter estimation: MLE and MAP (§8.3)** | 1 h |

## Block C: Information theory essentials (for DL-01 cross-entropy, GEN-04 KL divergence)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Read** | [UDL](https://udlbook.github.io/udlbook/) | Ch. 5 "Loss Functions" (it derives cross-entropy from maximum likelihood) and the book's appendix on probability (KL divergence) | 1 h |

## Block D: Statistics for evaluating models and running experiments (for CORE-04, CORE-11, EL-02, EL-05)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [Seeing Theory](https://seeing-theory.brown.edu/) | "Frequentist Inference" (confidence intervals, hypothesis testing), revisited with experiments in mind | 30 min |
| **Read** | [Think Stats](https://greenteapress.com/wp/think-stats-3e/) (Downey) | The chapters on estimation, hypothesis testing, and resampling. Python-first, with simulation instead of formulas. | 2 h |
| **Read** | [ISLP](https://www.statlearning.com/) | §5.2 (the bootstrap) and Ch. 13 "Multiple Testing" (why running 100 comparisons produces "significant" junk) | 1.5 h |
| *Go deeper* | [Stat 110](https://stat110.hsites.harvard.edu/) (Harvard) | The full probability course, if you want rigour | — |

## Self-check
1. Use Bayes' rule on a test with 99% sensitivity and 1% prevalence. What's P(disease | positive)?
2. Why does minimizing MSE correspond to MLE under Gaussian noise?
3. KL divergence is not symmetric. Give an intuition for what KL(p‖q) vs KL(q‖p) penalizes.
4. Model B beats model A by 0.4% accuracy on 2,000 test examples. How would you check whether that's real (bootstrap or a paired test)?
5. You compared 40 feature sets and 3 look "significant" at p < 0.05. What should you conclude?
