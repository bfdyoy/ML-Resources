# MATH-03: Probability & Statistics for ML (just-in-time)

| Track | Time | Level | Used by |
|---|---|---|---|
| Math | ~5 h (in pieces) | L1→L2 | CORE-03, GEN-04, EL-02, EL-03 |

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

## Self-check
1. Use Bayes' rule on a test with 99% sensitivity and 1% prevalence. What's P(disease | positive)?
2. Why does minimizing MSE correspond to MLE under Gaussian noise?
3. KL divergence is not symmetric. Give an intuition for what KL(p‖q) vs KL(q‖p) penalizes.
