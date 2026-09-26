# The From-Scratch Ladder

[← Exercises](README.md)

Implement the core algorithms of ML yourself, in **NumPy first, then PyTorch**. Each rung has a **check**: a concrete, testable
definition of done. Passing the check is what counts. "It runs" doesn't.

**Rules:** (1) Read the matching lesson/toolbox entry first. (2) Try for 30 minutes before looking at a reference.
(3) Keep all rungs in one repo (`ml-from-scratch/`) with a `tests/` folder. The repo becomes a portfolio piece.

**Reference implementations** (look only after you've tried): [labml.ai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) ·
[Géron's notebooks](https://github.com/ageron/handson-mlp) · [Deep-ML problems](https://www.deep-ml.com/problems) (many rungs exist there as auto-graded problems) ·
[rasbt/deeplearning-models](https://github.com/rasbt/deeplearning-models) · [Karpathy's repos](https://github.com/karpathy/nn-zero-to-hero)

---

## Level 1: Classical ML (NumPy only)
| # | Build | Check (definition of done) | Lesson |
|---|---|---|---|
| 1 | Linear regression via the normal equation, and via batch GD | Coefficients match `LinearRegression` to 1e-6 on `load_diabetes` | CORE-02 |
| 2 | Mini-batch SGD with momentum, and feature standardization | Converges on unscaled vs scaled data. Plot both loss curves. | CORE-02 |
| 3 | Ridge (closed form) and lasso (coordinate descent) | Match `Ridge`/`Lasso` coefficients to 1e-4 across 5 alphas | CORE-04 |
| 4 | Logistic regression with BCE loss and GD, plus L2 | Accuracy within 0.5% of `LogisticRegression`; gradient check passes | CORE-03 |
| 5 | Softmax regression (multiclass) | Accuracy within 1% of sklearn on `load_digits` | CORE-03 |
| 6 | Metrics: confusion matrix, precision/recall/F1, ROC curve, AUC, PR-AUC | Match `sklearn.metrics` exactly on 3 random prediction sets | CORE-03 |
| 7 | k-fold, stratified, and grouped cross-validation splitters | Folds are disjoint, cover all rows, and respect strata/groups (write assertions) | CORE-04 |
| 8 | k-nearest neighbours (vectorized distances, no loops) | Matches `KNeighborsClassifier` predictions exactly | toolbox 02 |
| 9 | Gaussian naive Bayes | Matches `GaussianNB` `predict_proba` to 1e-6 | toolbox 02 |
| 10 | Decision tree (CART, Gini/entropy, max_depth, min_samples) | Within 1% accuracy of `DecisionTreeClassifier` with the same params | CORE-05 |
| 11 | Random forest (bootstrap + feature subsampling) + OOB score | OOB score within 2% of sklearn's | CORE-05 |
| 12 | Gradient boosting for regression (fit residuals with your tree) | Train MSE decreases monotonically; test MSE within 5% of `GradientBoostingRegressor` | CORE-05 |
| 13 | k-means (with k-means++ init) | Same inertia as `KMeans(n_init=1)` given the same init | CORE-06 |
| 14 | PCA via SVD, with explained variance | Components match `PCA` up to sign | CORE-06 |
| 15 | Gaussian mixture model via EM (1-D, then diagonal N-D) | Log-likelihood increases every iteration; parameters close to `GaussianMixture` | CORE-06 |
| 16 | Permutation importance and a 1-D partial dependence plot | Match `sklearn.inspection` outputs | CORE-08 |
| 17 | Bootstrap confidence interval for a model metric | 95% CI covers the true value in ~95% of 200 simulated datasets | toolbox 01 |

## Level 2: Neural networks (NumPy → PyTorch)
| # | Build | Check | Lesson |
|---|---|---|---|
| 18 | Scalar autograd engine (micrograd-style) | Gradients match PyTorch on 5 random expressions ([micrograd](https://github.com/karpathy/micrograd) for reference) | DL-01 |
| 19 | MLP with vectorized forward/backward in NumPy | Finite-difference gradient check (relative error < 1e-6); >97% on MNIST | DL-01 |
| 20 | Optimizers: SGD, momentum, RMSProp, Adam, AdamW | Parameter trajectories match `torch.optim` for 100 steps | DL-03 |
| 21 | BatchNorm and LayerNorm forward + backward (NumPy) | Gradients match PyTorch autograd | DL-03 |
| 22 | Dropout, and weight initializations (Xavier/He) | Activation std stays ~constant across 20 layers with He init + ReLU | DL-03 |
| 23 | Conv2d forward/backward (im2col), max-pool | Outputs and gradients match `F.conv2d` / `F.max_pool2d` | DL-04 |
| 24 | ResNet-18 in PyTorch + training loop with AMP and a cosine schedule | >90% on CIFAR-10 | DL-04 |
| 25 | LSTM cell | Matches `nn.LSTMCell` outputs given the same weights | DL-05 |
| 26 | Character-level language model (bigram → MLP) | Dev NLL below the bigram baseline ([makemore](https://github.com/karpathy/makemore)) | DL-05 |

## Level 3: Transformers & LLMs (PyTorch)
| # | Build | Check | Lesson |
|---|---|---|---|
| 27 | Scaled dot-product attention with a causal mask | Matches `F.scaled_dot_product_attention` | DL-06 |
| 28 | Multi-head attention → transformer block → GPT | Overfits a tiny text; trains on Tiny Shakespeare to val loss < 1.6 ([nanoGPT](https://github.com/karpathy/nanoGPT)) | DL-06 |
| 29 | BPE tokenizer: train, encode, decode | `decode(encode(s)) == s` for 1,000 random strings; merges match [minbpe](https://github.com/karpathy/minbpe) on a test corpus | GEN-01 |
| 30 | Sampling: temperature, top-k, top-p | Empirical token frequencies match the target distribution (χ² test) | GEN-01 |
| 31 | KV cache for your GPT | Identical outputs with and without the cache; measure the speed-up | toolbox 07 |
| 32 | RoPE + RMSNorm + SwiGLU upgrade | Same or better val loss than rung 28 at equal compute | toolbox 07 |
| 33 | LoRA layer wrapping `nn.Linear` | Base weights frozen; merged weights give identical outputs; only ~1% of params trainable | GEN-02 |
| 34 | Instruction fine-tuning loop with loss masking on prompt tokens | Loss computed only on response tokens (assert on the mask) | GEN-02 |
| 35 | DPO loss | Matches TRL's `DPOTrainer` loss on a fixed batch ([TRL](https://github.com/huggingface/trl)) | GEN-02 |
| 36 | INT8 symmetric weight quantization of a linear layer | Output error < 1% relative; memory reduced ~4× | toolbox 07 |

## Level 4: Retrieval & RAG
| # | Build | Check | Lesson |
|---|---|---|---|
| 37 | BM25 ranking from scratch | Top-10 matches a reference implementation on a small corpus | toolbox 08 |
| 38 | Dense retrieval: embed, cosine top-k with NumPy, then with Faiss HNSW | Recall@10 of HNSW ≥ 0.95 of exact search | toolbox 08 |
| 39 | Retrieval evaluation: recall@k, MRR, NDCG@k | Matches hand-computed values on 3 toy examples | toolbox 08 |
| 40 | Minimal RAG pipeline + LLM-judge faithfulness eval | Judge agreement with your own labels (Cohen's κ) > 0.6 on 50 samples | GEN-03 |

## Level 5: Generative models & RL
| # | Build | Check | Lesson |
|---|---|---|---|
| 41 | VAE on MNIST | ELBO improves; interpolations in latent space look smooth | GEN-04 |
| 42 | DDPM on 2-D toy data, then on MNIST | 2-D samples match the data distribution visually and by MMD | GEN-04 |
| 43 | Tabular Q-learning on FrozenLake | Success rate > 70% on the slippery version | EL-03 |
| 44 | REINFORCE with a baseline on CartPole | Reaches 475+ average return | EL-03 |
| 45 | PPO (clipped objective + GAE) | Solves CartPole across 5 seeds; compare with [CleanRL](https://github.com/vwxyzjn/cleanrl) | EL-03 |

---

## Capstone rungs (optional, big)
- **Train a small LLM end to end** following [nanochat](https://github.com/karpathy/nanochat) or [CS336 assignment 1](https://github.com/stanford-cs336/assignment1-basics).
- **Write the forward pass of GPT-2 in C/CUDA** following [llm.c](https://github.com/karpathy/llm.c).
- **Reimplement one paper** from the [papers library](../papers/README.md) and compare against [labml.ai](https://github.com/labmlai/annotated_deep_learning_paper_implementations).
