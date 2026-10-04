# Study notes: the smooth ride

[← Back to the README](../README.md) · [Notation & symbols](notation.md)

The lessons tell you **what to study and where** (the best chapter, essay, or notebook for each step). The study notes are the
**explanation itself**, written for this curriculum. There's one note per lesson, and every note covers:

- **Where we are:** how this topic follows from the previous one, so nothing appears out of nowhere.
- **The ideas in plain words**, then **the math step by step**. Every formula is derived or motivated, and every symbol is defined ([notation guide](notation.md)).
- **Worked examples with real numbers**, small enough to do by hand.
- **Runnable code:** NumPy, scikit-learn, or PyTorch on CPU. Every printed number in a note was produced by its code, and the code is checked before each commit.
- **Pitfalls & misconceptions:** the mistakes people actually make.
- **A cheat sheet** you can revise from in five minutes.
- **Answer sketches** for every *Check your understanding* question in the lesson (folded, so you can try first).
- **Where this leads:** the bridge to the next note.

All 53 notes together take about **51 hours** of reading. That's about one hour per lesson.

---

## How to use them

1. **Before a lesson, read its note** (the lesson's step 0, about 1 h). It gives you the map: the intuition, the key equations, and how the pieces connect. Run the code cells as you go.
2. **Then do the lesson's steps** (book chapter, visual essay, notebook). You'll recognize everything, and the resources fill in the depth, the alternative explanations, and the practice.
3. **Answer the lesson's self-check questions**, *then* open the answer sketches at the bottom of the note.
4. **Later, revise from the cheat sheets.** Each one is the whole lesson on half a page.

If a note feels too fast, follow its **"You need"** links back to the exact section it builds on. If it feels too slow, skim to the math and the code.

---

## The route

The notes follow the same order as the [paths](../paths/), and each one links to the next, so you can read them start to finish like a book:

```
MATH-01 → MATH-02 → MATH-03
   → CORE-01 … CORE-12                         (Path 1)
   → DL-01 … DL-09                             (Path 2)
   → GEN-01 → 02 → 03 → 05 → 06 → 09           (Path 3, Part A: building LLM apps)
   → GEN-07 → 08                               (Part B: under the hood)
   → GEN-04 → 10                               (Part C: generative media)
   → PROD-01 … PROD-05                         (Path 4)
   → CV-01 … CV-04                             (Path 5)
   → EL-01 … EL-10                             (electives, any order)
```

The math notes are written so you can also dip into them *just in time*. Every later note links to the exact math section it uses.

---

## Index

### Math foundations
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| MATH-01 | [Linear Algebra for ML](math/01-linear-algebra.md) | 60 min | Matrices as maps, least squares as a projection, eigenvectors, PCA, SVD, conditioning |
| MATH-02 | [Calculus & Optimization for ML](math/02-calculus-optimization.md) | 60 min | Gradients, the chain rule, backprop by hand, why the condition number sets GD speed, momentum, convexity |
| MATH-03 | [Probability & Statistics for ML](math/03-probability-statistics.md) | 75 min | Bayes, where MSE and cross-entropy come from (MLE), KL divergence, error bars, paired tests, multiple testing |

### Path 1 · Core ML
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| CORE-01 | [The ML Workflow, End to End](core-ml/01-ml-workflow-end-to-end.md) | 40 min | Risk vs empirical risk, why the test set is used once, leakage (with a dramatic demo) |
| CORE-02 | [Linear Models & Gradient Descent](core-ml/02-linear-models-gradient-descent.md) | 50 min | Three derivations of least squares, R², standard errors, scaling and GD, collinearity |
| CORE-03 | [Classification & Evaluation Metrics](core-ml/03-classification-and-metrics.md) | 55 min | Log-odds, the log-loss gradient, cost-optimal thresholds, ROC vs PR, LDA vs logistic regression |
| CORE-04 | [Generalization, Validation & Regularization](core-ml/04-generalization-validation-regularization.md) | 60 min | The bias-variance derivation, CV variants, nested CV, ridge as SVD shrinkage, lasso's soft-thresholding, double descent |
| CORE-05 | [Trees, Random Forests & Gradient Boosting](core-ml/05-trees-and-ensembles.md) | 60 min | Impurity splits, the variance of averaged trees, OOB, boosting as gradient descent, XGBoost's leaf formula |
| CORE-06 | [Unsupervised Learning](core-ml/06-unsupervised-learning.md) | 55 min | PCA as minimum reconstruction error, k-means convergence, GMMs/EM, DBSCAN, reading t-SNE safely |
| CORE-07 | [Feature Engineering, Pipelines & Leakage](core-ml/07-feature-engineering-pipelines-leakage.md) | 50 min | Safe target encoding, cyclical features, TF-IDF, a field guide to leakage |
| CORE-08 | [Interpreting Models & Responsible ML](core-ml/08-interpretability-and-responsible-ml.md) | 55 min | Permutation importance, PDP assumptions, Shapley values computed exactly, the fairness impossibility |
| CORE-09 | [Kernels, SVMs, Nearest Neighbours & GPs](core-ml/09-kernels-svms-nearest-neighbours.md) | 70 min | The curse of dimensionality, margins and hinge loss, the dual and the kernel trick, random features, the GP posterior |
| CORE-10 | [Data-Centric ML](core-ml/10-data-centric-ml.md) | 55 min | Confident learning, what resampling does to probabilities, Cohen's κ, self-training, active learning |
| CORE-11 | [Calibration & Conformal Prediction](core-ml/11-uncertainty-calibration-conformal.md) | 60 min | ECE, temperature scaling, the full conformal coverage proof, CQR, the limits under shift |
| CORE-12 | [Hyperparameter Optimization](core-ml/12-hyperparameter-optimization.md) | 50 min | Why random beats grid, expected improvement, successive halving, the optimism of the best trial |

### Path 2 · Deep Learning
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| DL-01 | [Neural Networks from Scratch & Backprop](deep-learning/01-neural-networks-from-scratch.md) | 65 min | Why depth needs non-linearity, the softmax+CE gradient, matrix backprop, a 40-line autograd engine |
| DL-02 | [PyTorch Fluency](deep-learning/02-pytorch-fluency.md) | 45 min | Strides and views, broadcasting (and its silent bug), autograd mechanics, the canonical loop |
| DL-03 | [Training Deep Networks Well](deep-learning/03-training-deep-networks.md) | 70 min | Deriving He/Xavier init, residual highways, BatchNorm, Adam's bias correction, AdamW, the debugging recipe |
| DL-04 | [Convolutional Networks](deep-learning/04-cnns-computer-vision.md) | 55 min | Output sizes, parameter counts, receptive fields, equivariance, fine-tuning and domain shift |
| DL-05 | [Embeddings, Language Modeling & Sequences](deep-learning/05-embeddings-sequences-attention.md) | 60 min | Skip-gram, perplexity, why RNN gradients vanish, LSTM memory, attention as "look back" |
| DL-06 | [Transformers](deep-learning/06-transformers.md) | 75 min | Why √d_k, causal masks, permutation equivariance, the residual stream, the 12d² parameter rule |
| DL-07 | [Making Training Fast](deep-learning/07-performance-gpus-mixed-precision.md) | 55 min | The roofline, fusion, FP16 vs BF16, 16 bytes per parameter, checkpointing, the utilization drill |
| DL-08 | [Modern Architectures: RoPE, GQA, MoE, SSMs](deep-learning/08-modern-architectures-moe-ssm.md) | 70 min | RoPE's relative-position proof, KV-cache arithmetic, MoE routing and balance, selective SSMs |
| DL-09 | [Graph Neural Networks](deep-learning/09-graph-neural-networks.md) | 55 min | Message passing, GCN/SAGE/GAT, over-smoothing, the WL expressivity limit, transformers as GNNs |

### Path 3 · LLMs & Generative AI
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| GEN-01 | [How LLMs Are Built](llms-genai/01-how-llms-are-built.md) | 60 min | BPE from scratch, C ≈ 6ND, scaling laws, SFT → preferences → RLVR, sampling knobs |
| GEN-02 | [Adapting LLMs: Fine-tuning, LoRA & RAG](llms-genai/02-adapting-llms-finetuning-rag.md) | 60 min | The LoRA math and memory savings, forgetting, debugging RAG stage by stage, choosing a lever |
| GEN-03 | [Evaluating & Shipping LLM Apps](llms-genai/03-evaluating-llm-apps.md) | 50 min | Error analysis, eval levels, error bars, validating judges (TPR/TNR, correction), judge biases |
| GEN-05 | [Retrieval Engineering for RAG](llms-genai/05-retrieval-engineering.md) | 65 min | Ranking metrics, BM25, contrastive bi-encoders, cross-encoders, IVF/PQ/HNSW, RRF, HyDE |
| GEN-06 | [Agents, Tool Use & MCP](llms-genai/06-agents-tool-use.md) | 50 min | Workflows vs agents, error compounding, constrained decoding, MCP, pass@k vs pass^k |
| GEN-09 | [LLM Security & Safety](llms-genai/09-llm-security-safety.md) | 45 min | Why injection can't be prompted away, the lethal combination, defence in depth, red-team statistics |
| GEN-07 | [Post-Training: SFT, RLHF, DPO & Reasoning](llms-genai/07-post-training-alignment-reasoning.md) | 75 min | Bradley–Terry, the closed-form RLHF optimum, deriving DPO, GRPO, reward hacking |
| GEN-08 | [Efficient LLM Inference](llms-genai/08-efficient-llm-inference.md) | 60 min | Prefill vs decode, the decode speed limit, quantization and outliers, why speculative decoding is exact |
| GEN-04 | [Generative Models: VAEs, GANs & Diffusion](llms-genai/04-generative-models-diffusion.md) | 75 min | The ELBO, reparameterization, the GAN game, DDPM from scratch, CFG, latent diffusion |
| GEN-10 | [Advanced Diffusion & Flow Matching](llms-genai/10-advanced-diffusion-flow-matching.md) | 75 min | Scores, denoising score matching, the probability-flow ODE, flow matching, few-step sampling |

### Path 4 · ML in Production
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| PROD-01 | [ML System Design](production/01-ml-system-design.md) | 45 min | Framing and metrics, training/serving skew, batch vs online, label-free shift detection (PSI, domain classifier) |
| PROD-02 | [MLOps in Practice](production/02-mlops-in-practice.md) | 40 min | Reproducibility, the test layers, rollout, retrain triggers, champion/challenger gates |
| PROD-03 | [Testing, Monitoring & Drift](production/03-testing-monitoring-drift.md) | 50 min | Behavioural tests, the three drifts as probability, BBSE, why alerts fire at scale |
| PROD-04 | [Distributed Training at Scale](production/04-distributed-training.md) | 60 min | Ring all-reduce, ZeRO stages by the numbers, TP/PP/bubbles, gradient accumulation, scaling debugging |
| PROD-05 | [LLMOps](production/05-llmops-genai-platforms.md) | 45 min | Platform layers, traces, caching risks, routing economics, why the bill doubled |

### Path 5 · Computer Vision
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| CV-01 | [Object Detection & Segmentation](vision/01-object-detection-segmentation.md) | 60 min | IoU, box regression, NMS, focal loss, FPN, AP/mAP computed, U-Net |
| CV-02 | [ViTs & Self-Supervised Learning](vision/02-vision-transformers-self-supervised.md) | 60 min | Patchify, inductive bias vs data, NT-Xent, avoiding collapse, MAE's 75%, probes |
| CV-03 | [CLIP & Vision-Language Models](vision/03-clip-vision-language-models.md) | 55 min | The CLIP loss, zero-shot and prompt ensembles, SigLIP, the three VLM bridges, hallucination |
| CV-04 | [3D Vision & Neural Rendering](vision/04-3d-vision-neural-rendering.md) | 60 min | Camera projection, SfM, NeRF's volume rendering, positional encoding, Gaussian splatting |

### Electives
| ID | Notes | Read | What you'll understand |
|---|---|---|---|
| EL-01 | [Time-Series Forecasting](electives/01-time-series-forecasting.md) | 55 min | ACF, stationarity, strong baselines, ETS/ARIMA, rolling-origin evaluation, lag features |
| EL-02 | [Bayesian & Probabilistic ML](electives/02-bayesian-probabilistic-ml.md) | 55 min | Conjugate updates, credible vs confidence intervals, partial pooling, Metropolis, R-hat |
| EL-03 | [Reinforcement Learning](electives/03-reinforcement-learning.md) | 65 min | Bellman equations, TD, SARSA vs Q-learning (cliff walking), DQN, the policy-gradient theorem |
| EL-04 | [Recommender Systems](electives/04-recommender-systems.md) | 60 min | Matrix factorization and ALS, implicit feedback, two-tower and wide & deep, popularity bias |
| EL-05 | [Causal Inference & Uplift](electives/05-causal-inference-uplift.md) | 70 min | Potential outcomes, DAGs and colliders, IPW, DiD, IV, meta-learners, uplift curves |
| EL-06 | [Anomaly Detection](electives/06-anomaly-detection.md) | 50 min | Isolation forest, LOF, robust covariance, reconstruction error, alert budgets, context |
| EL-07 | [Mechanistic Interpretability](electives/07-mechanistic-interpretability.md) | 65 min | The residual stream, QK/OV circuits, induction heads, activation patching, superposition, SAEs |
| EL-08 | [GPU Programming](electives/08-gpu-programming.md) | 60 min | Warps, coalescing, tiled matmul, online softmax, FlashAttention implemented in NumPy |
| EL-09 | [Audio & Speech](electives/09-audio-and-speech.md) | 60 min | Aliasing, the STFT, mel, CTC with its forward DP, wav2vec 2.0, Whisper, WER |
| EL-10 | [Deep Learning for Tabular Data](electives/10-tabular-deep-learning.md) | 45 min | Why trees win (rotation and noise experiments), entity embeddings, TabPFN, fair benchmarks |

---

## Running the code

Every code cell runs on a laptop CPU in seconds. You need:

```
pip install numpy scipy scikit-learn pandas torch
```

The cells within one note share state, so run them top to bottom (paste them into a notebook, or `python -i`).
The [check script](../scripts/check_notes.py) runs every cell of every note: `python3 scripts/check_notes.py notes/`.
