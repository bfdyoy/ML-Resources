# ML-Resources

**A curated curriculum for learning machine learning in the easiest way that still makes sense.**

This repo doesn't host content. It picks the best free resources on the internet (book chapters with worked
examples, interactive visual essays, runnable notebooks, and videos where they're genuinely the best
explanation) and puts them in order as **lessons** and **learning paths**.

It's built for an **intermediate learner**: you know Python, you've trained a model or two, and you want real
understanding, not another "What is ML?" intro.

---

## How every lesson works

Each lesson is **5–12 hours** (most are 6–9) and follows the same four-beat rhythm:

1. **Intuition**: a visual essay, interactive demo, or short explainer, so you *see* the idea first (≤ 30 min).
2. **Read**: one primary **book chapter or written explainer**, with exact sections and a time estimate.
3. **Build**: a notebook, code-along, or exercise, because you learn ML by doing it.
4. **Check**: self-test questions (including a "debug this" scenario) and a small **mini-project** with a named dataset.

Videos are complements, not the only way in. Every step names **one** primary resource, so you're never
staring at a list of 12 links wondering where to start.

> **Intermediate fast-track:** Before each lesson, try its *Check your understanding* questions. If you can answer
> most of them confidently, do only the mini-project and move on.

---

## Learning paths

| Path | For | Lessons | Time |
|---|---|---|---|
| [1. Core ML Practitioner](paths/01-core-ml-practitioner.md) | Solid classical ML: workflow, metrics, validation, trees, features, interpretability | 8 | ~48 h |
| [2. Deep Learning Foundations](paths/02-deep-learning.md) | Backprop from scratch → PyTorch → training craft → CNNs → transformers | 6 | ~47 h |
| [3. LLMs & Generative AI](paths/03-llms-genai.md) | How LLMs are built, fine-tuning/LoRA/RAG, evals, diffusion | 4 | ~31 h |
| [4. ML in Production](paths/04-ml-in-production.md) | System design, MLOps, deployment, monitoring | 2 | ~18 h |
| [Electives](paths/05-electives.md) | Time series · Bayesian ML · Reinforcement learning | 3 | 8–12 h each |

### Recommended route

```
          ┌───────────────────────────────────────────┐
          │ Path 1: Core ML  (fast-track what you know)│
          └───────────────┬───────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
   Path 2: Deep Learning        Path 4: ML in Production
            │                   (can run in parallel)
            ▼
   Path 3: LLMs & GenAI
            │
            ▼
   Electives as needed ── Time series · Bayesian · RL
```

At **~7–8 hours/week**, the full route (Paths 1–4) takes about **5–6 months**, including capstones. Math is taught
**just in time**: lessons link to the specific block of [linear algebra](lessons/math/01-linear-algebra.md),
[calculus & optimization](lessons/math/02-calculus-optimization.md), or
[probability & statistics](lessons/math/03-probability-statistics.md) that you need, when you need it.

Track your progress in [PROGRESS.md](PROGRESS.md).

---

## Lesson index

### Core ML
| ID | Lesson | Primary resources |
|---|---|---|
| CORE-01 | [The ML Workflow, End to End](lessons/core-ml/01-ml-workflow-end-to-end.md) | Google Problem Framing · Géron Ch. 2 · ISLP Ch. 2 · Inria MOOC |
| CORE-02 | [Linear Models & Gradient Descent](lessons/core-ml/02-linear-models-gradient-descent.md) | MLU-Explain · ISLP Ch. 3 · Géron Ch. 4 |
| CORE-03 | [Classification & Evaluation Metrics](lessons/core-ml/03-classification-and-metrics.md) | MLU-Explain ×3 · ISLP Ch. 4 · Géron Ch. 3 |
| CORE-04 | [Generalization: Bias-Variance, Validation & Regularization](lessons/core-ml/04-generalization-validation-regularization.md) | MLU-Explain ×3 · ISLP Ch. 5–6 · Inria MOOC |
| CORE-05 | [Trees, Random Forests & Gradient Boosting](lessons/core-ml/05-trees-and-ensembles.md) | MLU-Explain ×2 · ISLP Ch. 8 · Géron Ch. 5–6 · StatQuest |
| CORE-06 | [Unsupervised Learning](lessons/core-ml/06-unsupervised-learning.md) | Setosa PCA · ISLP Ch. 12 · Géron Ch. 7–8 · Distill t-SNE · PAIR UMAP |
| CORE-07 | [Feature Engineering, Pipelines & Leakage](lessons/core-ml/07-feature-engineering-pipelines-leakage.md) | sklearn pitfalls · Kaggle Learn ×2 · Kuhn & Johnson |
| CORE-08 | [Interpreting Models & Responsible ML](lessons/core-ml/08-interpretability-and-responsible-ml.md) | Molnar's *Interpretable ML* · Google Fairness module |

### Deep Learning
| ID | Lesson | Primary resources |
|---|---|---|
| DL-01 | [Neural Networks from Scratch & Backprop](lessons/deep-learning/01-neural-networks-from-scratch.md) | 3Blue1Brown · Nielsen Ch. 2 · UDL Ch. 3–5 · Karpathy micrograd |
| DL-02 | [PyTorch Fluency](lessons/deep-learning/02-pytorch-fluency.md) | learnpytorch.io · D2L Builders' Guide |
| DL-03 | [Training Deep Networks Well](lessons/deep-learning/03-training-deep-networks.md) | UDL Ch. 6, 7, 9 · Distill Momentum · Karpathy's Recipe · CS231n |
| DL-04 | [Convolutional Networks & Computer Vision](lessons/deep-learning/04-cnns-computer-vision.md) | CNN Explainer · UDL Ch. 10–11 · CS231n · learnpytorch.io |
| DL-05 | [Embeddings, Language Modeling & Sequences](lessons/deep-learning/05-embeddings-sequences-attention.md) | Illustrated Word2vec · Jurafsky & Martin · Karpathy makemore |
| DL-06 | [Transformers](lessons/deep-learning/06-transformers.md) | 3Blue1Brown · Illustrated Transformer · Transformer Explainer · UDL Ch. 12 · Raschka · Karpathy GPT |

### LLMs & Generative AI
| ID | Lesson | Primary resources |
|---|---|---|
| GEN-01 | [How LLMs Are Built](lessons/llms-genai/01-how-llms-are-built.md) | Karpathy talks · Raschka Ch. 2, 5 · Illustrated DeepSeek-R1 |
| GEN-02 | [Adapting LLMs: Prompting, Fine-tuning, LoRA & RAG](lessons/llms-genai/02-adapting-llms-finetuning-rag.md) | HF LLM Course · Raschka Ch. 6–7, App. E · Eugene Yan |
| GEN-03 | [Evaluating & Shipping LLM Applications](lessons/llms-genai/03-evaluating-llm-apps.md) | Hamel Husain · Eugene Yan |
| GEN-04 | [Generative Models: VAEs, GANs & Diffusion](lessons/llms-genai/04-generative-models-diffusion.md) | UDL Ch. 14–18 · Illustrated Stable Diffusion · HF Diffusion course |

### Production
| ID | Lesson | Primary resources |
|---|---|---|
| PROD-01 | [ML System Design](lessons/production/01-ml-system-design.md) | Rules of ML · Chip Huyen's DMLS · FSDL |
| PROD-02 | [MLOps in Practice](lessons/production/02-mlops-in-practice.md) | Made With ML · MLOps Zoomcamp |

### Electives
| ID | Lesson | Primary resources |
|---|---|---|
| EL-01 | [Time Series Forecasting](lessons/electives/01-time-series-forecasting.md) | *Forecasting: Principles and Practice* (Python edition) |
| EL-02 | [Bayesian & Probabilistic ML](lessons/electives/02-bayesian-probabilistic-ml.md) | Bayesian Methods for Hackers · Statistical Rethinking |
| EL-03 | [Reinforcement Learning](lessons/electives/03-reinforcement-learning.md) | Sutton & Barto · HF Deep RL · Spinning Up · UDL Ch. 19 |

### Math (just in time)
| ID | Lesson | Primary resources |
|---|---|---|
| MATH-01 | [Linear Algebra](lessons/math/01-linear-algebra.md) | 3B1B Essence of LA · MML Ch. 2–4 |
| MATH-02 | [Calculus & Optimization](lessons/math/02-calculus-optimization.md) | 3B1B Essence of Calculus · MML Ch. 5, 7 |
| MATH-03 | [Probability & Statistics](lessons/math/03-probability-statistics.md) | Seeing Theory · MML Ch. 6, 8 |

---

## The core bookshelf

If you only bookmark a handful of things, bookmark these. All are free to read online.

| Book | Why | Used in |
|---|---|---|
| [*An Introduction to Statistical Learning* (Python ed.)](https://www.statlearning.com/) | The clearest classical-ML text: intuition first, with labs | Core ML |
| [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (Prince) | The best modern DL textbook for self-study, with a notebook per chapter | Deep Learning, GenAI |
| [*Mathematics for Machine Learning*](https://mml-book.github.io/) | Exactly the math you need, nothing more | Math |
| [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/) (Nielsen) | The gentlest written derivation of backprop | Deep Learning |
| [*Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) (Molnar) | Readable guide to SHAP, PDPs, and friends | Core ML |
| [Géron's notebooks](https://github.com/ageron/handson-mlp) & [Raschka's LLM code](https://github.com/rasbt/LLMs-from-scratch) | Free code for the two best practitioner books (the books themselves are paid) | Everywhere |

Full list with types, levels, costs, and verification dates: **[resources/catalog.md](resources/catalog.md)**.

---

## Repository structure

```
README.md                  you are here
PROGRESS.md                personal checklist
paths/                     learning paths (ordered lessons + capstones)
lessons/                   one file per lesson, grouped by track
resources/catalog.md       every resource, typed/tagged/verified (source of truth)
templates/                 lesson template
scripts/check_links.py     link checker (stdlib only), also run weekly by GitHub Actions
CLAUDE.md                  project instructions for Claude Code
.claude/rules/             quality bar, lesson format, link policy
.claude/skills/            research-resources · build-lesson · check-links · review-path
.claude/agents/            resource-scout · curriculum-architect · link-auditor · pedagogy-reviewer
```

## Maintaining and extending (with Claude Code)

This repo is set up so Claude Code can keep it growing and up to date:

- *"Research resources for **graph neural networks**"* → `research-resources` skill / `resource-scout` agent
- *"Write a lesson **DL-07 Graph Neural Networks**"* → `build-lesson` skill
- *"Check all links"* → `check-links` skill / `link-auditor` agent (also runs weekly in CI)
- *"Review Path 2 for gaps"* → `review-path` skill / `pedagogy-reviewer` agent

Rules everything follows: free first, verified links only, chapter-level precision, and no pirated mirrors. See [`.claude/rules/`](.claude/rules/).
