# ML-Resources

**A curated curriculum for learning machine learning in the easiest way that still makes sense.**

It picks the best free resources on the internet (book chapters with worked examples, interactive visual
essays, runnable notebooks, and videos where they're genuinely the best explanation) and puts them in order as
**lessons** and **learning paths**. Every lesson also has its own **study notes**: a written explanation of the
ideas and the math, step by step, with worked examples and runnable code, so the route from one topic to the next is smooth.

It's built for an **intermediate learner**: you know Python, you've trained a model or two, and you want real
understanding, not another "What is ML?" intro.

## Five layers

| Layer | What it is | Size |
|---|---|---|
| 🧭 **[Paths](#learning-paths) → [Lessons](#lesson-index)** | The guided route. What to study next, in what order, and how to check yourself. | 5 paths + electives · 53 lessons |
| 📘 **[Study notes](notes/README.md)** | The explanations, written for this repo: intuition, the math derived step by step, worked numbers, runnable code, pitfalls, cheat sheets, and answers to every self-check question. One per lesson, each bridging to the next. | 53 notes (~51 h) + [notation guide](notes/notation.md) |
| 🧰 **[Toolbox](toolbox/README.md)** | The reference shelf. Every concept in ML/DL/LLMs/MLOps, each mapped to *intuition → deeper reading → practice → paper*. | 12 domains · 29 free books |
| 📄 **[Papers](papers/README.md)** | Primary sources with verified arXiv links, a 30-paper must-read list, and a "read after lesson X" for each. | ~250 papers |
| 🏋️ **[Exercises](exercises/README.md)** | Practice at every scale: drills, puzzles, a 45-rung from-scratch ladder, university assignments, projects, interview prep. | 5 practice banks |

Use the lessons as your spine. Each one starts with its study notes (step 0) and ends with links into the toolbox, the papers, and the exercises for its topic.
Every one of the 600+ links in this repo has a verification record ([`resources/verified-urls.tsv`](resources/verified-urls.tsv)).

---

## How every lesson works

Each lesson is **6–13 hours** (most are 7–10) and follows the same rhythm:

0. **Primer**: the lesson's [study notes](notes/README.md) (~1 h). The ideas and the math step by step, with worked examples and code. Read them first, and check your answers against them at the end.
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
| [1. Core ML Practitioner](paths/01-core-ml-practitioner.md) | Classical ML done properly: workflow, metrics, validation, trees, features, interpretability, kernels, data quality, uncertainty, tuning | 12 | ~84 h |
| [2. Deep Learning Foundations](paths/02-deep-learning.md) | Backprop from scratch → PyTorch → training craft → CNNs → transformers → performance, modern architectures, GNNs | 9 | ~76 h |
| [3. LLMs & Generative AI](paths/03-llms-genai.md) | LLM apps (fine-tuning, RAG, retrieval, agents, evals, security), post-training & inference, diffusion & flow matching | 10 | ~91 h |
| [4. ML in Production](paths/04-ml-in-production.md) | System design, MLOps, testing & drift, distributed training, LLMOps | 5 | ~46 h |
| [5. Computer Vision](paths/05-computer-vision.md) | Detection & segmentation, ViTs & self-supervised learning, CLIP & VLMs, 3D | 4 | ~34 h |
| [Electives](paths/06-electives.md) | Time series · Bayesian · RL · RecSys · Causal · Anomaly detection · Mech interp · GPU programming · Audio · Tabular DL | 10 | 6–13 h each |

### Recommended route

```
          ┌───────────────────────────────────────────┐
          │ Path 1: Core ML  (fast-track what you know)│
          └───────────────┬───────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
   Path 2: Deep Learning        Path 4: ML in Production
            │                   (runs in parallel; PROD-04/05
     ┌──────┴───────┐            wait for DL-07 / GEN-03)
     ▼              ▼
 Path 3: LLMs    Path 5: Computer Vision
 & GenAI
     │
     ▼
 Electives as needed ── Time series · Bayesian · RL · RecSys · Causal · Anomaly · Mech interp · GPU · Audio · Tabular DL
```

At **~7–8 hours/week**, the **essential route** takes about **6–7 months**, including capstones: CORE-01…08, DL-01…06, Path 3 Part A, and
PROD-01…03. Everything (Paths 1–5) is roughly a **year-long programme**. The extended lessons are clearly marked in each path, so you can
take them when you need them. Math is taught
**just in time**: lessons link to the specific block of [linear algebra](lessons/math/01-linear-algebra.md),
[calculus & optimization](lessons/math/02-calculus-optimization.md), or
[probability & statistics](lessons/math/03-probability-statistics.md) that you need, when you need it.

Track your progress in [PROGRESS.md](PROGRESS.md).

---

## Lesson index

### Core ML
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| CORE-01 | [The ML Workflow, End to End](lessons/core-ml/01-ml-workflow-end-to-end.md) | [📘](notes/core-ml/01-ml-workflow-end-to-end.md) | Google Problem Framing · Géron Ch. 2 · ISLP Ch. 2 · Inria MOOC |
| CORE-02 | [Linear Models & Gradient Descent](lessons/core-ml/02-linear-models-gradient-descent.md) | [📘](notes/core-ml/02-linear-models-gradient-descent.md) | MLU-Explain · ISLP Ch. 3 · Géron Ch. 4 |
| CORE-03 | [Classification & Evaluation Metrics](lessons/core-ml/03-classification-and-metrics.md) | [📘](notes/core-ml/03-classification-and-metrics.md) | MLU-Explain ×3 · ISLP Ch. 4 · Géron Ch. 3 |
| CORE-04 | [Generalization: Bias-Variance, Validation & Regularization](lessons/core-ml/04-generalization-validation-regularization.md) | [📘](notes/core-ml/04-generalization-validation-regularization.md) | MLU-Explain ×3 · ISLP Ch. 5–6 · Inria MOOC |
| CORE-05 | [Trees, Random Forests & Gradient Boosting](lessons/core-ml/05-trees-and-ensembles.md) | [📘](notes/core-ml/05-trees-and-ensembles.md) | MLU-Explain ×2 · ISLP Ch. 8 · Géron Ch. 5–6 · StatQuest |
| CORE-06 | [Unsupervised Learning](lessons/core-ml/06-unsupervised-learning.md) | [📘](notes/core-ml/06-unsupervised-learning.md) | Setosa PCA · ISLP Ch. 12 · Géron Ch. 7–8 · Distill t-SNE · PAIR UMAP |
| CORE-07 | [Feature Engineering, Pipelines & Leakage](lessons/core-ml/07-feature-engineering-pipelines-leakage.md) | [📘](notes/core-ml/07-feature-engineering-pipelines-leakage.md) | sklearn pitfalls · Kaggle Learn ×2 · Kuhn & Johnson |
| CORE-08 | [Interpreting Models & Responsible ML](lessons/core-ml/08-interpretability-and-responsible-ml.md) | [📘](notes/core-ml/08-interpretability-and-responsible-ml.md) | Molnar's *Interpretable ML* · Google Fairness module |
| CORE-09 | [Kernel Methods, SVMs, Nearest Neighbours & GPs](lessons/core-ml/09-kernels-svms-nearest-neighbours.md) | [📘](notes/core-ml/09-kernels-svms-nearest-neighbours.md) | ISLP Ch. 9 · CS229 notes · Distill GP explorer · sklearn |
| CORE-10 | [Data-Centric ML: Label Quality, Imbalance & Few Labels](lessons/core-ml/10-data-centric-ml.md) | [📘](notes/core-ml/10-data-centric-ml.md) | MIT DCAI + labs · cleanlab · Lilian Weng (human data, semi-supervised, active learning) |
| CORE-11 | [Uncertainty: Calibration & Conformal Prediction](lessons/core-ml/11-uncertainty-calibration-conformal.md) | [📘](notes/core-ml/11-uncertainty-calibration-conformal.md) | sklearn calibration · Guo et al. · Angelopoulos & Bates · MAPIE |
| CORE-12 | [Hyperparameter Optimization](lessons/core-ml/12-hyperparameter-optimization.md) | [📘](notes/core-ml/12-hyperparameter-optimization.md) | Inria MOOC · sklearn search · Bayesian optimization · Optuna · DL Tuning Playbook |

### Deep Learning
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| DL-01 | [Neural Networks from Scratch & Backprop](lessons/deep-learning/01-neural-networks-from-scratch.md) | [📘](notes/deep-learning/01-neural-networks-from-scratch.md) | 3Blue1Brown · Nielsen Ch. 2 · UDL Ch. 3–5 · Karpathy micrograd |
| DL-02 | [PyTorch Fluency](lessons/deep-learning/02-pytorch-fluency.md) | [📘](notes/deep-learning/02-pytorch-fluency.md) | learnpytorch.io · D2L Builders' Guide |
| DL-03 | [Training Deep Networks Well](lessons/deep-learning/03-training-deep-networks.md) | [📘](notes/deep-learning/03-training-deep-networks.md) | UDL Ch. 6, 7, 9 · Distill Momentum · Karpathy's Recipe · CS231n |
| DL-04 | [Convolutional Networks & Computer Vision](lessons/deep-learning/04-cnns-computer-vision.md) | [📘](notes/deep-learning/04-cnns-computer-vision.md) | CNN Explainer · UDL Ch. 10–11 · CS231n · learnpytorch.io |
| DL-05 | [Embeddings, Language Modeling & Sequences](lessons/deep-learning/05-embeddings-sequences-attention.md) | [📘](notes/deep-learning/05-embeddings-sequences-attention.md) | Illustrated Word2vec · Jurafsky & Martin · Karpathy makemore |
| DL-06 | [Transformers](lessons/deep-learning/06-transformers.md) | [📘](notes/deep-learning/06-transformers.md) | 3Blue1Brown · Illustrated Transformer · Transformer Explainer · UDL Ch. 12 · Raschka · Karpathy GPT |
| DL-07 | [Making Training Fast: GPUs, Mixed Precision & Profiling](lessons/deep-learning/07-performance-gpus-mixed-precision.md) | [📘](notes/deep-learning/07-performance-gpus-mixed-precision.md) | Horace He · PyTorch tuning guide, torch.compile, profiler · How to Scale Your Model |
| DL-08 | [Modern Architectures: RoPE, GQA, MoE & SSMs](lessons/deep-learning/08-modern-architectures-moe-ssm.md) | [📘](notes/deep-learning/08-modern-architectures-moe-ssm.md) | Raschka's architecture comparison · HF (positional encoding, MoE) · Lilian Weng · Mamba |
| DL-09 | [Graph Neural Networks](lessons/deep-learning/09-graph-neural-networks.md) | [📘](notes/deep-learning/09-graph-neural-networks.md) | Distill ×2 · Hamilton's GRL book · UvA tutorial · PyG |

### Computer Vision
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| CV-01 | [Object Detection & Segmentation](lessons/vision/01-object-detection-segmentation.md) | [📘](notes/vision/01-object-detection-segmentation.md) | Lilian Weng's detection series · D2L CV chapter · U-Net · Ultralytics/Detectron2 |
| CV-02 | [Vision Transformers & Self-Supervised Learning](lessons/vision/02-vision-transformers-self-supervised.md) | [📘](notes/vision/02-vision-transformers-self-supervised.md) | ViT · UvA tutorials · Lilian Weng (contrastive) · SSL Cookbook |
| CV-03 | [Multimodal: CLIP & Vision-Language Models](lessons/vision/03-clip-vision-language-models.md) | [📘](notes/vision/03-clip-vision-language-models.md) | CLIP · HF VLM posts ×2 · Lilian Weng (VLMs) |
| CV-04 | [3D Vision & Neural Rendering](lessons/vision/04-3d-vision-neural-rendering.md) | [📘](notes/vision/04-3d-vision-neural-rendering.md) | Szeliski Ch. 2, 14 · NeRF · 3D Gaussian Splatting |

### LLMs & Generative AI
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| GEN-01 | [How LLMs Are Built](lessons/llms-genai/01-how-llms-are-built.md) | [📘](notes/llms-genai/01-how-llms-are-built.md) | Karpathy talks · Raschka Ch. 2, 5 · Illustrated DeepSeek-R1 |
| GEN-02 | [Adapting LLMs: Prompting, Fine-tuning, LoRA & RAG](lessons/llms-genai/02-adapting-llms-finetuning-rag.md) | [📘](notes/llms-genai/02-adapting-llms-finetuning-rag.md) | HF LLM Course · Raschka Ch. 6–7, App. E · Eugene Yan |
| GEN-03 | [Evaluating & Shipping LLM Applications](lessons/llms-genai/03-evaluating-llm-apps.md) | [📘](notes/llms-genai/03-evaluating-llm-apps.md) | Hamel Husain · Eugene Yan |
| GEN-04 | [Generative Models: VAEs, GANs & Diffusion](lessons/llms-genai/04-generative-models-diffusion.md) | [📘](notes/llms-genai/04-generative-models-diffusion.md) | UDL Ch. 14–18 · Illustrated Stable Diffusion · HF Diffusion course |
| GEN-05 | [Retrieval Engineering for RAG](lessons/llms-genai/05-retrieval-engineering.md) | [📘](notes/llms-genai/05-retrieval-engineering.md) | IR book Ch. 6, 8, 11 · Sentence Transformers · Faiss · RAG_Techniques |
| GEN-06 | [Agents: Tool Use, Structured Outputs & MCP](lessons/llms-genai/06-agents-tool-use.md) | [📘](notes/llms-genai/06-agents-tool-use.md) | Anthropic "Building Effective Agents" · Lilian Weng · HF Agents Course · MCP |
| GEN-07 | [Post-Training: SFT, RLHF, DPO & Reasoning](lessons/llms-genai/07-post-training-alignment-reasoning.md) | [📘](notes/llms-genai/07-post-training-alignment-reasoning.md) | RLHF Book · HF (RLHF, DPO, RLOO) · Raschka · TRL |
| GEN-08 | [Efficient LLM Inference](lessons/llms-genai/08-efficient-llm-inference.md) | [📘](notes/llms-genai/08-efficient-llm-inference.md) | HF KV cache · Lilian Weng · Visual Guide to Quantization · vLLM |
| GEN-09 | [LLM Security & Safety](lessons/llms-genai/09-llm-security-safety.md) | [📘](notes/llms-genai/09-llm-security-safety.md) | Simon Willison · OWASP LLM Top 10 · Lilian Weng (attacks, reward hacking) |
| GEN-10 | [Advanced Diffusion & Flow Matching](lessons/llms-genai/10-advanced-diffusion-flow-matching.md) | [📘](notes/llms-genai/10-advanced-diffusion-flow-matching.md) | Step-by-Step Diffusion · Yang Song · MIT 6.S184 · HF Diffusion course |

### Production
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| PROD-01 | [ML System Design](lessons/production/01-ml-system-design.md) | [📘](notes/production/01-ml-system-design.md) | Rules of ML · Chip Huyen's DMLS · FSDL |
| PROD-02 | [MLOps in Practice](lessons/production/02-mlops-in-practice.md) | [📘](notes/production/02-mlops-in-practice.md) | Made With ML · MLOps Zoomcamp |
| PROD-03 | [Testing, Monitoring & Drift](lessons/production/03-testing-monitoring-drift.md) | [📘](notes/production/03-testing-monitoring-drift.md) | ML Test Score · DMLS Ch. 8–9 · Evidently course |
| PROD-04 | [Distributed Training at Scale](lessons/production/04-distributed-training.md) | [📘](notes/production/04-distributed-training.md) | Ultra-Scale Playbook · Lilian Weng · PyTorch DDP/FSDP · LLM-Training-Puzzles |
| PROD-05 | [LLMOps](lessons/production/05-llmops-genai-platforms.md) | [📘](notes/production/05-llmops-genai-platforms.md) | Chip Huyen · applied-llms.org · Evidently LLM course · LLM Zoomcamp |

### Electives
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| EL-01 | [Time Series Forecasting](lessons/electives/01-time-series-forecasting.md) | [📘](notes/electives/01-time-series-forecasting.md) | *Forecasting: Principles and Practice* (Python edition) |
| EL-02 | [Bayesian & Probabilistic ML](lessons/electives/02-bayesian-probabilistic-ml.md) | [📘](notes/electives/02-bayesian-probabilistic-ml.md) | Bayesian Methods for Hackers · Statistical Rethinking |
| EL-03 | [Reinforcement Learning](lessons/electives/03-reinforcement-learning.md) | [📘](notes/electives/03-reinforcement-learning.md) | Sutton & Barto · HF Deep RL · Spinning Up · UDL Ch. 19 |
| EL-04 | [Recommender Systems](lessons/electives/04-recommender-systems.md) | [📘](notes/electives/04-recommender-systems.md) | Google RecSys course · fastbook Ch. 8 · Microsoft Recommenders · SASRec |
| EL-05 | [Causal Inference & Uplift](lessons/electives/05-causal-inference-uplift.md) | [📘](notes/electives/05-causal-inference-uplift.md) | Causal Inference for the Brave and True · Brady Neal · EconML |
| EL-06 | [Anomaly & Outlier Detection](lessons/electives/06-anomaly-detection.md) | [📘](notes/electives/06-anomaly-detection.md) | sklearn outlier detection · MIT DCAI · PyOD |
| EL-07 | [Mechanistic Interpretability](lessons/electives/07-mechanistic-interpretability.md) | [📘](notes/electives/07-mechanistic-interpretability.md) | Transformer Circuits · ARENA · TransformerLens · Toy Models of Superposition |
| EL-08 | [GPU Programming for ML](lessons/electives/08-gpu-programming.md) | [📘](notes/electives/08-gpu-programming.md) | GPU-Puzzles · GPU MODE · FlashAttention · LeetGPU/Tensara |
| EL-09 | [Audio & Speech](lessons/electives/09-audio-and-speech.md) | [📘](notes/electives/09-audio-and-speech.md) | HF Audio Course · Whisper |
| EL-10 | [Deep Learning for Tabular Data](lessons/electives/10-tabular-deep-learning.md) | [📘](notes/electives/10-tabular-deep-learning.md) | Grinsztajn et al. · fastbook Ch. 9 · TabPFN |

### Math (just in time)
| ID | Lesson | Notes | Primary resources |
|---|---|---|---|
| MATH-01 | [Linear Algebra](lessons/math/01-linear-algebra.md) | [📘](notes/math/01-linear-algebra.md) | 3B1B Essence of LA · MML Ch. 2–4 |
| MATH-02 | [Calculus & Optimization](lessons/math/02-calculus-optimization.md) | [📘](notes/math/02-calculus-optimization.md) | 3B1B Essence of Calculus · MML Ch. 5, 7 |
| MATH-03 | [Probability & Statistics](lessons/math/03-probability-statistics.md) | [📘](notes/math/03-probability-statistics.md) | Seeing Theory · MML Ch. 6, 8 · Think Stats · ISLP Ch. 13 |

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
notes/                     study notes: one written explanation per lesson (math, worked examples, code, answers)
toolbox/                   12-domain reference shelf + free bookshelf
papers/                    ~250 verified papers by topic + must-read list + how to read papers
exercises/                 drills, from-scratch ladder, assignments, projects, interview prep
resources/catalog.md       resources used in lessons, typed and tagged
resources/verified-urls.tsv  how each of the 600+ URLs was verified
templates/                 lesson and study-notes templates
scripts/check_links.py     live link checker (also runs weekly in GitHub Actions)
scripts/audit_urls.py      fails if any URL lacks a verification record
scripts/check_notes.py     runs every code cell in the notes and lints their math
CLAUDE.md                  project instructions for Claude Code
.claude/rules/             quality bar, lesson format, link policy
.claude/skills/            research-resources · build-lesson · write-notes · expand-toolbox · check-links · review-path
.claude/agents/            resource-scout · curriculum-architect · link-auditor · pedagogy-reviewer
```

## Maintaining and extending (with Claude Code)

This repo is set up so Claude Code can keep it growing and up to date:

- *"Research resources for **graph neural networks**"* → `research-resources` skill / `resource-scout` agent
- *"Write a lesson **DL-07 Graph Neural Networks**"* → `build-lesson` skill
- *"Write or update the **study notes** for DL-08"* → `write-notes` skill
- *"Add papers and resources on **model merging** to the toolbox"* → `expand-toolbox` skill
- *"Check all links"* → `check-links` skill / `link-auditor` agent (also runs weekly in CI)
- *"Review Path 2 for gaps"* → `review-path` skill / `pedagogy-reviewer` agent

Rules everything follows: free first, verified links only, chapter-level precision, and no pirated mirrors. See [`.claude/rules/`](.claude/rules/).
