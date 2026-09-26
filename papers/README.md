# Papers Library

About **240 papers**, grouped by topic and ordered so each one builds on the ones before it. Every arXiv link was
checked against the exact ID and title (see [`resources/verified-urls.tsv`](../resources/verified-urls.tsv)).

> **You don't need to read all of these.** Papers are the *third* layer of the curriculum. First the lesson's book chapter,
> then the [toolbox](../toolbox/README.md) explainer, and then the paper, once you want the original argument or the details.

## How to read a paper (read this first)

1. Read [S. Keshav, *How to Read a Paper*](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf) (3 pages). Use its **three-pass method**:
   - **Pass 1 (10 min):** title, abstract, intro, section headings, conclusion. Answer: *what problem, what's the claim, is it relevant?*
   - **Pass 2 (1 h):** the figures, the method, the main results. Skip the proofs. Write 5 bullet notes.
   - **Pass 3 (3–5 h, only for key papers):** re-derive or re-implement it. This is where you actually learn.
2. **Find an explainer first.** For most famous papers a blog post (Alammar, Lilian Weng, Distill, the HF blog) explains the
   idea better than the paper does. The [toolbox](../toolbox/README.md) links these next to each topic.
3. **Find the code.** Look at the [annotated paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
   (labml.ai), the Hugging Face `transformers` docs, or the official repo. Reading code alongside the paper roughly halves the time it takes.
4. Keep a one-line note per paper in [PROGRESS.md](../PROGRESS.md): *"Paper X: showed Y by doing Z. Surprising: W."*

## Legend
- ⭐ **Must-read.** Read at least passes 1–2.
- **Level:** L1 accessible · L2 needs the matching lesson · L3 research-level
- **After:** the lesson or toolbox page that gives you enough background first

## The 30 must-reads (a minimal path through the field)

If you only read 30 papers, read these, in this order.

| # | Paper | Year | Why |
|---|---|---|---|
| 1 | [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) | 2016 | Why GBMs win on tabular data |
| 2 | [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | 2022 | Know when *not* to use deep learning |
| 3 | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | The default optimizer |
| 4 | [Batch Normalization](https://arxiv.org/abs/1502.03167) | 2015 | Normalization, and the "why does this train?" question |
| 5 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2015 | Residual connections are everywhere now |
| 6 | [Deep Double Descent](https://arxiv.org/abs/1912.02292) | 2019 | Modern generalization |
| 7 | [Efficient Estimation of Word Representations (word2vec)](https://arxiv.org/abs/1301.3781) | 2013 | Embeddings |
| 8 | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | 2014 | The birth of attention |
| 9 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The Transformer |
| 10 | [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Pretrain → fine-tune |
| 11 | [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | In-context learning at scale |
| 12 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Predictable scaling |
| 13 | [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) | 2022 | How to spend compute |
| 14 | [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 2020 | Transformers for vision |
| 15 | [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | Multimodal contrastive learning |
| 16 | [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | RLHF, the base-model-to-assistant recipe |
| 17 | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | RLHF without RL |
| 18 | [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 2022 | Reasoning via intermediate steps |
| 19 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | RL with verifiable rewards → reasoning models |
| 20 | [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Cheap fine-tuning |
| 21 | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 2020 | Grounding LLMs in documents |
| 22 | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 2023 | How LLMs actually use long context |
| 23 | [ReAct](https://arxiv.org/abs/2210.03629) | 2022 | Reasoning + acting = agents |
| 24 | [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | IO-aware thinking about GPUs |
| 25 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 2023 | How LLMs are served |
| 26 | [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) | 2013 | Latent-variable generative models |
| 27 | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | Diffusion |
| 28 | [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) | 2017 | The workhorse RL algorithm (also in RLHF) |
| 29 | [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874) | 2017 | Explaining models |
| 30 | [Hidden Technical Debt in Machine Learning Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) | 2015 | Why ML in production is hard |

## Full library by topic

| File | Topics | Papers |
|---|---|---|
| [01 Optimization, training & generalization](01-optimization-training-generalization.md) | optimizers, LR schedules, init, normalization, regularization, generalization theory | 29 |
| [02 Computer vision](02-computer-vision.md) | CNNs, detection, segmentation, ViT, self-supervised, multimodal, 3D | 29 |
| [03 NLP, transformers & LLMs](03-nlp-transformers-llms.md) | embeddings, seq2seq, transformers, pretraining, scaling, architectures, data | 38 |
| [04 Post-training, reasoning & agents](04-alignment-reasoning-agents.md) | instruction tuning, RLHF/DPO, reasoning, prompting, agents, evaluation | 34 |
| [05 Retrieval & RAG](05-retrieval-rag.md) | dense/sparse retrieval, embeddings, vector search, RAG variants | 21 |
| [06 Efficiency & systems](06-efficiency-systems.md) | PEFT, quantization, distillation, attention kernels, serving, parallelism | 24 |
| [07 Generative models](07-generative-models.md) | VAEs, GANs, flows, diffusion, flow matching | 20 |
| [08 Reinforcement learning](08-reinforcement-learning.md) | value-based, policy gradient, model-based, games | 14 |
| [09 Tabular, time series, recsys & causal](09-tabular-timeseries-recsys-causal.md) | boosting, imbalanced data, HPO, forecasting, recommenders, causal ML | 22 |
| [10 Interpretability, uncertainty & responsible ML](10-interpretability-uncertainty-responsible.md) | SHAP/LIME, saliency, mech interp, calibration, conformal, documentation | 11 |
| [11 ML in production](11-ml-in-production.md) | tech debt, testing, operationalization | 5 |

## Where to find more
- [labml.ai annotated paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations): 60+ papers implemented in PyTorch, with side-by-side notes
- [dair-ai/ML-Papers-Explained](https://github.com/dair-ai/ML-Papers-Explained) and [ML-Papers-of-the-Week](https://github.com/dair-ai/ML-Papers-of-the-Week): short summaries, a good way to keep up
- [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM): an LLM paper index
- [eugeneyan/applied-ml](https://github.com/eugeneyan/applied-ml): industry papers and blog posts on ML in production
