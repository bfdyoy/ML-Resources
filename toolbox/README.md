# The Toolbox

The **lessons** give you a guided path. The **toolbox** is the reference shelf you come back to: every concept in ML, deep
learning, LLMs, and MLOps, each with the best way to understand it and to practice it.

For a written, step-by-step explanation of a concept that a lesson teaches (with the math and runnable code), go to that lesson's
**[study notes](../notes/README.md)**. The toolbox points outward; the notes explain in place.

**How each entry is organized:**

| Column | What it gives you |
|---|---|
| **Concept** | The thing you want to understand |
| **Start here** | The fastest route to intuition: a visual essay, an illustrated blog post, or a short explainer (≤ 30 min) |
| **Go deeper** | A book chapter or long-form text with worked examples |
| **Practice** | Code you run or write yourself: a notebook, an exercise, a library tutorial |
| **Paper** | The original or definitive paper (see the [papers library](../papers/README.md)) |

Everything is free unless marked `[paid]`. Every link was verified (see [`resources/verified-urls.tsv`](../resources/verified-urls.tsv)).

## Domains

| # | Page | Covers |
|---|---|---|
| 01 | [Math & statistics](01-math-and-stats.md) | linear algebra, calculus, matrix calculus, probability, statistics, information theory, optimization |
| 02 | [Classical ML algorithms](02-classical-ml.md) | linear/logistic regression, GLMs, kNN, naive Bayes, SVMs & kernels, trees, bagging, boosting, clustering, mixtures, dimensionality reduction, Gaussian processes |
| 03 | [Data, features & evaluation](03-data-features-evaluation.md) | Python, NumPy & pandas foundations, EDA, cleaning, encoding, leakage, imbalance, validation schemes, metrics, calibration, conformal prediction, HPO, data-centric AI, fairness |
| 04 | [Deep learning fundamentals](04-deep-learning-fundamentals.md) | backprop, autodiff, losses, initialization, optimizers, schedules, normalization, regularization, debugging, mixed precision, PyTorch |
| 05 | [Architectures](05-architectures.md) | MLPs, CNNs, ResNets, RNN/LSTM, attention, transformers, positional encodings, MoE, state-space models, GNNs, autoencoders |
| 06 | [Computer vision](06-computer-vision.md) | augmentation, transfer learning, detection, segmentation, ViTs, self-supervised learning, CLIP & VLMs, 3D |
| 07 | [NLP & LLMs](07-nlp-and-llms.md) | tokenization, embeddings, language modeling, pretraining, scaling laws, decoding, fine-tuning, PEFT, RLHF/DPO, reasoning models, quantization, inference |
| 08 | [RAG, agents & LLM evaluation](08-rag-agents-evals.md) | retrieval fundamentals, embeddings, vector search, chunking, reranking, advanced RAG, prompting, structured output, tools, agents, MCP, evals, LLM security |
| 09 | [Generative models](09-generative-models.md) | VAEs, GANs, normalizing flows, diffusion, score matching, guidance, latent diffusion, flow matching |
| 10 | [Reinforcement learning](10-reinforcement-learning.md) | MDPs, bandits, DP, MC/TD, Q-learning, DQN, policy gradients, PPO, model-based RL, RLHF connection |
| 11 | [MLOps, systems & scale](11-mlops-and-systems.md) | system design, pipelines, experiment tracking, serving, monitoring & drift, testing, CI/CD, GPUs, profiling, distributed training, LLM serving |
| 12 | [Specialized topics](12-specialized-topics.md) | time series, recommender systems, causal inference, anomaly detection, tabular deep learning, audio & speech, interpretability (mechanistic), AI safety basics |

## The free bookshelf

Every book below is free to read online (legally, from the author or publisher). They're sorted roughly from most approachable to hardest.

| Book | Best for | Level |
|---|---|---|
| [An Introduction to Statistical Learning (Python)](https://www.statlearning.com/) | Classical ML, intuition first | L1–L2 |
| [Think Stats](https://greenteapress.com/wp/think-stats-3e/) / [Think Bayes](https://greenteapress.com/wp/think-bayes/) (Downey) | Stats and Bayes, taught through Python code | L1 |
| [Seeing Theory](https://seeing-theory.brown.edu/) | Visual probability and statistics | L1 |
| [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) (Nielsen) | Backprop, gently | L1–L2 |
| [Dive into Deep Learning](https://d2l.ai/) | DL with runnable code | L2 |
| [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Prince) | Modern DL, best figures | L2 |
| [Mathematics for Machine Learning](https://mml-book.github.io/) | The math, just enough | L2 |
| [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (Molnar) | Explaining models | L2 |
| [Feature Engineering and Selection](https://bookdown.org/max/FES) (Kuhn & Johnson) | Features | L2 |
| [Forecasting: Principles and Practice (Python)](https://otexts.com/fpppy/) | Time series | L1–L2 |
| [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) | Causal inference in Python | L2 |
| [Causal Inference: The Mixtape](https://mixtape.scunning.com/) | Causal inference, econometrics view | L2 |
| [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) (Brady Neal) | Causal inference, ML view | L2 |
| [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) | Search and retrieval (RAG fundamentals) | L2 |
| [Speech and Language Processing (3rd ed.)](https://web.stanford.edu/~jurafsky/slp3/) | NLP and LLMs | L2 |
| [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (Sutton & Barto) | RL | L2 |
| [RLHF Book](https://rlhfbook.com/) (Lambert) | Post-training LLMs | L2–L3 |
| [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) (Hamilton) | GNNs | L2–L3 |
| [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) (Szeliski) | Classical + modern CV | L2–L3 |
| [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow, Bengio, Courville) | Classic DL theory (2016) | L2–L3 |
| [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) (Bishop) | Rigorous modern DL | L2–L3 |
| [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) (Bishop) | Probabilistic classical ML | L3 |
| [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) | ISLP's rigorous big sibling | L3 |
| [Probabilistic Machine Learning](https://probml.github.io/pml-book/) (Murphy) | Encyclopaedic probabilistic ML | L3 |
| [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/mackay/itila/book.html) (MacKay) | Info theory + Bayesian ML | L3 |
| [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) (Boyd & Vandenberghe) | Optimization theory | L3 |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google DeepMind) | LLMs on accelerators | L3 |
| [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face) | Distributed LLM training | L3 |
| [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (Bekman) | Training infra in practice | L3 |
