# Resource Catalog

Every resource used in a lesson is listed here. Lessons should only link to resources in this file.

**Legend**
- **Type:** book · book-chapter · visual-essay · interactive · course · course-notes · video · notebook · blog · docs
- **Cost:** `free` · `free-online/paid-print` · `paid`
- **Level:** L1 intro · L2 intermediate · L3 advanced
- **Verified:** `F` = page fetched successfully · `S` = exact URL confirmed in search results · date = YYYY-MM

> Why these and not others? See [`.claude/rules/resource-quality.md`](../.claude/rules/resource-quality.md).
> The short version is that we prefer well-written chapters with worked examples and interactive visuals, and use video as a complement.

---

## 1. Books (free online editions)

These are the backbone of the curriculum. Where possible, lessons reuse a few of these books so you don't have to keep switching between authors.

| Resource | Author(s) | Why it's here | Type | Cost | Level | Used in | Verified |
|---|---|---|---|---|---|---|---|
| [An Introduction to Statistical Learning, with Applications in Python (ISLP)](https://www.statlearning.com/) · [free PDF](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html) · [Python labs](https://www.statlearning.com/resources-python) | James, Witten, Hastie, Tibshirani, Taylor | The clearest statistical-learning text around. Intuition first, light math, labs in every chapter. | book | free | L1–L2 | CORE-02…07 | S 2026-09 |
| [Understanding Deep Learning (UDL)](https://udlbook.github.io/udlbook/) · [notebooks repo](https://github.com/udlbook/udlbook) | Simon J.D. Prince (MIT Press 2023) | The best modern DL textbook for self-study. Superb figures, and a Colab notebook for every chapter. | book | free | L2 | DL-01, DL-03…06, GEN-04 | S+F 2026-09 |
| [Dive into Deep Learning (D2L)](https://d2l.ai/) | Zhang, Lipton, Li, Smola | Every concept comes with runnable PyTorch code, inline. Good second view to set beside UDL. | book | free | L2 | DL-02…06 | S 2026-09 |
| [Mathematics for Machine Learning (MML)](https://mml-book.github.io/) · [PDF](https://mml-book.github.io/book/mml-book.pdf) | Deisenroth, Faisal, Ong | Exactly the math ML uses, nothing more. Used just-in-time from the math lessons. | book | free | L2 | MATH-01…03 | S 2026-09 |
| [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) | Michael Nielsen | Still the gentlest *written* derivation of backprop, with small, readable NumPy code. | book | free | L1–L2 | DL-01, DL-03 | S 2026-09 |
| [The Hundred-Page Machine Learning Book](https://themlbook.com/) | Andriy Burkov | Dense 100-page overview, good for review. "Read first, buy later" model. | book | free-online/paid-print | L1–L2 | Go deeper | S 2026-09 |
| [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html) | François Fleuret | A 170-page, phone-sized DL summary. Great for revision. | book | free | L2 | Go deeper | S 2026-09 |
| [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) | Chris Bishop & Hugh Bishop | Rigorous, beautifully organised modern DL text. The "go deeper" reference. | book | free-online/paid-print | L2–L3 | Go deeper | S 2026-09 |
| [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/) | Kevin Murphy | Encyclopaedic probabilistic view of ML, for when you want the full story. | book | free | L3 | EL-02, Go deeper | S 2026-09 |
| [Interpretable Machine Learning (3rd ed.)](https://christophm.github.io/interpretable-ml-book/) | Christoph Molnar | The standard, very readable guide to PDP, permutation importance, SHAP, LIME… | book | free-online/paid-ebook | L2 | CORE-08 | S 2026-09 |
| [Feature Engineering and Selection](https://bookdown.org/max/FES) | Max Kuhn & Kjell Johnson | A practical, example-driven book on feature engineering. Code is in R, but the ideas are language-agnostic. | book | free | L2 | CORE-07 | S 2026-09 |
| [Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) | Sutton & Barto | *The* RL textbook. Clear and patient, with lots of examples. | book | free | L2 | EL-03 | S 2026-09 |
| [Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/) · [R original (fpp3)](https://otexts.com/fpp3/) | Hyndman, Athanasopoulos et al. | The friendliest forecasting book, now with a Python edition (Nixtla). | book | free | L1–L2 | EL-01 | S 2026-09 |
| [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) | Jurafsky & Martin | The canonical NLP text, updated for LLMs. Chapter-level reading on embeddings and transformers. | book | free | L2 | DL-05, GEN-01 | S 2026-09 |
| [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) | Cameron Davidson-Pilon | Bayesian inference taught computation-first, as notebooks. | book / notebook | free | L2 | EL-02 | S 2026-09 |

## 2. Books (paid, with official free code)

| Resource | Author(s) | Why it's here | Type | Cost | Level | Used in | Verified |
|---|---|---|---|---|---|---|---|
| [Hands-On Machine Learning with Scikit-Learn and PyTorch](https://github.com/ageron/handson-mlp) (notebooks free) | Aurélien Géron (O'Reilly, 2025) | The best practitioner book. The free notebooks alone are excellent. (Older TF edition: [handson-ml3](https://github.com/ageron/handson-ml3).) | book + notebook | paid (notebooks free) | L1–L2 | CORE-01, CORE-03, CORE-05, DL-03 | S+F 2026-09 |
| [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) (code free) | Sebastian Raschka (Manning) | Builds GPT step by step: tokenizer → attention → pretraining → fine-tuning → LoRA. | book + notebook | paid (code free) | L2 | DL-06, GEN-01, GEN-02 | F 2026-09 |
| [Deep Learning for Coders with fastai & PyTorch (fastbook)](https://github.com/fastai/fastbook) | Howard & Gugger | The full book is free as notebooks. Top-down and practical. | book + notebook | free (notebooks) | L1–L2 | DL-02, DL-04 | F 2026-09 |
| [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) · [free summaries](https://github.com/chiphuyen/dmls-book) | Chip Huyen (O'Reilly) | The best book on ML in production: data, features, deployment, drift, monitoring. | book | paid (summaries free) | L2 | PROD-01, PROD-02 | S 2026-09 |
| [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) · [resources repo](https://github.com/chiphuyen/aie-book) | Chip Huyen (O'Reilly 2025) | Building apps on foundation models: evaluation, RAG, fine-tuning, inference. | book | paid | L2 | GEN-02, GEN-03 | S 2026-09 |

## 3. Visual essays & interactive explainers

Start lessons with these. They build the mental picture before the reading.

| Resource | Author | Topic | Cost | Used in | Verified |
|---|---|---|---|---|---|
| [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) | Amazon MLU | Linear regression | free | CORE-02 | F(repo) 2026-09 |
| [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/) | Amazon MLU | Logistic regression | free | CORE-03 | F(repo) 2026-09 |
| [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) | Amazon MLU | Metrics | free | CORE-03 | F(repo) 2026-09 |
| [MLU-Explain: ROC & AUC](https://mlu-explain.github.io/roc-auc/) | Amazon MLU | Metrics | free | CORE-03 | F(repo) 2026-09 |
| [MLU-Explain: Train, Test & Validation Sets](https://mlu-explain.github.io/train-test-validation/) | Amazon MLU | Validation | free | CORE-04 | F(repo) 2026-09 |
| [MLU-Explain: Bias-Variance Tradeoff](https://mlu-explain.github.io/bias-variance/) | Amazon MLU | Generalization | free | CORE-04 | S 2026-09 |
| [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) | Amazon MLU | Generalization (modern) | free | CORE-04, DL-03 | F(repo) 2026-09 |
| [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/) | Amazon MLU | Trees | free | CORE-05 | F(repo) 2026-09 |
| [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/) | Amazon MLU | Ensembles | free | CORE-05 | F(repo) 2026-09 |
| [Principal Component Analysis Explained Visually](https://setosa.io/ev/principal-component-analysis/) | Setosa | PCA | free | CORE-06 | S 2026-09 |
| [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) | Wattenberg, Viégas, Johnson (Distill) | t-SNE pitfalls | free | CORE-06 | S 2026-09 |
| [Understanding UMAP](https://pair-code.github.io/understanding-umap/) | Google PAIR | UMAP vs t-SNE | free | CORE-06 | S 2026-09 |
| [Why Momentum Really Works](https://distill.pub/2017/momentum/) | Gabriel Goh (Distill) | Optimization | free | DL-03, MATH-02 | S 2026-09 |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | Polo Club, Georgia Tech | CNNs | free | DL-04 | S 2026-09 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Polo Club, Georgia Tech | A live GPT-2 running in your browser | free | DL-06 | S 2026-09 |
| [Seeing Theory](https://seeing-theory.brown.edu/) | Brown University | Probability & statistics | free | MATH-03 | S 2026-09 |

## 4. Long-form explainers (blogs)

| Resource | Author | Topic | Used in | Verified |
|---|---|---|---|---|
| [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) | Jay Alammar | Embeddings | DL-05 | S 2026-09 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Jay Alammar | Transformers | DL-06 | S 2026-09 |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Jay Alammar | Decoder-only LMs | DL-06, GEN-01 | S 2026-09 |
| [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) | Jay Alammar | Transfer learning in NLP | DL-06 | S 2026-09 |
| [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | Jay Alammar | Latent diffusion | GEN-04 | S 2026-09 |
| [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) | Jay Alammar | Reasoning models, RL post-training | GEN-01 | S 2026-09 |
| [Understanding and Coding Self-Attention, Multi-Head, Causal & Cross-Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) | Sebastian Raschka | Attention from scratch | DL-06 | S 2026-09 |
| [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) | Sebastian Raschka | LoRA | GEN-02 | S 2026-09 |
| [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Lilian Weng | Diffusion math | GEN-04 | S 2026-09 |
| [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) | Andrej Karpathy | Practical training & debugging | DL-03 | S 2026-09 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | Harvard NLP | Line-by-line transformer code | DL-06 | S 2026-09 |
| [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/) | Eugene Yan | Evals, RAG, guardrails, caching | GEN-02, GEN-03 | S 2026-09 |
| [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) · [Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) | Hamel Husain | LLM evaluation in practice | GEN-03 | S 2026-09 |
| [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) | Martin Zinkevich (Google) | 43 hard-won production rules | PROD-01 | S 2026-09 |

## 5. Courses & course notes (free)

| Resource | Provider | Why it's here | Level | Used in | Verified |
|---|---|---|---|---|---|
| [Machine Learning Crash Course (2024 rebuild)](https://developers.google.com/machine-learning/crash-course) · [Fairness module](https://developers.google.com/machine-learning/crash-course/fairness) | Google | Short, interactive modules with 130+ exercises | L1 | CORE-01, CORE-08 | S 2026-09 |
| [Introduction to ML Problem Framing](https://developers.google.com/machine-learning/problem-framing) | Google | A one-hour course on deciding whether (and how) to use ML | L1 | CORE-01, PROD-01 | S 2026-09 |
| [Machine learning in Python with scikit-learn (MOOC)](https://inria.github.io/scikit-learn-mooc/) | Inria / scikit-learn core devs | The best "use sklearn *properly*" course: pipelines, CV, tuning, pitfalls | L1–L2 | CORE-01, CORE-04, CORE-07 | S+F 2026-09 |
| [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) · [Feature Engineering](https://www.kaggle.com/learn/feature-engineering) | Kaggle | Short, practical, browser notebooks | L1–L2 | CORE-05, CORE-07 | S 2026-09 |
| [Learn PyTorch for Deep Learning (online book)](https://www.learnpytorch.io/) | Daniel Bourke / ZTM | Code-first PyTorch, with every section free to read | L1–L2 | DL-02, DL-04 | S+F 2026-09 |
| [Practical Deep Learning for Coders](https://course.fast.ai/) | fast.ai | Top-down, build-first DL | L1–L2 | DL-04 | S 2026-09 |
| [CS231n: Deep Learning for Computer Vision (notes)](https://cs231n.github.io/) | Stanford | Classic, very clear written notes on CNNs and training | L2 | DL-03, DL-04 | S 2026-09 |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | Hugging Face | Transformers, tokenizers, fine-tuning, reasoning models | L2 | GEN-02 | S 2026-09 |
| [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) | Hugging Face | Hands-on diffusers notebooks | L2 | GEN-04 | S 2026-09 |
| [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | Hugging Face | Hands-on deep RL | L2 | EL-03 | S 2026-09 |
| [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/user/introduction.html) | OpenAI | Concise written intro to policy-gradient RL, with clean code | L2–L3 | EL-03 | S 2026-09 |
| [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) · [2025 lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) | Stanford | Advanced: build and train an LM end to end | L3 | GEN-01 (deeper) | S 2026-09 |
| [Made With ML](https://madewithml.com/) · [repo](https://github.com/GokuMohandas/Made-With-ML) | Goku Mohandas | ML + software engineering, end to end | L2 | PROD-02 | S+F 2026-09 |
| [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | FSDL | Lectures and labs on shipping DL products | L2 | PROD-01 | S 2026-09 |
| [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) | DataTalks.Club | Free 9-week project course: tracking, orchestration, deployment, monitoring | L2 | PROD-02 | S 2026-09 |
| [Statistical Rethinking 2026](https://github.com/rmcelreath/stat_rethinking_2026) | Richard McElreath | The most human Bayesian stats course out there | L2 | EL-02 | S 2026-09 |

## 6. Video (complements, never the only way in)

| Resource | Creator | Topic | Length | Used in | Verified |
|---|---|---|---|---|---|
| [Essence of Linear Algebra (playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | 3Blue1Brown | Linear algebra intuition | ~3 h | MATH-01 | S 2026-09 |
| [Essence of Calculus (playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | 3Blue1Brown | Calculus intuition | ~3 h | MATH-02 | S 2026-09 |
| [But what is a neural network?](https://www.3blue1brown.com/lessons/neural-networks/) · [Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent/) · [Backprop calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus/) | 3Blue1Brown | NN intuition (lesson pages have the video + text) | ~1 h | DL-01 | S 2026-09 |
| [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt/) · [Attention in transformers](https://www.3blue1brown.com/lessons/attention/) | 3Blue1Brown | Transformer intuition | ~1 h | DL-06 | S 2026-09 |
| [Gradient Boost & XGBoost (playlist)](https://www.youtube.com/playlist?list=PLZ5DHV9_5h9vQwAImmNi1RfoTtSuOUjwM) | StatQuest | Boosting step by step | ~2 h | CORE-05 | S 2026-09 |
| [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | Andrej Karpathy | Build micrograd → makemore → GPT → tokenizer | ~15 h | DL-01, DL-05, DL-06, GEN-01 | F 2026-09 |
| ↳ [Lecture 1: micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | Karpathy | Backprop from scratch | 2.5 h | DL-01 | F 2026-09 |
| ↳ [Lecture 2: makemore bigram](https://www.youtube.com/watch?v=PaCmpygFfXo) | Karpathy | Language modeling basics | 2 h | DL-05 | F 2026-09 |
| ↳ [Lecture 3: makemore MLP](https://youtu.be/TCH_1BHY58I) | Karpathy | Embeddings, MLP LM, train/dev/test | 1.25 h | DL-05 | F 2026-09 |
| ↳ [Lecture 4: Activations, Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc) | Karpathy | Network health diagnostics | 2 h | DL-03 | F 2026-09 |
| ↳ [Lecture 7: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Karpathy | Transformer from scratch | 2 h | DL-06 | F 2026-09 |
| ↳ [Lecture 8: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | Karpathy | BPE tokenization | 2.2 h | GEN-01 | F 2026-09 |
| [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Karpathy | LLM overview | 1 h | GEN-01 | S 2026-09 |
| [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | Karpathy | Full training stack: pretraining → SFT → RL | 3.5 h | GEN-01 | S 2026-09 |

## 7. Docs used as "Build" references

| Resource | Used in | Verified |
|---|---|---|
| [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) | CORE-* | S 2026-09 |
| [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) | CORE-07 | S 2026-09 |
| [Python code for Murphy's PML (pyprobml)](https://github.com/probml/pyprobml) | EL-02 | S 2026-09 |
