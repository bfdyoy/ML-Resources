# Resource Catalog

Every resource used in a **lesson** is listed here. The wider reference material lives in the [toolbox](../toolbox/README.md),
the [papers library](../papers/README.md), and the [exercises](../exercises/README.md). How every URL in the repo was verified is recorded in
[`verified-urls.tsv`](verified-urls.tsv).

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

## 8. Deep links into catalogued resources

Chapter/section URLs used directly in lessons. The parent resource is listed above.

| Deep link | Parent resource | Used in | Verified |
|---|---|---|---|
| [Nielsen Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html) · [Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html) · [Ch. 3](http://neuralnetworksanddeeplearning.com/chap3.html) · [Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html) · [Ch. 5](http://neuralnetworksanddeeplearning.com/chap5.html) | Neural Networks and Deep Learning | DL-01, DL-03 | S 2026-09 (Ch. 2); others follow the site's `chapN.html` scheme |
| [Molnar: SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) | Interpretable Machine Learning | CORE-08 | S 2026-09 |
| [MLU-Explain: Double Descent 2](https://mlu-explain.github.io/double-descent2/) | MLU-Explain | CORE-04 | F(repo) 2026-09 |
| [learnpytorch.io 00 Fundamentals](https://www.learnpytorch.io/00_pytorch_fundamentals/) · [01 Workflow](https://www.learnpytorch.io/01_pytorch_workflow/) · [04 Custom Datasets](https://www.learnpytorch.io/04_pytorch_custom_datasets/) · [07 Experiment Tracking](https://www.learnpytorch.io/07_pytorch_experiment_tracking/) · [08 Paper Replicating](https://www.learnpytorch.io/08_pytorch_paper_replicating/) · [09 Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/) | Learn PyTorch for Deep Learning | DL-02, DL-04, Path 2, PROD-02 | S (00, 01) + F(repo) 2026-09 |

## 9. Resources used by the full-coverage lessons

Added with CORE-09…12, DL-07…09, CV-01…04, GEN-05…10, PROD-03…05, EL-04…10 and the extensions to existing lessons.
Papers linked from lessons are catalogued in the [papers library](../papers/README.md). The `Verified` column uses the method codes from
[`verified-urls.tsv`](verified-urls.tsv).

| Resource | Used in | Verified |
|---|---|---|
| [Stat 110](https://stat110.hsites.harvard.edu/) | MATH-03 | S 2026-09 |
| [Think Stats](https://greenteapress.com/wp/think-stats-3e/) | MATH-03 | S 2026-09 |
| [Deep-ML](https://www.deep-ml.com/problems) | CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, CORE-08, CORE-09, DL-01, DL-02, DL-03, DL-04, DL-05, DL-06, EL-01, EL-02, EL-03, GEN-01, GEN-02, GEN-03, GEN-04, PROD-01, PROD-02, CV-01 | S 2026-09 |
| [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) | CORE-09 | S 2026-09 |
| [Bishop, PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) | CORE-09 | S 2026-09 |
| [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf) | CORE-09 | S 2026-09 |
| [scikit-learn: Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html) | CORE-09 | R 2026-09 |
| [scikit-learn: Kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) | CORE-09 | R 2026-09 |
| [scikit-learn: Nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html) | CORE-09 | R 2026-09 |
| [scikit-learn: SVM](https://scikit-learn.org/stable/modules/svm.html) | CORE-09 | R 2026-09 |
| [StatQuest: Support Vector Machines, Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) | CORE-09 | S 2026-09 |
| [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) | CORE-09 | S 2026-09 |
| [cleanlab](https://github.com/cleanlab/cleanlab) | CORE-10 | G 2026-09 |
| [dcai-lab](https://github.com/dcai-course/dcai-lab) | CORE-10 | G 2026-09 |
| [imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) | CORE-10 | S 2026-09 |
| [Lilian Weng: Active learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/) | CORE-10 | R 2026-09 |
| [Lilian Weng: Semi-supervised learning](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/) | CORE-10 | R 2026-09 |
| [Lilian Weng: Thinking about High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) | CORE-10 | R 2026-09 |
| [MIT Introduction to Data-Centric AI](https://dcai.csail.mit.edu/) | CORE-10, EL-06, PROD-03 | S 2026-09 |
| [sklearn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) | CORE-10 | R 2026-09 |
| [conformal-prediction notebooks](https://github.com/aangelopoulos/conformal-prediction) | CORE-11 | G 2026-09 |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | CORE-11 | G 2026-09 |
| [sklearn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | CORE-11 | R 2026-09 |
| [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | CORE-12 | G 2026-09 |
| [Optuna](https://github.com/optuna/optuna) | CORE-12 | G 2026-09 |
| [sklearn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) | CORE-12 | R 2026-09 |
| [colah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | DL-05 | S 2026-09 |
| [Karpathy: The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) | DL-05 | R 2026-09 |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | DL-07, EL-08 | G 2026-09 |
| [Horace He: Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | DL-07, EL-08 | S 2026-09 |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | DL-07, EL-08, GEN-08, PROD-04 | S 2026-09 |
| [PyTorch Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | DL-07 | R 2026-09 |
| [PyTorch: Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | DL-07 | R 2026-09 |
| [PyTorch: torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | DL-07 | R 2026-09 |
| [Tim Dettmers: Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) | DL-07 | S 2026-09 |
| [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | DL-07, PROD-04 | S 2026-09 |
| [HF: Designing positional encoding](https://huggingface.co/blog/designing-positional-encoding) | DL-08 | R 2026-09 |
| [HF: Mixture of Experts Explained](https://huggingface.co/blog/moe) | DL-08 | R 2026-09 |
| [Lilian Weng: The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | DL-08 | R 2026-09 |
| [Raschka: The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) | DL-08 | S 2026-09 |
| [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) | DL-08 | G 2026-09 |
| [CS224W course notes](https://snap-stanford.github.io/cs224w-notes/) | DL-09 | S 2026-09 |
| [CS224W: Machine Learning with Graphs](https://cs224w.stanford.edu/) | DL-09 | S 2026-09 |
| [Distill: A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) | DL-09 | S 2026-09 |
| [Distill: Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/) | DL-09 | S 2026-09 |
| [Hamilton: Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) | DL-09 | S 2026-09 |
| [HF: Introduction to Graph Machine Learning](https://huggingface.co/blog/intro-graphml) | DL-09 | R 2026-09 |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | DL-09 | G 2026-09 |
| [UvA DL Tutorial: GNNs](https://uvadlc-notebooks.readthedocs.io/) | DL-09, CV-02 | S 2026-09 |
| [Detectron2](https://github.com/facebookresearch/detectron2) | CV-01 | G 2026-09 |
| [HF Community CV Course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome) | CV-01, CV-03 | S 2026-09 |
| [Lilian Weng: Object Detection for Dummies](https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/) | CV-01 | R 2026-09 |
| [Lilian Weng: Object Detection for Dummies, part 2](https://lilianweng.github.io/posts/2017-12-15-object-recognition-part-2/) | CV-01 | R 2026-09 |
| [Lilian Weng: Object Detection for Dummies, part 3](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/) | CV-01 | R 2026-09 |
| [Lilian Weng: Object Detection for Dummies, part 4](https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/) | CV-01 | R 2026-09 |
| [Szeliski, *Computer Vision: Algorithms and Applications*](https://szeliski.org/Book/) | CV-01, CV-04 | S 2026-09 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | CV-01 | G 2026-09 |
| [Lilian Weng: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) | CV-02 | R 2026-09 |
| [Lilian Weng: Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) | CV-02 | R 2026-09 |
| [HF: A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining) | CV-03 | R 2026-09 |
| [HF: Vision Language Models Explained](https://huggingface.co/blog/vlms) | CV-03 | R 2026-09 |
| [Lilian Weng: Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) | CV-03 | R 2026-09 |
| [Foundations of Computer Vision](https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/) | CV-04 | S 2026-09 |
| [Lilian Weng: Scaling Laws, Carefully](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/) | GEN-01 | R 2026-09 |
| [nanochat (Karpathy)](https://github.com/karpathy/nanochat) | GEN-01 | G 2026-09 |
| [HF Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) | GEN-03 | G 2026-09 |
| [Lilian Weng: Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) | GEN-03 | R 2026-09 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | GEN-03 | G 2026-09 |
| [Faiss (Meta)](https://github.com/facebookresearch/faiss) | GEN-05 | G 2026-09 |
| [HF: Embedding Quantization](https://huggingface.co/blog/embedding-quantization) | GEN-05 | R 2026-09 |
| [HF: MTEB](https://huggingface.co/blog/mteb) | GEN-05 | R 2026-09 |
| [Lilian Weng: How to Build an Open-Domain QA System](https://lilianweng.github.io/posts/2020-10-29-odqa/) | GEN-05 | R 2026-09 |
| [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) | GEN-05 | G 2026-09 |
| [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) | GEN-05 | G 2026-09 |
| [Sentence Transformers docs](https://sbert.net/) | GEN-05 | S 2026-09 |
| [Anthropic courses](https://github.com/anthropics/courses) | GEN-06 | G 2026-09 |
| [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | GEN-06 | S 2026-09 |
| [GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | GEN-06 | G 2026-09 |
| [HF AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) | GEN-06 | S 2026-09 |
| [HF: Evaluating structured outputs](https://huggingface.co/blog/evaluation-structured-outputs) | GEN-06 | R 2026-09 |
| [Lilian Weng: Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) | GEN-06 | R 2026-09 |
| [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | GEN-06 | R 2026-09 |
| [MCP docs](https://modelcontextprotocol.io/) | GEN-06 | S 2026-09 |
| [OpenAI Cookbook](https://github.com/openai/openai-cookbook) | GEN-06 | G 2026-09 |
| [HF smol-course](https://github.com/huggingface/smol-course) | GEN-07 | G 2026-09 |
| [HF: Fine-tune with DPO (TRL)](https://huggingface.co/blog/dpo-trl) | GEN-07 | R 2026-09 |
| [HF: Illustrating RLHF](https://huggingface.co/blog/rlhf) | GEN-07 | R 2026-09 |
| [HF: Open-R1](https://huggingface.co/blog/open-r1) | GEN-07 | R 2026-09 |
| [HF: RLOO](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo) | GEN-07 | R 2026-09 |
| [Lilian Weng: Reward Hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | GEN-07, GEN-09 | R 2026-09 |
| [Lilian Weng: Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) | GEN-07 | R 2026-09 |
| [Raschka: Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms) | GEN-07 | S 2026-09 |
| [Reasoning from Scratch](https://github.com/rasbt/reasoning-from-scratch) | GEN-07 | G 2026-09 |
| [RLHF Book](https://rlhfbook.com/) | GEN-07 | S 2026-09 |
| [TRL (Hugging Face)](https://github.com/huggingface/trl) | GEN-07 | G 2026-09 |
| [Grootendorst: A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) | GEN-08 | S 2026-09 |
| [HF: 4-bit quantization & QLoRA with bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes) | GEN-08 | R 2026-09 |
| [HF: Assisted Generation](https://huggingface.co/blog/assisted-generation) | GEN-08 | R 2026-09 |
| [HF: KV Caching Explained](https://huggingface.co/blog/kv-cache) | GEN-08 | R 2026-09 |
| [HF: Optimizing your LLM in production](https://huggingface.co/blog/optimize-llm) | GEN-08 | R 2026-09 |
| [Lilian Weng: Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | GEN-08 | R 2026-09 |
| [MIT 6.5940 EfficientML](https://hanlab.mit.edu/courses/2026-fall-65940) | GEN-08 | S 2026-09 |
| [vLLM](https://github.com/vllm-project/vllm) | GEN-08 | G 2026-09 |
| [Lilian Weng: Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) | GEN-09 | R 2026-09 |
| [OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) | GEN-09 | S 2026-09 |
| [Simon Willison: Prompt injection series](https://simonwillison.net/series/prompt-injection/) | GEN-09 | S 2026-09 |
| [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | GEN-10 | G 2026-09 |
| [Lilian Weng: Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) | GEN-10 | R 2026-09 |
| [MIT 6.S184: Flow Matching & Diffusion](https://diffusion.csail.mit.edu/2026/index.html) | GEN-10 | S 2026-09 |
| [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) | GEN-10 | S 2026-09 |
| [Evidently (open source)](https://github.com/evidentlyai/evidently) | PROD-03 | G 2026-09 |
| [Evidently: ML Observability course](https://www.evidentlyai.com/ml-observability-course) | PROD-03 | S 2026-09 |
| [Google Cloud: MLOps pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | PROD-03 | S 2026-09 |
| [Hidden Technical Debt in ML Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) | PROD-03 | S 2026-09 |
| [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) | PROD-03 | S 2026-09 |
| [Lilian Weng: How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/) | PROD-04 | R 2026-09 |
| [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles) | PROD-04 | G 2026-09 |
| [PyTorch: DDP series intro](https://pytorch.org/tutorials/beginner/ddp_series_intro.html) | PROD-04 | R 2026-09 |
| [PyTorch: DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | PROD-04 | R 2026-09 |
| [PyTorch: FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | PROD-04 | R 2026-09 |
| [Stas Bekman: ML Engineering Open Book](https://github.com/stas00/ml-engineering) | PROD-04 | G 2026-09 |
| [Chip Huyen: Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) | PROD-05 | S 2026-09 |
| [Chip Huyen: Common pitfalls when building generative AI applications](https://huyenchip.com/2025/01/16/ai-engineering-pitfalls.html) | PROD-05 | S 2026-09 |
| [Evidently: LLM evaluation course](https://www.evidentlyai.com/llm-evaluations-course) | PROD-05 | S 2026-09 |
| [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) | PROD-05 | G 2026-09 |
| [What We've Learned From A Year of Building with LLMs](https://applied-llms.org/) | PROD-05 | S 2026-09 |
| [Chronos (Amazon)](https://github.com/amazon-science/chronos-forecasting) | EL-01 | G 2026-09 |
| [GluonTS](https://github.com/awslabs/gluonts) | EL-01 | G 2026-09 |
| [Kaggle Learn: Time Series](https://www.kaggle.com/learn/time-series) | EL-01 | S 2026-09 |
| [TimesFM (Google)](https://github.com/google-research/timesfm) | EL-01 | G 2026-09 |
| [Evidently system-design case studies](https://www.evidentlyai.com/ml-system-design) | EL-04, PROD-05 | S 2026-09 |
| [Google: Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) | EL-04 | S 2026-09 |
| [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) | EL-04, GEN-05 | S 2026-09 |
| [Jay Alammar: Skip-gram for recommendations](https://jalammar.github.io/skipgram-recommender-talk/) | EL-04 | R 2026-09 |
| [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) | EL-04 | G 2026-09 |
| [Brady Neal: Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) | EL-05 | S 2026-09 |
| [Brave and True: Meta Learners chapter](https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html) | EL-05 | S 2026-09 |
| [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) | EL-05 | S 2026-09 |
| [Causal Inference: The Mixtape](https://mixtape.scunning.com/) | EL-05 | S 2026-09 |
| [EconML](https://github.com/py-why/EconML) | EL-05 | G 2026-09 |
| [PyOD](https://github.com/yzhao062/pyod) | EL-06 | G 2026-09 |
| [sklearn: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html) | EL-06 | R 2026-09 |
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | EL-07 | S 2026-09 |
| [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0) | EL-07, GEN-07, GEN-09 | G 2026-09 |
| [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) | EL-07 | S 2026-09 |
| [Distill: The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) | EL-07 | S 2026-09 |
| [Thinking Like Transformers (raspy)](https://github.com/srush/raspy) | EL-07 | G 2026-09 |
| [Transformer Circuits Exercises](https://transformer-circuits.pub/2021/exercises/index.html) | EL-07 | S 2026-09 |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | EL-07 | G 2026-09 |
| [GPU MODE lectures](https://github.com/gpu-mode/lectures) | EL-08 | G 2026-09 |
| [LeetGPU](https://leetgpu.com/) | EL-08 | S 2026-09 |
| [llm.c (Karpathy)](https://github.com/karpathy/llm.c) | EL-08 | G 2026-09 |
| [Stanford CS336 assignment 2](https://github.com/stanford-cs336/assignment2-systems) | EL-08, PROD-04 | G 2026-09 |
| [Tensara problems](https://github.com/tensara/problems) | EL-08 | G 2026-09 |
| [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) | EL-09 | S 2026-09 |
| [Hugging Face Transformers (Whisper docs)](https://github.com/huggingface/transformers) | EL-09 | G 2026-09 |
| [tabular-benchmark](https://github.com/LeoGrin/tabular-benchmark) | EL-10 | G 2026-09 |
