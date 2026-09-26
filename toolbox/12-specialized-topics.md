# Toolbox 12: Specialized Topics

[← Toolbox](README.md) · Papers: [09](../papers/09-tabular-timeseries-recsys-causal.md), [10](../papers/10-interpretability-uncertainty-responsible.md)

## Time-series forecasting
Guided version: [EL-01](../lessons/electives/01-time-series-forecasting.md)

| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Decomposition, stationarity, ACF | [Kaggle: Time Series](https://www.kaggle.com/learn/time-series) | [FPP (Python)](https://otexts.com/fpppy/) | FPP exercises | — |
| Baselines, ETS, ARIMA | — | [FPP (Python)](https://otexts.com/fpppy/) · [FPP3 (R)](https://otexts.com/fpp3/) | Beat seasonal-naive on a real series | — |
| ML & global models (GBMs with lags, DeepAR) | [Kaggle: Time Series](https://www.kaggle.com/learn/time-series) (hybrid models) | [GluonTS](https://github.com/awslabs/gluonts) docs | LightGBM with lag features vs DeepAR | [DeepAR](https://arxiv.org/abs/1704.04110), [N-BEATS](https://arxiv.org/abs/1905.10437), [TFT](https://arxiv.org/abs/1912.09363) |
| Skepticism about transformers for forecasting | — | [Are Transformers Effective?](https://arxiv.org/abs/2205.13504) ([code](https://github.com/cure-lab/LTSF-Linear)) | Reproduce DLinear vs a transformer | — |
| Foundation models for forecasting | — | [Chronos](https://github.com/amazon-science/chronos-forecasting) · [TimesFM](https://github.com/google-research/timesfm) | Zero-shot Chronos vs your tuned model | [Chronos](https://arxiv.org/abs/2403.07815), [TimesFM](https://arxiv.org/abs/2310.10688) |

## Recommender systems
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Candidate generation → scoring → re-ranking | [Google: Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) | [Evidently case studies](https://www.evidentlyai.com/ml-system-design) (filter: recommender systems) | — | — |
| Collaborative filtering & matrix factorization | [Google recsys course](https://developers.google.com/machine-learning/recommendation) (matrix factorization) | [fastbook](https://github.com/fastai/fastbook) Ch. 8 (collaborative filtering) | MF on MovieLens from scratch | [NCF](https://arxiv.org/abs/1708.05031) |
| Deep & sequential recommenders | [Alammar: Skipgram recommender talk](https://jalammar.github.io/skipgram-recommender-talk/) | [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) (notebooks for dozens of algorithms) | SASRec on MovieLens | [Wide & Deep](https://arxiv.org/abs/1606.07792), [DeepFM](https://arxiv.org/abs/1703.04247), [SASRec](https://arxiv.org/abs/1808.09781), [BERT4Rec](https://arxiv.org/abs/1904.06690), [DLRM](https://arxiv.org/abs/1906.00091) |
| Evaluation for recommenders | — | [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) (evaluation notebooks) · [IR book](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) Ch. 8 | Offline NDCG vs a simulated A/B test | — |

## Causal inference & uplift
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Potential outcomes, confounding, DAGs | [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) (Part I) | [Brady Neal: Intro to Causal Inference](https://www.bradyneal.com/causal-inference-course) · [The Mixtape](https://mixtape.scunning.com/) | The notebooks in Brave and True | — |
| Matching, IPW, diff-in-diff, IV, RDD | [Brave and True](https://matheusfacure.github.io/python-causality-handbook/) | [The Mixtape](https://mixtape.scunning.com/) | Diff-in-diff on a public policy dataset | — |
| Heterogeneous effects, meta-learners, double ML | [Brave and True](https://matheusfacure.github.io/python-causality-handbook/) (Part II) | [EconML](https://github.com/py-why/EconML) docs | T-/X-learner uplift model with EconML | [Double ML](https://arxiv.org/abs/1608.00060), [Meta-learners](https://arxiv.org/abs/1706.03461) |

## Anomaly & outlier detection
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Isolation forest, LOF, one-class SVM | [sklearn: Novelty and outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html) | [PyOD](https://github.com/yzhao062/pyod) (40+ detectors, benchmarks) | Compare 3 detectors on credit-card fraud data |
| Outliers & distribution shift as data problems | [MIT DCAI](https://dcai.csail.mit.edu/) (outliers & shift lecture) | — | — |

## Tabular deep learning
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| When deep learning helps on tables (and when it doesn't) | [fastbook](https://github.com/fastai/fastbook) Ch. 9 (tabular) | [Grinsztajn et al.](https://arxiv.org/abs/2207.08815) | Entity embeddings vs GBM on the same data | [TabPFN](https://arxiv.org/abs/2207.01848) |

## Audio & speech
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Audio data, spectrograms, ASR, TTS | [HF Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) | [SLP3](https://web.stanford.edu/~jurafsky/slp3/) (speech chapters) | Fine-tune Whisper on a small language dataset | [wav2vec 2.0](https://arxiv.org/abs/2006.11477), [Whisper](https://arxiv.org/abs/2212.04356), [WaveNet](https://arxiv.org/abs/1609.03499) |

## Interpretability of neural networks (mechanistic)
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Circuits, residual stream, induction heads | [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | [ARENA](https://github.com/callummcdougall/ARENA_3.0) interpretability chapter | [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) on GPT-2 small | [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) |
| Feature visualization & attribution | [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) | [Molnar](https://christophm.github.io/interpretable-ml-book/) (neural network chapters) | Integrated gradients on an image classifier | [Integrated Gradients](https://arxiv.org/abs/1703.01365), [Sanity checks](https://arxiv.org/abs/1810.03292) |

## Uncertainty
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Calibration & conformal prediction | [sklearn: Calibration](https://scikit-learn.org/stable/modules/calibration.html) | [Conformal prediction intro](https://arxiv.org/abs/2107.07511) | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) intervals on a regression model | [Calibration](https://arxiv.org/abs/1706.04599) |
| Bayesian deep learning basics | — | [Probabilistic ML, Book 2](https://probml.github.io/pml-book/) (Bayesian neural networks chapter) | MC-dropout uncertainty on a small regression | — |

## AI safety & security (for practitioners)
| Concept | Start here | Go deeper |
|---|---|---|
| LLM application security | [OWASP Top 10 for LLM Apps](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) | [Simon Willison: prompt injection](https://simonwillison.net/series/prompt-injection/) |
| Alignment & reward hacking | [Lilian Weng: Reward Hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) | [RLHF Book](https://rlhfbook.com/) · [Constitutional AI](https://arxiv.org/abs/2212.08073) |
| Hands-on safety & evals curriculum | [ARENA](https://github.com/callummcdougall/ARENA_3.0) | — |
