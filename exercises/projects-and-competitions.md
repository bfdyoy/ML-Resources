# Projects & Competitions

[← Exercises](README.md)

## Project ideas by level
Each project should end with a **README, a model card** ([template idea: Model Cards](https://arxiv.org/abs/1810.03993)), and a short write-up of what didn't work.

### Level 1: Classical ML (after Path 1)
| Project | Skills | Dataset idea |
|---|---|---|
| Price prediction with a full pipeline + SHAP | pipelines, GBMs, interpretability | Ames Housing (OpenML) |
| Churn model with cost-based threshold selection | imbalance, metrics, calibration | Telco churn (public Kaggle datasets) |
| Customer segmentation report | clustering, PCA/UMAP, communication | UCI Online Retail |
| Demand forecasting with backtesting | time-series CV, ETS/ARIMA vs GBM | UCI Bike Sharing |
| Uncertainty-aware regression | conformal prediction with [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | California housing |

### Level 2: Deep learning (after Path 2)
| Project | Skills | Dataset idea |
|---|---|---|
| Fine-grained image classifier + Grad-CAM error analysis | transfer learning, interpretability | Oxford-IIIT Pets |
| Small GPT on a domain corpus, with your own tokenizer | transformers, tokenization, scaling experiments | Public-domain books (Project Gutenberg) |
| Paper replication (ResNet / ViT / SimCLR) with ablations | reading papers, experiments | CIFAR-10 |
| Object detector for a custom class | detection, labeling, augmentation | Your own photos (label ~200 images) |
| Recommender on MovieLens (MF → SASRec) | embeddings, ranking metrics | MovieLens |

### Level 3: LLMs & production (after Paths 3–4)
| Project | Skills | Idea |
|---|---|---|
| Domain RAG assistant with an eval harness | retrieval, evals, LLM-as-judge | Docs of a library you use |
| LoRA fine-tune vs RAG vs prompting comparison | PEFT, evaluation design | A narrow task (e.g. SQL generation) |
| Tool-using agent with guardrails | agents, security (OWASP LLM Top 10) | A research or data-analysis assistant |
| End-to-end MLOps system | tracking, CI/CD, serving, drift monitoring | Any Level-1 model, "productionized" |
| GRPO on math word problems | RL for LLMs, reward design | GSM8K subset with a 0.5–1.5B model |

## Kaggle: how to use it well
- Start with [Kaggle Learn](https://www.kaggle.com/learn) micro-courses. Then join the **Getting Started** and **Playground Series** competitions
  (search for them on Kaggle's competitions page). They're low-stakes and always open.
- After each competition, **read the top-3 public write-ups** in the discussion forum and reproduce one trick. That's where most of the learning is.
- Use competitions to practice **validation strategy** (does your CV correlate with the leaderboard?) and **leakage hunting**, not just to chase the leaderboard.

## Portfolio checklist
- [ ] 3–5 polished repos (one per path capstone) with clear READMEs
- [ ] At least one project with a **deployed** demo (Gradio/Streamlit, or an API)
- [ ] At least one **paper replication** with an ablation table
- [ ] At least one **evaluation-heavy** LLM project
- [ ] Your [from-scratch ladder](from-scratch-ladder.md) repo, with passing tests
