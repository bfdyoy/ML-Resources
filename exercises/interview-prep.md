# Interview & System-Design Prep

[← Exercises](README.md)

Even if you're not job hunting, answering interview questions without notes is a great way to find gaps in your knowledge.

## Question banks & solved problems
| Resource | What's inside | Level |
|---|---|---|
| [Chip Huyen: Introduction to ML Interviews Book](https://huyenchip.com/ml-interviews-book/) ([repo](https://github.com/chiphuyen/ml-interviews-book)) | ~200 questions on math, ML, DL, and ML systems, plus how interviews work | L2 |
| [Deep Learning Interviews (Kashani & Ivry)](https://arxiv.org/abs/2201.00650) | Hundreds of *fully solved* problems: information theory, calculus, NN math, CNNs, and more (free PDF on arXiv) | L2→L3 |
| [alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews) | A study guide covering coding, ML breadth/depth, and ML system design | L2 |
| [andrewekhalel/MLQuestions](https://github.com/andrewekhalel/MLQuestions) | Common ML and CV interview questions, with answers | L1→L2 |
| [Deep-ML](https://www.deep-ml.com/problems) | ML coding problems (implement X from scratch) | L1→L3 |

## ML system design practice
1. Pick a case study from the [Evidently database of 800 ML/LLM system designs](https://www.evidentlyai.com/ml-system-design) or [applied-ml](https://github.com/eugeneyan/applied-ml).
2. **Before reading it**, spend 30 minutes designing it yourself: problem framing → metrics → data → features → model → serving → monitoring.
3. Read the company's write-up and compare. Note what you missed.
4. Frameworks for structuring your answer: [CS329S](https://stanford-cs329s.github.io/), the [DMLS summaries](https://github.com/chiphuyen/dmls-book), and the [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml).

## Self-test: can you explain these without notes?
- Bias–variance trade-off, and where double descent fits in
- Why L1 gives sparsity; ridge vs lasso geometry
- Precision/recall trade-off; ROC-AUC vs PR-AUC under imbalance
- How gradient boosting works, step by step
- Backprop through a 2-layer MLP, on a whiteboard
- Why BatchNorm behaves differently at train and test time
- Self-attention: shapes of Q, K, and V; why divide by √d; the causal mask
- KV cache: what it stores and how much memory it takes
- LoRA: where the parameter savings come from
- RAG failure modes and how to evaluate each stage
- RLHF vs DPO: what each one optimizes
- Data drift vs concept drift; how to detect each without labels
