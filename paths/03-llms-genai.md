# Path 3: LLMs & Generative AI

**Goal:** Understand how LLMs and diffusion models are built, then adapt, evaluate, and ship applications on top of them.
**Duration:** ~81 h (≈ 11 weeks; Part A alone ≈ 46 h) · **Level:** L2→L3 · **Primary resources:** Raschka's *LLMs from Scratch* code, Hugging Face courses, UDL

**Prerequisites:** [Path 2](02-deep-learning.md), at least DL-02, DL-03, DL-05, DL-06.

### Part A: Building LLM applications
| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [GEN-01 How LLMs Are Built](../lessons/llms-genai/01-how-llms-are-built.md) | Explain tokenization, pretraining, and post-training | 9 h |
| 2 | [GEN-02 Adapting LLMs: Prompting, Fine-tuning, LoRA & RAG](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md) | Choose and apply the right adaptation technique | 9 h |
| 3 | [GEN-03 Evaluating & Shipping LLM Applications](../lessons/llms-genai/03-evaluating-llm-apps.md) | Build rigorous evals and production guardrails | 5 h |
| 4 | [GEN-05 Retrieval Engineering for RAG](../lessons/llms-genai/05-retrieval-engineering.md) | Build and measure a strong retriever (BM25, embeddings, ANN, reranking) | 9 h |
| 5 | [GEN-06 Agents: Tool Use, Structured Outputs & MCP](../lessons/llms-genai/06-agents-tool-use.md) | Build agents and workflows that you can evaluate | 9 h |
| 6 | [GEN-09 LLM Security & Safety](../lessons/llms-genai/09-llm-security-safety.md) | Threat-model and red-team LLM apps | 5 h |

### Part B: Under the hood
| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 7 | [GEN-07 Post-Training: SFT, RLHF, DPO & Reasoning](../lessons/llms-genai/07-post-training-alignment-reasoning.md) | Align and fine-tune models with preferences and verifiable rewards | 10 h |
| 8 | [GEN-08 Efficient LLM Inference](../lessons/llms-genai/08-efficient-llm-inference.md) | Quantize, cache, batch, and serve models efficiently | 8 h |

### Part C: Generative media
| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 9 | [GEN-04 Generative Models: VAEs, GANs & Diffusion](../lessons/llms-genai/04-generative-models-diffusion.md) | Explain and train diffusion models | 8 h |
| 10 | [GEN-10 Advanced Diffusion & Flow Matching](../lessons/llms-genai/10-advanced-diffusion-flow-matching.md) | Score models, guidance, latent diffusion, and flow matching | 9 h |

Part A is the practical core. Parts B and C can be taken in either order. Part C only needs Path 2 (GEN-04 can follow straight after DL-04).
GEN-08 assumes [DL-07](../lessons/deep-learning/07-performance-gpus-mixed-precision.md). GEN-07 is easier after [EL-03 RL](../lessons/electives/03-reinforcement-learning.md).

## Capstone
**Ship a small LLM-powered product with an eval harness.** A domain assistant (RAG and/or a LoRA fine-tune, optionally agentic) with:
a retrieval eval set and a leaderboard (GEN-05), a test set of 50+ questions, error analysis, assertion and LLM-judge evals (with measured judge
agreement), a red-team suite (GEN-09), a documented comparison of at least 2 approaches, and a simple UI (Gradio/Streamlit).
The eval report matters more than the UI.

**Next:** [Path 4: ML in Production](04-ml-in-production.md).
