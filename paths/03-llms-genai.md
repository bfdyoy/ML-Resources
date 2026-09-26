# Path 3: LLMs & Generative AI

**Goal:** Understand how LLMs and diffusion models are built, then adapt, evaluate, and ship applications on top of them.
**Duration:** ~31 h (≈ 4–5 weeks) · **Level:** L2→L3 · **Primary resources:** Raschka's *LLMs from Scratch* code, Hugging Face courses, UDL

**Prerequisites:** [Path 2](02-deep-learning.md), at least DL-02, DL-03, DL-05, DL-06.

| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [GEN-01 How LLMs Are Built](../lessons/llms-genai/01-how-llms-are-built.md) | Explain tokenization, pretraining, and post-training (SFT, preference tuning, RL for reasoning) | 9 h |
| 2 | [GEN-02 Adapting LLMs: Prompting, Fine-tuning, LoRA & RAG](../lessons/llms-genai/02-adapting-llms-finetuning-rag.md) | Choose and apply the right adaptation technique | 9 h |
| 3 | [GEN-03 Evaluating & Shipping LLM Applications](../lessons/llms-genai/03-evaluating-llm-apps.md) | Build rigorous evals and production guardrails | 5 h |
| 4 | [GEN-04 Generative Models: VAEs, GANs & Diffusion](../lessons/llms-genai/04-generative-models-diffusion.md) | Explain and train diffusion models | 8 h |

GEN-04 is independent of GEN-01–03. You can do it right after DL-04 if image generation interests you more.

## Capstone
**Ship a small LLM-powered product with an eval harness.** A domain assistant (RAG and/or a LoRA fine-tune) with: a test
set of 50+ questions, error analysis, assertion and LLM-judge evals (with measured judge agreement), a documented
comparison of at least 2 approaches, and a simple UI (Gradio/Streamlit). The eval report matters more than the UI.

**Next:** [Path 4: ML in Production](04-ml-in-production.md).
