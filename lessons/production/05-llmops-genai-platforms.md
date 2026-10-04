# PROD-05: LLMOps: Running LLM Applications in Production

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Production | ~8 h | L2→L3 | PROD-02, GEN-03 (GEN-05/06/08 help) |

## Why this matters
LLM products bring new operational problems. Behaviour changes whenever you edit a prompt or the provider updates a model. Costs scale
with tokens. Quality needs continuous evals, and every request should be traceable. LLMOps applies MLOps discipline to these systems:
eval-driven development, tracing, caching, routing, guardrails, and cost control.

## Learning goals
By the end you can:
- Sketch the architecture of a GenAI platform: context construction, guardrails, model gateway/router, caching, orchestration, and observability.
- Set up eval-driven development: offline eval sets that gate prompt or model changes, and online feedback signals.
- Trace and debug LLM calls end to end (inputs, retrieved context, tool calls, outputs, latency, cost).
- Reduce cost and latency with caching, model routing, prompt compression, and self-hosting trade-offs.
- Avoid the common pitfalls of building GenAI apps.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: LLMOps: Running GenAI Applications in Production](../../notes/production/05-llmops-genai-platforms.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Chip Huyen: Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) | The whole post | 1 h |
| 2 | **Read** | [What We've Learned From A Year of Building with LLMs](https://applied-llms.org/) | The operational and strategic sections (you read the tactical one in GEN-02/03) | 1.5 h |
| 3 | **Read** | [Chip Huyen: Common pitfalls when building generative AI applications](https://huyenchip.com/2025/01/16/ai-engineering-pitfalls.html) | The whole post | 30 min |
| 4 | **Read + Build** | [Evidently: LLM evaluation course](https://www.evidentlyai.com/llm-evaluations-course) | Test datasets, comparing prompts and models, and tracing | 2 h |
| 5 | **Build** | [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) | The monitoring module (or the equivalent in your stack): add tracing, user feedback capture, and a cost dashboard to your GEN-03 app | 2 h |

## Check your understanding
1. Why should a prompt change go through the same review and eval gate as a code change?
2. What should a trace record for a RAG + tools request so you can debug it a week later?
3. Exact caching vs semantic caching: what are the risks of each?
4. When does routing between a small and a large model pay off, and how do you decide where to route?
5. What online signals (explicit and implicit) tell you quality is dropping?
6. *(debug)* Your API bill doubled overnight with no traffic increase. List the four most likely causes.

## Mini-project (capstone for LLM production)
**Task:** Take your GEN-03 app to "production grade": versioned prompts, an eval gate in CI, request tracing, a feedback button,
a cost/latency dashboard, caching, and the GEN-09 red-team suite.
**Deliverable:** An architecture diagram, the CI config, and screenshots of traces and the dashboard.

## Go deeper
- [Eugene Yan: Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/).
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) (Chip Huyen) `[paid]`: the architecture and user-feedback chapters.
- [Evidently: 800 ML & LLM system-design case studies](https://www.evidentlyai.com/ml-system-design) (filter: Generative AI & LLM).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md) (LLMOps row) · [Toolbox 08: RAG, agents & evals](../../toolbox/08-rag-agents-evals.md).
- **Papers:** [ML in production](../../papers/11-ml-in-production.md) · [Post-training, reasoning & agents](../../papers/04-alignment-reasoning-agents.md) (evaluating LLMs).
- **Implement it yourself:** a semantic cache (embed the query, similarity threshold, TTL). Measure the hit rate and the wrong-answer rate on your eval set.
- **Drills:** more in [exercises/](../../exercises/README.md).
