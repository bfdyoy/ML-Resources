# GEN-03: Evaluating & Shipping LLM Applications

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~5 h | L2 | GEN-02 |

## Why this matters
Unsuccessful LLM products almost always have the same root cause: **no systematic evaluation**. "It looked good
on five examples" is not an eval. This lesson turns vibe checks into a measurable loop of error analysis → evals →
improvement, plus the guardrails you need in production.

## Learning goals
By the end you can:
- Do error analysis on LLM outputs: read traces, categorize failures, and prioritize.
- Build the three levels of evals: assertion/unit tests, LLM-as-judge (validated against human labels), and human review.
- Explain why LLM-as-judge needs its own validation, and how to measure judge agreement.
- Apply production patterns: guardrails, caching, defensive UX, and collecting user feedback.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) | The whole post | 1 h |
| 2 | **Read** | [Hamel Husain: AI Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) | Skim the questions and read the ones about error analysis and LLM-as-judge | 45 min |
| 3 | **Read** | [Eugene Yan: Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/) | The sections on evals, guardrails, caching, defensive UX, and user feedback | 1 h |
| 4 | **Build** | Your GEN-02 assistant | Label 50 outputs by hand, write failure categories, add assertion tests, build an LLM judge, and measure its agreement with your labels | 2 h |

## Check your understanding
1. Why should error analysis come *before* writing evals?
2. What are three ways an LLM-as-judge can be biased, and how would you detect each?
3. What's the difference between an offline eval set and online metrics? Why do you need both?
4. When is a simple assertion (e.g. "the output parses as JSON") better than a model-graded eval?
5. *(debug)* Your judge rates 95% of answers "good", but users complain constantly. What do you investigate?

## Mini-project
**Task:** Build an eval harness for your GEN-02 assistant that you can rerun: a test set, assertions, an LLM judge with
measured agreement, and a single summary score. Change one thing (the prompt, chunk size, or model) and show the eval catching a regression.
**Deliverable:** A script/notebook plus a before/after table.

## Go deeper
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) (Chip Huyen) `[paid]`: the evaluation-methodology chapters.
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml): these "classic ML" production rules still apply to LLM products.
- [Lilian Weng: Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/): causes, detection, and evaluation of hallucination.
- [HF Evaluation Guidebook](https://github.com/huggingface/evaluation-guidebook) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): benchmark-style evaluation of models.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 08: RAG, agents & evals](../../toolbox/08-rag-agents-evals.md), for every concept in this lesson, with alternatives.
- **Papers:** [Post-training, reasoning & agents](../../papers/04-alignment-reasoning-agents.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 39–40.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
