# Path 4: ML in Production

**Goal:** Design ML systems end to end and operate them reliably: tracking, pipelines, serving, monitoring, retraining.
**Duration:** ~46 h + capstone (≈ 6–7 weeks) · **Level:** L2 · **Primary resources:** Rules of ML, Chip Huyen's DMLS, Made With ML, MLOps Zoomcamp

> 📘 **Study notes:** every lesson starts with its written explanation (step 0, ~1 h): the math step by step, worked examples, runnable code, and answers to the self-check. Read in order, the notes form one continuous walkthrough: [notes index](../notes/README.md).
>
> 🗓️ **As a course:** [Course 4: ML in Production](../courses/04-ml-in-production.md) turns this path into a week-by-week plan: warm-ups, [labs](../labs/README.md), [playbook](../playbook/README.md) readings, project milestones, a midterm, and a capstone rubric.

**Prerequisites:** [Path 1](01-core-ml-practitioner.md). Git, basic Docker, and some command-line comfort.

| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [PROD-01 ML System Design](../lessons/production/01-ml-system-design.md) | Write a solid ML design doc; reason about data, serving, and drift | 7 h |
| 2 | [PROD-02 MLOps in Practice](../lessons/production/02-mlops-in-practice.md) | Ship a tracked, tested, containerized, monitored model | 13 h |
| 3 | [PROD-03 Testing, Monitoring & Drift](../lessons/production/03-testing-monitoring-drift.md) | Test data and models; detect and respond to drift | 8 h |
| 4 | [PROD-04 Distributed Training at Scale](../lessons/production/04-distributed-training.md) | Train beyond one GPU (DDP, FSDP/ZeRO, 5D parallelism) | 10 h |
| 5 | [PROD-05 LLMOps](../lessons/production/05-llmops-genai-platforms.md) | Run LLM apps in production: evals in CI, tracing, cost | 8 h |

PROD-04 needs [DL-07](../lessons/deep-learning/07-performance-gpus-mixed-precision.md) and [GEN-01](../lessons/llms-genai/01-how-llms-are-built.md). PROD-05 needs [GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md).

## Capstones
- **Classic ML system:** described in [PROD-02](../lessons/production/02-mlops-in-practice.md#mini-project-capstone-for-the-production-path): take an earlier
model to "production", with a pipeline, registry, API, CI, and drift monitoring, extended with the PROD-03 tests.
- **LLM system:** described in [PROD-05](../lessons/production/05-llmops-genai-platforms.md#mini-project-capstone-for-llm-production).
