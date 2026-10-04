# Course 4: ML in Production, a 7-week syllabus

[← Courses](README.md) · Path: [Path 4: ML in Production](../paths/04-ml-in-production.md) · Labs: [index](../labs/README.md) · Cards: [production deck](flashcards/README.md)

> **Pace** ≈ 8 h/week for 7 weeks. **Before you start:** Course 1 (at least through CORE-07). PROD-04 and PROD-05 also need parts of Courses 2 and 3.
> **Running project:** **one system, end to end** (Made With ML and Full Stack Deep Learning). Take your Core ML capstone model to production. It gets a little more "real" each week.

The [weekly loop](README.md#32-the-weekly-loop-about-8-hours) applies every week.

## How this course is shaped

- **Deploy on day 1, improve forever after** (ML Zoomcamp puts deployment in module 5; fast.ai deploys in lesson 2). Week 1 ships an ugly but working endpoint.
  Every later week replaces one manual step with an automated, tested one.
- **The system is the unit, not the model** (FSDL). Milestones are about data, CI, monitoring and rollback, and not about accuracy.
- **Rules before tools** ([Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)). Each week names the rule it applies, and only then the tool.
- **Write the design doc first** (Chip Huyen's DMLS, linked from PROD-01). The doc is a living artifact, updated every week.

---

## Week 1: Whole game: ship an ugly endpoint + system design
- **Whole game (2–3 h):** wrap your Core ML model in FastAPI. Containerize it, run it, and `curl` it. No tests, no CI yet.
- **Do:** [PROD-01](../lessons/production/01-ml-system-design.md) with its [notes](../notes/production/01-ml-system-design.md).
- **Project:** a 2-page design doc for *this* system. Mark every manual step in red.
- **Explain it back:** what training/serving skew could affect your system, specifically.

## Week 2: MLOps, part 1: pipeline, tracking, registry
- **Warm-up:** PROD-01 + CORE-07 cards. *Interleave:* "Name a feature that is easy to compute in training and hard to compute at serving time."
- **Do:** [PROD-02](../lessons/production/02-mlops-in-practice.md), first half.
- **Project:** a training pipeline that logs runs and registers the model. The endpoint loads from the registry.

## Week 3: MLOps, part 2: CI, containers, tests
- **Warm-up:** PROD-02 + CORE-04 cards. *Interleave:* "Which test catches a silently changed feature order between training and serving?"
- **Do:** PROD-02, second half.
- **Project:** GitHub Actions runs unit tests, a data-schema check, and a "model beats baseline" gate.

## Week 4: Testing, monitoring & drift
- **Warm-up:** PROD-02 + CORE-11 cards. *Interleave:* "Your calibrated model drifts. Which breaks first: ranking (AUC) or calibration (ECE)?"
- **Do:** [PROD-03](../lessons/production/03-testing-monitoring-drift.md).
- **Lab:** [Lab 12](../labs/12-calibration-conformal-drift/README.md), part C (PSI by hand).
- **Playbook:** [the classifier two-sample test](../playbook/05-outside-the-box.md#2-a-classifier-is-a-two-sample-test).
- **Project:** a scheduled drift job. Simulate a shift and show it firing.

## Week 5: LLMOps
- **Warm-up:** PROD-03 + GEN-03 cards. *Interleave:* "PSI is 0.3 on a feature with low importance. Do you retrain?"
- **Do:** [PROD-05](../lessons/production/05-llmops-genai-platforms.md).
- **Project (if you did Course 3):** the eval gate in CI for your LLM assistant.

## Week 6: Distributed training
- **Warm-up:** PROD-05 + DL-07 cards. *Interleave:* "Gradient accumulation and data parallelism both enlarge the effective batch. What does each one cost?"
- **Do:** [PROD-04](../lessons/production/04-distributed-training.md).
- **Project:** the one-page "7B on 64 GPUs" plan from the lesson.

## Week 7: **Capstone**
- **Interleaved quiz:** 25 cards from PROD-01…05, CORE-07 and CORE-11.
- **Capstone:** the [path capstones](../paths/04-ml-in-production.md#capstones) (rubric below).

---

## Midterm checkpoint (end of week 3)

A **demo-day check:** a stranger clones the repo and gets a prediction with one command, CI is green, and the README has an architecture diagram.
No score. It either works or you fix it before week 4.

## Capstone rubric (week 7)

| Criterion | What "2" looks like |
|---|---|
| Reproducible training | One command; data and code versioned; the run is tracked |
| Serving | A containerized API with input validation, a health check, and the model version in the response |
| CI/CD | Tests for code, data schema and model quality gate every merge |
| Monitoring | Drift and data-quality checks on a schedule, with a simulated incident that was caught |
| Rollback | A documented and tested way back to the previous model |
| Design doc | Kept up to date; risks and their mitigations listed |

Pass: ≥ 9/12. Score your system against the ML Test Score rubric from PROD-03 as well.

## Assessment summary

| Component | Weight |
|---|---|
| Weekly self-checks from memory | 15% |
| Demo-day check | pass/fail gate |
| Capstone | 85% |
