# PROD-02: MLOps in Practice

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Production | ~12 h (spread over 2–3 weeks) | L2 | PROD-01, basic Docker & Git |

## Why this matters
MLOps is the engineering discipline that makes ML repeatable: experiment tracking, reproducible pipelines, testing,
model serving, CI/CD, and monitoring. This lesson is deliberately hands-on. You'll ship something.

## Learning goals
By the end you can:
- Track experiments and register models (e.g. MLflow).
- Turn a notebook into a tested, packaged, reproducible training pipeline, with orchestration.
- Serve a model behind an API, and containerize it.
- Set up monitoring for data drift and model performance.
- Explain CI/CD and continual training for ML systems.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read + Build** | [Made With ML](https://madewithml.com/) ([repo](https://github.com/GokuMohandas/Made-With-ML)) | Work through the course's develop → deploy → iterate flow: data, training, tracking, testing, serving, CI/CD | 6 h |
| 2 | **Build** | [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) (DataTalks.Club) | Modules on experiment tracking, orchestration, deployment, and monitoring. Do the homework for at least 2 of them. | 6 h |

**Notes for the learner:** You don't have to finish both. Made With ML teaches the *principles* with a clean
reference project. The Zoomcamp gives you *reps* with different tools. If time is short, do Made With ML fully and
the Zoomcamp's monitoring module.

## Check your understanding
1. What's the minimum you need to log so an experiment is reproducible?
2. What kinds of tests does an ML system need, beyond unit tests of the code? (Think data, model, and behaviour.)
3. What triggers a retrain in a continual-training setup, and what gates a new model's promotion?
4. *(debug)* Drift monitoring alerts on a feature, but model performance hasn't changed. Do you retrain? Why or why not?

## Mini-project (capstone for the Production path)
**Task:** Take one of your earlier models (e.g. CORE-05 or DL-04) to "production": a tracked training pipeline, a model registry,
a FastAPI service in Docker, a GitHub Actions workflow that runs tests, and a drift report comparing training data with new data.
**Deliverable:** A public repo with a README architecture diagram.

## Go deeper
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) `[paid]`: Ch. 9 (continual learning and testing in production) and Ch. 10 (infrastructure and tooling).
- [learnpytorch.io 09: Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/): a lightweight deployment demo.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md), for every concept in this lesson, with alternatives.
- **Papers:** [ML in production](../../papers/11-ml-in-production.md). Start with the ⭐ ones.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
