# PROD-03: Testing, Monitoring & Drift in Production ML

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Production | ~7 h | L2 | PROD-01, PROD-02 |

## Why this matters
Models fail silently. The data shifts, an upstream pipeline changes a unit, or a feedback loop drags performance down, and nothing
crashes. Testing ML systems (data, model, and code) and monitoring them in production is how you find out before your users or your
revenue do.

## Learning goals
By the end you can:
- Apply the ML Test Score rubric to a system, covering data tests, model tests, infrastructure tests, and monitoring tests.
- Write data validation checks (schemas, ranges, distributions) and model behaviour tests (invariance, directional, minimum-functionality).
- Tell covariate shift, label shift, and concept drift apart, and detect each, with and without labels.
- Choose monitoring metrics and alert thresholds that are actionable rather than noisy.
- Decide when to retrain: scheduled, triggered, or continual.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) | The whole paper (it's short): all 28 tests | 1 h |
| 2 | **Read** | [Designing ML Systems: summaries](https://github.com/chiphuyen/dmls-book) | Ch. 8 (data distribution shifts and monitoring) and Ch. 9 (continual learning and testing in production) | 1 h |
| 3 | **Read + Build** | [Evidently: ML Observability course](https://www.evidentlyai.com/ml-observability-course) | The modules on ML monitoring metrics, data quality, and drift detection | 2 h |
| 4 | **Build** | [Made With ML](https://madewithml.com/) | The testing lesson (code, data, and model tests with pytest) | 1.5 h |
| 5 | **Build** | [Evidently](https://github.com/evidentlyai/evidently) | A drift report on your PROD-02 model with a synthetic shift, plus a test suite that fails the build on drift | 1.5 h |

## Check your understanding
1. Give one example each of covariate shift, label shift, and concept drift in a fraud-detection system.
2. How can you monitor model quality when labels arrive weeks later?
3. Why do statistical drift tests fire constantly on large datasets, and how do you make alerts actionable?
4. What is an invariance test for a sentiment model? And a directional test?
5. What is training–serving skew, and what test catches it?
6. *(debug)* Drift alerts are firing on 12 features at once after a deployment, but the model's accuracy is unchanged. What's your first hypothesis?

## Mini-project
**Task:** Score your PROD-02 system against the ML Test Score (be honest), then implement the three missing tests with the highest value.
Add a daily drift and data-quality job, and simulate a shift to show it catching the problem.
**Deliverable:** The rubric table (before/after), the tests in CI, and a drift dashboard screenshot.

## Go deeper
- [Hidden Technical Debt in ML Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) · [Challenges in Deploying ML](https://arxiv.org/abs/2011.09926) · [Operationalizing ML](https://arxiv.org/abs/2209.09125)
- [MIT DCAI](https://dcai.csail.mit.edu/): the "Class Imbalance, Outliers, and Distribution Shift" lecture.
- [Google Cloud: MLOps pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning): continuous training.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md) (the MLOps lifecycle).
- **Papers:** [ML in production](../../papers/11-ml-in-production.md). Start with the ⭐ ones.
- **Implement it yourself:** a population-stability-index (PSI) and KS-test drift checker in NumPy. Compare it with Evidently's output.
- **Drills:** [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) monitoring homework · more in [exercises/](../../exercises/README.md).
