# PROD-01: ML System Design

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Production | ~6 h | L2 | CORE-01…07 (DL optional) |

## Why this matters
A model in a notebook creates no value. ML system design is about everything *around* the model: data pipelines,
labels, features, training cadence, serving, monitoring, and feedback loops. These decide whether an ML product works.
It's also a standard interview topic.

## Learning goals
By the end you can:
- Frame a business problem as an ML system: objectives, constraints (latency, cost, privacy), and success metrics.
- Design the data side: sources, labeling, sampling, class imbalance, and train/serve feature consistency.
- Choose between batch and online prediction, and understand the trade-offs of model compression and edge deployment.
- Explain data distribution shift (covariate, label, concept), and how to detect it.
- Apply Google's "Rules of ML" heuristics (e.g. start with simple heuristics, get the pipeline right first).

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Zinkevich, Google) | All 43 rules. Read "Before Machine Learning" and "Phase I" twice. | 1.5 h |
| 2 | **Read** | [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) (Chip Huyen) `[paid]`, or the free [chapter summaries](https://github.com/chiphuyen/dmls-book) | Ch. 2 (intro to ML systems design), Ch. 4 (training data), Ch. 7 (deployment), Ch. 8 (data distribution shifts and monitoring) | 3 h (book) / 1 h (summaries) |
| 3 | **Watch/Read** | [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | Lectures "Development Infrastructure & Tooling", "Deployment", and "Continual Learning". Read the lecture notes on the course site. | 1.5 h |

**Free-only route:** If you skip the paid book, read the DMLS summaries + FSDL notes + the Rules of ML. That covers
about 80% of the concepts.

## Check your understanding
1. Why is "start with a heuristic" often the right first step, even for an ML team?
2. What's training-serving skew? Give two concrete ways it happens.
3. Batch vs online prediction: what do you gain and lose with each?
4. How would you detect covariate shift in production *without* labels?
5. *(design)* Sketch the system for a "similar products" recommender for an online shop: data, model, serving, monitoring, and feedback.

## Mini-project
**Task:** Write a 2-page design doc for an ML system you'd actually like to build. Use this structure: problem → metrics →
data → features → model baseline → serving → monitoring → risks.
**Deliverable:** A markdown design doc (a good portfolio piece).

## Go deeper
- [Google MLCC](https://developers.google.com/machine-learning/crash-course): the "Production ML systems" module
- [Made With ML](https://madewithml.com/): the "Design" section
