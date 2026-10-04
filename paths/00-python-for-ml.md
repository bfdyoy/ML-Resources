# Path 0: Python for ML (optional on-ramp)

**Goal:** Make Python, NumPy and pandas fluent enough that they stop being the hard part, so that Path 1 can be about ML rather than about shapes, merges, and notebooks that don't rerun.
**Duration:** ~32 h (≈ 5 weeks at 6–7 h/week) · **Level:** L1→L2 · **Primary books:** *Python for Data Analysis* (McKinney, free online) + *From Python to NumPy* (Rougier, free) + *Python Data Science Handbook* (VanderPlas, free)

> 📘 **Study notes:** every lesson starts with its written explanation (step 0, ~1 h): the ideas, worked examples, runnable code, and answers to the self-check. Read in order, the notes form one continuous walkthrough: [notes index](../notes/README.md).
>
> 🗓️ **As a course:** [Course 0: Python for ML](../courses/00-python-for-ml.md) turns this path into a 5-week plan with a running dataset, labs, warm-ups, and a mini-capstone.

> **Skip test:** if you can answer ≥ 80% of each lesson's *Check your understanding* from memory, and [Lab 01](../labs/01-numpy-vectorization/README.md) and [Lab 02](../labs/02-pandas-wrangling/README.md)
> pass on your first attempt, go straight to [Path 1](01-core-ml-practitioner.md).

| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [PY-01 Python for ML Engineers](../lessons/python/01-python-for-ml-engineers.md) | Predict shared-data bugs; stream data with generators; write decorators, context managers and dataclasses; implement the `Dataset` protocol | 6 h |
| 2 | [PY-02 NumPy & Vectorized Thinking](../lessons/python/02-numpy-vectorized-thinking.md) | Replace loops with broadcasting, reductions, masks and `einsum`; write numerically stable array code | 7 h |
| 3 | [PY-03 pandas & Data Wrangling](../lessons/python/03-pandas-data-wrangling.md) | Group, merge, and reshape safely; build time features that don't leak | 7 h |
| 4 | [PY-04 EDA & Visualization](../lessons/python/04-eda-visualization.md) | Audit a dataset, choose the right chart, and avoid misleading statistics and plots | 6 h |
| 5 | [PY-05 Engineering ML Code](../lessons/python/05-engineering-ml-code.md) | Package, test, profile, and reproduce your ML code | 6 h |

## Capstone
**A reproducible mini-analysis repo** on one public dataset you'll keep using in Path 1: a `src/` package with the cleaning and feature code, `pytest` tests,
an EDA report with 6 plots (each with a question and a one-sentence answer), a pinned environment, and one command that reproduces everything.
The rubric is in [Course 0](../courses/00-python-for-ml.md#mini-capstone-rubric).

**Next:** [Path 1: Core ML Practitioner](01-core-ml-practitioner.md).
