# PY-04: Exploratory Data Analysis & Visualization

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Python | ~6 h | L1→L2 | PY-03 |

## Why this matters
EDA is where you find the leak, the broken sensor, the label that means something different after a policy change, and the feature that will carry the model.
Good plots answer a specific question; bad ones decorate a report. Knowing which chart answers which question (and which charts mislead) is a core ML skill,
and it's the first step of Karpathy's recipe: "become one with the data".

## Learning goals
By the end you can:
- **Run** a structured EDA: data dictionary, missingness, distributions, relationships with the target, time behaviour, and data-quality checks.
- **Choose** the right chart for the question (distribution, comparison, relationship, composition, change over time).
- **Recognize** misleading plots (truncated axes, overplotting, area distortion, bad colour scales) and fix them.
- **Use** summary statistics robustly (median/IQR vs mean/std, correlation vs dependence, Simpson's paradox).
- **Write** each plot's takeaway in one sentence, and turn surprises into checks or features.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: EDA & visualization](../../notes/python/04-eda-visualization.md) | An EDA checklist, robust statistics, the "same statistics, different data" trap, Simpson's paradox, and the matplotlib object model, with runnable examples. Read it first. | ~1 h |
| 1 | **Intuition** | [*Fundamentals of Data Visualization*](https://clauswilke.com/dataviz/) (Wilke, free) | Ch. 5 "Directory of visualizations": a visual menu of chart types by question | 20 min |
| 2 | **Read** | [*Fundamentals of Data Visualization*](https://clauswilke.com/dataviz/) | Ch. 7 "Visualizing distributions: histograms and density plots", Ch. 12 "Visualizing associations among two or more quantitative variables", Ch. 17 "The principle of proportional ink", Ch. 18 "Handling overlapping points", Ch. 19 "Common pitfalls of color use" | 2 h |
| 3 | **Read** | [*Python Data Science Handbook*](https://jakevdp.github.io/PythonDataScienceHandbook/) | Ch. 4 "Visualization with Matplotlib": the introduction (figures vs axes, the two interfaces), simple line and scatter plots, histograms, and subplots | 1 h |
| 4 | **Build** | [Kaggle Learn: Data Visualization](https://www.kaggle.com/learn/data-visualization) + the mini-project | Do the exercises that are new to you, then the EDA report on your dataset | 2 h |

**Notes for the learner:** use matplotlib's object-oriented interface (`fig, ax = plt.subplots()`) from the start. Every plot in your report should come with the
*question* it answers and the *answer* in one sentence. If you can't write that sentence, the plot doesn't belong in the report.

## Check your understanding
1. Name four datasets with identical means, variances and correlation that look completely different. What does that teach about summary statistics?
2. Your target is heavily right-skewed. Which summary statistics and which plots do you use, and why?
3. Over all customers, the new feature looks *negatively* correlated with churn, but within every region it's positive. What is this called, and what do you do?
4. When does a scatter plot of 1,000,000 points mislead, and what are three fixes?
5. A bar chart's y-axis starts at 95 instead of 0. When is that misleading, and when is a non-zero baseline fine?
6. What are the first five checks you run on a new dataset before plotting anything?
7. *(debug)* A feature's histogram has a huge spike at exactly 0 (or -1, or 999). Name two likely explanations and how you'd confirm each.

## Mini-project
**Task:** a 6-plot EDA report on your running dataset. Each plot gets a question, a one-sentence answer, and a follow-up (a data check to add, a feature idea, or a question for a domain expert).
Include one plot of the target over time and one of missingness.
**Dataset:** your Course 0 dataset.
**Deliverable:** a notebook (or a rendered HTML report) a colleague could read in 10 minutes.

## Go deeper
- [*Fundamentals of Data Visualization*](https://clauswilke.com/dataviz/), Ch. 9 "Visualizing many distributions at once" (boxplots, violins, ridgelines) and Part II "Principles of figure design".
- [*Python for Data Analysis*](https://wesmckinney.com/book/), Ch. 9 "Plotting and Visualization" (matplotlib and seaborn from pandas).
- [Scientific Python Lectures](https://lectures.scientific-python.org/index.html): "Matplotlib: plotting".

## Math refresher
- [MATH-03 Probability & statistics](../math/03-probability-statistics.md), block A (summary statistics and distributions).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) ("Problem framing & data work").
- **Papers:** none.
- **Implement it yourself:** the [Course 0 syllabus](../../courses/00-python-for-ml.md), week 4.
- **Drills:** [Kaggle Learn: Data Visualization](https://www.kaggle.com/learn/data-visualization) · more in [exercises/](../../exercises/README.md).
