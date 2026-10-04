# Course 0: Python for ML, a 5-week syllabus

[← Courses](README.md) · Path: [Path 0: Python for ML](../paths/00-python-for-ml.md) · Labs: [index](../labs/README.md) · Cards: [python deck](flashcards/README.md)

> **Pace** ≈ 7 h/week for 5 weeks. **Who it's for:** you can write Python scripts, but NumPy broadcasting, `groupby().transform`, and writing tests still feel shaky.
> **Skip test:** if you can answer ≥ 80% of each lesson's *Check your understanding* from memory, and Labs 01–02 pass on your first try, go straight to [Course 1](01-core-ml.md).
> **Running dataset:** one dataset threaded through all five weeks (the Software Carpentry approach). Pick one public table with dates, categories, numbers and some missing values. A bike-sharing, taxi-trip, or energy-use dataset works well.

The [weekly loop](README.md#32-the-weekly-loop-about-8-hours) applies every week.

## How this course is shaped

- **Short read → do cycles** (Kaggle Learn, CS50P). Every note section ends in a cell to run and modify. Every lab runs in seconds and tells you exactly which test fails.
- **One dataset all the way through** (Software Carpentry's inflammation data). The data stays familiar, so all your attention goes to the *technique*.
- **Vectorized thinking is the core skill** (Rougier's *From Python to NumPy*, VanderPlas's *Python Data Science Handbook*). Week 2 is about replacing loops with array expressions, and checking that both give the same answer.
- **Engineer from the start** (MIT's *Missing Semester*, CS50P's check50). Week 5 turns the notebook into a package with tests. It's the bridge to the [labs](../labs/README.md), which all run on `pytest`.

---

## Week 1: Python for ML engineers
- **Whole game (1 h):** load your dataset with pandas, make one plot, fit `LinearRegression` on two columns, and print the score. Thirty lines, copied and adapted. Don't polish anything.
- **Do:** [PY-01](../lessons/python/01-python-for-ml-engineers.md) with its [notes](../notes/python/01-python-for-ml-engineers.md).
- **Explain it back:** what a generator saves you when you stream a 20 GB file.

## Week 2: NumPy and vectorized thinking
- **Warm-up:** PY-01 cards. *Interleave:* "A function with a mutable default argument `def f(x, seen=[])`: what goes wrong on the second call?"
- **Do:** [PY-02](../lessons/python/02-numpy-vectorized-thinking.md).
- **Lab:** [Lab 01: vectorization](../labs/01-numpy-vectorization/README.md).
- **Project:** rewrite one loop over your dataset as a vectorized expression. Time both, and assert they agree.

## Week 3: pandas and data wrangling
- **Warm-up:** PY-02 + PY-01 cards. *Interleave:* "Arrays of shape (3,1) and (4,): what shape is their sum?"
- **Do:** [PY-03](../lessons/python/03-pandas-data-wrangling.md).
- **Lab:** [Lab 02: pandas wrangling](../labs/02-pandas-wrangling/README.md).
- **Project:** a cleaning script that turns your raw file into a tidy table, with a one-line check per assumption (`assert df.id.is_unique`, and so on).

## Week 4: EDA and visualization
- **Warm-up:** PY-03 + PY-02 cards. *Interleave:* "`groupby().agg` vs `groupby().transform`: which one keeps the original number of rows?"
- **Do:** [PY-04](../lessons/python/04-eda-visualization.md).
- **Project:** an EDA report: 6 plots, each with a one-sentence takeaway and the question it raised.

## Week 5: Engineering ML code + **mini-capstone**
- **Warm-up:** PY-04 + PY-03 cards. *Interleave:* "A merge silently doubled your row count. Which argument would have caught it?"
- **Do:** [PY-05](../lessons/python/05-engineering-ml-code.md).
- **Mini-capstone (rubric below):** the week 1–4 work as a small repo: `src/` package, tests, pinned environment, one command to reproduce the report.

---

## Mini-capstone rubric

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Structure | One notebook | Notebook + helper module | `src/` package, notebook only calls functions |
| Correctness checks | None | A few asserts | `pytest` tests for cleaning and features, including one edge case |
| Vectorization | Row loops | Mostly vectorized | No Python loops over rows; a timing comparison included |
| Reproducibility | Doesn't rerun | Reruns by hand | Pinned environment + one command + fixed seeds |
| Communication | No text | Plots only | Each plot has a takeaway; the README says what to look at first |

Pass: ≥ 7/10.
