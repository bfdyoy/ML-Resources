# PY-05: Engineering ML Code: Projects, Tests & Reproducibility

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Python | ~6 h | L1→L2 | PY-01…PY-04 |

## Why this matters
Notebooks are great for exploring and terrible as the only home of your code: hidden state, out-of-order cells, copy-pasted functions, no tests.
Research code that can't be rerun can't be trusted, and production ML (Path 4) assumes you already package, test and version your work.
This lesson is the bridge: the habits that make every later lab, project and capstone reproducible.

## Learning goals
By the end you can:
- **Structure** an ML project (`src/` package, `notebooks/`, `tests/`, `data/` kept out of git, a pinned environment) and install it in editable mode.
- **Write** `pytest` tests for data transforms and model code: shape and dtype checks, invariants, small known-answer cases, and fixtures.
- **Make** runs reproducible: seeds, pinned dependencies, configs saved with results, and deterministic data splits.
- **Profile** code to find the real bottleneck before optimizing it.
- **Use** git for experiments (small commits, branches, `.gitignore` for data and artifacts).

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: engineering ML code](../../notes/python/05-engineering-ml-code.md) | Project layout, testing patterns for ML (with runnable pytest-style tests), seeding and determinism, profiling, and a pre-commit checklist. Read it first. | ~1 h |
| 1 | **Intuition** | [Software Carpentry: Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/) | The episodes "Defensive Programming" and "Debugging": assertions, test-driven habits, and a debugging procedure, on a small dataset | 40 min |
| 2 | **Read** | [The Good Research Code Handbook](https://goodresearch.dev/) (Mineault, free) | "Setting up your project", "Keeping things tidy", "Writing decoupled code", and "Testing your code" | 1.5 h |
| 3 | **Read** | [pytest: Get Started](https://docs.pytest.org/en/stable/getting-started.html) | The whole page (test discovery, assertions, `pytest.raises`, fixtures via `tmp_path`) | 30 min |
| 4 | **Read** | [The Missing Semester](https://missing.csail.mit.edu/) (MIT) | The lectures "Version Control (Git)" and "Debugging and Profiling" (read the notes; the videos are optional) | 1 h |
| 5 | **Build** | The mini-project below | Package your Course 0 work with tests | 1.5 h |

**Notes for the learner:** you'll use all of this immediately: every [lab](../../labs/README.md) in this repo is a small package with `pytest` tests, and the
Path 4 capstone grades reproducibility. Start small: one `src/` package, three tests, one `requirements.txt`, and one command that reproduces your result.

## Check your understanding
1. Why does `pip install -e .` beat `sys.path.append("..")` in notebooks?
2. Name three kinds of test that make sense for ML code, given that you can't assert "the model is accurate" exactly.
3. You set `np.random.seed(0)` at the top of a notebook, yet results change between runs. Give three reasons.
4. What should *not* go into git in an ML project, and where does it go instead?
5. Your training script is slow. What do you run before changing any code, and what do you look for in its output?
6. What is a pytest fixture for? Give an ML example.
7. *(debug)* A colleague can't reproduce your notebook's result: different numbers, same code. List the checks you'd go through, in order.

## Mini-project
**Task:** turn your Course 0 work (cleaning, features, EDA) into a small repo: `src/<name>/` with the cleaning and feature functions, `tests/` with at least five
`pytest` tests (including one edge case and one `pytest.raises`), a pinned environment file, a `Makefile` or script that reproduces the report with one command, and a README.
**Dataset:** your Course 0 dataset.
**Deliverable:** a public (or private) git repo that a stranger can clone, install, and rerun. This is the Course 0 mini-capstone ([rubric](../../courses/00-python-for-ml.md#mini-capstone-rubric)).

## Go deeper
- [CS50's Introduction to Programming with Python](https://cs50.harvard.edu/python/syllabus/), lecture and problem set "Unit Tests".
- [The Good Research Code Handbook](https://goodresearch.dev/): "Writing good documentation" and "Making coding social".
- [Made With ML](https://madewithml.com/): its lessons on developing and testing ML code carry these habits into a full ML system (Path 4).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 11: MLOps & systems](../../toolbox/11-mlops-and-systems.md) ("The MLOps lifecycle").
- **Papers:** none.
- **Implement it yourself:** read `scripts/make_lab_stubs.py` and any lab's `test_lab*.py` in [labs/](../../labs/README.md), which use the same patterns.
- **Drills:** write the tests first for one function of your next lab, CS336-style, before implementing it.
