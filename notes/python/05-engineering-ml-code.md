# PY-05 notes: Engineering ML Code: Projects, Tests & Reproducibility

[← Lesson PY-05](../../lessons/python/05-engineering-ml-code.md) · [All notes](../README.md) · [← PY-04 notes](04-eda-visualization.md) · Next: [MATH-01 notes →](../math/01-linear-algebra.md)

> **Reading time** ≈ 60 min. **You need:** [PY-01 notes §4–§6](01-python-for-ml-engineers.md#4-decorators-wrap-a-function-to-add-behaviour) (decorators, context managers, dataclasses) and [PY-03 notes §6](03-pandas-data-wrangling.md#6-time-features-without-leaking-the-future) (leak-free features).

---

## Where we are

PY-01 to PY-04 produced useful code: cleaning functions, features, an EDA audit. Right now it probably lives in notebook cells.
This note moves it into a package with tests, makes its results reproducible, and shows how to find what's slow. It's the minimum engineering that every lab, project and capstone in the rest of the repo assumes.

---

## 1. A project layout that scales from a weekend to a team

```text
my-project/
├── pyproject.toml          # package metadata + pinned dependencies (or requirements.txt / environment.yml)
├── README.md               # what it does, how to install, the one command that reproduces the results
├── src/my_project/         # importable code: data.py, features.py, train.py, evaluate.py
├── tests/                  # pytest tests, mirroring src/
├── notebooks/              # exploration only; they import from src/, they don't define logic
├── configs/                # experiment configs (YAML/JSON/dataclasses)
├── data/                   # NOT in git: raw/, interim/, processed/
└── outputs/                # NOT in git: models, figures, metrics (or use an experiment tracker)
```

Install the package in **editable mode** (`pip install -e .`), so notebooks and scripts import `my_project.features` from anywhere, and edits take effect immediately.
No `sys.path.append("..")`, and no copy-pasted functions drifting apart between notebooks.
The rule that keeps this clean: **notebooks call functions, they don't define the logic**. When a cell becomes useful, move it into `src/`, and write a test for it.

## 2. What to test in ML code

You can't assert "the model is accurate" exactly, but most ML bugs are in deterministic code around the model, and that code *can* be tested. Five kinds of test cover most of it:

| Kind | Example |
|---|---|
| **Known answer** | A metric on a 4-element input whose value you computed by hand |
| **Invariant / property** | Scaled features have mean 0; probabilities sum to 1; a transform is shape-preserving; shuffling the rows doesn't change a group mean |
| **Edge case** | Empty input, a single row, all-missing column, unseen category, constant feature |
| **Contract** | The function raises a clear error on bad input (wrong dtype, missing column, negative age) |
| **Smoke / regression** | Training for 1 epoch on 100 rows runs and beats a dummy baseline; a fixed seed reproduces a stored metric |

Here are tests for a small feature function, written as plain functions so they run anywhere. pytest discovers and runs functions named `test_*` in files named `test_*.py` in exactly this form:

```python
import hashlib
import cProfile
import io
import pstats
import numpy as np
import pandas as pd

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with income_per_member = income / household_size; household_size must be >= 1."""
    if "income" not in df or "household_size" not in df:
        raise KeyError("need columns income and household_size")
    if (df["household_size"] < 1).any():
        raise ValueError("household_size must be >= 1")
    out = df.copy()
    out["income_per_member"] = out["income"] / out["household_size"]
    return out

def test_known_answer():
    out = add_ratio_features(pd.DataFrame({"income": [100.0, 90.0], "household_size": [2, 3]}))
    assert out["income_per_member"].tolist() == [50.0, 30.0]

def test_does_not_mutate_input_and_keeps_rows():
    df = pd.DataFrame({"income": [1.0, 2.0, 3.0], "household_size": [1, 1, 1]})
    before = df.copy()
    out = add_ratio_features(df)
    pd.testing.assert_frame_equal(df, before)
    assert len(out) == len(df) and list(out.index) == list(df.index)

def test_edge_cases_and_contract():
    assert len(add_ratio_features(pd.DataFrame({"income": [], "household_size": []}))) == 0
    for bad, err in [(pd.DataFrame({"income": [1.0]}), KeyError),
                     (pd.DataFrame({"income": [1.0], "household_size": [0]}), ValueError)]:
        try:
            add_ratio_features(bad)
        except err:
            pass
        else:
            raise AssertionError(f"expected {err.__name__}")

for t in [test_known_answer, test_does_not_mutate_input_and_keeps_rows, test_edge_cases_and_contract]:
    t()
print("3 tests passed")
```

In a real `tests/test_features.py`, the contract test is shorter with `pytest.raises`, and shared inputs come from **fixtures**: functions that build test data once and are injected by name:

```py
import pytest
from my_project.features import add_ratio_features

@pytest.fixture
def households():
    return pd.DataFrame({"income": [100.0, 90.0], "household_size": [2, 3]})

def test_known_answer(households):
    assert add_ratio_features(households)["income_per_member"].tolist() == [50.0, 30.0]

def test_rejects_zero_size():
    with pytest.raises(ValueError, match=">= 1"):
        add_ratio_features(pd.DataFrame({"income": [1.0], "household_size": [0]}))
```

Every [lab](../../labs/README.md) in this repo is built this way, and its tests are worth reading as examples: they compare against `scikit-learn`, PyTorch autograd, finite differences, and hand-computed values.

## 3. Reproducibility: seeds are necessary, not sufficient

**Pass random generators explicitly.** A global seed at the top of a notebook is fragile: any extra random call anywhere (a new plot with jitter, a library that samples internally)
shifts every number drawn after it. Give each consumer its own seeded generator, or its own `random_state`:

```python
def split_global(n):
    return np.flatnonzero(np.random.rand(n) < 0.5)   # "sample about half the rows", using hidden global state

np.random.seed(0); a = split_global(10)
np.random.seed(0); _ = np.random.rand(); b = split_global(10)   # one innocent extra call...
print(a, b, "same split:", np.array_equal(a, b))

def split_explicit(n, rng):
    return np.flatnonzero(rng.random(n) < 0.5)

c = split_explicit(10, np.random.default_rng(0))
_ = np.random.rand()                               # unrelated global draws no longer matter
d = split_explicit(10, np.random.default_rng(0))
print(c, d, "same split:", np.array_equal(c, d))
```

With the global seed, one extra `np.random.rand()` call changes the sampled rows from `[4 6 9]` to `[3 5 8]`. With an explicit `default_rng(0)` passed in, both runs give `[1 2 3]`, whatever else draws from the global state.

**Make the train/test split stable as the data grows.** A random permutation re-deals every row when new data arrives, so yesterday's test rows can land in today's training set.
Assign each row by a **hash of its stable ID** instead: the same ID always lands on the same side.

```python
def in_test_set(row_id, test_fraction=0.2):
    h = int(hashlib.md5(str(row_id).encode()).hexdigest(), 16)
    return (h % 10_000) < test_fraction * 10_000

ids_v1 = range(1000)
ids_v2 = range(1500)                              # the dataset grew
test_v1 = {i for i in ids_v1 if in_test_set(i)}
test_v2 = {i for i in ids_v2 if in_test_set(i)}
print(len(test_v1), len(test_v2), "old test rows still in test:", test_v1 <= test_v2)
```

**210** of the first 1,000 IDs are in the test set (about 20%). After the data grows to 1,500, **every one of them is still there** (`True`), and the new rows are split in about the same proportion (314 in total).
Python's built-in `hash()` is randomized per process for strings, so use a real hash function such as `hashlib`.

**Seeds aren't the whole story.** Also pin the library versions (results change between releases), save the config and the git commit hash next to every result,
and remember that GPU kernels can be non-deterministic unless you ask for determinism (`torch.use_deterministic_algorithms(True)`, at some speed cost).

## 4. Profile before you optimize

Guessing which line is slow is usually wrong. `cProfile` counts calls and time per function, and `pstats` sorts them:

```python
def slow_feature(values):
    return [sum(values[max(0, i - 50):i + 1]) / len(values[max(0, i - 50):i + 1]) for i in range(len(values))]

def fast_feature(values):
    x = np.asarray(values, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    lo = np.maximum(0, idx - 50)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)

def pipeline(values):
    a = slow_feature(values)
    b = fast_feature(values)
    return np.allclose(a, b)

vals = list(np.random.default_rng(0).normal(size=20_000))
prof = cProfile.Profile()
prof.enable()
ok = pipeline(vals)
prof.disable()
stream = io.StringIO()
pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats()
ranked = [line.split("(")[-1].rstrip(")") for line in stream.getvalue().splitlines()
          if line.rstrip().endswith(("(slow_feature)", "(fast_feature)"))]
print("results agree:", ok, "| ranked by cumulative time:", ranked)
```

Both versions compute the same trailing-window mean (`results agree: True`), and the profile ranks `slow_feature` above `fast_feature` by cumulative time. That's where to spend effort.
For line-level detail, use `line_profiler`. For memory, `tracemalloc` or `memray`. For PyTorch, `torch.profiler` ([DL-07](../../lessons/deep-learning/07-performance-gpus-mixed-precision.md)).

## 5. Git for ML projects

- **Commit code, configs and small metadata. Don't commit data, models or outputs.** A starting `.gitignore`: `data/`, `outputs/`, `*.ckpt`, `*.pt`, `.ipynb_checkpoints/`, `__pycache__/`, `.env`.
  Version data with DVC, a data lake with snapshots, or at least a checksum and a path recorded in the README.
- **Small commits with messages that say why** ("features: drop leaky store_mean, use past-only mean"). One experiment per branch is fine.
- **Notebooks diff badly.** Clear outputs before committing, or pair them with scripts (jupytext), and keep the logic in `src/` anyway.
- **Never commit secrets.** Keep API keys in environment variables or a `.env` file that is git-ignored.

## 6. A pre-commit checklist

1. Tests pass (`pytest`), including the new test for what you just changed.
2. Code is formatted and linted (`ruff format`, `ruff check`), so diffs show logic, not whitespace.
3. The one command in the README still reproduces the result from a clean clone.
4. No data, models, secrets or notebook outputs in the diff.
5. The commit message says what changed and why.

---

## Pitfalls & misconceptions

- **Logic defined in notebooks.** Copies drift apart and hidden state (cells run out of order) makes results irreproducible. Move the logic to `src/`, and import it.
- **"I can't test ML code."** Most bugs are in deterministic data and feature code. Test known answers, invariants, edge cases and contracts, plus a smoke test that the model beats a dummy baseline.
- **One global seed means reproducibility.** Extra random calls shift everything downstream, and library versions, GPU kernels and data order matter too. Pass explicit generators, and pin versions.
- **Random splits on growing data** leak old test rows into training. Use hash-based or time-based splits.
- **Optimizing without profiling** speeds up the wrong thing. Profile, fix the top entry, and profile again.
- **Committing data or secrets.** Git keeps them in the history forever, even after you delete the file. Ignore them from the start.

## Cheat sheet

| Need | Tool / habit |
|---|---|
| Importable code | `src/` layout + `pip install -e .` |
| Tests | `pytest`: `test_*.py`, plain `assert`, `pytest.raises`, fixtures, `tmp_path` |
| What to test | Known answers, invariants, edge cases, contracts, smoke/regression |
| Randomness | `rng = np.random.default_rng(seed)` passed in; `random_state=` everywhere |
| Stable splits | Hash of a stable ID (`hashlib`), or time-based |
| Reproducible runs | Pinned environment, saved config + git hash, one command |
| Profiling | `cProfile` + `pstats`; `line_profiler`; `tracemalloc`; `torch.profiler` |
| Hygiene | `.gitignore` data/outputs/secrets; formatter + linter; small commits |

## Answer sketches for the lesson's self-check

<details>
<summary>1. <code>pip install -e .</code> vs <code>sys.path.append("..")</code></summary>

An editable install makes the package importable from any working directory, with the right name, its dependencies declared, and edits picked up immediately.
Path hacks depend on where the notebook was started, break in scripts and CI, and encourage copy-pasting (§1).
</details>

<details>
<summary>2. Three kinds of test for ML code</summary>

Any three of: known-answer tests on tiny inputs; invariants (shape preserved, probabilities sum to 1, no mutation of the input); edge cases (empty, single row, unseen category);
contracts (clear errors on bad input); smoke/regression tests (one epoch runs and beats a dummy baseline; a fixed seed reproduces a stored metric) (§2).
</details>

<details>
<summary>3. Seeded at the top, yet results change</summary>

(1) Extra or reordered random calls consume the global stream, which shifts later draws (§3). (2) Libraries with their own generators, or a missing `random_state` (scikit-learn, PyTorch, Python's `random`).
(3) Non-deterministic sources: GPU kernels, multithreaded reductions, data-loading order, unpinned library versions, or data that changed underneath you.
</details>

<details>
<summary>4. What not to commit</summary>

Raw and processed data, trained models and checkpoints, generated outputs and figures, notebook outputs, environment folders, and secrets.
Data goes to storage with versioning (DVC, snapshots, checksums), models and metrics go to an experiment tracker or registry, and secrets go to environment variables or a git-ignored `.env` (§5).
</details>

<details>
<summary>5. Before optimizing a slow training script</summary>

Profile it (`cProfile`/`pstats`, or `torch.profiler` for GPU work), and look at the functions with the largest cumulative time and call counts: often data loading, Python loops in feature code, or repeated host↔device copies, rather than the model (§4).
</details>

<details>
<summary>6. What a pytest fixture is for</summary>

It builds shared test inputs (or resources) once and injects them into tests by parameter name, with optional setup and teardown. An ML example: a small, seeded DataFrame or a tiny trained model reused by many tests, or `tmp_path` for a temporary directory to save and reload a model (§2).
</details>

<details>
<summary>7. (debug) A colleague gets different numbers from the same code</summary>

In order: the same data (checksums, row counts, date of the extract)? The same library versions (diff the `pip freeze` output)? The same config and code commit?
Explicit seeds for every source of randomness, including the data split (§3)? Notebook hidden state (rerun from a fresh kernel, top to bottom)? Hardware non-determinism (GPU vs CPU)?
</details>

## Where this leads

Next: [MATH-01 notes](../math/01-linear-algebra.md). The Python foundations are in place. The route continues with the just-in-time math, and then Path 1 starts the ML itself with [CORE-01](../core-ml/01-ml-workflow-end-to-end.md),
where the habits from this track (leak-free features, honest splits, tested pipelines) carry straight over.
