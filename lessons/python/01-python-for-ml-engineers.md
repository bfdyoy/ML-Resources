# PY-01: Python for ML Engineers

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Python | ~6 h | L1→L2 | You can write functions, loops and classes in Python |

## Why this matters
ML code fails in Python-specific ways long before it fails in math-specific ways: a mutable default argument that remembers last call's data, a "copy" that is
really a shared reference, a 20 GB file loaded into memory when a generator would stream it. The Python features that ML libraries are built from
(iterators, decorators, context managers, dataclasses, the data model behind `len()` and `[]`) are also what make PyTorch's `Dataset`, `nn.Module`, `torch.no_grad()` and scikit-learn's API make sense.

## Learning goals
By the end you can:
- **Explain** Python's name-binding model (names point to objects; mutation vs rebinding) and predict when two variables share data.
- **Write** generators and use `itertools` to stream data lazily instead of materializing lists.
- **Implement** a decorator (timing, caching) and a context manager (timing, temporarily setting state), and recognize both in ML libraries.
- **Use** `dataclasses` and type hints to make configs and records explicit.
- **Implement** the special methods (`__len__`, `__getitem__`, `__iter__`, `__call__`) that make your objects work like PyTorch datasets and modules.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Python for ML engineers](../../notes/python/01-python-for-ml-engineers.md) | Names and objects, iterators and generators, closures and decorators, context managers, dataclasses, the data model, with runnable examples and answers to the questions below. Read it first. | ~1 h |
| 1 | **Intuition** | [Python Like You Mean It](https://www.pythonlikeyoumeanit.com/) (Soklaski) | Module 2 "Essentials of Python": the sections on iterables, generators and comprehensions, and functions. Written for scientists who will use NumPy next. | 45 min |
| 2 | **Read** | [The Python Tutorial](https://docs.python.org/3/tutorial/), [§9 Classes](https://docs.python.org/3/tutorial/classes.html) | §9.1–9.3 (scopes and namespaces: the name-binding model), then §9.8 Iterators, §9.9 Generators, §9.10 Generator expressions | 60 min |
| 3 | **Read** | [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html) | "Iterators", "Generator expressions and list comprehensions", "Generators", and "The itertools module" | 45 min |
| 4 | **Read** | [*Python for Data Analysis*, 3rd ed.](https://wesmckinney.com/book/) (McKinney, free online) | Ch. 3 "Built-in Data Structures, Functions, and Files": skim the data structures, read the parts on functions, generators, and errors carefully | 45 min |
| 5 | **Build** | The notes' code cells + the mini-project below | Re-type the examples, then change them until they break | 90 min |

**Notes for the learner:** if most of this is familiar, do the self-check first and read only the notes sections you miss. The point is not syntax trivia:
it's to make the libraries you'll use (PyTorch's `Dataset`/`DataLoader`, `torch.no_grad()`, `functools.lru_cache`, scikit-learn estimators) feel like ordinary Python.

## Check your understanding
1. `a = [1, 2, 3]; b = a; b.append(4)`. What is `a`, and why? What would change if the second statement were `b = a + [4]`?
2. Why is `def f(x, cache={})` a bug in most code, and when is it a (deliberate) trick?
3. What's the memory difference between `sum([x * x for x in range(10**8)])` and `sum(x * x for x in range(10**8))`? What does a generator give up in exchange?
4. Write a decorator `@timed` that prints how long a function call took. Why does it need `functools.wraps`?
5. `torch.no_grad()` is a context manager. Sketch how you'd write a context manager that sets a global flag on entry and *always* restores it on exit, even after an exception.
6. Which special methods must a class implement so that `len(ds)`, `ds[3]` and `for x in ds` all work? Which of them does a PyTorch map-style `Dataset` need?
7. *(debug)* A data-cleaning function gives correct results the first time it's called in a notebook, and wrong results on the second call with the same input. Name two likely causes, and how you'd check each.

## Mini-project
**Task:** write a small, typed, streaming data reader: a `@dataclass` config (path, batch size, columns), a generator that yields batches of parsed rows from a large CSV
without loading it all, a `@timed` decorator on the processing function, and a `Dataset`-like class with `__len__` and `__getitem__` over an in-memory sample.
**Dataset:** any CSV over 100 MB (e.g. a year of a public taxi-trips or bike-sharing dataset), or generate one with NumPy.
**Deliverable:** a `.py` module plus a notebook that shows the memory use of the streaming reader (it stays flat) against `pd.read_csv` on the full file.

## Go deeper
- [Python Like You Mean It](https://www.pythonlikeyoumeanit.com/), Module 4 "Object Oriented Programming": special methods and inheritance, with exercises.
- [CS50's Introduction to Programming with Python](https://cs50.harvard.edu/python/syllabus/): the problem sets for "Exceptions", "Libraries", "Unit Tests" and "Object-Oriented Programming", if you want auto-graded practice.
- [Kaggle Learn: Python](https://www.kaggle.com/learn/python): quick browser exercises, useful as a warm-up if any basics are rusty.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) ("Python, NumPy & pandas foundations").
- **Papers:** none. This lesson is engineering, not research.
- **Implement it yourself:** the [Course 0 syllabus](../../courses/00-python-for-ml.md) (week 1), and the [labs](../../labs/README.md), which all run on these features.
- **Drills:** [Kaggle Learn: Python](https://www.kaggle.com/learn/python) exercises · more in [exercises/](../../exercises/README.md).
