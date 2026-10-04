# PY-02: NumPy & Vectorized Thinking

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Python | ~7 h | L1→L2 | PY-01 |

## Why this matters
Every ML library speaks arrays. Thinking in whole-array operations (broadcasting, reductions, boolean masks, fancy indexing) instead of Python loops
makes code 10–1000× faster *and* shorter, and it is exactly how you'll read and write PyTorch tensor code, attention masks, and loss functions later.
Most "shape errors" in deep learning are NumPy misunderstandings in disguise.

## Learning goals
By the end you can:
- **Explain** what an array is in memory (a buffer plus dtype, shape and strides) and predict when an operation returns a **view** vs a **copy**.
- **Apply** the broadcasting rules to predict the result shape of any elementwise operation, and use `None`/`np.newaxis` to line up axes on purpose.
- **Replace** loops with ufuncs, reductions along an axis (`keepdims`), boolean masks, fancy indexing, `cumsum` tricks and `einsum`.
- **Write** numerically stable versions of common ML functions (softmax, log-sum-exp, pairwise distances).
- **Measure** a speed-up honestly (`%timeit`), and check that the vectorized result equals the loop version.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: NumPy & vectorized thinking](../../notes/python/02-numpy-vectorized-thinking.md) | Memory model, broadcasting rules with worked shapes, axis reductions, masks and fancy indexing, stability tricks, `einsum`, with timed demos. Read it first. | ~1 h |
| 1 | **Intuition** | [NumPy manual: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) | The whole page: the two rules and the figures of stretched arrays | 20 min |
| 2 | **Read** | [*From Python to NumPy*](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) (Rougier, free) | Ch. "Anatomy of an array" (memory layout, views and copies) and Ch. "Code vectorization" (uniform, temporal and spatial vectorization: Game of Life, Mandelbrot) | 2 h |
| 3 | **Read** | [*Python Data Science Handbook*](https://jakevdp.github.io/PythonDataScienceHandbook/) (VanderPlas, free) | Ch. 2 "Introduction to NumPy": the sections on computation with ufuncs, aggregations, broadcasting, comparisons/masks, fancy indexing, and sorting | 1.5 h |
| 4 | **Build** | [numpy-100](https://github.com/rougier/numpy-100) + [Lab 01: vectorization](../../labs/01-numpy-vectorization/README.md) | numpy-100 exercises 1–50 (skip ones you can do instantly), then make Lab 01's tests pass | 2 h |

**Notes for the learner:** every time you write a `for` loop over rows, stop and ask "which axis is this loop over, and which reduction or broadcast replaces it?"
Then keep the loop version as a test oracle. The Lab 01 tests do exactly that.

## Check your understanding
1. Arrays of shapes `(3, 1)` and `(4,)` are added. What is the result's shape, and why? What about `(3,)` and `(4,)`?
2. `b = a[::2]` and `c = a[[0, 2, 4]]`. Which one is a view and which a copy? How can you check?
3. Why does `x - x.max(axis=1, keepdims=True)` make softmax stable, and what goes wrong without `keepdims=True`?
4. Write the pairwise squared-distance matrix between `X (n, d)` and `Y (m, d)` with no loops. What is its peak memory, compared with `((X[:, None] - Y[None]) ** 2).sum(-1)`?
5. How do you compute a moving average of window `w` in O(n) without a loop?
6. What does `np.einsum("bij,bjk->bik", A, B)` compute? Write it another way.
7. *(debug)* Your vectorized loss gives a different number from the loop version, but only on some inputs. Name two likely causes (think dtypes and broadcasting) and how you'd check each.

## Mini-project
**Task:** vectorize a slow, realistic routine: k-nearest-neighbour classification on 10,000 points, or a per-row feature computation on your Course 0 dataset.
Write the loop version first as the oracle, then the vectorized version, then assert they agree and time both at three sizes.
**Dataset:** `sklearn.datasets.load_digits` (kNN), or your running dataset.
**Deliverable:** a notebook with a timing table (loop vs vectorized, three input sizes) and a one-paragraph explanation of *where* the speed-up comes from.

## Go deeper
- [*From Python to NumPy*](https://www.labri.fr/perso/nrougier/from-python-to-numpy/), Ch. "Problem vectorization": rethinking an algorithm, not just its loops.
- [Scientific Python Lectures](https://lectures.scientific-python.org/index.html): "NumPy: creating and manipulating numerical data", then "Advanced NumPy" (strides, structured arrays).
- [*Python for Data Analysis*](https://wesmckinney.com/book/), Ch. 4 "NumPy Basics" and Appendix A "Advanced NumPy".

## Math refresher
- [MATH-01 Linear algebra](../math/01-linear-algebra.md), block A: vectors, matrices and matrix products, which `@` and `einsum` implement.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) ("Python, NumPy & pandas foundations").
- **Papers:** none.
- **Implement it yourself:** [Lab 01](../../labs/01-numpy-vectorization/README.md), then [from-scratch ladder](../../exercises/from-scratch-ladder.md) rung 8 (vectorized kNN).
- **Drills:** [numpy-100](https://github.com/rougier/numpy-100) · more in [exercises/](../../exercises/README.md).
