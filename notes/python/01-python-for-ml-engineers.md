# PY-01 notes: Python for ML Engineers

[← Lesson PY-01](../../lessons/python/01-python-for-ml-engineers.md) · [All notes](../README.md) · Next: [PY-02 notes →](02-numpy-vectorized-thinking.md)

> **Reading time** ≈ 60 min. **You need:** working Python. Nothing from the other notes.

---

## Where we are

This is the first note of the route. Before arrays, models and gradients, it fixes the Python ideas that ML libraries are built on, because those ideas explain
behaviour you'll otherwise find surprising: shared data, lazy streams, `with torch.no_grad():`, and why `model(x)` works at all.

---

## 1. Names point to objects

A Python variable is a **name bound to an object**, not a box holding a value. Assignment never copies. It binds another name to the same object.
Whether a change is visible through the other name depends on one question: did you **mutate** the object, or **rebind** the name?

```python
a = [1, 2, 3]
b = a                 # two names, one list
b.append(4)           # mutation: visible through both names
print(a, a is b)

b = b + [5]           # `+` builds a NEW list; rebinding b leaves a alone
print(a, b, a is b)

import copy
nested = [[0, 0], [0, 0]]
shallow = list(nested)            # new outer list, SAME inner lists
deep = copy.deepcopy(nested)
nested[0][0] = 99
print(shallow[0][0], deep[0][0])
```

The output is `[1, 2, 3, 4] True`, then `[1, 2, 3, 4] [1, 2, 3, 4, 5] False`, then `99 0`. A shallow copy copies only the outer container.
The same distinction comes back with NumPy **views vs copies** ([PY-02 notes §2](02-numpy-vectorized-thinking.md#2-views-copies-and-strides)) and pandas' chained-assignment trap.

**The mutable default argument.** Default values are evaluated **once**, when the `def` runs, not on every call. A mutable default is therefore shared across calls:

```python
def add_row(row, rows=[]):        # BUG: one list shared by every call
    rows.append(row)
    return rows

print(add_row("a"), add_row("b"))

def add_row_fixed(row, rows=None):
    rows = [] if rows is None else rows
    rows.append(row)
    return rows

print(add_row_fixed("a"), add_row_fixed("b"))
```

The buggy version prints `['a', 'b'] ['a', 'b']`: the second call sees the first call's data. The fixed version prints `['a'] ['b']`.
(The same sharing is occasionally used *deliberately* as a cache, but `functools.lru_cache` says so far more clearly. See §4.)

## 2. Iterators and generators: stream instead of load

An **iterable** is anything you can loop over. Looping calls `iter(obj)` to get an **iterator**, then `next()` until `StopIteration`.
A **generator function** (a function with `yield`) returns an iterator that computes values **lazily**, one at a time, keeping only its local state in memory.

```python
import sys

def read_batches(n_rows, batch_size):
    """Pretend to stream a huge file: yields lists of rows without ever holding them all."""
    batch = []
    for i in range(n_rows):
        batch.append(i)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

gen = read_batches(10, 4)
print(next(gen), next(gen), list(gen))

squares_list = [x * x for x in range(1_000_000)]
squares_gen = (x * x for x in range(1_000_000))
print(f"list: {sys.getsizeof(squares_list) / 1e6:.1f} MB   generator: {sys.getsizeof(squares_gen)} bytes")
print(sum(squares_gen) == sum(squares_list), sum(squares_gen))
```

`read_batches` yields `[0, 1, 2, 3]`, `[4, 5, 6, 7]`, then the leftover `[8, 9]`. The list of a million squares takes **8.4 MB** just for its pointers (more for the integers themselves).
The generator object takes **about 200 bytes** (the exact figure depends on the Python version), whatever the length. The price: a generator can be consumed **only once**. The last line prints `True 0`, because the generator was already exhausted by the first `sum`.

`itertools` provides lazy building blocks: `islice` (take the first n), `chain` (concatenate streams), `batched` (Python 3.12+), `accumulate`, `groupby`.
PyTorch's `IterableDataset` is exactly this idea: a dataset that streams.

## 3. Functions are objects: closures

Functions can be passed around, stored, and created inside other functions. An inner function **closes over** the variables of the enclosing scope, so it remembers them after the outer function has returned:

```python
def make_scaler(mean, std):
    def scale(x):
        return (x - mean) / std        # mean and std are remembered from the enclosing call
    return scale

standardize_age = make_scaler(40.0, 10.0)
print(standardize_age(55.0), standardize_age.__closure__ is not None)
```

`standardize_age(55.0)` is **1.5**. A fitted scikit-learn transformer is the object-oriented version of the same idea: parameters learned in `fit`, used later in `transform`.

## 4. Decorators: wrap a function to add behaviour

A decorator is a function that takes a function and returns a new one. `@timed` above a `def` is shorthand for `f = timed(f)`.

```python
import functools
import time

def timed(fn):
    @functools.wraps(fn)                  # keep fn's name and docstring on the wrapper
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        wrapper.last_seconds = time.perf_counter() - t0
        return out
    return wrapper

@timed
def slow_sum(n):
    """Sum 0..n-1 with a Python loop."""
    total = 0
    for i in range(n):
        total += i
    return total

print(slow_sum(100_000), slow_sum.__name__, slow_sum.__doc__, slow_sum.last_seconds > 0)

@functools.lru_cache(maxsize=None)        # a decorator from the standard library: memoization
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print(fib(80), fib.cache_info().hits)
```

`slow_sum(100_000)` returns **4999950000**, and thanks to `functools.wraps` the wrapper still reports the name `slow_sum` and its docstring (without it, both would be the wrapper's).
`lru_cache` turns the exponential recursive Fibonacci into a linear one: `fib(80)` = **23416728348467685** instantly, with **78** cache hits.
In ML code you'll meet decorators as `@torch.no_grad()`, `@torch.compile`, `@pytest.fixture`, `@dataclass`, and `@property`.

## 5. Context managers: set up, and always clean up

A `with` block calls `__enter__` on entry and `__exit__` on exit, **even when an exception is raised inside**. That's how files get closed and how `torch.no_grad()` restores gradient tracking.
`contextlib.contextmanager` lets you write one as a generator: the code before `yield` is the setup, the code after it (in `finally`) is the teardown.

```python
from contextlib import contextmanager

SETTINGS = {"grad_enabled": True}

@contextmanager
def no_grad():
    old = SETTINGS["grad_enabled"]
    SETTINGS["grad_enabled"] = False
    try:
        yield
    finally:
        SETTINGS["grad_enabled"] = old    # runs even if the block raised

with no_grad():
    inside = SETTINGS["grad_enabled"]
try:
    with no_grad():
        raise ValueError("boom")
except ValueError:
    pass
print(inside, SETTINGS["grad_enabled"])
```

Inside the block the flag is `False`. After both blocks, including the one that raised, it's back to `True`. Without the `try/finally`, an exception would leave the flag stuck at `False`,
the kind of bug that makes evaluation silently run without gradients or with the wrong mode.

## 6. Dataclasses and type hints: make structure explicit

Configs passed around as dicts invite typos (`cfg["learning_rte"]`) and hide what a function needs. A `@dataclass` generates `__init__`, `__repr__` and `__eq__` from annotated fields:

```python
from dataclasses import dataclass, field, asdict, replace

@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 10
    layers: tuple[int, ...] = (256, 128)
    tags: list[str] = field(default_factory=list)     # mutable defaults need a factory (§1!)

base = TrainConfig()
fast = replace(base, epochs=2, lr=1e-3)
print(fast)
print(asdict(fast)["layers"], base == TrainConfig())
```

`frozen=True` makes instances immutable (an accidental `cfg.lr = ...` raises an error), `replace` makes modified copies for experiments, and `asdict` turns a config into a plain dict to log with your results.
Type hints (`lr: float`) aren't enforced at runtime, but editors and `mypy` use them to catch mistakes before you run anything.

## 7. The data model: how `len(x)`, `x[i]` and `model(x)` work

Python's syntax calls **special methods** ("dunder" methods). Implement them and your object works with the built-in syntax:

| Syntax | Calls |
|---|---|
| `len(ds)` | `ds.__len__()` |
| `ds[i]` | `ds.__getitem__(i)` |
| `for x in ds` | `ds.__iter__()` (or falls back to `__getitem__` with 0, 1, 2, … until `IndexError`) |
| `model(x)` | `model.__call__(x)` |
| `a + b`, `a @ b` | `a.__add__(b)`, `a.__matmul__(b)` |

```python
class SquaresDataset:
    """A map-style dataset, the same protocol as torch.utils.data.Dataset."""
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        return i, i * i            # (input, target)

class Affine:
    def __init__(self, w, b):
        self.w, self.b = w, b
    def __call__(self, x):         # nn.Module defines __call__, which runs hooks and then your forward()
        return self.w * x + self.b

ds = SquaresDataset(5)
model = Affine(2.0, 1.0)
print(len(ds), ds[3], [y for _, y in ds], [model(x) for x, _ in ds])
```

`len(ds)` is 5, `ds[3]` is `(3, 9)`, iteration works through `__getitem__` alone, and `model(x)` gives `[1.0, 3.0, 5.0, 7.0, 9.0]`.
PyTorch's map-style `Dataset` needs exactly `__len__` and `__getitem__`. The `DataLoader` then handles batching and shuffling. And `nn.Module.__call__` is why you call `model(x)`, not `model.forward(x)`.

---

## Pitfalls & misconceptions

- **"Assignment copies."** It never does. `b = a` shares the object. Copy explicitly (`list(a)`, `a.copy()`, `copy.deepcopy(a)`), and know whether you need shallow or deep.
- **Mutable defaults** (`def f(x, cache={})`, or a dataclass field `= []`) are shared between calls. Use `None` plus a fresh object, or `field(default_factory=list)`.
- **Reusing an exhausted generator.** The second pass over a generator yields nothing, silently. Materialize it with `list()` if you need two passes, or recreate it.
- **Decorators without `functools.wraps`** hide the original name and docstring, which breaks logging, debugging, and tools that inspect signatures.
- **Cleanup outside `finally`.** Without `try/finally` (or a context manager), an exception skips your teardown and leaves global state changed.
- **Calling `model.forward(x)` directly** in PyTorch skips the hooks that `__call__` runs. Call `model(x)`.

## Cheat sheet

| Need | Python tool |
|---|---|
| Independent copy | `list(x)` / `x.copy()` (shallow), `copy.deepcopy(x)` (deep) |
| Stream big data | Generator function with `yield`; `itertools.islice`, `chain` |
| Remember parameters | Closure, or a class with `__call__` |
| Add timing/caching/logging | Decorator with `functools.wraps`; `functools.lru_cache` |
| Guaranteed cleanup | `with` + `contextlib.contextmanager` + `try/finally` |
| Typed config | `@dataclass(frozen=True)`, `replace`, `asdict`; `field(default_factory=...)` |
| Dataset protocol | `__len__`, `__getitem__` (map-style); `__iter__` (iterable-style) |
| Callable object | `__call__` |

## Answer sketches for the lesson's self-check

<details>
<summary>1. <code>b = a; b.append(4)</code> vs <code>b = a + [4]</code></summary>

`a` is `[1, 2, 3, 4]`: `b` and `a` are two names for one list, and `append` mutates it. With `b = a + [4]`, `+` creates a new list and rebinds `b`, so `a` stays `[1, 2, 3]` (§1).
</details>

<details>
<summary>2. Why <code>def f(x, cache={})</code> is usually a bug</summary>

The default dict is created once, when `def` executes, and every call that doesn't pass `cache` shares it. Data from one call leaks into the next (§1).
It's a deliberate (if obscure) memoization trick when the sharing is the point. `functools.lru_cache` does the same thing explicitly (§4).
</details>

<details>
<summary>3. List comprehension vs generator expression in <code>sum</code></summary>

The list version materializes 10⁸ integers (gigabytes) before summing. The generator version holds one value at a time, so its memory is constant (§2).
The generator gives up random access, `len()`, and the ability to iterate twice.
</details>

<details>
<summary>4. A <code>@timed</code> decorator, and why <code>functools.wraps</code></summary>

See §4: the wrapper records `time.perf_counter()` before and after calling `fn(*args, **kwargs)` and returns the result.
Without `functools.wraps`, the decorated function's `__name__`, `__doc__` and signature become the wrapper's, which confuses logs, debuggers, `help()`, and frameworks that inspect functions.
</details>

<details>
<summary>5. A context manager that always restores a flag</summary>

A generator decorated with `@contextmanager`: save the old value, set the new one, `yield` inside `try`, and restore the old value in `finally` (§5).
The `finally` runs whether the `with` block finishes normally or raises.
</details>

<details>
<summary>6. Special methods for <code>len</code>, indexing, and iteration</summary>

`__len__`, `__getitem__`, and `__iter__` (iteration also falls back to `__getitem__` with 0, 1, 2, … until `IndexError`) (§7).
A PyTorch map-style `Dataset` needs `__len__` and `__getitem__`. An `IterableDataset` needs `__iter__`.
</details>

<details>
<summary>7. (debug) Correct on the first call, wrong on the second</summary>

(1) A **mutable default argument**, or module-level state that the function appends to: check the signature for `=[]`/`={}` and look for globals (§1).
(2) The function **mutates its input in place** (e.g. `df.drop(..., inplace=True)` or modifying a list argument), so the second call receives already-modified data: check `id()`s, or compare the input before and after the call.
Also consider a **consumed generator** passed in as the input (§2).
</details>

## Where this leads

Next: [PY-02 notes](02-numpy-vectorized-thinking.md). We've been working with Python objects one at a time. NumPy replaces millions of them with one block of memory and whole-array operations.
That is both the speed-up and the new set of surprises (views, broadcasting) to learn.
