# PY-02 notes: NumPy & Vectorized Thinking

[← Lesson PY-02](../../lessons/python/02-numpy-vectorized-thinking.md) · [All notes](../README.md) · [← PY-01 notes](01-python-for-ml-engineers.md) · Next: [PY-03 notes →](03-pandas-data-wrangling.md)

> **Reading time** ≈ 70 min. **You need:** [PY-01 notes §1](01-python-for-ml-engineers.md#1-names-point-to-objects) (names vs objects, shallow copies).

---

## Where we are

PY-01 dealt with Python objects one at a time. An ML dataset is millions of numbers, and a Python list of floats stores each one as a separate object.
NumPy stores them as **one contiguous block of typed memory** and runs loops in compiled code. This note covers the mental model that makes that fast and safe: memory and views, broadcasting, axes, and numerical stability.

---

## 1. Why vectorize: the loop moves into C

A Python `for` loop does type checks, object lookups and reference counting for every element. A NumPy operation does one type check and then a tight compiled loop over raw memory.

```python
import time
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=1_000_000)

t0 = time.perf_counter()
loop_total = 0.0
for v in x:
    loop_total += v * v
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
vec_total = float(np.dot(x, x))
t_vec = time.perf_counter() - t0

print(np.isclose(loop_total, vec_total), f"speed-up on this machine: {t_loop / t_vec:.0f}x")
```

Both give the same sum of squares (up to float rounding, hence `isclose`). The vectorized version is typically **hundreds of times** faster. The exact ratio depends on your machine, so run it.
**The habit:** keep the loop version as a *test oracle* and assert that the vectorized version agrees, exactly as [Lab 01](../../labs/01-numpy-vectorization/README.md)'s tests do.

## 2. Views, copies and strides

An array is a **buffer** of bytes plus metadata: `dtype` (how to read the bytes), `shape`, and **strides** (how many bytes to step to reach the next element along each axis).
Basic slicing (`a[1:5]`, `a[::2]`, `a[:, 0]`) and reshaping just create new metadata over the **same buffer**: a **view**. Fancy indexing (`a[[0, 2]]`) and boolean masks (`a[a > 0]`) must gather scattered elements, so they return a **copy**.

```python
a = np.arange(12, dtype=np.int64).reshape(3, 4)
print(a.strides)                       # 4 columns * 8 bytes to the next row, 8 bytes to the next column
col = a[:, 1]                          # a view
col[0] = 100
print(a[0, 1], np.shares_memory(a, col), col.strides)

picked = a[[0, 2]]                     # fancy indexing: a copy
picked[0, 0] = -1
print(a[0, 0], np.shares_memory(a, picked))
print(a.T.strides, a.T.flags["C_CONTIGUOUS"])
```

`a.strides` is `(32, 8)`. Writing through the column view changes `a` (`a[0, 1]` becomes **100**). Writing into the fancy-indexed copy doesn't (`a[0, 0]` stays **0**).
The transpose is a free view with swapped strides `(8, 32)`, which is why it's no longer C-contiguous. Some operations (and PyTorch's `.view()`) then need an explicit contiguous copy.
**Rule:** if you want an independent array, call `.copy()`. If you modify a slice, you modify the original.

## 3. Broadcasting: the two rules

When two arrays meet in an elementwise operation, NumPy compares their shapes **from the right**:

1. If the arrays have different numbers of dimensions, prepend 1s to the shorter shape.
2. Along each axis, the sizes must be **equal**, or **one of them must be 1**. A size-1 axis is stretched (virtually, without copying) to match.

*Worked shapes:*

| Shapes | Aligned from the right | Result |
|---|---|---|
| `(3, 1)` and `(4,)` | `(3, 1)` vs `(1, 4)` | `(3, 4)` |
| `(5, 3)` and `(3,)` | `(5, 3)` vs `(1, 3)` | `(5, 3)`: subtract a per-column mean |
| `(5, 3)` and `(5,)` | `(5, 3)` vs `(1, 5)` | **error**: 3 ≠ 5. You need `(5, 1)`, i.e. `v[:, None]` |
| `(8, 1, 6)` and `(7, 1)` | `(8, 1, 6)` vs `(1, 7, 1)` | `(8, 7, 6)` |

```python
X = rng.normal(size=(5, 3))
col_means = X.mean(axis=0)             # shape (3,)
row_means = X.mean(axis=1)             # shape (5,)
print((X - col_means).shape, (X - row_means[:, None]).shape)
try:
    X - row_means
except ValueError as e:
    print("error:", str(e)[:60])

u, v = np.arange(3)[:, None], np.arange(4)
print((u * 10 + v))                    # an outer "sum table" with no loops
```

`X - col_means` works (shape `(5, 3)`), `X - row_means` raises "operands could not be broadcast together", and `row_means[:, None]` (shape `(5, 1)`) fixes it.
`None` (or `np.newaxis`) inserts a size-1 axis: it's how you **line up axes on purpose**. Pairwise operations (distances, attention scores) are broadcasting between `(n, 1, d)` and `(1, m, d)`.

## 4. Axes and `keepdims`

`axis=k` means "collapse axis k". For a `(n_samples, n_features)` matrix, `axis=0` aggregates over samples (one value per feature) and `axis=1` over features (one value per sample).
`keepdims=True` leaves the collapsed axis with size 1, so the result broadcasts straight back against the original. That's what you want for normalizing:

```python
Z = rng.normal(size=(4, 3))
row_sum_flat = Z.sum(axis=1)                  # (4,)
row_sum_keep = Z.sum(axis=1, keepdims=True)   # (4, 1)
print(row_sum_flat.shape, row_sum_keep.shape, np.allclose((Z / row_sum_keep).sum(axis=1), 1))
```

## 5. Masks, fancy indexing, and "find" operations

Boolean masks select, count and replace without loops. `np.where(cond, a, b)` is a vectorized if/else. `argsort` and `argpartition` give *indices*, which you then use to gather other arrays.

```python
scores = np.array([0.2, 0.9, 0.4, 0.95, 0.1, 0.7])
labels = np.array([0, 1, 0, 1, 0, 0])
mask = scores > 0.5
print(mask.sum(), labels[mask], np.where(mask, "flag", "ok"))

top2 = np.argpartition(-scores, 2)[:2]          # the 2 largest, in no particular order: O(n)
top2 = top2[np.argsort(-scores[top2])]          # then sort just those: O(k log k)
print(top2, scores[top2])

clipped = scores.copy()
clipped[clipped < 0.3] = 0.3                    # masked assignment (or np.clip)
print(clipped)
```

There are **3** scores above 0.5, with labels `[1 1 0]`. The top two indices are `[3 1]` (scores 0.95 and 0.9). `argpartition` + a small sort is how you take a top-k from millions of scores without fully sorting them.

## 6. Numerical stability: where vectorized code goes wrong

Floats have finite range and precision. Three traps come up constantly in ML code.

**(a) Overflow in `exp`.** `softmax(z) = exp(z) / sum(exp(z))`, and `exp(1000)` is `inf`. Since softmax is unchanged when you subtract a constant from every entry, subtract the max first.
The same idea gives a stable **log-sum-exp**: `logsumexp(z) = m + log(sum(exp(z − m)))` with `m = max(z)`.

```python
z = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum()
stable = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print(naive, stable.round(4))
```

The naive version returns `[nan nan nan]` (inf / inf). The stable one returns `[0.09 0.2447 0.6652]`, the same as for `[0, 1, 2]`.

**(b) Catastrophic cancellation.** Subtracting two nearly equal large numbers loses most significant digits. The fast pairwise-distance formula `‖x‖² + ‖y‖² − 2x·y` suffers from this for points that are far from the origin and close together:

```python
p = np.array([[1e4, 1e4]], dtype=np.float32)
q = p + np.float32(0.01)                        # true squared distance: 2 * 0.01^2 = 0.0002
expanded = (p ** 2).sum(1) + (q ** 2).sum(1) - 2 * (p * q).sum(1)
direct = ((p - q) ** 2).sum(1)
print(expanded, direct)
```

In float32, the expanded formula returns **0.0** (it even goes *negative* for other inputs, which is why Lab 01 clips at 0). The direct difference gives about **0.0002**, close to the true value.
Fixes: center the data first, use float64, or use the direct formula when memory allows.

**(c) Integer wraparound.** Images often load as `uint8` (0–255). Arithmetic stays in `uint8` and **wraps around** silently:

```python
img = np.array([10, 200], dtype=np.uint8)
print(img - 20, img.astype(np.int16) - 20, img / 255.0)
```

`img - 20` gives `[246 180]`: 10 − 20 wrapped around to 246. Convert to a float (or a wider int) before doing arithmetic on pixel values.

## 7. Three tricks worth memorizing

**Window sums with `cumsum`:** the sum of `x[i:i+w]` is `c[i+w] − c[i]`, where `c` is the cumulative sum with a leading 0. That's O(n) for every window at once.

**Group sums with `bincount`:** with integer group ids `g`, `np.bincount(g, weights=x)` is a vectorized "group by and sum". Divide by `np.bincount(g)` for group means.

**`einsum` for any product-and-sum:** name the axes, and repeated letters are multiplied, while letters missing from the output are summed.

```python
x = np.arange(10, dtype=float)
c = np.concatenate([[0.0], np.cumsum(x)])
print((c[3:] - c[:-3]) / 3)                     # moving average, window 3

g = np.array([0, 1, 0, 2, 1, 0])
vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print(np.bincount(g, weights=vals) / np.bincount(g))

A, B = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 4, 5))
print(np.allclose(np.einsum("bij,bjk->bik", A, B), A @ B), np.einsum("ii->", np.eye(3)))
```

The moving average of 0…9 with window 3 is `[1 2 3 4 5 6 7 8]`. The group means are `[3.33 3.5 4.]`. And `"bij,bjk->bik"` is a **batched matrix product** (the same as `A @ B` on 3-D arrays). `"ii->"` is the trace.
Attention scores in a transformer are `einsum("bhqd,bhkd->bhqk", Q, K)`.

---

## Pitfalls & misconceptions

- **"Slicing copies."** Basic slicing returns a view, so writing to it changes the original. Fancy and boolean indexing return copies, so writing to *those* changes nothing in the original.
- **Accidental broadcasting.** `(n,)` minus `(n, 1)` gives an `(n, n)` matrix, not an error. Losses computed this way look plausible and are wrong. Assert shapes (`assert y_pred.shape == y.shape`).
- **Forgetting `keepdims`** when normalizing along axis 1 leads to either an error or a silent broadcast against the wrong axis.
- **Naive `exp`** in softmax, sigmoid, or likelihoods overflows. Use max-subtraction and `logsumexp`/`logaddexp`.
- **`uint8` arithmetic wraps around.** Convert images to float before doing math.
- **float32 vs float64.** Mixing them silently upcasts. Using float32 for gradient checks gives misleadingly large errors ([Lab 08](../../labs/08-mlp-backprop/README.md)).

## Cheat sheet

| Need | NumPy |
|---|---|
| Independent array | `a.copy()`; check with `np.shares_memory(a, b)` |
| Line up axes | `v[:, None]`, `v[None, :]`, `np.expand_dims` |
| Broadcasting rule | Align shapes from the right; each axis equal or 1 |
| Reduce and re-broadcast | `x.sum(axis=k, keepdims=True)` |
| Vectorized if | `np.where(cond, a, b)`; masks `a[a > t]` |
| Top-k | `np.argpartition(-s, k)[:k]`, then sort those |
| Stable softmax / LSE | Subtract the max; `scipy.special.logsumexp` |
| Window sums | Differences of `cumsum` |
| Group sums | `np.bincount(g, weights=x)` |
| Batched products | `@` on stacks; `np.einsum("bij,bjk->bik", A, B)` |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Shapes <code>(3, 1)</code> + <code>(4,)</code>, and <code>(3,)</code> + <code>(4,)</code></summary>

Aligning from the right, `(4,)` becomes `(1, 4)`, and each axis is either equal or 1, so the result is `(3, 4)` (an outer sum). `(3,)` and `(4,)` fail: 3 ≠ 4 and neither is 1 (§3).
</details>

<details>
<summary>2. <code>a[::2]</code> vs <code>a[[0, 2, 4]]</code></summary>

`a[::2]` is basic slicing: a view with a doubled stride. `a[[0, 2, 4]]` is fancy indexing: a copy. Check with `np.shares_memory(a, b)`, or `b.base is a` (§2).
</details>

<details>
<summary>3. Why subtracting the max stabilizes softmax, and why <code>keepdims</code></summary>

Softmax is invariant to adding a constant to every entry of a row, and after subtracting the row max the largest exponent is `exp(0) = 1`, so nothing overflows (§6a).
`keepdims=True` keeps the max as an `(n, 1)` column, so it broadcasts across each row. Without it, `(n, C) − (n,)` either errors or (when n = C) silently subtracts along the wrong axis (§4).
</details>

<details>
<summary>4. Pairwise squared distances without loops, and memory</summary>

`(X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2 * X @ Y.T`, clipped at 0. Peak memory is O(n·m) for the result.
The direct broadcast `((X[:, None] - Y[None]) ** 2).sum(-1)` materializes an `(n, m, d)` array first, d times more memory, but it's numerically safer for close points (§6b, Lab 01).
</details>

<details>
<summary>5. An O(n) moving average</summary>

Take the cumulative sum with a leading zero, `c`. The window sums are `c[w:] − c[:-w]`. Divide by `w` (§7).
</details>

<details>
<summary>6. <code>np.einsum("bij,bjk->bik", A, B)</code></summary>

A batched matrix product: for each batch index b, `A[b] @ B[b]`. The same as `A @ B` or `np.matmul(A, B)` on 3-D arrays (§7).
</details>

<details>
<summary>7. (debug) The vectorized loss disagrees with the loop on some inputs</summary>

(1) **Accidental broadcasting**: e.g. `y` has shape `(n,)` and `y_pred` has shape `(n, 1)`, so the difference is `(n, n)`. Check: assert the shapes match before reducing.
(2) **dtype problems**: integer inputs (integer division, `uint8` wraparound) or float32 cancellation/overflow on large values. Check the `dtype` of every input, and compare on float64 copies (§6).
</details>

## Where this leads

Next: [PY-03 notes](03-pandas-data-wrangling.md). NumPy arrays are positional and of a single type. Real tables have named columns of mixed types, missing values, and keys to join on.
pandas adds an **index** on top of NumPy, which brings label alignment: a new superpower and a new class of bugs.
