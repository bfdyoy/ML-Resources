# Lab 01: NumPy vectorization

[← Labs](../README.md) · Lesson: [PY-02 NumPy & vectorized thinking](../../lessons/python/02-numpy-vectorized-thinking.md) · Notes: [PY-02 notes](../../notes/python/02-numpy-vectorized-thinking.md)

**Time** ≈ 1.5 h · **You'll practise:** broadcasting, reductions with `keepdims`, fancy indexing, `cumsum` tricks, `einsum`.

| Function | What it does | The trick |
|---|---|---|
| `pairwise_sq_dists` | all-pairs squared distances | expand the square; one matrix product |
| `softmax` | stable softmax | subtract the max first |
| `one_hot` | labels → indicator matrix | fancy indexing |
| `moving_average` | sliding-window mean | difference of cumulative sums |
| `standardize` | column z-scores | broadcasting; guard zero variance |
| `batched_quadratic_form` | `x_i^T A x_i` per row | `einsum` |
| `knn_predict` | k-NN classifier | `argpartition` + vote counting |

```bash
pytest labs/01-numpy-vectorization          # your exercise.py
```

**Rule for this lab:** no Python `for` loops over rows. The tests compare your answers with loop versions.

**Bonus:** time `pairwise_sq_dists` against a double loop for 2,000 × 2,000 points. Then explain why the expanded formula
can lose precision for points that are very close together, compared with `((X[:, None] - Y[None]) ** 2).sum(-1)`. What does that version cost in memory?
