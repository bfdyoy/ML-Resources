# Labs: build it yourself, with tests

[← Back to the README](../README.md) · [Courses](../courses/README.md) · [From-scratch ladder](../exercises/from-scratch-ladder.md)

The labs take the **test-driven assignment** format of Stanford's CS336 and the ARENA bootcamp, and the from-scratch-then-check habit of CS231n.
Each lab gives you **function stubs and a failing test suite**. You're done when the tests pass.
The tests compare your code with a trusted reference: `scikit-learn`, PyTorch autograd, finite differences, or a value worked out by hand.

| Lab | Builds | Lesson | Course week |
|---|---|---|---|
| [01 NumPy vectorization](01-numpy-vectorization/README.md) | pairwise distances, stable softmax, `cumsum` windows, `einsum`, kNN | PY-02 | [C0](../courses/00-python-for-ml.md) w2 |
| [02 pandas wrangling](02-pandas-wrangling/README.md) | group stats, safe merges, leak-free lags and rolling means | PY-03 | C0 w3 |
| [03 Linear regression](03-linear-regression/README.md) | least squares, gradient + check, gradient descent, ridge | CORE-02 | [C1](../courses/01-core-ml.md) w2 |
| [04 Logistic regression & metrics](04-logistic-regression-metrics/README.md) | stable loss, logistic GD, P/R/F1, AUC by ranks, cost threshold | CORE-03 | C1 w3 |
| [05 CV, trees & target encoding](05-trees-and-cross-validation/README.md) | k-fold / group k-fold, Gini split, out-of-fold target encoding | CORE-04/05/07 | C1 w4, w5, w8 |
| [06 k-means & PCA](06-kmeans-pca/README.md) | Lloyd, k-means++, PCA via SVD, reconstruction | CORE-06 | C1 w7 |
| [07 micrograd](07-micrograd/README.md) | a scalar autograd engine | DL-01 | [C2](../courses/02-deep-learning.md) w1 |
| [08 MLP backprop](08-mlp-backprop/README.md) | vectorized backprop, softmax-CE gradient, numeric gradient check | DL-01 | C2 w2 |
| [09 Attention](09-attention/README.md) | masked scaled dot-product attention, multi-head self-attention | DL-06 | C2 w8 |
| [10 BPE tokenizer](10-bpe-tokenizer/README.md) | byte-level BPE training, encode, decode | GEN-01 | [C3](../courses/03-llms-genai.md) w1 |
| [11 Retrieval metrics](11-retrieval-metrics/README.md) | BM25, recall@k, MRR, NDCG, reciprocal rank fusion | GEN-05 | C3 w4 |
| [12 Calibration, conformal & drift](12-calibration-conformal-drift/README.md) | ECE, split conformal (sets and intervals), PSI | CORE-11, PROD-03 | C1 w12, [C4](../courses/04-ml-in-production.md) w4 |
| [13 Detection metrics](13-detection-metrics/README.md) | box IoU, NMS, average precision | CV-01 | [C5](../courses/05-computer-vision.md) w1 |

## How to work a lab

```bash
pip install numpy scipy scikit-learn pandas torch pytest
pytest labs/03-linear-regression                 # run one lab against YOUR exercise.py: all red at first
pytest labs/03-linear-regression -k gradient     # one test at a time
pytest labs/03-linear-regression -x --pdb        # stop at the first failure and inspect it
```

1. Read the lab's README and the docstrings in `exercise.py`. Each docstring lists **subgoals**: the named steps of the procedure.
2. Implement one function, run its test, and repeat. Read a failing test before you read the solution: the test usually shows the exact expected numbers.
3. Stuck for more than ~20 minutes? Use the [stuck protocol](../courses/README.md#34-when-youre-stuck-the-protocol). Then peek at **one function** of `solution.py`, close it, and write it from memory.
4. Done? Do the **Bonus** in the lab's README. Two weeks later, redo the lab from a fresh stub (`scripts/make_lab_stubs.py --force labs/<lab>`). This is spaced retrieval for code.

The scaffolding fades as the labs go on. Lab 01 docstrings almost give the answer, and some functions are already given as worked examples (micrograd's `__add__`).
Later labs give only the contract and a hint.

## For maintainers

- `solution.py` is the source of truth. `exercise.py` is generated from it by `python3 scripts/make_lab_stubs.py` (function bodies become
  `raise NotImplementedError`, docstrings are kept, and functions marked `(given)` are kept whole).
- CI (`.github/workflows/labs.yml`) runs `LAB_IMPL=solution pytest labs/` and `scripts/make_lab_stubs.py --check`.
- Tests must run in seconds on a CPU, use seeded randomness, and compare against an independent reference wherever one exists.
