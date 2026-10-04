# MATH-01 notes: Linear Algebra for ML

[← Lesson MATH-01](../../lessons/math/01-linear-algebra.md) · [All notes](../README.md) · [Notation](../notation.md) · [← PY-05 notes](../python/05-engineering-ml-code.md) (optional Path 0) · Next: [MATH-02 notes →](02-calculus-optimization.md)

> **Reading time** ≈ 60 min (in three blocks, matching the lesson). **You need:** high-school algebra and a little Python.
> Run the code cells as you go. Every number on this page is printed by the code.

---

## Where we are

Almost every object in machine learning is a vector or a matrix:

- a data set is a matrix $X$ with one row per example;
- a linear model is the product $Xw$;
- a neural-network layer is a matrix multiply followed by a squish;
- an embedding is a vector, and "similar" means "pointing in a similar direction";
- PCA is "find the directions the data stretches along".

This page builds the handful of ideas you need, always with the **geometric picture first**, then the formula, then a check in NumPy.

---

## Block A: Vectors, matrices, and what a matrix *does*

### A1. A vector is an arrow and a list of numbers

$x = (x_1, \dots, x_d)$ is a point (or an arrow from the origin) in $d$-dimensional space. In ML, a vector is usually one example's features:
(age, income, number of purchases) is a point in $\mathbb{R}^3$. We write vectors as **columns** by default.

Two operations make a space "linear":

- **Adding** arrows: place them tip to tail.
- **Scaling** an arrow by a number: stretch or flip it.

A **linear combination** of vectors $v_1, \dots, v_k$ is $c_1 v_1 + \dots + c_k v_k$. The set of all such combinations is their **span**.
If none of the vectors can be built from the others, they are **linearly independent**.

### A2. A matrix is a function that moves space

A matrix $A \in \mathbb{R}^{m \times n}$ takes an $n$-vector and returns an $m$-vector: $y = Ax$. The key fact that makes this easy to picture:

> **The columns of $A$ are where the basis vectors land.** $Ax$ is "take $x_1$ of column 1, plus $x_2$ of column 2, …".

```math
A x = \begin{bmatrix} | & | \\ a_1 & a_2 \\ | & | \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = x_1 a_1 + x_2 a_2
```

So $Ax$ is a linear combination of the columns of $A$. **The set of all outputs $Ax$ is the column space of $A$.** Remember this; it is
the whole story of least squares (Block B).

There are three ways to read a matrix product $C = AB$. All three are useful:

| View | Formula | When it helps |
|---|---|---|
| Entry by entry | $C_{ij} = \sum_k A_{ik} B_{kj}$ (row $i$ of $A$ · column $j$ of $B$) | Computing by hand; attention scores |
| Column by column | column $j$ of $C$ = $A$ times (column $j$ of $B$) | Applying a layer to a batch of vectors |
| Sum of outer products | $C = \sum_k (\text{col}_k A)(\text{row}_k B)$ | Low-rank ideas, SVD, LoRA |

Geometrically, $AB$ means "**do $B$ first, then $A$**", that is, function composition. That's why $AB \neq BA$ in general.

### A3. Shapes are your best debugging tool

$(m \times n)(n \times p) = (m \times p)$. The inner dimensions must match. In ML code:

- $X$ is $(N \times d)$, with $N$ examples and $d$ features.
- $w$ is $(d \times 1)$, so $Xw$ is $(N \times 1)$: one prediction per example.
- A layer with weight matrix $W$ of shape $(d_\text{out} \times d_\text{in})$ maps a batch $X$ to $X W^\top$, of shape $(N \times d_\text{out})$. PyTorch's `nn.Linear` stores $W$ exactly this way.

### A4. Rank: how many dimensions survive

The **rank** of $A$ is the number of independent columns, which equals the dimension of the column space, which also equals the number of independent rows.

- A $3 \times 3$ matrix of rank 3 maps space onto all of space. It is **invertible**: you can undo it.
- Rank 2 squashes 3D space onto a plane. Information is lost, so it can't be undone.
- **Rank 1** squashes everything onto a single line. Every rank-1 matrix is an outer product $u v^\top$: "measure how much of $v$ is in $x$, then output that many copies of $u$":

```math
(u v^\top) x = u \, (v^\top x)
```

This last identity is the seed of low-rank approximation (SVD, Block C) and of LoRA fine-tuning (GEN-02), where a weight update is $\Delta W = BA$ with a small inner dimension.

### A5. Solving $Ax = b$, and the inverse

If $A$ is square and full rank, $Ax = b$ has exactly one solution, $x = A^{-1} b$. In code you almost **never form $A^{-1}$**.
You call `np.linalg.solve(A, b)`, which is faster and more accurate. When $A$ is not square (more equations than unknowns, which is
the usual case with data), there is typically **no** exact solution. Then we look for the *closest* one. That is Block B.

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

A = np.array([[2., 1.],
              [1., 3.]])
x = np.array([1., 2.])
print("A @ x            =", A @ x)
print("as column combo  =", x[0] * A[:, 0] + x[1] * A[:, 1])   # same thing

u, v = np.array([1., 2.]), np.array([3., 1.])
R1 = np.outer(u, v)
print("rank of u v^T    =", np.linalg.matrix_rank(R1))
print("(u v^T) x == u (v.x):", np.allclose(R1 @ x, u * (v @ x)))

b = np.array([3., 5.])
print("solve A x = b    =", np.linalg.solve(A, b))
```

---

## Block B: Geometry: dot products, norms, projections

### B1. The dot product measures alignment

```math
a \cdot b = a^\top b = \sum_i a_i b_i = \lVert a \rVert \, \lVert b \rVert \cos\theta
```

The left-hand formula is how you compute it. The right-hand one is what it **means**:

- the dot product is large and positive when the arrows point the same way;
- it is zero when they are perpendicular (**orthogonal**);
- it is negative when they point in opposite directions.

**Cosine similarity**, used for embeddings and retrieval, is just the cosine part, $\cos\theta = \frac{a^\top b}{\lVert a\rVert\lVert b\rVert}$.

*Worked example.* Take $a = (3, 4)$ and $b = (4, 3)$. Then $a^\top b = 12 + 12 = 24$ and $\lVert a\rVert = \lVert b\rVert = 5$, so $\cos\theta = 24/25 = 0.96$ and $\theta \approx 16.3°$.

In attention (DL-06), the score between a query $q$ and a key $k$ is exactly $q^\top k$: "how much does this key match what I'm looking for?"

### B2. Norms measure size

- **L2 (Euclidean):** $\lVert x\rVert_2 = \sqrt{\sum_i x_i^2}$. Ridge regression penalizes $\lVert w\rVert_2^2$.
- **L1:** $\lVert x\rVert_1 = \sum_i \lvert x_i \rvert$. The lasso penalizes this, and its pointy "diamond" shape is why lasso gives exact zeros (CORE-04).
- **L∞:** $\max_i \lvert x_i\rvert$.

The distance between two points is $\lVert a - b\rVert$. For unit vectors, $\lVert a-b\rVert_2^2 = 2 - 2\cos\theta$, so
**ranking by Euclidean distance and ranking by cosine similarity agree once vectors are normalized**. That fact matters in vector databases (GEN-05).

### B3. Projection: the closest point on a line

To project $b$ onto the line spanned by $a$, find the multiple $\hat{c}\,a$ closest to $b$. The error $b - \hat{c}a$ must be
**perpendicular** to $a$; otherwise you could slide along the line and get closer. Write that condition and solve:

```math
a^\top (b - \hat{c}\, a) = 0 \;\Longrightarrow\; \hat{c} = \frac{a^\top b}{a^\top a}, \qquad \text{proj}_a(b) = \frac{a^\top b}{a^\top a}\, a
```

### B4. Least squares *is* a projection

Now the main event. Data: a matrix $X$ ($N \times d$, with $N \gg d$) and targets $y \in \mathbb{R}^N$. We want weights $w$ with $Xw \approx y$.

- Every possible prediction vector $Xw$ lies in the **column space** of $X$, a $d$-dimensional flat sheet inside $\mathbb{R}^N$.
- $y$ is (almost surely) **not** on that sheet; noise pushes it off.
- The best we can do is the point $\hat{y} = X\hat{w}$ on the sheet that is closest to $y$, which is the **projection** of $y$ onto the column space.

Same argument as before: the residual $r = y - X\hat w$ must be perpendicular to **every column** of $X$:

```math
X^\top (y - X\hat{w}) = 0 \;\Longrightarrow\; \boxed{X^\top X \,\hat{w} = X^\top y} \quad\text{(the normal equations)}
```

If $X^\top X$ is invertible (the columns are independent), $\hat w = (X^\top X)^{-1}X^\top y$. You'll derive the same equation again in
[MATH-02](02-calculus-optimization.md) by setting a gradient to zero. Two roads, one answer: geometry and calculus agree.

**Worked example by hand.** Fit $y = w_0 + w_1 x$ to the points $(0, 1), (1, 2), (2, 2)$.

```math
X = \begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\quad y=\begin{bmatrix}1\\2\\2\end{bmatrix},\quad
X^\top X = \begin{bmatrix}3&3\\3&5\end{bmatrix},\quad X^\top y = \begin{bmatrix}5\\6\end{bmatrix}
```

```math
(X^\top X)^{-1} = \frac{1}{6}\begin{bmatrix}5&-3\\-3&3\end{bmatrix}
\;\Rightarrow\;
\hat w = \frac{1}{6}\begin{bmatrix}25-18\\-15+18\end{bmatrix} = \begin{bmatrix}7/6\\1/2\end{bmatrix}
```

The predictions are $\hat y = (7/6,\ 5/3,\ 13/6)$ and the residuals are $r = (-1/6,\ 1/3,\ -1/6)$. Check the perpendicularity: $r$ sums to 0
(it's orthogonal to the column of ones), and $0\cdot(-\tfrac16) + 1\cdot\tfrac13 + 2\cdot(-\tfrac16) = 0$ (it's orthogonal to the $x$ column). ✔

**A free consequence:** because the residual is orthogonal to the column of ones, **OLS residuals with an intercept always sum to zero**.

```python
X = np.array([[1., 0.], [1., 1.], [1., 2.]])
y = np.array([1., 2., 2.])
w = np.linalg.solve(X.T @ X, X.T @ y)          # normal equations
r = y - X @ w
print("w        =", w, " (7/6 =", 7/6, ")")
print("residual =", r)
print("X^T r    =", X.T @ r, "<- orthogonal to every column")
print("lstsq    =", np.linalg.lstsq(X, y, rcond=None)[0])   # what you'd use in practice
```

### B5. Orthonormal bases make everything easy

A set of vectors $q_1, \dots, q_k$ is **orthonormal** if each has length 1 and every pair is perpendicular. Stack them as the columns of $Q$,
and $Q^\top Q = I$. Projections then need no inverse: $\text{proj}(b) = Q Q^\top b$. **Orthogonal matrices** (square $Q$ with $Q^\top Q = I$)
are rotations and reflections. They preserve lengths and angles, so they never amplify errors. This is why numerical libraries love
them (QR decomposition, SVD).

---

## Block C: Eigenvectors, eigendecomposition, SVD

### C1. Eigenvectors: directions a matrix only stretches

Most vectors get knocked off their line when you multiply by $A$. An **eigenvector** $v$ stays on its line; it just gets scaled by its **eigenvalue** $\lambda$:

```math
A v = \lambda v
```

*Worked example.* Take

```math
A = \begin{bmatrix}2&1\\1&2\end{bmatrix}.
```

Try $v = (1, 1)$: $Av = (3, 3) = 3v$, so $\lambda = 3$. Try $v = (1, -1)$: $Av = (1, -1)$, so $\lambda = 1$.
Geometrically, $A$ stretches space by 3 along the diagonal and leaves the anti-diagonal alone.

### C2. Symmetric matrices: the spectral theorem

Covariance matrices, $X^\top X$, kernel matrices, and Hessians are all **symmetric** ($A = A^\top$). For symmetric matrices, the nicest possible thing is true:

> **Spectral theorem.** A real symmetric $A$ has real eigenvalues and an orthonormal set of eigenvectors, so
> $A = Q \Lambda Q^\top$, with $Q$ orthogonal and $\Lambda$ diagonal.

So **every symmetric matrix is a rotation, then an axis-aligned stretch, then the rotation back**. For a symmetric $A$:

- it is **positive semi-definite** (PSD), meaning $x^\top A x \ge 0$ for all $x$, exactly when all its eigenvalues are $\ge 0$;
- $X^\top X$ is always PSD, because $x^\top X^\top X x = \lVert Xx\rVert^2 \ge 0$.

### C3. PCA = eigenvectors of the covariance matrix

Center the data (subtract each column's mean) to get $X_c$. The sample covariance is $S = \frac{1}{N-1}X_c^\top X_c$. The variance of the data
**projected onto a unit direction $u$** is

```math
\operatorname{Var}(X_c u) = \frac{1}{N-1}\lVert X_c u\rVert^2 = u^\top S\, u .
```

PCA asks which direction maximizes this. Write $u$ in the eigenbasis of $S$: $u^\top S u = \sum_i \lambda_i\, (q_i^\top u)^2$, a weighted average
of eigenvalues, because the weights $(q_i^\top u)^2$ sum to 1. That average is largest when all the weight sits on the biggest eigenvalue. So:

> **The first principal component is the top eigenvector of $S$, and the variance it captures is the top eigenvalue.**
> The next PC is the best direction orthogonal to the first, which is the second eigenvector, and so on.

The "explained variance ratio" printed by scikit-learn is $\lambda_k / \sum_i \lambda_i$.

### C4. SVD: eigen-decomposition for any matrix

Every matrix, of any shape, factors as

```math
X = U \Sigma V^\top, \qquad U^\top U = I,\; V^\top V = I,\; \Sigma = \operatorname{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge 0).
```

Read it right to left as a story: **rotate** ($V^\top$), **stretch** each axis by $\sigma_i$ ($\Sigma$), **rotate** again ($U$).
It connects to Block C2 directly: $X^\top X = V \Sigma^2 V^\top$. So:

- the right singular vectors $V$ are the eigenvectors of $X^\top X$, which are the **principal directions** when $X$ is centered;
- the eigenvalues of $X^\top X$ are $\sigma_i^2$, and the PCA variances are $\sigma_i^2/(N-1)$.

That's why scikit-learn's `PCA` is implemented with an SVD: it's more accurate than forming $X^\top X$.

### C5. Low-rank approximation (Eckart–Young)

Written as a sum of rank-1 pieces, $X = \sum_i \sigma_i u_i v_i^\top$. Keep only the first $k$ terms and you get $X_k$, **the best rank-$k$
approximation** of $X$ in the least-squares sense. The error you make is exactly the energy you dropped:

```math
\lVert X - X_k \rVert_F^2 = \sum_{i > k} \sigma_i^2 .
```

This one theorem explains image compression with SVD, PCA dimensionality reduction, latent semantic analysis, recommender matrix factorization (EL-04),
and why LoRA works (weight updates have low "intrinsic rank", GEN-02).

### C6. The condition number: how touchy is a problem?

$\kappa(A) = \sigma_{\max}/\sigma_{\min}$. A large $\kappa$ means some directions are stretched far more than others. The consequences:

- **solving** $Ax=b$ amplifies small errors in $b$ by up to $\kappa$;
- **gradient descent** on a quadratic bowl whose Hessian has a large $\kappa$ zig-zags slowly ([MATH-02](02-calculus-optimization.md) §B3);
- collinear features make $X^\top X$ badly conditioned, so OLS coefficients become unstable (CORE-02). Ridge regression adds $\lambda I$ to $X^\top X$,
  which lifts every eigenvalue by $\lambda$ and fixes the conditioning.

```python
A = np.array([[2., 1.], [1., 2.]])
lam, Q = np.linalg.eigh(A)                       # eigh: for symmetric matrices
print("eigenvalues      =", lam)
print("eigenvectors (columns):\n", Q)
print("Q diag(lam) Q^T == A:", np.allclose(Q @ np.diag(lam) @ Q.T, A))

# PCA three ways on correlated 2-D data
rng = np.random.default_rng(0)
Z = rng.normal(size=(500, 2)) @ np.array([[3., 0.], [1.5, 0.5]])   # stretched, tilted cloud
Xc = Z - Z.mean(axis=0)
S = Xc.T @ Xc / (len(Xc) - 1)
evals, evecs = np.linalg.eigh(S)
U, sig, Vt = np.linalg.svd(Xc, full_matrices=False)
from sklearn.decomposition import PCA
pca = PCA().fit(Z)
print("cov eigenvalues (desc)  =", evals[::-1])
print("sigma^2/(N-1)           =", sig**2 / (len(Xc) - 1))
print("sklearn explained_var   =", pca.explained_variance_)
print("top PC (up to sign) eig =", evecs[:, -1], " svd =", Vt[0], " sklearn =", pca.components_[0])

# Eckart–Young: rank-1 approximation error equals the dropped singular value squared
X1 = sig[0] * np.outer(U[:, 0], Vt[0])
print("||Xc - X1||_F^2 =", np.sum((Xc - X1) ** 2), " sigma_2^2 =", sig[1] ** 2)
print("condition number of S =", np.linalg.cond(S))
```

---

## Pitfalls & misconceptions

- **"Inverting a matrix is how you solve equations."** In code, use `solve` or `lstsq`. Explicit inverses are slow and lose precision.
- **Forgetting to center before PCA.** Without centering, the first "component" mostly points at the mean of the data.
- **Forgetting to scale before PCA.** A feature measured in grams beats one measured in kilograms just because its numbers are bigger. Standardize first unless the units are comparable.
- **Sign ambiguity.** Eigenvectors and singular vectors are defined only up to sign ($v$ and $-v$ are both valid). Different libraries can return different signs. That's not a bug.
- **Row vs column conventions.** Math texts write $Wx$ for one column vector. Code applies layers to a batch of row vectors, so it's `X @ W.T`. If shapes don't match, write the shapes down.

## Cheat sheet

| Idea | Formula | One-line meaning |
|---|---|---|
| Matrix-vector product | $Ax = \sum_j x_j a_j$ | A combination of the columns |
| Dot product | $a^\top b = \lVert a\rVert\lVert b\rVert\cos\theta$ | Alignment |
| Projection onto $a$ | $\frac{a^\top b}{a^\top a}a$ | Closest point on a line |
| Normal equations | $X^\top X w = X^\top y$ | Residual ⟂ column space |
| Eigenpair | $Av = \lambda v$ | A direction that is only stretched |
| Spectral theorem | $A = Q\Lambda Q^\top$ (symmetric $A$) | Rotate, stretch, rotate back |
| SVD | $X = U\Sigma V^\top$ | Works for any matrix |
| Best rank-$k$ approximation | Keep the top $k$ singular triplets | Error $=\sum_{i>k}\sigma_i^2$ |
| Condition number | $\sigma_{\max}/\sigma_{\min}$ | How much errors get amplified |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What does it mean geometrically for a matrix to have rank 1?</summary>

All outputs lie on a single line through the origin (the column space is 1-D). The matrix is an outer product $uv^\top$: it measures the
component of the input along $v$ and outputs that many copies of $u$. Everything orthogonal to $v$ is sent to zero.
</details>

<details>
<summary>2. Why is the least-squares solution a projection?</summary>

All possible predictions $Xw$ form the column space of $X$. The least-squares prediction is the point in that subspace closest to $y$,
and the closest point in a subspace is the orthogonal projection: the residual is perpendicular to the subspace, $X^\top(y - X\hat w) = 0$.
</details>

<details>
<summary>3. What do the singular values of a data matrix tell you about PCA?</summary>

For centered $X$, $\sigma_i^2/(N-1)$ is the variance along the $i$-th principal direction, so $\sigma_i^2/\sum_j\sigma_j^2$ is the explained-variance ratio.
A fast drop-off means the data is close to low-dimensional. The number of non-zero singular values is the rank.
</details>

## Where this leads

- **Next in math:** [MATH-02 Calculus & Optimization](02-calculus-optimization.md). You'll re-derive the normal equations by calculus, and see why the condition number controls how fast gradient descent converges.
- **Used directly in:** [CORE-02 notes](../core-ml/02-linear-models-gradient-descent.md) (least squares), [CORE-06 notes](../core-ml/06-unsupervised-learning.md) (PCA), [DL-06 notes](../deep-learning/06-transformers.md) (attention is dot products), [GEN-02 notes](../llms-genai/02-adapting-llms-finetuning-rag.md) (LoRA is low rank).
