# MATH-01: Linear Algebra for ML (just-in-time)

| Track | Time | Level | Used by |
|---|---|---|---|
| Math | ~6 h (in pieces) | L1→L2 | CORE-02, CORE-06, DL-01+ |

**How to use this page:** Don't do it all up front. When a lesson links here, do only the matching block, and come back later.

## Block A: Vectors, matrices, and transformations (for CORE-02, DL-01)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Ch. 1–4 (vectors, span, linear transformations, matrix multiplication as composition) | 45 min |
| **Read** | [MML book](https://mml-book.github.io/) ([PDF](https://mml-book.github.io/book/mml-book.pdf)) | Ch. 2 "Linear Algebra" §2.1–2.3 (systems of equations, matrices, solving linear systems), §2.6–2.7 (basis, rank, linear mappings) | 1.5 h |

## Block B: Geometry (dot products, norms, projections) (for CORE-02 least squares, DL-06 attention)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | "Dot products and duality" | 15 min |
| **Read** | [MML book](https://mml-book.github.io/) | Ch. 3 "Analytic Geometry": norms, inner products, angles, orthogonal projections (§3.8) | 1 h |

## Block C: Eigen-decomposition and SVD (for CORE-06 PCA)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | "Change of basis", "Eigenvectors and eigenvalues" | 35 min |
| **Read** | [MML book](https://mml-book.github.io/) | Ch. 4 "Matrix Decompositions": eigendecomposition (§4.2, §4.4), SVD (§4.5) | 1.5 h |

## Self-check
1. What does it mean geometrically for a matrix to have rank 1?
2. Why is the least-squares solution a projection?
3. What do the singular values of a data matrix tell you about PCA?
