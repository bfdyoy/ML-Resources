# EL-08 notes: GPU Programming for ML

[← Lesson EL-08](../../lessons/electives/08-gpu-programming.md) · [All notes](../README.md) · [← EL-07 notes](07-mechanistic-interpretability.md) · Next: [EL-09 notes →](09-audio-and-speech.md)

> **Reading time** ≈ 60 min. **You need:** [DL-07 notes](../deep-learning/07-performance-gpus-mixed-precision.md) (the roofline, arithmetic intensity, fusion) and [DL-06 notes](../deep-learning/06-transformers.md) §3 (attention's memory cost). The code here is CPU-only NumPy that *simulates* GPU memory behaviour, so it runs anywhere. The Triton kernel is shown for reading, not run.

---

## Where we are

DL-07 treated the GPU as a box with a FLOP rate and a bandwidth. This lesson opens it: how thousands of threads are organized, where data lives, and the three or four patterns (coalescing, tiling, fusion, online algorithms) behind every fast kernel, up to FlashAttention.

---

## 1. The execution model

- A **kernel** is launched over a **grid** of **thread blocks**. Each block runs on one **streaming multiprocessor (SM)**, and its threads can share fast memory and synchronize.
- The hardware executes threads in groups of 32 called **warps**, in lockstep (**SIMT**): one instruction, 32 threads, each on its own data.
- **Warp divergence:** if threads in the same warp take different branches of an `if`, the warp runs **both** paths one after the other, with the inactive threads masked off. A 50/50 split halves the throughput for that code. Branch on values that are uniform across a warp where you can.
- **Occupancy:** how many warps are resident per SM, relative to the maximum. While one warp waits hundreds of cycles for memory, the scheduler runs another. High occupancy hides latency. It's limited by registers and shared memory per thread or block.

## 2. The memory hierarchy

| Level | Size (order of magnitude) | Bandwidth / latency | Scope |
|---|---|---|---|
| Registers | ~256 KB per SM | fastest | per thread |
| Shared memory / L1 (SRAM) | ~100–250 KB per SM | ~10+ TB/s aggregate, low latency | per block |
| L2 cache | tens of MB | — | whole GPU |
| HBM (global memory) | tens of GB | ~2–3 TB/s, hundreds of cycles | whole GPU |

**Almost all kernel optimization means moving data from HBM fewer times**, and reusing it from SRAM and registers as often as possible.

### 2.1 Coalescing

HBM is read in aligned segments (say 32- or 128-byte transactions). If the 32 threads of a warp read **consecutive** addresses (thread $k$ reads element $k$), the warp's request is served by **one or a few** transactions: that's **coalesced**.
If each thread reads with a large stride (thread $k$ reads element $32k$, as when walking down a column of a row-major matrix), the same 32 values need **up to 32** separate transactions, which wastes most of the bandwidth. Map threads so that neighbouring threads touch neighbouring memory.

**Bank conflicts:** shared memory is split into 32 banks. If several threads in a warp hit different addresses in the *same* bank, the accesses serialize. Padding arrays (for example, 33 columns instead of 32) is the classic fix.

---

## 3. Tiled matrix multiplication

$C = AB$ with $A\in\mathbb{R}^{M\times K}$ and $B\in\mathbb{R}^{K\times N}$. **Naive:** each thread computes one $C_{ij}$, reading a row of $A$ and a column of $B$ from HBM: $2K$ loads per output, so about $2MNK$ loads in total. Every value of $A$ is loaded $N$ times.

**Tiled:** each block computes a $T\times T$ tile of $C$. It loops over $K$ in chunks, loading a $T\times T$ tile of $A$ and one of $B$ into **shared memory** once, after which every thread in the block reuses them $T$ times. The HBM traffic drops to about

```math
\text{loads} \approx \frac{2MNK}{T},
```

so the **arithmetic intensity rises by a factor of $T$** (DL-07 §1.1). With register tiling (each thread computing several outputs), tensor cores, and double-buffering (loading the next tile while computing on the current one), you get cuBLAS-level performance. The code counts the loads both ways.

**Debug: a hand-written matmul is 10× slower than cuBLAS.** Check:

1. **Memory access pattern:** are the global loads coalesced? (The profiler shows the memory throughput and the sectors per request.)
2. **Tiling and reuse:** is it reading straight from HBM in the inner loop, instead of from shared memory or register tiles? Are there bank conflicts?
3. **Hardware utilization:** is it using **tensor cores** (BF16/FP16/TF32 MMA instructions)? Is occupancy limited by register or shared-memory use? Are the tile sizes and launch configuration sensible?

Profile with Nsight Compute: achieved memory bandwidth, compute throughput, and occupancy.

---

## 4. Fused, numerically stable softmax

Softmax over a row needs the **max** (for stability, MATH-03: $e^{x - m}$ never overflows) and the **sum** of the exponentials. Naively, that's three passes over HBM (max, sum, normalize), with intermediate tensors written in between.
A **fused** kernel loads the row once into SRAM or registers and does everything there.

Rows too long to fit at once use the **online softmax**, which keeps a running max $m$ and a running sum $\ell$ and **rescales** the sum whenever the max increases:

```math
m' = \max(m, x),\qquad \ell' = \ell\,e^{m - m'} + e^{x - m'} .
```

At the end, $\text{softmax}(x_i) = e^{x_i - m}/\ell$. One streaming pass computes exact statistics. That's the key to FlashAttention.

## 5. FlashAttention

Standard attention writes the $N\times N$ score matrix $S = QK^\top$ to HBM, reads it back for the softmax, writes $P$, and reads it again for $PV$. That's $O(N^2)$ memory and $O(N^2)$ HBM traffic. For long sequences, attention is **memory-bound** (DL-07).

**FlashAttention:**

1. Splits $Q$ into row blocks, and $K, V$ into column blocks, each sized to fit in SRAM.
2. For each $Q$ block, streams over the $K, V$ blocks: computes the block's scores in SRAM, updates the **online softmax** statistics ($m$, $\ell$) per row, and **rescales and accumulates** the output, $O \leftarrow O\,e^{m - m'} + e^{S - m'}V_\text{block}$.
3. Writes only the final $O$ (size $N\times d$) and the per-row log-sum-exp statistics to HBM.

The **$N\times N$ matrix never exists in HBM**, so memory is $O(N)$ and HBM traffic falls sharply. The result is *exact* attention, not an approximation, typically 2–4× faster, and it allows much longer contexts.
For the **backward pass**, instead of storing $P$ (an $N^2$ matrix), it **recomputes** the attention blocks from $Q$, $K$, $V$ and the saved per-row statistics. That's extra FLOPs, but FLOPs are cheap and memory traffic isn't. The same trade as gradient checkpointing (DL-07 §3.2).

```python
import numpy as np
rng = np.random.default_rng(0)

# --- HBM loads: naive vs tiled matmul (counting, not timing) ------------------------------------
def naive_loads(M, N, K): return M * N * 2 * K                     # each output reads a row and a column
def tiled_loads(M, N, K, T):
    tiles = (M // T) * (N // T) * (K // T)                         # tile-pairs loaded into shared memory
    return tiles * 2 * T * T
M = N = K = 1024
for T in [1, 16, 64, 128]:
    print(f"tile {T:3d}: HBM loads {tiled_loads(M, N, K, T):.2e}  (naive {naive_loads(M, N, K):.2e}), arithmetic intensity x{naive_loads(M,N,K)/tiled_loads(M,N,K,T):.0f}")

# --- Coalescing: 128-byte segments touched by one warp (32 threads x 4-byte floats) ----------------------
def segments(stride):
    addrs = np.arange(32) * stride * 4
    return len(np.unique(addrs // 128))
for stride in [1, 2, 8, 32]:
    print(f"stride {stride:2d}: {segments(stride):2d} memory transactions for 32 floats")

# --- Online softmax == two-pass softmax ----------------------------------------------------------------------
x = rng.normal(0, 5, 1000)
m, l = -np.inf, 0.0
for v in x:                                                         # one streaming pass
    m_new = max(m, v); l = l * np.exp(m - m_new) + np.exp(v - m_new); m = m_new
print("online softmax matches:", np.allclose(np.exp(x - m) / l, np.exp(x - x.max()) / np.exp(x - x.max()).sum()))
```

```python
# --- FlashAttention's forward algorithm in NumPy: block-wise, never forming the N x N matrix ---------------------------
N, d, Bq, Bk = 256, 32, 64, 64
Q, K, V = rng.normal(size=(N, d)), rng.normal(size=(N, d)), rng.normal(size=(N, d))
scale = 1 / np.sqrt(d)

S = Q @ K.T * scale                                                   # reference: full attention
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
ref = P @ V

out = np.zeros((N, d))
for qs in range(0, N, Bq):
    q = Q[qs:qs + Bq]
    m = np.full(len(q), -np.inf); l = np.zeros(len(q)); o = np.zeros((len(q), d))
    for ks in range(0, N, Bk):
        s = q @ K[ks:ks + Bk].T * scale                               # a Bq x Bk block, lives in "SRAM"
        m_new = np.maximum(m, s.max(1))
        p = np.exp(s - m_new[:, None])
        corr = np.exp(m - m_new)                                      # rescale the old statistics
        l = l * corr + p.sum(1)
        o = o * corr[:, None] + p @ V[ks:ks + Bk]
        m = m_new
    out[qs:qs + Bq] = o / l[:, None]
print("block-wise (FlashAttention-style) == full attention:", np.allclose(out, ref))
print(f"largest score block held at once: {Bq}x{Bk} = {Bq*Bk:,} values, vs the full matrix {N*N:,}")
```

A Triton kernel for a fused row softmax, for reading (it needs a GPU and `triton`, so it isn't executed here):

```py
import triton, triton.language as tl

@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)                          # one program instance per row
    cols = tl.arange(0, BLOCK)                      # BLOCK >= n_cols, a power of 2
    x = tl.load(in_ptr + row * row_stride + cols, mask=cols < n_cols, other=-float("inf"))
    x = x - tl.max(x, axis=0)                       # numerically stable
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)                   # everything stays in registers: one HBM read, one write
    tl.store(out_ptr + row * row_stride + cols, y, mask=cols < n_cols)
```

---

## Pitfalls & misconceptions

- **Optimizing FLOPs in a memory-bound kernel.**
- **Uncoalesced access** from the wrong thread-to-data mapping (rows vs columns).
- **Benchmarking without warm-up and synchronization**, or on tiny sizes.
- **Ignoring tensor cores**, which need the right dtypes and tile shapes.
- **"FlashAttention approximates attention."** It's exact. Only the order of operations changes.

## Cheat sheet

| Item | Rule / formula |
|---|---|
| Warp | 32 threads in lockstep; divergence serializes the branches |
| Coalescing | neighbouring threads → neighbouring addresses |
| Tiling | HBM loads $\approx 2MNK/T$; intensity × $T$ |
| Online softmax | $m' = \max(m,x)$, $\ell' = \ell e^{m-m'} + e^{x-m'}$ |
| FlashAttention | tile Q/K/V in SRAM, online softmax, never store $N\times N$; recompute in backward |
| Debug a slow kernel | coalescing → shared-memory tiling/bank conflicts → tensor cores/occupancy |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why warps matter; what is warp divergence?</summary>

The hardware schedules and executes 32 threads together on one instruction stream. If they branch differently, both paths execute in sequence with threads masked off, which reduces throughput in proportion to the divergence.
</details>

<details>
<summary>2. What makes a global-memory access coalesced?</summary>

The threads of a warp access consecutive, aligned addresses, so their requests combine into a minimal number of memory transactions (the demo: 1 transaction for stride 1, 32 for stride 32).
</details>

<details>
<summary>3. How does tiling through shared memory reduce HBM traffic?</summary>

Each tile of A and B is loaded from HBM once and reused by all the threads in the block, about $T$ times, so the total HBM loads fall from about $2MNK$ to about $2MNK/T$, raising the arithmetic intensity by $T$.
</details>

<details>
<summary>4. Writing softmax as a fused, numerically stable kernel.</summary>

Load the row into registers or SRAM once, subtract the row max, exponentiate, sum, divide, and write once. For long rows, stream with the online softmax (a running max and a rescaled running sum).
</details>

<details>
<summary>5. Why FlashAttention never materializes N×N, and what it recomputes.</summary>

It processes Q, K, V in SRAM-sized blocks, using the online softmax to combine the block results exactly, so only $O(N)$ outputs and statistics go to HBM. In the backward pass, it recomputes the attention blocks from Q, K, V and the saved log-sum-exp statistics, instead of storing the $N\times N$ probabilities.
</details>

<details>
<summary>6. A hand-written matmul is 10× slower than cuBLAS.</summary>

Check the global-memory coalescing, the shared-memory and register tiling (and bank conflicts), and tensor-core use, occupancy, and launch configuration, using Nsight Compute metrics.
</details>

## Where this leads

Next: [EL-09 notes](09-audio-and-speech.md). Back to data modalities: audio is a 1-D signal that ML mostly treats as an image (the spectrogram) or as a sequence of tokens.
