# PROD-04 notes: Distributed Training at Scale

[← Lesson PROD-04](../../lessons/production/04-distributed-training.md) · [All notes](../README.md) · [← PROD-03 notes](03-testing-monitoring-drift.md) · Next: [PROD-05 notes →](05-llmops-genai-platforms.md)

> **Reading time** ≈ 60 min. **You need:** [DL-07 notes](../deep-learning/07-performance-gpus-mixed-precision.md) §3 (the 16 bytes/param accounting, activations, checkpointing) and [DL-06 notes](../deep-learning/06-transformers.md) §3 (transformer costs).

---

## Where we are

DL-07 showed that a 7B model needs about 112 GB of training state, more than any single GPU. And even when a model fits, one GPU is too slow for trillions of tokens. This lesson covers the ways to split training across GPUs, and the communication arithmetic that decides which split to use.

---

## 1. Memory, revisited

Per parameter, with AdamW and mixed precision (DL-07 §3.1): **2** (BF16 weights) + **2** (gradients) + **12** (FP32 master weights, $m$, $v$) = **16 bytes**. Plus activations.

*7B model:* $7\times10^9\times16 = 112$ GB of static state.

---

## 2. Data parallelism (DDP)

Each of $N$ GPUs holds a **full copy** of the model, processes a **different mini-batch**, computes its gradients, and then all the GPUs **average** their gradients, so every copy takes the identical update. The averaging is an **all-reduce**.

### 2.1 Ring all-reduce and its cost

Arrange the GPUs in a ring and split the gradient (size $S$ bytes) into $N$ chunks. In a **reduce-scatter** phase ($N - 1$ steps), each GPU ends up owning the full sum of one chunk. Then an **all-gather** phase ($N - 1$ steps) circulates the summed chunks. Each GPU sends:

```math
2\,\frac{N-1}{N}\,S \ \text{bytes} \quad(\to 2S \text{ for large } N),
```

**independent of $N$**. That's why ring all-reduce scales: the time is about $2S/\text{bandwidth}$ whatever the cluster size (plus latency terms).

### 2.2 Overlap with the backward pass

Gradients become available **layer by layer, from the last layer backwards**. DDP groups them into **buckets** and starts all-reducing a bucket as soon as it's ready, while backprop continues on earlier layers. Communication then hides behind compute, as long as compute per step is large enough (a big enough per-GPU batch).

**DDP's limit:** every GPU still holds all 16 bytes/param, so DDP alone can't train a model that doesn't fit on one GPU.

---

## 3. ZeRO / FSDP: shard the redundant state

In DDP, all $N$ GPUs store identical optimizer states, gradients, and weights: $N$-fold redundancy. **ZeRO** shards them:

| Stage | Sharded across GPUs | Bytes/param per GPU (mixed-precision Adam) | 7B on 8 GPUs |
|---|---|---|---|
| DDP | nothing | $16$ | 112 GB |
| ZeRO-1 | optimizer states (12 B) | $2 + 2 + 12/N$ | 38.5 GB |
| ZeRO-2 | + gradients | $2 + (2 + 12)/N$ | 26.25 GB |
| ZeRO-3 / FSDP | + weights | $16/N$ | 14 GB |

- **The cost of ZeRO-3/FSDP:** before each layer's forward (and again in backward), the GPUs **all-gather** that layer's weight shards, use them, and discard them. Gradients are **reduce-scattered** instead of all-reduced.
  The communication is about 1.5× DDP's, and it overlaps with compute through prefetching. Activations aren't sharded, so combine this with activation checkpointing.
- **The rule of thumb:** use DDP if the model fits; FSDP or ZeRO-3 when it doesn't, or when you need room for bigger batches.

---

## 4. Model parallelism: splitting the computation itself

### 4.1 Tensor parallelism (TP)

Split each big **matrix** across GPUs. In an MLP, split the first matrix by **columns** (each GPU computes a slice of the hidden units, with no communication needed), and the second by **rows** (each GPU produces a partial sum). Then **all-reduce** the outputs.
Attention heads split naturally across GPUs. That means **an all-reduce of activations in every layer, forward and backward**: a lot of communication, on the critical path, every step. It needs the very fast **intra-node** interconnect (NVLink, hundreds of GB/s), not the inter-node network (tens of GB/s). Hence **TP stays inside a node** (typically TP ≤ 8).

### 4.2 Pipeline parallelism (PP)

Split the **layers** into $p$ stages on different GPUs, and pass the activations from stage to stage. A naive pipeline leaves most GPUs idle. Splitting each batch into $m$ **micro-batches** keeps the stages busy, but the start and end of each step still have a **bubble** of idle time:

```math
\text{bubble fraction} \approx \frac{p - 1}{m + p - 1}.
```

With $p = 8$ and $m = 32$, about 18% is idle. More micro-batches shrink the bubble (with more activation memory), and interleaved schedules reduce it further. The communication is only point-to-point activations between neighbouring stages, so PP **tolerates slower inter-node links**.

### 4.3 Sequence/context and expert parallelism

- **Sequence / context parallelism:** split the *sequence* dimension. This is needed for very long contexts, where the activations of a single sequence don't fit. Ring attention passes the key/value blocks around.
- **Expert parallelism:** place different MoE experts (DL-08 §4) on different GPUs, and route tokens to them with **all-to-all** communication.

### 4.4 "5D parallelism"

Large runs **compose** them: TP inside each node × PP across nodes × DP/FSDP across replicas × context parallelism for long sequences × EP for MoE. The design follows the bandwidth hierarchy: **the most communication-hungry split goes on the fastest links.**

---

## 5. Gradient accumulation

Run $k$ micro-batches forward and backward, **summing the gradients** (the PyTorch `+=` from DL-02), then take one optimizer step. The gradient is **mathematically identical** to that of a $k\times$ larger batch (for losses averaged over examples, and except for BatchNorm, whose statistics are per micro-batch).

- **What it gives:** large-batch optimization dynamics under a memory limit.
- **What it doesn't give:** any speed-up. It's sequential: $k$ passes take $k\times$ the time, with no extra parallelism. And it doesn't give the BatchNorm behaviour of a true large batch.

---

## 6. Debugging poor scaling (the lesson's question 6: 8 GPUs give 3×)

Scaling efficiency is $3/8 = 37.5\%$. Here are the likely bottlenecks and how to measure each:

1. **The input pipeline:** the data loader can't feed 8× the throughput (CPU decoding, slow storage). Measure loader throughput alone, and watch GPU idle gaps in the profiler.
2. **Communication not hidden:** the per-GPU batch is small, so compute is too short to overlap the all-reduce, or the interconnect is slow (PCIe or Ethernet instead of NVLink or InfiniBand). The profiler shows NCCL kernels on the critical path. Measure the step time against the per-GPU batch size.
3. **Synchronization and stragglers:** one slow GPU or rank (thermal throttling, uneven data shards, variable sequence lengths) makes everyone wait at every all-reduce. Compare per-rank step times.
4. **Host-side overhead:** logging and `.item()` calls, or checkpointing on every rank.
5. **Not actually 8× the work:** the global batch size was kept fixed, so each GPU gets a tiny batch with low utilization.

A good practice is to plot **throughput (samples/s) vs number of GPUs** against the ideal line. Weak scaling (a fixed batch *per GPU*) is the fair test.

```python
import numpy as np, torch
rng = np.random.default_rng(0); torch.manual_seed(0)

# --- Memory per GPU for each ZeRO stage --------------------------------------------------------
def per_gpu_gb(P, N, stage):
    if stage == 0: b = 16
    elif stage == 1: b = 2 + 2 + 12 / N
    elif stage == 2: b = 2 + (2 + 12) / N
    else: b = 16 / N
    return P * b / 1e9
for P in [7e9, 70e9]:
    print(f"{P/1e9:.0f}B on 8 GPUs:", {f"ZeRO-{s}" if s else "DDP": round(per_gpu_gb(P, 8, s), 1) for s in range(4)}, "GB/GPU (+activations)")

# --- Ring all-reduce simulated: exact sum, and bytes sent per GPU ---------------------------------------
N, S = 4, 12                                       # 4 workers, gradient of 12 numbers
grads = [rng.normal(size=S) for _ in range(N)]
chunks = [np.array_split(g.copy(), N) for g in grads]
sent = 0
for step in range(N - 1):                          # reduce-scatter
    for r in range(N):
        c = (r - step) % N
        chunks[(r + 1) % N][c] = chunks[(r + 1) % N][c] + chunks[r][c]; sent += len(chunks[r][c])
for step in range(N - 1):                          # all-gather
    for r in range(N):
        c = (r + 1 - step) % N
        chunks[(r + 1) % N][c] = chunks[r][c].copy(); sent += len(chunks[r][c])
result = [np.concatenate(ch) for ch in chunks]
print("every worker holds the exact sum:", all(np.allclose(r, sum(grads)) for r in result))
print(f"values sent per worker: {sent / N:.0f}  vs formula 2(N-1)/N*S = {2 * (N - 1) / N * S:.0f}")

# --- Pipeline bubble ---------------------------------------------------------------------------------------
for p, m in [(4, 4), (8, 8), (8, 32), (8, 128)]:
    print(f"stages p={p}, micro-batches m={m:3d}: bubble {(p - 1) / (m + p - 1):.0%}")
```

```python
# --- Gradient accumulation == large batch (for a mean loss) ----------------------------------------------
model = torch.nn.Linear(10, 1); X, y = torch.randn(64, 10), torch.randn(64, 1)
loss_fn = torch.nn.MSELoss()
model.zero_grad(); loss_fn(model(X), y).backward()
g_big = model.weight.grad.clone()
model.zero_grad()
for xb, yb in zip(X.chunk(4), y.chunk(4)):
    (loss_fn(model(xb), yb) / 4).backward()           # divide by the number of micro-batches
print("accumulated gradient == full-batch gradient:", torch.allclose(g_big, model.weight.grad, atol=1e-6))
```

---

## Pitfalls & misconceptions

- **Expecting DDP to fit a bigger model.** It replicates everything. Use FSDP or ZeRO.
- **Tensor parallelism across nodes** over slow links.
- **Forgetting to divide the loss by the number of accumulation steps** (or using a sum loss inconsistently).
- **Comparing scaling at a fixed global batch.** The per-GPU work shrinks, so efficiency looks terrible.
- **Ignoring activations** in the memory budget. ZeRO doesn't shard them, so use checkpointing.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Training state | 16 B/param (mixed-precision AdamW) |
| Ring all-reduce traffic | $2\frac{N-1}{N}S$ per GPU |
| ZeRO-1/2/3 per GPU | $4 + 12/N$ / $2 + 14/N$ / $16/N$ B/param |
| Pipeline bubble | $(p-1)/(m+p-1)$ |
| TP | per-layer all-reduce → keep it on NVLink, inside a node |
| Gradient accumulation | same gradient as a $k\times$ batch, $k\times$ the time |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Memory for a 7B model with AdamW (mixed precision), and ZeRO savings.</summary>

About 112 GB without sharding (16 B/param), plus activations. On 8 GPUs: ZeRO-1 ≈ 38.5 GB, ZeRO-2 ≈ 26.3 GB, ZeRO-3 ≈ 14 GB per GPU (the demo prints these).
</details>

<details>
<summary>2. What all-reduce does in DDP, and why it overlaps backward.</summary>

It sums (and then averages) the gradients across GPUs, so all the replicas apply the same update. Gradients for later layers are ready first, so bucketed all-reduces start while backprop continues through earlier layers, hiding the communication behind compute.
</details>

<details>
<summary>3. Tensor vs pipeline parallelism: what each splits and costs.</summary>

TP splits each layer's matrices across GPUs, which costs activation all-reduces in every layer, forward and backward (high bandwidth needed). PP splits the layers into stages, which costs pipeline bubbles of idle time (reduced with micro-batches) plus point-to-point activation transfers.
</details>

<details>
<summary>4. Why keep TP inside a node?</summary>

Its frequent, blocking all-reduces in every layer need the very high bandwidth and low latency of NVLink within a node. Inter-node links are far slower, so TP across nodes would stall compute.
</details>

<details>
<summary>5. Gradient accumulation vs a truly larger batch.</summary>

It gives the same averaged gradient (large-batch optimization) under a memory cap. It gives no speed-up (the micro-batches run sequentially), and BatchNorm still sees micro-batch statistics.
</details>

<details>
<summary>6. 1→8 GPUs gives 3×: bottlenecks and how to measure.</summary>

The data loader (benchmark it alone; GPU idle gaps), unhidden communication from small per-GPU batches or slow links (NCCL time in the profiler; vary the batch), stragglers and synchronization (per-rank step times), host overhead (logging and sync calls), and a fixed global batch shrinking the per-GPU work (use weak scaling) (§6).
</details>

## Where this leads

Next: [PROD-05 notes](05-llmops-genai-platforms.md), the last lesson of Path 4. Most teams won't train models at this scale, but they will *run* LLM applications. LLMOps applies everything from Paths 3 and 4 to that job.
