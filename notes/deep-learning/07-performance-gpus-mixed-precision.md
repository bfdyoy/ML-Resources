# DL-07 notes: Making Training Fast: GPUs, Mixed Precision, Compilation

[← Lesson DL-07](../../lessons/deep-learning/07-performance-gpus-mixed-precision.md) · [All notes](../README.md) · [← DL-06 notes](06-transformers.md) · Next: [DL-08 notes →](08-modern-architectures-moe-ssm.md)

> **Reading time** ≈ 55 min. **You need:** [DL-06 notes](06-transformers.md) §3 (attention cost, parameter counts) and [DL-03 notes](03-training-deep-networks.md) §3 (Adam's state).

---

## Where we are

You can build and train a transformer. Now the practical question: **why is it slow, and why is it out of memory?** The answers come from a little arithmetic about hardware.

A GPU does arithmetic extremely fast, but moving data to and from its memory is comparatively slow, and launching work from Python has a fixed cost. Every optimization in this lesson attacks one of those three limits.

---

## 1. Three regimes: compute, memory bandwidth, overhead

Every operation is limited by one of three things:

1. **Compute-bound:** the arithmetic units are busy. Only fewer FLOPs, or faster FLOPs (lower precision, tensor cores), help.
2. **Memory-bandwidth-bound:** the units sit waiting for data from HBM (the GPU's main memory). Moving fewer bytes helps: fusion, and lower precision.
3. **Overhead-bound:** the GPU is idle while Python, the framework, or kernel launches catch up. Doing fewer, bigger operations helps: larger batches, `torch.compile`, CUDA graphs.

### 1.1 Arithmetic intensity and the roofline

**Arithmetic intensity** (AI) is the number of FLOPs per byte moved. A GPU with peak compute $P$ (FLOP/s) and bandwidth $B$ (bytes/s) can reach at most:

```math
\text{attainable FLOP/s} = \min\big(P,\ B \times \text{AI}\big).
```

The **ridge point** $P/B$ is the AI you need to keep the compute units busy. For a modern datacenter GPU, it's in the **hundreds of FLOPs per byte** (for example, about $10^{15}$ FLOP/s of BF16 compute against about $3\times10^{12}$ B/s gives about 300).

- **Matrix multiply** of two $n\times n$ BF16 matrices: $2n^3$ FLOPs over about $3n^2\times2$ bytes, so $\text{AI} \approx n/3$. For $n = 4096$, AI ≈ 1365, **compute-bound**. Big matmuls are what GPUs are built for.
- **Element-wise op** (say `x.cos()`) on FP32: 1 FLOP per element, against 4 bytes read + 4 written, so AI ≈ 0.125. That's **thousands of times below the ridge**: memory-bound. The arithmetic is free, and you pay for the data movement.

So in a transformer, the matmuls are efficient, while the LayerNorms, activations, dropout, softmax, and residual adds are memory-bound "glue" that can take a surprising share of the time.

### 1.2 Operator fusion

`x.cos().cos()` as two kernels reads $x$, writes a temporary, reads it again, and writes the result: 4 memory trips. **Fused** into one kernel, it reads $x$ once and writes once: 2 trips, with **identical FLOPs**. For memory-bound ops, the time is proportional to the bytes moved, so fusion gives about 2× here.
`torch.compile` finds such chains automatically and generates fused kernels. FlashAttention is the most famous hand-fused kernel: it computes the softmax attention in on-chip memory tiles, and never writes the $T\times T$ matrix to HBM.

---

## 2. Number formats and mixed precision

| Format | Sign/exponent/mantissa bits | Max | Smallest normal | ~Decimal digits |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | 7 |
| FP16 | 1 / 5 / 10 | 65,504 | $6.1\times10^{-5}$ | 3.3 |
| BF16 | 1 / 8 / 7 | $3.4\times10^{38}$ | $1.2\times10^{-38}$ | 2.4 |

- **Exponent bits set the range; mantissa bits set the precision.**
- **FP16** has decent precision but a tiny range. Small gradients (say $10^{-8}$) **underflow to zero**. **Loss scaling** fixes this: multiply the loss by a large factor $S$ before backward, so the gradients are scaled up into FP16's range, then divide by $S$ before the optimizer step (skipping steps that overflow). `GradScaler` does this dynamically.
- **BF16** keeps FP32's 8 exponent bits, so it has **the same range** and gradients don't underflow: **no loss scaling needed**. It pays with lower precision, which neural nets tolerate well. It's the default on modern GPUs.

**Mixed precision** (`torch.autocast`) runs matmuls and convolutions in BF16/FP16 (fast tensor cores, half the bytes). It keeps precision-sensitive operations (reductions, softmax, the loss) in FP32, and keeps an **FP32 master copy** of the weights for the update.
(The update $\theta - \eta g$ often changes the weights by less than BF16's resolution, about $\theta/128$. Updates done in BF16 would round to zero. The demo shows it.)

---

## 3. Where the memory goes

### 3.1 The static part: about 16 bytes per parameter

For AdamW with mixed precision, the common accounting per parameter is:

| Item | Bytes |
|---|---|
| BF16 weights (used in forward/backward) | 2 |
| BF16 gradients | 2 |
| FP32 master weights | 4 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| **Total** | **16** |

So a **1B-parameter model needs about 16 GB before a single activation**, and a 7B model needs about 112 GB, which is more than one 80 GB GPU. That's why sharding the optimizer state (ZeRO/FSDP, [PROD-04](../production/04-distributed-training.md)) exists.
Inference alone, in BF16, needs only the 2 bytes per parameter.

### 3.2 The dynamic part: activations

Backprop needs the forward activations (DL-01 §3.1). Their memory scales with **batch × sequence length × layers × width**, plus the $T^2$ attention terms if the attention isn't fused. For long sequences or big batches, activations dominate.

**Gradient checkpointing** stores only some activations (for example, each block's input) and **recomputes** the rest during the backward pass. It costs about one extra forward pass (roughly 30% more compute), and it cuts activation memory from $O(L)$ layers' worth to about one block's worth plus the checkpoints.
(With $\sqrt{L}$ evenly spaced checkpoints, the total is $O(\sqrt{L})$.) It trades compute for memory.

---

## 4. Finding the bottleneck: the 35%-utilization drill

Low GPU utilization almost always comes from one of three causes:

1. **The input pipeline:** the GPU waits for the data loader (decoding JPEGs, augmentations on too few CPU workers). *Test:* time a pass over the data loader alone, with no model. If it's about as slow as training, this is the bottleneck.
   *Fix:* more `num_workers`, `pin_memory=True`, pre-processed or cached data, GPU-side augmentation.
2. **Host–device synchronization and Python overhead:** a `.item()`, `print(loss)`, or `.cpu()` every step forces the CPU to wait for the GPU, and lots of tiny ops mean launch overhead dominates.
   *Test:* the profiler shows gaps between kernels, and `cudaStreamSynchronize` or `aten::item` calls.
   *Fix:* log every N steps, use bigger batches, `torch.compile`, CUDA graphs.
3. **Slow or memory-bound kernels:** unfused glue ops, FP32 where BF16 would do, an unfused attention implementation.
   *Test:* the profiler's top kernels by time.
   *Fix:* autocast, `torch.compile`, `F.scaled_dot_product_attention` (which dispatches to FlashAttention).

**Always measure with warm-up:** the first iterations include compilation and memory allocation. Also synchronize the GPU (`torch.cuda.synchronize()`) before reading a timer, because GPU work is asynchronous.

```python
import torch, time, math
torch.manual_seed(0)

# --- Arithmetic intensity: matmul vs elementwise ------------------------------------------
def ai_matmul(n, bytes_per=2): return 2 * n**3 / (3 * n * n * bytes_per)
print(f"AI matmul n=4096 (bf16): {ai_matmul(4096):.0f} FLOP/byte;  AI cos() fp32: {1/8:.3f} FLOP/byte")

# Measured on this machine's CPU: big matmul achieves far more FLOP/s than an elementwise op
n = 1024
a, b = torch.randn(n, n), torch.randn(n, n)
x = torch.randn(n * n * 8)
for _ in range(3): a @ b; x.cos()                        # warm-up
t0 = time.perf_counter(); [a @ b for _ in range(10)]; t_mm = (time.perf_counter() - t0) / 10
t0 = time.perf_counter(); [x.cos() for _ in range(10)]; t_ew = (time.perf_counter() - t0) / 10
print(f"matmul: {2 * n**3 / t_mm / 1e9:7.1f} GFLOP/s   elementwise cos: {x.numel() / t_ew / 1e9:7.2f} GFLOP/s")

# --- FP16 underflow vs BF16 range; BF16 update rounding --------------------------------------
g = torch.tensor(1e-8)
print("1e-8 as fp16:", g.half().item(), "  as bf16:", g.bfloat16().item())
print("fp16 max:", torch.finfo(torch.float16).max, "  bf16 max:", f"{torch.finfo(torch.bfloat16).max:.2e}")
w = torch.tensor(1.0)
print("1.0 + 1e-3 in bf16:", (w.bfloat16() + 1e-3).item(), " in fp32:", (w + 1e-3).item(),
      "  <- why master weights stay in fp32")

# --- Memory accounting for AdamW + mixed precision -----------------------------------------
def train_mem_gb(params, bytes_per_param=16): return params * bytes_per_param / 1e9
for p in [124e6, 1e9, 7e9, 70e9]:
    print(f"{p/1e9:6.3f}B params: ~{train_mem_gb(p):7.1f} GB static training state, {p*2/1e9:6.1f} GB bf16 weights for inference")
```

```python
# --- Gradient checkpointing: same gradients, recomputed forward -----------------------------------
from torch.utils.checkpoint import checkpoint
calls = {"n": 0}
def block(x, W):
    calls["n"] += 1
    return torch.tanh(x @ W)
W = torch.randn(64, 64, requires_grad=True); x = torch.randn(8, 64)
y = block(block(x, W), W).sum(); y.backward(); g_plain = W.grad.clone(); W.grad = None
plain_calls = calls["n"]; calls["n"] = 0
y = checkpoint(lambda t: block(block(t, W), W), x, use_reentrant=False).sum(); y.backward()
print("same gradient:", torch.allclose(g_plain, W.grad), f"| block calls: plain {plain_calls}, checkpointed {calls['n']} (forward recomputed)")
```

---

## Pitfalls & misconceptions

- **Timing GPU code without synchronizing**, or without a warm-up.
- **Calling `.item()` every step** in the training loop.
- **FP16 without a `GradScaler`.** Gradients silently underflow.
- **Measuring "it got faster" on a single noisy run.** Repeat, and report the median.
- **Optimizing compute when the data loader is the bottleneck.** Profile first.

## Cheat sheet

| Item | Value / rule |
|---|---|
| Roofline | $\min(P,\ B\cdot\text{AI})$; ridge point = $P/B$ |
| AI of an $n\times n$ BF16 matmul | $\approx n/3$ |
| AI of an elementwise FP32 op | $\approx 1/8$ |
| BF16 vs FP16 | Same range as FP32 vs a bigger mantissa; FP16 needs loss scaling |
| AdamW mixed-precision memory | $\approx 16$ bytes/param + activations |
| Checkpointing | about +30% compute for much less activation memory |
| Low utilization | data loader / sync & overhead / slow kernels |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Arithmetic intensity: why a big matmul is compute-bound and an elementwise op memory-bound.</summary>

AI = FLOPs per byte moved. A matmul does $O(n^3)$ work on $O(n^2)$ data, so its AI grows with $n$ (about $n/3$ in BF16), above the GPU's ridge point. An elementwise op does about 1 FLOP per 8 bytes, far below the ridge, so it waits on memory.
</details>

<details>
<summary>2. Why fusing x.cos().cos() helps with identical FLOPs.</summary>

Unfused, it moves data 4 times (read, write the temporary, read the temporary, write). Fused, it moves data twice. Memory-bound ops take time proportional to the bytes moved, so fusion is about 2× faster for the same arithmetic.
</details>

<details>
<summary>3. Why BF16 usually doesn't need loss scaling.</summary>

It has FP32's 8-bit exponent, so it has the same dynamic range, and small gradients don't underflow. FP16's 5-bit exponent can't represent values below about $6\times10^{-5}$ as normal numbers, hence loss scaling.
</details>

<details>
<summary>4. Training memory for a 1B model with AdamW in mixed precision.</summary>

About 16 bytes/param: 2 (BF16 weights) + 2 (BF16 gradients) + 4 (FP32 master) + 4 + 4 (Adam m, v). That's about 16 GB, plus activations, which depend on batch size, sequence length, and checkpointing.
</details>

<details>
<summary>5. What does gradient checkpointing trade?</summary>

Extra compute (recomputing the forward pass of the checkpointed segments during backward, roughly +30%) for a large reduction in stored activation memory.
</details>

<details>
<summary>6. GPU utilization at 35%: three culprits and how to tell them apart.</summary>

(1) The data loader: time it alone, and look for GPU idle gaps while the CPU workers are busy. (2) Sync and Python overhead: `.item()`/print calls and many tiny kernels, which show as gaps and sync calls in the profiler. (3) Inefficient kernels: the profiler's top ops are unfused or FP32. Fixes are in §4.
</details>

## Where this leads

Next: [DL-08 notes](08-modern-architectures-moe-ssm.md). Many "modern architecture" choices are really *performance* choices in disguise. GQA shrinks the KV cache, MoE adds parameters without adding FLOPs per token, and SSMs avoid quadratic attention.
With this lesson's arithmetic, you'll see exactly why each one exists.
