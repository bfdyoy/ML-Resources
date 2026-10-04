# GEN-08 notes: Efficient LLM Inference

[← Lesson GEN-08](../../lessons/llms-genai/08-efficient-llm-inference.md) · [All notes](../README.md) · [← GEN-07 notes](07-post-training-alignment-reasoning.md) · Next: [GEN-04 notes →](04-generative-models-diffusion.md)

> **Reading time** ≈ 60 min. **You need:** [DL-07 notes](../deep-learning/07-performance-gpus-mixed-precision.md) (the roofline, memory-bound vs compute-bound, number formats) and [DL-08 notes](../deep-learning/08-modern-architectures-moe-ssm.md) §3 (the KV cache, GQA).

---

## Where we are

Training happens once, and inference happens billions of times. Serving cost and latency are governed by a few numbers you can compute on paper:

- bytes of weights;
- bytes of KV cache;
- memory bandwidth;
- batch size.

Every technique in this lesson moves one of those.

---

## 1. Prefill vs decode

Generation has two phases with opposite profiles:

- **Prefill:** process the whole prompt in one forward pass. All $T$ tokens go through each matmul together, which gives high arithmetic intensity. **It's compute-bound.** It sets the **time to first token (TTFT)**, which grows with prompt length (and quadratically in the attention part).
- **Decode:** generate one token per forward pass. Each step must **read every weight from memory** to do a matrix–*vector* product for a single token: about 2 FLOPs per 2 bytes in BF16, an arithmetic intensity of about 1, far below the GPU's ridge point of hundreds. **It's memory-bandwidth-bound.**

### 1.1 The decode speed limit

At batch size 1, every token needs one full read of the weights (plus the KV cache). So:

```math
\text{tokens/s} \lesssim \frac{\text{memory bandwidth}}{\text{bytes of weights} + \text{bytes of KV cache read}} .
```

*Example:* a 7B model in BF16 (14 GB) on a GPU with 3.35 TB/s bandwidth gives at most about $3.35\times10^{12}/1.4\times10^{10} \approx 240$ tokens/s, however many FLOPs the GPU has. **Shrink the bytes** (quantization) or **share the read** across many sequences (batching).

### 1.2 Why batching helps decode

With $B$ sequences decoding together, one read of the weights serves $B$ tokens. The arithmetic intensity grows about $B$-fold, so throughput rises almost linearly until either the computation becomes compute-bound or the **KV cache fills the memory**.
Batching barely slows each step at first, so it's close to free throughput. Its limit is memory for the caches.

---

## 2. The KV cache

From DL-08 §3.1, the cache per sequence is $2\cdot n_\text{layers}\cdot T\cdot n_\text{kv}\cdot d_\text{head}\cdot\text{bytes}$.

*The lesson's question 2:* 32 layers, 8 KV heads, $d_\text{head} = 128$, a 32k context, BF16:

```math
2 \times 32 \times 32768 \times 8 \times 128 \times 2 \text{ bytes} = 4.29\times10^{9}\ \text{bytes} \approx 4.3\ \text{GB per sequence}.
```

Ten concurrent 32k-context users need 43 GB of cache, which can exceed the weights. The tools for this:

- **GQA/MQA** (fewer KV heads; DL-08 §3.2).
- **Paged attention (vLLM):** store the cache in fixed-size *blocks*, the way an OS stores memory in pages. There's no need to reserve the maximum length up front, so fragmentation waste falls from very high to near zero, and many more sequences fit.
- **Prefix caching:** requests that share a prefix (a long system prompt, the same document) reuse its cached keys and values, skipping that part of the prefill.
- **KV quantization** (8-bit or 4-bit caches) and eviction or sliding windows.

---

## 3. Weight quantization

### 3.1 The basic maps

**Absmax (symmetric) INT8:** for a group of weights, $s = \max\lvert w\rvert/127$, $q = \operatorname{round}(w/s)$, and to dequantize, $\hat w = s\,q$.
**Zero-point (asymmetric):** map $[\min, \max]$ onto $[0, 255]$ with a scale and an offset. This is better for skewed ranges.
The rounding error per weight is at most $s/2$, so **one large value in a group inflates $s$ and wrecks the precision for everyone else in that group**. Hence small **groups** (for example 128 weights, each with its own scale) for 4-bit formats.

### 3.2 Outlier features

In LLMs beyond a few billion parameters, a handful of hidden **dimensions** carry activations 10–100× larger than the rest, consistently across tokens. Naive per-tensor INT8 quantization of the activations either clips those outliers (destroying the model) or uses a scale so large that every normal value rounds to zero.
Fixes:

- keep the outlier dimensions in 16-bit (LLM.int8());
- move the difficulty from activations to weights with a per-channel rescaling (SmoothQuant);
- or quantize **weights only** (the common 4-bit approach), keeping activations in 16-bit.

### 3.3 GPTQ vs AWQ

- **GPTQ** quantizes the weights **column by column**, and after each column, **updates the remaining columns to compensate** for the rounding error. It minimizes each layer's output error $\lVert WX - \hat WX\rVert^2$ on calibration data, using second-order (Hessian $\approx XX^\top$) information.
- **AWQ** (activation-aware) observes that a small fraction of weight channels, those that multiply large activations, matter most. It **scales up those salient channels before quantizing**, so they lose less relative precision. There's no back-propagation or reconstruction step, so it's simple and robust.
- **GGUF** is a file format (for llama.cpp) bundling various k-quant schemes for CPU and Apple-silicon inference.

**Always measure the quality impact** on *your* task (perplexity plus task evals). 4-bit is usually close to the 16-bit model, while 3-bit and below degrade noticeably.

---

## 4. Speculative decoding

Decode is memory-bound, so a forward pass that checks $k+1$ tokens costs about the same as one that generates a single token. Speculative decoding exploits this:

1. A cheap **draft** model (or extra heads, or n-gram lookup) proposes $k$ tokens.
2. The big **target** model scores all of them in **one** forward pass.
3. Accept the draft tokens left to right with the rejection rule below. At the first rejection, resample from a corrected distribution. You get at least one new token per target pass, and up to $k+1$.

**Why it's lossless:** for a draft token $x$ with draft probability $q(x)$ and target probability $p(x)$, accept it with probability $\min\big(1, p(x)/q(x)\big)$. On rejection, sample from the residual $\operatorname{norm}\big(\max(0, p - q)\big)$. Then the probability of ending up with $x$ is

```math
q(x)\min\Big(1,\frac{p(x)}{q(x)}\Big) + \Big(1 - \sum_{x'}\min(p(x'), q(x'))\Big)\frac{\max(0, p(x) - q(x))}{\sum_{x'}\max(0, p(x')-q(x'))} = p(x),
```

because $\sum\min(p,q) + \sum\max(0, p-q) = 1$. **The output distribution is exactly the target model's.** The speed-up depends on the **acceptance rate** (how well the draft matches the target), the draft's cost, and $k$. With a per-token acceptance rate $\alpha$, the expected number of tokens per target pass is $\frac{1-\alpha^{k+1}}{1-\alpha}$.

---

## 5. Continuous batching and serving engines

**Static batching** waits for a batch to fill, then runs until the *longest* sequence finishes. Slots freed by short sequences sit idle, and new requests wait.
**Continuous (in-flight) batching** schedules at the **iteration level**: after each decode step, finished sequences leave and waiting ones join. Utilization stays high, and so does throughput, with lower queueing latency.
Combined with paged KV memory and chunked prefill (prefill split into pieces and interleaved with decode steps, so long prompts don't stall everyone), that's the core of vLLM, SGLang, and TensorRT-LLM.

**Debug (the lesson's question 7): a 4-bit model, fast on short prompts, but TTFT explodes on long ones.** Long prompts are dominated by **prefill**, which is compute-bound and has quadratic attention. 4-bit weights help decode (memory-bound), not prefill. Many 4-bit kernels even dequantize on the fly, which adds compute.
What helps:

- an efficient attention kernel (FlashAttention);
- **prefix caching** for repeated contexts;
- chunked prefill (to protect other users' latency);
- a faster compute path for prefill (FP8 or BF16 kernels);
- shorter prompts (retrieval instead of stuffing everything in);
- splitting prefill and decode onto different workers.

```python
import numpy as np
rng = np.random.default_rng(0)

# --- Decode ceiling and KV cache arithmetic ---------------------------------------------------
bw = 3.35e12                                            # bytes/s
for name, params, bpp in [("7B bf16", 7e9, 2), ("7B 4-bit", 7e9, 0.5), ("70B 4-bit", 70e9, 0.5)]:
    print(f"{name:10s}: weights {params*bpp/1e9:5.1f} GB -> at most ~{bw / (params*bpp):5.0f} tokens/s at batch 1")
kv = 2 * 32 * 32768 * 8 * 128 * 2
print(f"KV cache, 32 layers x 8 kv heads x 128 dims x 32k tokens, bf16: {kv/1e9:.2f} GB per sequence")

# --- Absmax INT8 and the outlier problem --------------------------------------------------------
def absmax_q(w, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(w).max() / qmax
    return np.round(w / s).clip(-qmax, qmax) * s
w = rng.normal(0, 0.02, 4096)
w_out = w.copy(); w_out[7] = 3.0                          # one outlier
for name, x in [("no outlier", w), ("one outlier", w_out)]:
    err = np.abs(absmax_q(x) - x)[np.arange(len(x)) != 7].mean()
    zeroed = np.mean(absmax_q(x)[np.arange(len(x)) != 7] == 0)
    print(f"INT8 per-tensor, {name:11s}: mean abs error {err:.5f}, normal weights rounded to 0: {zeroed:.0%}")
groups = w_out.reshape(-1, 128)                           # per-group scales (128 weights each)
deq = np.concatenate([absmax_q(g, bits=4) for g in groups])
print(f"4-bit with group size 128, outlier present: mean abs error {np.abs(deq - w_out)[np.arange(len(w_out)) != 7].mean():.5f}")
```

```python
# --- Speculative sampling is lossless: empirical check -------------------------------------------
V = 6
p = rng.dirichlet(np.ones(V))           # target distribution
q = rng.dirichlet(np.ones(V))           # draft distribution
def spec_sample():
    x = rng.choice(V, p=q)
    if rng.random() < min(1, p[x] / q[x]): return x, True
    resid = np.maximum(p - q, 0); return rng.choice(V, p=resid / resid.sum()), False
samples = [spec_sample() for _ in range(200_000)]
emp = np.bincount([s for s, _ in samples], minlength=V) / len(samples)
print("target p      :", p.round(3))
print("spec. sampling:", emp.round(3), f"  acceptance rate {np.mean([a for _, a in samples]):.2f} (theory {np.minimum(p, q).sum():.2f})")
alpha, k = 0.8, 4
print(f"expected tokens per target pass at alpha={alpha}, k={k}: {(1 - alpha**(k+1)) / (1 - alpha):.2f}")
```

---

## Pitfalls & misconceptions

- **Judging speed by FLOPs.** Decode is bandwidth-bound: count bytes.
- **Quantizing without evaluating** on your task. Perplexity alone can hide task regressions.
- **Forgetting the KV cache** in memory planning, especially at long context × concurrency.
- **Benchmarking only throughput or only latency.** Report TTFT, inter-token latency, and throughput at a stated concurrency.
- **Thinking speculative decoding changes the outputs.** With the correct rejection rule, it doesn't.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Decode ceiling (batch 1) | bandwidth / (weight bytes + KV bytes) |
| KV cache | $2\cdot L\cdot T\cdot n_\text{kv}\cdot d_\text{head}\cdot$ bytes per value |
| Absmax quantization | $s = \max\lvert w\rvert/(2^{b-1}-1)$, $q = \operatorname{round}(w/s)$ |
| Speculative acceptance | $\min(1, p/q)$; residual $\propto \max(0, p-q)$ |
| Expected tokens per pass | $(1-\alpha^{k+1})/(1-\alpha)$ |
| Prefill vs decode | compute-bound (TTFT) vs memory-bound (tokens/s) |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why is decoding memory-bound, and why does batching help?</summary>

Each step reads all the weights to process one token per sequence, so its arithmetic intensity is about 1 FLOP/byte, far below the ridge point. Batching reuses each weight read for $B$ tokens, raising the intensity and throughput almost linearly until compute or KV memory becomes the limit.
</details>

<details>
<summary>2. KV cache for 32 layers, 8 KV heads, head dim 128, 32k context, BF16.</summary>

$2\times32\times32768\times8\times128\times2 \approx 4.3$ GB per sequence.
</details>

<details>
<summary>3. What are outlier features, and why do they break naive INT8?</summary>

A few hidden dimensions with consistently huge activations. A per-tensor absmax scale set by them makes the normal values round to 0, while clipping them destroys the model. Fix it with mixed-precision outliers, SmoothQuant rescaling, per-group scales, or weight-only quantization (the demo shows the per-tensor collapse).
</details>

<details>
<summary>4. GPTQ vs AWQ.</summary>

GPTQ: layer-wise reconstruction. It quantizes the weights sequentially and updates the remaining weights to minimize the output error, using second-order information from calibration data. AWQ: protects the salient weight channels (those with large activations) by scaling them before quantization, without reconstruction.
</details>

<details>
<summary>5. Why is speculative decoding lossless, and what sets the speed-up?</summary>

The accept rule $\min(1, p/q)$, plus resampling from $\max(0, p-q)$ on rejection, yields exactly the target distribution (§4 identity; the demo checks it empirically). The speed-up depends on the acceptance rate (how well the draft matches the target), the draft's cost, and the speculation length $k$.
</details>

<details>
<summary>6. What does continuous batching add?</summary>

Iteration-level scheduling: sequences join and leave the running batch after every step, so slots freed by finished requests are refilled immediately. Higher utilization and throughput, and lower queueing latency than static batches that wait for the longest sequence.
</details>

<details>
<summary>7. 4-bit model: fast on short prompts, TTFT explodes on long ones.</summary>

Prefill dominates long prompts. It's compute-bound with quadratic attention, and 4-bit weights don't speed it up (on-the-fly dequantization may even slow it). Use FlashAttention, prefix caching, chunked prefill, a faster precision for prefill, shorter or retrieved context, or separate prefill and decode workers.
</details>

## Where this leads

That completes Part B. Next is Part C, generative media: [GEN-04 notes](04-generative-models-diffusion.md). Text generation was "predict the next token". Images need a different idea: learn to **turn noise into data**. VAEs, GANs, and diffusion are three answers.
