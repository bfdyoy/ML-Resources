# GEN-08: Efficient LLM Inference: KV Cache, Quantization, Speculative Decoding & Serving

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~8 h | L3 | GEN-01, DL-07 (DL-08 helps) |

## Why this matters
Inference cost and latency decide whether an LLM feature is viable. Most of the speed-ups come from a small set of ideas: the KV cache
and the ways to shrink it, weight quantization, batching, paged memory, and speculative decoding. You need these to pick a serving
stack, size hardware, or run a model on a laptop.

## Learning goals
By the end you can:
- Explain prefill vs decode, why decoding is memory-bandwidth-bound, and what sets tokens/sec.
- Compute KV-cache memory, and explain how GQA, paging, and prefix caching reduce it.
- Explain INT8/4-bit weight quantization (absmax/zero-point, GPTQ, AWQ, GGUF), and measure the quality impact.
- Explain speculative/assisted decoding, and why it doesn't change the output distribution.
- Benchmark a serving engine (continuous batching) against naive generation, for throughput and latency.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [HF: KV Caching Explained](https://huggingface.co/blog/kv-cache) | The whole post | 30 min |
| 2 | **Read** | [Lilian Weng: Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | The whole post (distillation, quantization, pruning, sparsity, architecture tricks) | 1.5 h |
| 3 | **Read** | [Grootendorst: A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) | The whole post | 1 h |
| 4 | **Read** | [HF: Optimizing your LLM in production](https://huggingface.co/blog/optimize-llm) + [HF: Assisted Generation](https://huggingface.co/blog/assisted-generation) | Both posts | 1 h |
| 5 | **Build** | `transformers` + [bitsandbytes via HF](https://huggingface.co/blog/4bit-transformers-bitsandbytes) | Load one model in FP16, 8-bit, and 4-bit. Measure memory, tokens/sec, and perplexity on a held-out text. | 1.5 h |
| 6 | **Build** | [vLLM](https://github.com/vllm-project/vllm) | Benchmark vLLM against plain `generate` at batch sizes 1/8/32: throughput, time-to-first-token, p95 latency | 1.5 h |
| 7 | *Read (optional)* | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) | The inference chapter (latency/throughput rooflines for serving) | 1 h |

## Check your understanding
1. Why is decoding one token at a time memory-bound, and why does batching help?
2. Compute the KV-cache size for a 32-layer model with 8 KV heads, head dim 128, a 32k context, in BF16.
3. What are outlier features, and why do they make naive INT8 quantization of LLMs fail?
4. GPTQ vs AWQ: what does each optimize?
5. Why is speculative decoding lossless? What determines its speed-up?
6. What does continuous batching do that static batching doesn't?
7. *(debug)* Your 4-bit model is fast on short prompts but time-to-first-token explodes on long ones. What is dominating, and what can help?

## Mini-project
**Task:** Produce a "deployment sheet" for one open model: memory and speed for FP16/INT8/4-bit, the perplexity change, the throughput curve under vLLM,
and the speed-up from speculative decoding with a small draft model.
**Deliverable:** A table plus a one-paragraph recommendation for (a) a laptop, (b) one 24 GB GPU, and (c) a high-traffic API.

## Go deeper
- Papers: [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) · [Speculative decoding](https://arxiv.org/abs/2211.17192) · [LLM.int8()](https://arxiv.org/abs/2208.07339) · [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00978) · [FlashAttention](https://arxiv.org/abs/2205.14135)
- [MIT 6.5940 EfficientML](https://hanlab.mit.edu/courses/2026-fall-65940): the LLM-deployment lectures and labs.
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 07: NLP & LLMs](../../toolbox/07-nlp-and-llms.md) (efficient inference) · [Toolbox 11: MLOps, systems & scale](../../toolbox/11-mlops-and-systems.md).
- **Papers:** [Efficiency & systems](../../papers/06-efficiency-systems.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 31 and 36 (KV cache, INT8 quantization).
- **Drills:** more in [exercises/](../../exercises/README.md).
