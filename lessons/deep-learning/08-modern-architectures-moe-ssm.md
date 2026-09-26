# DL-08: Modern Architectures: Positional Encodings, Efficient Attention, MoE & State-Space Models

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~7 h | L3 | DL-06 |

## Why this matters
The 2017 transformer is not what runs today. Modern LLMs use RoPE, RMSNorm, SwiGLU, grouped-query attention, and often
mixture-of-experts layers. State-space models (Mamba) are a serious alternative for long sequences. You need to know these
changes to read current model cards and papers, and to make sensible architecture choices yourself.

## Learning goals
By the end you can:
- Explain sinusoidal, learned, and rotary (RoPE) position encodings, and how RoPE-based context extension works.
- Explain MQA/GQA and sliding-window attention, and their effect on the KV cache.
- Explain a sparse MoE layer: router, top-k experts, load balancing, and the "total vs active parameters" distinction.
- Explain the idea behind state-space models and selective scans (Mamba), and when they beat attention.
- Read a modern model's architecture table and name each design choice.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Raschka: The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) | The whole article. It's the map for this lesson. | 1.5 h |
| 2 | **Read** | [HF: Designing positional encoding](https://huggingface.co/blog/designing-positional-encoding) | The whole post (it builds RoPE from first principles) | 45 min |
| 3 | **Read** | [Lilian Weng: The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | The sections on positional encoding, long context, and efficient/sparse attention. Use it as a reference; don't read end to end. | 1 h |
| 4 | **Read** | [HF: Mixture of Experts Explained](https://huggingface.co/blog/moe) | The whole post | 45 min |
| 5 | **Read** | [Mamba](https://arxiv.org/abs/2312.00752) | §1–3: the selection mechanism and the hardware-aware scan. Skim the experiments. | 1.5 h |
| 6 | **Build** | Your DL-06 nanoGPT | Swap in RoPE, RMSNorm, and SwiGLU, then GQA. Compare val loss and KV-cache size at equal compute. | 1.5 h |

## Check your understanding
1. Why does RoPE encode *relative* position even though it's applied to each token independently?
2. How much does GQA with 8 KV heads (out of 32 query heads) shrink the KV cache?
3. An MoE model has 47B total and 13B active parameters. What does each number determine: memory, compute, or quality?
4. Why do MoE models need a load-balancing loss? What happens without it?
5. What makes Mamba's "selective" SSM different from S4, and why does that matter for language?
6. *(debug)* After adding MoE to your model, the loss is spiky and a few experts get almost all the tokens. What do you check?

## Mini-project
**Task:** Build "nanoGPT-2026": your DL-06 model with RoPE + RMSNorm + SwiGLU + GQA, plus an optional top-2 MoE MLP. Train both variants
on the same data and compute budget.
**Deliverable:** An ablation table (val loss, tokens/sec, parameter count, KV-cache bytes/token) and a paragraph on the trade-offs.

## Go deeper
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) · [GQA](https://arxiv.org/abs/2305.13245) · [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Mamba-2 (Transformers are SSMs)](https://arxiv.org/abs/2405.21060): connects SSMs and attention.
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437): MoE and multi-head latent attention at frontier scale.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 05: Architectures](../../toolbox/05-architectures.md) (transformers section).
- **Papers:** [NLP, transformers & LLMs](../../papers/03-nlp-transformers-llms.md) (modern architecture components).
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 31–32.
- **Drills:** [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) · more in [exercises/](../../exercises/README.md).
