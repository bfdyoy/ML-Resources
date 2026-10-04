# DL-06: Transformers

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~10 h | L2→L3 | DL-05 |

## Why this matters
The transformer is the architecture behind LLMs, modern vision models (ViT), speech, and protein models. If you
understand self-attention, positional information, residual streams, and layer norm, you can read most modern ML
papers.

## Learning goals
By the end you can:
- Explain self-attention (Q, K, V, scaled dot product, softmax), multi-head attention, and causal masking.
- Draw a transformer block (attention → MLP, with residuals and LayerNorm), and explain what each part does.
- Explain how position is added (learned, sinusoidal, and RoPE at a conceptual level).
- Tell encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) models apart, and say what each is used for.
- Implement a small GPT from scratch and train it on text.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Transformers](../../notes/deep-learning/06-transformers.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Intuition** | [3Blue1Brown: Transformers](https://www.3blue1brown.com/lessons/gpt/) → [Attention in transformers](https://www.3blue1brown.com/lessons/attention/) | Both lessons | 1 h |
| 2 | **Read** | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Alammar) | The whole post | 45 min |
| 3 | **Intuition** | [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Type your own prompts. Inspect the attention maps and the temperature. | 30 min |
| 4 | **Read** | [UDL](https://udlbook.github.io/udlbook/) (Prince) | Ch. 12 "Transformers" (with the notebooks) | 2 h |
| 5 | **Read + Build** | [Raschka: Understanding and Coding Self-Attention…](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) | Code each attention variant in PyTorch as you read | 1 h |
| 6 | **Build** | [Karpathy Lecture 7: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Code along, and train on Tiny Shakespeare | 2.5 h |

**Notes for the learner:** This is the most important lesson in the curriculum, so give it time. The order is on purpose:
*see it* (3B1B), *read it* (Alammar, then UDL), *poke it* (Explainer), *code it* (Raschka, Karpathy).

## Check your understanding
1. Why divide by `sqrt(d_k)` in scaled dot-product attention?
2. What does the causal mask do, and why is it needed for training a decoder in parallel?
3. Self-attention is permutation-equivariant. Why does that force us to add positional information?
4. What's the time and memory cost of attention with respect to sequence length? Why does that matter for long contexts?
5. What role does the residual stream play? What would break without residual connections?
6. *(debug)* Your mini-GPT's training loss drops fast but the samples are gibberish, and validation loss is rising. What's happening?

## Mini-project
**Task:** Train your from-scratch GPT on a small corpus you care about (song lyrics, a public-domain book, code). Then
try one architectural change (e.g. more heads vs more layers, or RoPE vs learned positions) and compare validation loss.
**Deliverable:** Training curves, samples, and a paragraph on what changed.

## Go deeper
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/): the original encoder-decoder, line by line.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) and [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [Build a LLM From Scratch](https://github.com/rasbt/LLMs-from-scratch): Ch. 3 (attention) and Ch. 4 (GPT model), a careful written alternative to Karpathy's video.
- [D2L](https://d2l.ai/): chapter "Attention Mechanisms and Transformers" (including vision transformers)

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 05: Architectures](../../toolbox/05-architectures.md), for every concept in this lesson, with alternatives.
- **Papers:** [NLP, transformers & LLMs](../../papers/03-nlp-transformers-llms.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 27–28.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
