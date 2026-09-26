# EL-07: Mechanistic Interpretability of Neural Networks

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~10 h | L3 | DL-06, CORE-08 |

## Why this matters
Feature importance (CORE-08) tells you *what* a model uses. Mechanistic interpretability asks *how* the network computes it, by reverse-engineering
circuits inside transformers. It's central to AI safety research, and it's a great way to understand transformers deeply: residual streams,
attention heads, induction heads, and superposition.

## Learning goals
By the end you can:
- Describe the residual-stream view of a transformer, and the QK/OV decomposition of attention heads.
- Explain induction heads, and how they enable in-context learning.
- Use activation patching and logit attribution to locate a behaviour inside a model.
- Explain superposition and polysemantic neurons, and why sparse autoencoders are used to find features.
- Be appropriately skeptical of saliency-style explanations.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | The whole article: zero-, one-, and two-layer attention-only transformers | 3 h |
| 2 | **Build** | [Transformer Circuits Exercises](https://transformer-circuits.pub/2021/exercises/index.html) | Work the pen-and-paper exercises | 1.5 h |
| 3 | **Build** | [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0) | The transformer-interpretability chapter: [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) intro, induction heads, activation patching | 4 h |
| 4 | **Read** | [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) | §1–3 and the key figures | 1.5 h |

**Notes for the learner:** ARENA is the best hands-on resource here. Its exercises have tests and solutions, so work through them in order.

## Check your understanding
1. What does "the residual stream is a communication channel" mean? How do layers read from and write to it?
2. What do the QK and OV circuits of a head each determine?
3. How does an induction head use a previous-token head to copy patterns?
4. What does activation patching measure, and why patch from a "clean" run into a "corrupted" one?
5. Why does superposition make individual neurons hard to interpret?
6. *(debug)* A saliency map for your classifier looks identical when you randomize the model's weights. What does that tell you?

## Mini-project
**Task:** Find induction heads in GPT-2 small with TransformerLens (use repeated random-token sequences), confirm them by ablation, and write
a short "circuit report" with attention-pattern plots.
**Deliverable:** A notebook and a 1-page report.

## Go deeper
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) · [Integrated Gradients](https://arxiv.org/abs/1703.01365)
- [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) · [Distill: The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/)

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (interpretability of neural networks).
- **Papers:** [Interpretability, uncertainty & responsible ML](../../papers/10-interpretability-uncertainty-responsible.md) (mechanistic interpretability section).
- **Implement it yourself:** direct logit attribution for one prompt, by hand from cached activations.
- **Drills:** [Thinking Like Transformers (raspy)](https://github.com/srush/raspy) · more in [exercises/](../../exercises/README.md).
