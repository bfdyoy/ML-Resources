# GEN-10: Advanced Diffusion: Score Matching, Guidance, Latent Diffusion & Flow Matching

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~9 h | L3 | GEN-04 · math: [MATH-03](../math/03-probability-statistics.md) (Gaussians), [MATH-02](../math/02-calculus-optimization.md) |

## Why this matters
GEN-04 got you to DDPM. Modern image, video, and audio generators go further. They use the score-based/SDE view, classifier-free
guidance, latent spaces, transformer backbones (DiT), and increasingly **flow matching**, which is simpler to train and faster to sample.
This lesson gets you to the point where you can read current generative-model papers and train a modern model yourself.

## Learning goals
By the end you can:
- Explain score matching, and how diffusion is a discretized SDE (forward noising, reverse-time sampling).
- Compare DDPM, DDIM, and ODE samplers, and the step-count/quality trade-off.
- Implement classifier-free guidance, and explain the effect of the guidance scale.
- Explain latent diffusion (VAE latent + denoiser + text conditioning) and DiT backbones.
- Derive and implement conditional flow matching / rectified flow, and compare them with diffusion.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929) | The whole tutorial (diffusion and flow matching, minimal prerequisites) | 2 h |
| 2 | **Read** | [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) | The whole post (score matching → SDEs) | 1.5 h |
| 3 | **Read + Build** | [MIT 6.S184: Flow Matching & Diffusion](https://diffusion.csail.mit.edu/2026/index.html) | Lecture notes on flow matching, score matching, and guidance, with the matching labs | 3 h |
| 4 | **Build** | [HF Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) | The units on fine-tuning and guidance, and on Stable Diffusion (latent diffusion with 🤗 Diffusers) | 1.5 h |
| 5 | *Read (optional)* | [Lilian Weng: Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) | The whole post | 1 h |

## Check your understanding
1. What is the score function, and why can you learn it without knowing the normalizing constant?
2. How are DDPM's noise prediction and score estimation related?
3. Why does DDIM allow far fewer sampling steps than DDPM with the same trained model?
4. What does classifier-free guidance combine, and what goes wrong at very high guidance scales?
5. In flow matching, what is the regression target for the velocity field along a straight-line path?
6. Why does running diffusion in a VAE latent space make high-resolution generation feasible?
7. *(debug)* Your flow-matching model trains fine, but samples with 4 Euler steps look terrible while 100 steps look good. Why, and what fixes it?

## Mini-project
**Task:** Train the same small class-conditional model on MNIST or Fashion-MNIST two ways, as DDPM and as flow matching, with CFG for both.
Compare sample quality (visual + FID if you can) against the number of sampling steps.
**Deliverable:** Sample grids at 1/4/16/64 steps for both, and a guidance-scale sweep.

## Go deeper
- Papers: [Score SDE](https://arxiv.org/abs/2011.13456) · [CFG](https://arxiv.org/abs/2207.12598) · [Latent diffusion](https://arxiv.org/abs/2112.10752) · [DiT](https://arxiv.org/abs/2212.09748) · [EDM](https://arxiv.org/abs/2206.00364)
- [Flow Matching](https://arxiv.org/abs/2210.02747) · [Rectified flow](https://arxiv.org/abs/2209.03003) · [Consistency models](https://arxiv.org/abs/2303.01469)
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) + [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 09: Generative models](../../toolbox/09-generative-models.md).
- **Papers:** [Generative models](../../papers/07-generative-models.md) (diffusion & flow matching section).
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rung 42. Then convert it to flow matching (about 10 lines change).
- **Drills:** [MIT 6.S184 labs](https://diffusion.csail.mit.edu/2026/index.html) · more in [exercises/](../../exercises/README.md).
