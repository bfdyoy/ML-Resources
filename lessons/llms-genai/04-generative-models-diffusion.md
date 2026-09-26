# GEN-04: Generative Models: VAEs, GANs & Diffusion

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| LLMs & GenAI | ~8 h | L2→L3 | DL-04, CORE-06 · math: [MATH-03](../math/03-probability-statistics.md) (Gaussians, KL divergence) |

## Why this matters
Image, audio, and video generation run on diffusion models, and the ideas behind them (latent spaces, variational
bounds, denoising) show up all over modern ML. Learning VAEs → GANs → diffusion in that order makes each step feel natural.

## Learning goals
By the end you can:
- Explain what a generative model learns, and how explicit-likelihood models differ from implicit ones.
- Explain a VAE: encoder, decoder, reparameterization trick, and the ELBO (reconstruction + KL).
- Explain GAN training as a two-player game, and its failure modes (mode collapse, instability).
- Explain diffusion: the forward noising process, learning to denoise, sampling, and classifier-free guidance.
- Describe latent diffusion (Stable Diffusion): VAE latent space + U-Net/transformer denoiser + text conditioning.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [UDL](https://udlbook.github.io/udlbook/) (Prince) | Ch. 14 "Unsupervised Learning" (the taxonomy), Ch. 17 "Variational Autoencoders", Ch. 15 "Generative Adversarial Networks" | 2.5 h |
| 2 | **Read** | [UDL](https://udlbook.github.io/udlbook/) | Ch. 18 "Diffusion Models", with the notebooks | 2 h |
| 3 | **Intuition** | [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) (Alammar) | The whole post | 45 min |
| 4 | **Build** | [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) | Unit 1 (intro to 🤗 Diffusers, and a diffusion model from scratch) | 2.5 h |

**Notes for the learner:** UDL is the primary text here because it treats all four generative families with
consistent notation and excellent figures. Skip Ch. 16 (normalizing flows) on a first pass.

## Check your understanding
1. Why can't you backprop through sampling directly, and how does the reparameterization trick fix that?
2. What do the two terms of the ELBO each encourage?
3. What is mode collapse in GANs, and why doesn't a VAE suffer from it the same way?
4. In diffusion, what exactly does the network predict at each step (noise vs clean image), and why are those two equivalent?
5. Why does Stable Diffusion run diffusion in a latent space instead of on pixels?
6. *(debug)* Your diffusion model's samples are blurry blobs after training. Name three things to check.

## Mini-project
**Task:** Train a small unconditional diffusion model on 32×32 images (e.g. Fashion-MNIST or a butterfly subset, as in the HF course),
then add class conditioning. Show samples at several training checkpoints.
**Deliverable:** A notebook plus a sample grid over time.

## Go deeper
- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): a dense but complete mathematical tour.
- [Géron, *Hands-On ML*, Ch. 18 "Autoencoders, GANs, and Diffusion Models"](https://github.com/ageron/handson-mlp): a practical implementation angle.
- [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) (Bishop): the chapters on generative models, for a rigorous treatment.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 09: Generative models](../../toolbox/09-generative-models.md), for every concept in this lesson, with alternatives.
- **Papers:** [Generative models](../../papers/07-generative-models.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 41–42.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
