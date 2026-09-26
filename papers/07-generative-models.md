# Papers: Generative Models

[← Papers library](README.md) · Background: [GEN-04](../lessons/llms-genai/04-generative-models-diffusion.md), [GEN-10](../lessons/llms-genai/10-advanced-diffusion-flow-matching.md) · [Toolbox: Generative models](../toolbox/09-generative-models.md)

## VAEs, GANs & flows
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114) | 2013 | ⭐ The reparameterization trick and the ELBO. | L2 | GEN-04 |
| [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937) | 2017 | Discrete latents, the precursor to image tokenizers. | L3 | GEN-04 |
| [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) | 2014 | ⭐ The original GAN. | L2 | GEN-04 |
| [DCGAN](https://arxiv.org/abs/1511.06434) | 2015 | The first GAN recipe that trained reliably. | L2 | GEN-04 |
| [Wasserstein GAN](https://arxiv.org/abs/1701.07875) | 2017 | A better training objective. Math-heavy. | L3 | GEN-04 |
| [A Style-Based Generator Architecture (StyleGAN)](https://arxiv.org/abs/1812.04948) | 2018 | High-fidelity faces, and style mixing. | L3 | GEN-04 |
| [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/abs/1912.02762) | 2019 | ⭐ The review to read on normalizing flows. | L3 | GEN-04 |

## Diffusion & flow matching
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929) | 2024 | ⭐ **Start here.** The simplest correct introduction to diffusion and flow matching. | L1 | GEN-04 |
| [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) | 2022 | ⭐ VAE → hierarchical VAE → diffusion, derived carefully. | L2 | GEN-04 |
| [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) | 2020 | ⭐ The paper that made diffusion work. | L2 | GEN-04 |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 2020 | Unifies score matching and diffusion. Read it with [Yang Song's blog post](https://yang-song.net/blog/2021/score/). | L3 | GEN-10 |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2022 | ⭐ The guidance trick every text-to-image model uses. | L2 | GEN-10 |
| [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 2021 | ⭐ Stable Diffusion: diffusion in a VAE latent space. | L2 | GEN-04 |
| [Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL·E 2)](https://arxiv.org/abs/2204.06125) | 2022 | CLIP prior + diffusion decoder. | L3 | GEN-10 |
| [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) | 2022 | Separates out the design choices of diffusion models. A research favourite. | L3 | GEN-10 |
| [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) | 2022 | Transformers replace the U-Net, as in modern image and video models. | L3 | GEN-10 |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 | ⭐ A simpler training objective, now very common. | L3 | GEN-10 |
| [Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003) | 2022 | Straight paths allow few-step sampling. | L3 | GEN-10 |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023 | One-step generation. | L3 | GEN-10 |
| [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) | 2024 | A long tutorial with a reference library ([facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)). | L3 | GEN-10 |
