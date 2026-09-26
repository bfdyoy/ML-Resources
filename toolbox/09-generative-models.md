# Toolbox 09: Generative Models

[← Toolbox](README.md) · Guided version: [GEN-04](../lessons/llms-genai/04-generative-models-diffusion.md) · Papers: [07 Generative models](../papers/07-generative-models.md)

**Whole-field resources:** [UDL](https://udlbook.github.io/udlbook/) Ch. 14–18 (primary text) · [CS236 notes](https://deepgenerativemodels.github.io/notes/) (Stanford) · [MIT 6.S184: Flow Matching & Diffusion](https://diffusion.csail.mit.edu/2026/index.html) (lecture notes + labs) · [HF Diffusion Course](https://github.com/huggingface/diffusion-models-class)

| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| The generative-model zoo (what's explicit vs implicit likelihood) | [UDL](https://udlbook.github.io/udlbook/) Ch. 14 | [CS236 notes](https://deepgenerativemodels.github.io/notes/) intro | — | — |
| Autoregressive models (PixelCNN → GPT) | [Karpathy: makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) | [CS236 notes](https://deepgenerativemodels.github.io/notes/) (autoregressive models) | Your DL-05 makemore *is* one | — |
| VAEs, the ELBO, reparameterization | [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 17 · [CS236 notes](https://deepgenerativemodels.github.io/notes/) (VAE) | VAE on MNIST; interpolate in latent space | [VAE](https://arxiv.org/abs/1312.6114), [VQ-VAE](https://arxiv.org/abs/1711.00937) |
| GANs, and why they're unstable | [Lilian Weng: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 15 | DCGAN on a small image set; watch for mode collapse | [GAN](https://arxiv.org/abs/1406.2661), [DCGAN](https://arxiv.org/abs/1511.06434), [WGAN](https://arxiv.org/abs/1701.07875), [StyleGAN](https://arxiv.org/abs/1812.04948) |
| Normalizing flows | [Lilian Weng: Flow-based Deep Generative Models](https://lilianweng.github.io/posts/2018-10-13-flow-models/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 16 · [UvA flows tutorial](https://uvadlc-notebooks.readthedocs.io/) | RealNVP on 2-D toy data | [NF review](https://arxiv.org/abs/1912.02762) |
| Diffusion (DDPM) | [Step-by-Step Diffusion tutorial](https://arxiv.org/abs/2406.08929) · [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 18 · [Understanding Diffusion Models](https://arxiv.org/abs/2208.11970) | [HF: The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | [DDPM](https://arxiv.org/abs/2006.11239) |
| Score matching & SDEs | [Yang Song: Generative Modeling by Estimating Gradients](https://yang-song.net/blog/2021/score/) | [MIT 6.S184 notes](https://diffusion.csail.mit.edu/2026/index.html) | Score model on 2-D data with Langevin sampling | [Score SDE](https://arxiv.org/abs/2011.13456), [EDM](https://arxiv.org/abs/2206.00364) |
| Guidance (classifier-free) | [HF Diffusion Course](https://github.com/huggingface/diffusion-models-class) unit on guidance | [MIT 6.S184 notes](https://diffusion.csail.mit.edu/2026/index.html) | Add CFG to your class-conditional model | [CFG](https://arxiv.org/abs/2207.12598) |
| Latent diffusion / Stable Diffusion | [Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | [HF: Stable Diffusion with Diffusers](https://huggingface.co/blog/stable_diffusion) | Generate with Diffusers; swap schedulers | [LDM](https://arxiv.org/abs/2112.10752), [DiT](https://arxiv.org/abs/2212.09748) |
| Flow matching & rectified flow | [MIT 6.S184 notes](https://diffusion.csail.mit.edu/2026/index.html) | [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) | [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) examples | [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified flow](https://arxiv.org/abs/2209.03003), [Consistency models](https://arxiv.org/abs/2303.01469) |
| Video diffusion | — | [Lilian Weng: Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/) | — | — |
| Multimodal generation (text→image) | [Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | [HF CV course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome) (generative unit) | — | [DALL·E 2](https://arxiv.org/abs/2204.06125) |

## Code to learn from
- [labml.ai implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations): DDPM, GANs, VAEs, and more, annotated line by line.
- [openai/improved-diffusion](https://github.com/openai/improved-diffusion): a clean reference DDPM codebase.
