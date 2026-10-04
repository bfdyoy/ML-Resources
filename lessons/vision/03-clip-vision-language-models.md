# CV-03: Multimodal Models: CLIP & Vision-Language Models

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Computer Vision | ~8 h | L3 | CV-02, GEN-01 |

## Why this matters
CLIP-style embeddings power zero-shot classification, image search, content moderation, and the text conditioning of image
generators. Vision-language models (VLMs) that "see and chat" are built by connecting a vision encoder to an LLM. Knowing how
these pieces fit together lets you pick, fine-tune, and evaluate multimodal systems.

## Learning goals
By the end you can:
- Explain CLIP's contrastive image–text objective, and how zero-shot classification works with prompt templates.
- Compare the softmax (CLIP) and sigmoid (SigLIP) contrastive losses.
- Describe the main VLM designs: cross-attention into a frozen LM (Flamingo), a Q-Former bridge (BLIP-2), and a projection + instruction tuning (LLaVA).
- Build image–text retrieval with CLIP embeddings, and evaluate it with recall@k.
- Fine-tune or prompt a small open VLM for a task, and name the common failure modes (hallucinated objects, counting, OCR).

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Multimodal: CLIP & Vision-Language Models](../../notes/vision/03-clip-vision-language-models.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [CLIP](https://arxiv.org/abs/2103.00020) | §1–2 (approach, Figure 1, pseudocode) and §3.1 (zero-shot transfer) | 1.5 h |
| 2 | **Read** | [HF: A Dive into Vision-Language Models](https://huggingface.co/blog/vision_language_pretraining) | The whole post (the pretraining strategies) | 45 min |
| 3 | **Read** | [HF: Vision Language Models Explained](https://huggingface.co/blog/vlms) | The whole post (model families, benchmarks, fine-tuning) | 45 min |
| 4 | **Read** | [Lilian Weng: Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) | The four approaches to wiring vision into LMs | 1 h |
| 5 | **Build** | CLIP via `transformers` | Zero-shot classification on your CV-02 dataset with 3 prompt templates, then text→image retrieval with recall@k | 1.5 h |
| 6 | **Build** | A small open VLM | Run it on 30 images for captioning and VQA. Categorize the errors. | 1 h |

## Check your understanding
1. What does CLIP's loss push together and pull apart? Why do large batches help?
2. Why does prompt engineering ("a photo of a {label}") change zero-shot accuracy?
3. What is SigLIP's sigmoid loss, and why does it scale better than softmax over the batch?
4. LLaVA vs BLIP-2 vs Flamingo: where does each put the trainable parameters?
5. Why do VLMs hallucinate objects that aren't in the image?
6. *(debug)* CLIP zero-shot accuracy on your medical images is near chance. Why, and what are your options?

## Mini-project
**Task:** Build a small image search engine: embed 5–10k images with CLIP, index them with Faiss, and support text queries. Evaluate recall@10
on 30 hand-written queries. Then compare CLIP with a SigLIP model.
**Deliverable:** A demo notebook (or Gradio app) and a results table.

## Go deeper
- [SigLIP](https://arxiv.org/abs/2303.15343) · [Flamingo](https://arxiv.org/abs/2204.14198) · [BLIP-2](https://arxiv.org/abs/2301.12597) · [LLaVA](https://arxiv.org/abs/2304.08485)
- [HF Community CV Course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome): the multimodal units.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 06: Computer vision](../../toolbox/06-computer-vision.md) (CLIP & vision-language models).
- **Papers:** [Computer vision](../../papers/02-computer-vision.md) (vision + language section).
- **Implement it yourself:** the symmetric CLIP loss in 10 lines of PyTorch; verify it on a toy batch.
- **Drills:** more in [exercises/](../../exercises/README.md).
