# CV-03 notes: Multimodal: CLIP & Vision-Language Models

[← Lesson CV-03](../../lessons/vision/03-clip-vision-language-models.md) · [All notes](../README.md) · [← CV-02 notes](02-vision-transformers-self-supervised.md) · Next: [CV-04 notes →](04-3d-vision-neural-rendering.md)

> **Reading time** ≈ 55 min. **You need:** [CV-02 notes](02-vision-transformers-self-supervised.md) §2 (contrastive learning), [GEN-05 notes](../llms-genai/05-retrieval-engineering.md) §1 and §3 (recall@k, bi-encoders), and [GEN-02 notes](../llms-genai/02-adapting-llms-finetuning-rag.md) (fine-tuning LLMs).

---

## Where we are

CV-02 aligned two *views of an image*. CLIP aligns an **image with its caption**. That one change gives:

- a **shared embedding space** for images and text;
- **zero-shot classification** (name the classes in words, no training);
- image–text **search**;
- the vision encoder used inside most **vision-language models (VLMs)**, which let an LLM "see".

---

## 1. CLIP: contrastive image–text pretraining

There are two encoders: an image encoder $f$ (a ViT or ResNet) and a text encoder $g$ (a transformer), each followed by a projection into a shared $d$-dimensional space, with L2-normalized outputs.
For a batch of $N$ (image, caption) pairs, compute the $N\times N$ similarity matrix $S_{ij} = \langle f(x_i), g(t_j)\rangle\cdot e^{s}$, where $s$ is a **learned** log-temperature. The loss is a **symmetric cross-entropy**:

```math
\mathcal{L} = \frac{1}{2}\Big[\underbrace{\frac1N\sum_i -\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}}_{\text{image}\to\text{text}} + \underbrace{\frac1N\sum_j -\log\frac{e^{S_{jj}}}{\sum_i e^{S_{ij}}}}_{\text{text}\to\text{image}}\Big].
```

- **Pushed together:** matching pairs (the diagonal). **Pulled apart:** every non-matching pair in the batch ($N^2 - N$ negatives).
- **Large batches help** (CLIP used 32k), because more negatives make a harder, more informative task. With a larger batch, a caption must beat 32k distractors, not 256.
- **Data:** about 400M web image–alt-text pairs. Noisy but enormous, and very diverse.

### 1.1 Zero-shot classification

To classify among $K$ classes, embed $K$ text prompts ("a photo of a {label}"), and pick the class whose text embedding has the highest cosine similarity with the image embedding. The **text encoder writes the classifier's weights**.

**Why the prompt wording matters:** the text encoder learned from *captions*, so "a photo of a dog" is in-distribution, while the bare word "dog" is less so. Context also **disambiguates** ("crane" the bird vs the machine; "a photo of a boxer, a type of dog").
**Prompt ensembling** (averaging the embeddings of many templates, like "a photo of a {}", "a blurry photo of a {}", "a drawing of a {}") reduces template noise and typically adds a few points of accuracy.

---

## 2. SigLIP: a sigmoid instead of a softmax

CLIP's softmax normalizes over **the whole batch**. Every logit needs every other logit in its row and column, which means a global, all-gathered similarity matrix across devices, and a loss that depends on batch composition. **SigLIP** treats each pair **independently**, as binary classification: "is this a matching pair?"

```math
\mathcal{L} = -\frac{1}{N}\sum_{i,j}\log\sigma\big(z_{ij}\,(t\,\langle x_i, y_j\rangle + b)\big),\qquad z_{ij} = \begin{cases}+1 & i = j\\ -1 & i\ne j\end{cases}
```

with a learned temperature $t$ and bias $b$. The bias handles the huge imbalance (one positive vs $N-1$ negatives per row).

**Why it scales better:** there's no batch-wide normalization, so each device computes the loss on its own chunks while passing embeddings around in a ring. Memory is lower, and it works well at both small and very large batch sizes.

---

## 3. Vision-language models: three ways to connect a vision encoder to an LLM

| Design | Bridge | What's trained | Notes |
|---|---|---|---|
| **Flamingo** | **Gated cross-attention** layers inserted *between* the frozen LM's layers, attending to visual tokens (via a Perceiver resampler) | The new cross-attention layers + the resampler; the vision encoder and LM are frozen | Handles interleaved image–text sequences; strong few-shot |
| **BLIP-2** | **Q-Former**: a small transformer whose learnable queries extract a fixed number of visual tokens, aligned to text | Mainly the Q-Former (the image encoder and LM are frozen) | Parameter-efficient bridge |
| **LLaVA** | A simple **projection** (linear or MLP) mapping image-patch features into the LM's token-embedding space; visual tokens are fed in like words | Stage 1: the projector only. Stage 2: projector + LLM, **visual instruction tuning** (often LoRA) | Simple and very effective; the template for most open VLMs |

In every design, a **pretrained CLIP/SigLIP-style vision encoder** supplies the visual features. The real work is the bridge, plus instruction data.

### 3.1 Why VLMs hallucinate objects

- **Language priors dominate:** the LM has learned strong co-occurrence statistics ("kitchen" → "refrigerator", "dining table" → "fork"). When the visual evidence is weak, the fluent prior wins.
- **Lossy visual tokens:** fixed-resolution encoders and token compression drop small details, so the model guesses.
- **Training data:** captions and instruction data that mention unseen objects, or reward descriptive richness, teach the model to embellish.
- **Weaknesses specific to the architecture:** counting, reading small text (OCR), and spatial relations, because contrastive pretraining aligns *global* semantics, not precise positions or counts.

Mitigations: higher-resolution or tiled inputs, grounding (asking for boxes or citations), preference tuning against hallucinations, and verification with a detector.

---

## 4. When zero-shot fails: domain gap

CLIP on medical images (X-rays, histology) often scores **near chance**. Web image–caption data contains few such images, and even fewer with expert captions. Class names like "pneumothorax" have weak text embeddings for this purpose, and the visual features don't separate subtle clinical findings. Options:

1. **A linear probe or few-shot adapter** on frozen CLIP features: cheap, and a good test of whether the features carry the signal at all.
2. **Fine-tune the image encoder** (full or LoRA) on labeled domain data.
3. **A domain-specific CLIP** pretrained on domain image–text pairs (radiology reports, pathology captions), or SSL pretraining on unlabeled domain images (CV-02).
4. **Better prompts** with clinical descriptions. That helps a little, and it can't create features that aren't there.

## 5. Image–text retrieval and its evaluation

Embed the whole gallery once, embed the query (text or image), and take the nearest neighbours by cosine similarity (with ANN at scale, GEN-05 §5). Evaluate with **recall@k** in both directions (text→image and image→text) on a held-out set of pairs.

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)

# --- Synthetic paired embeddings: images and captions share a latent "concept" -----------------------
N, d = 256, 64
concept = torch.randn(N, d)
img = F.normalize(concept + 1.5 * torch.randn(N, d), dim=1)
txt = F.normalize(concept + 1.5 * torch.randn(N, d), dim=1)

def clip_loss(img, txt, logit_scale=torch.tensor(1 / 0.07).log()):
    S = img @ txt.T * logit_scale.exp()
    targets = torch.arange(len(img))
    return (F.cross_entropy(S, targets) + F.cross_entropy(S.T, targets)) / 2

def siglip_loss(img, txt, t=10.0, b=-10.0):
    logits = img @ txt.T * t + b
    z = 2 * torch.eye(len(img)) - 1                         # +1 on the diagonal, -1 elsewhere
    return -F.logsigmoid(z * logits).sum() / len(img)

shuffled = txt[torch.randperm(N)]
print(f"CLIP loss   aligned {clip_loss(img, txt):.3f} | shuffled {clip_loss(img, shuffled):.3f}")
print(f"SigLIP loss aligned {siglip_loss(img, txt):.3f} | shuffled {siglip_loss(img, shuffled):.3f}")

def recall_at_k(q, gallery, k):
    ranks = (q @ gallery.T).argsort(dim=1, descending=True)[:, :k]
    return (ranks == torch.arange(len(q))[:, None]).any(1).float().mean().item()
print(f"text->image recall@1 {recall_at_k(txt, img, 1):.2f}, recall@5 {recall_at_k(txt, img, 5):.2f}")
```

```python
# --- Zero-shot classification with prompt ensembling (synthetic) ------------------------------------------
K = 10
class_concept = torch.randn(K, d)
labels = torch.randint(0, K, (2000,))
images = F.normalize(class_concept[labels] + 3.0 * torch.randn(2000, d), dim=1)
def text_embed(template_noise):                     # each template adds its own distortion of the class text
    return F.normalize(class_concept + template_noise * torch.randn(K, d), dim=1)
single = text_embed(0.6)
ensemble = F.normalize(torch.stack([text_embed(0.6) for _ in range(16)]).mean(0), dim=1)
for name, W in [("one template", single), ("16-template ensemble", ensemble)]:
    acc = ((images @ W.T).argmax(1) == labels).float().mean().item()
    print(f"zero-shot accuracy, {name}: {acc:.3f}")
# (Synthetic: the size of the gain depends on how noisy individual templates are. On real CLIP models,
#  ensembling typically adds a few points.)
```

---

## Pitfalls & misconceptions

- **Comparing CLIP embeddings from different models** (or unnormalized embeddings).
- **Trusting zero-shot accuracy on specialist domains.** Always run a linear probe as a sanity check.
- **Assuming a VLM's description is grounded.** Verify objects, counts, and text.
- **Fine-tuning CLIP with tiny batches and the softmax loss.** It's unstable and has few negatives. Consider SigLIP-style or adapter approaches.
- **Evaluating retrieval in one direction only.**

## Cheat sheet

| Item | Formula / rule |
|---|---|
| CLIP loss | symmetric cross-entropy over the $N\times N$ similarity matrix, learned temperature |
| Zero-shot | argmax over the cosine with prompt embeddings ("a photo of a {label}") |
| Prompt ensembling | average the normalized embeddings over templates |
| SigLIP | $-\sum_{ij}\log\sigma(z_{ij}(t\langle x_i,y_j\rangle + b))$, with no batch normalization |
| VLM bridges | cross-attention (Flamingo), Q-Former (BLIP-2), projection + instruction tuning (LLaVA) |
| Domain gap | linear probe → fine-tune → domain CLIP / SSL |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What does CLIP's loss push together and pull apart; why do large batches help?</summary>

It pulls matching image–caption pairs together and pushes all the non-matching pairs in the batch apart, in both directions. Larger batches supply more (and harder) negatives, which makes the task more informative.
</details>

<details>
<summary>2. Why does prompt wording change zero-shot accuracy?</summary>

The text encoder learned from caption-style text. Templates like "a photo of a {label}" are in-distribution and disambiguate class names. Averaging many templates reduces noise (the demo shows the ensemble beating a single template).
</details>

<details>
<summary>3. SigLIP's sigmoid loss, and why it scales better.</summary>

Each image–text pair is a binary match/no-match classification with a learned temperature and bias. There's no softmax normalization across the batch, so no global similarity matrix is needed, it's memory-efficient, it can be computed in device-local chunks, and it's robust to batch size.
</details>

<details>
<summary>4. Where LLaVA, BLIP-2, and Flamingo put the trainable parameters.</summary>

LLaVA: a projector (then the projector + the LLM, during visual instruction tuning). BLIP-2: a Q-Former between the frozen image encoder and the frozen LLM. Flamingo: new gated cross-attention layers (plus a resampler) inserted into a frozen LM.
</details>

<details>
<summary>5. Why do VLMs hallucinate objects?</summary>

Strong language priors override weak or lossy visual evidence. Low-resolution visual tokens drop details. Training data rewards descriptive embellishment. And contrastive vision features are poor at counts, positions, and small text.
</details>

<details>
<summary>6. CLIP zero-shot near chance on medical images.</summary>

A domain gap: few medical images and expert captions in the web pretraining data, and weak text embeddings for clinical terms. Options: a linear probe or few-shot adapter (to check the signal), fine-tuning, a domain-specific CLIP, or SSL pretraining on domain images. Better prompts help only marginally.
</details>

## Where this leads

Next: [CV-04 notes](04-3d-vision-neural-rendering.md). So far every image has been flat. CV-04 adds the third dimension: cameras as projection matrices, and NeRF and Gaussian splats as learned 3D scenes, rendered with differentiable maths.
