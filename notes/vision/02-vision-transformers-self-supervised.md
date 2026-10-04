# CV-02 notes: Vision Transformers & Self-Supervised Learning

[← Lesson CV-02](../../lessons/vision/02-vision-transformers-self-supervised.md) · [All notes](../README.md) · [← CV-01 notes](01-object-detection-segmentation.md) · Next: [CV-03 notes →](03-clip-vision-language-models.md)

> **Reading time** ≈ 60 min. **You need:** [DL-06 notes](../deep-learning/06-transformers.md) (the transformer), [DL-04 notes](../deep-learning/04-cnns-computer-vision.md) (inductive biases, augmentation), and [GEN-05 notes](../llms-genai/05-retrieval-engineering.md) §3 (the InfoNCE loss).

---

## Where we are

Two ideas reshaped vision after 2020:

1. **Vision transformers:** treat an image as a sequence of patches and use the DL-06 transformer, almost unchanged.
2. **Self-supervised learning (SSL):** learn strong representations from **unlabeled** images by inventing a training task from the data itself.

Together, they give general-purpose backbones that need few labels downstream.

---

## 1. The Vision Transformer (ViT)

**Pipeline:**

1. Split a $224\times224$ image into $16\times16$ patches, giving $14\times14 = 196$ patches.
2. Flatten each patch ($16\cdot16\cdot3 = 768$ values) and project it linearly to $d$ dimensions. That's one "token" per patch. (It's equivalent to a convolution with kernel 16 and stride 16.)
3. Prepend a learnable **[CLS]** token, and add learned **position embeddings**.
4. Run a standard transformer encoder (bidirectional attention, no mask).
5. Classify from the final [CLS] token (or from the mean of all the patch tokens).

**Cost:** attention over $N$ patches is $O(N^2)$. Halving the patch size quadruples $N$ and multiplies the attention cost by 16. That's why patches are 14–16 pixels, and why hierarchical variants (Swin) use local windows.

### 1.1 Inductive bias and data hunger

CNNs build in **locality** (small kernels), **translation equivariance** (weight sharing), and a **hierarchy** of growing receptive fields (DL-04). A ViT has almost none of that:

- every patch can attend to every other patch from layer 1;
- position is just a learned embedding;
- there's no built-in notion that nearby patches are related.

**The consequence:** on small or medium datasets, ViTs underperform CNNs, because they must *learn* locality from data. Given huge datasets (hundreds of millions of images), or strong augmentation and regularization plus distillation (DeiT), or self-supervised pretraining, ViTs match or beat CNNs, and they scale better.
Less bias means more data needed, and a higher ceiling.

---

## 2. Contrastive self-supervised learning (SimCLR, MoCo)

### 2.1 The idea

Create **two augmented views** of the same image (random crop, colour jitter, blur, flip). Train an encoder so that the two views of the same image map to **nearby** embeddings (a *positive* pair), while views of **different** images map far apart (*negatives*).

### 2.2 NT-Xent / InfoNCE

A batch of $N$ images gives $2N$ views. For a positive pair $(i, j)$ with normalized embeddings $z$, cosine similarity $\text{sim}$, and temperature $\tau$:

```math
\ell_{i,j} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k\ne i}\exp(\text{sim}(z_i, z_k)/\tau)} .
```

It's a $(2N-1)$-way classification: "which of the other views is my partner?" The other $2N - 2$ views are the negatives.

- **Why batch size matters:** more negatives make the task harder and more informative, and the loss is a tighter bound on the mutual information between the views. SimCLR used batches of 4096. **MoCo** gets many negatives without huge batches, by keeping a **queue** of embeddings from past batches, encoded by a slowly updated **momentum encoder** so they stay consistent.
- **Why augmentations matter most:** they *define* what the representation becomes invariant to. If the two views could be matched by a trivial cue, the network learns that cue instead of semantics.
  The classic example: without **colour jitter**, two crops of the same image share a colour histogram, and the network can match pairs by colour alone. The contrastive loss drops fast, but the features are useless for recognition (the lesson's debug question). Strong **random cropping** forces matching parts to wholes, and colour distortion removes the colour shortcut.

---

## 3. Non-contrastive SSL (BYOL, DINO): no negatives needed

Without negatives, the trivial solution is **collapse**: map every image to the same vector, and the two views match perfectly. BYOL and DINO avoid it through **asymmetry**:

- an **online/student** network predicts the output of a **target/teacher** network;
- the teacher's weights are an **exponential moving average (EMA)** of the student's, and **no gradient flows through the teacher** (stop-gradient);
- **BYOL** adds a *predictor* head on the online side only. **DINO** **centres** the teacher's outputs (subtracting a running mean, which prevents one dimension from dominating) and **sharpens** them with a low temperature (which prevents a uniform output). Centring and sharpening balance each other.

The slow-moving teacher provides targets that are stable but not constant, and the asymmetric architecture means the collapsed solution is not an attractor of these training dynamics. DINO's ViT attention maps famously segment objects with no labels at all.

## 4. Masked image modeling (MAE)

Mask a large fraction of the patches. The **encoder sees only the visible patches** (cheap). A light **decoder** reconstructs the pixels of the masked ones, and the loss is the MSE on the masked patches only.

**Why mask 75% when BERT masks 15%?** Images are highly **redundant** spatially. With 15% masked, a missing patch can be filled in by interpolating its neighbours, a low-level task that teaches little. Masking 75% removes most of the local context, so the model must understand objects and scenes to reconstruct them.
Language is dense in information: each word carries a lot, so 15% is already hard. (A side benefit: the encoder processes only 25% of the patches, so pretraining is about 3–4× faster.)

---

## 5. Evaluating representations

- **Linear probe:** freeze the encoder, and train only a linear classifier on its features. It measures **how linearly separable the semantic information already is** in the frozen representation, which is the quality of the features themselves.
- **k-NN evaluation:** classify each test image by the labels of its nearest training embeddings. There's no training at all, so it's even more direct.
- **Fine-tuning:** update everything. It measures how good a *starting point* the encoder is, which can hide weak frozen features. (MAE features probe poorly but fine-tune excellently. Contrastive and DINO features probe well.)

### 5.1 Supervised vs self-supervised backbones

- **Supervised ImageNet backbones:** strong for natural-image classification near ImageNet's label space.
- **SSL backbones (DINOv2 and similar):** more general features (good for retrieval, segmentation, depth, and *domains far from ImageNet*). They can be pretrained on **your own unlabeled domain data** (medical, satellite, industrial), which is often the biggest win when labels are scarce.

```python
import torch, torch.nn.functional as F, math
torch.manual_seed(0)

# --- Patchify: an image becomes a sequence of tokens ---------------------------------------
img = torch.randn(1, 3, 224, 224)
P = 16
patches = img.unfold(2, P, P).unfold(3, P, P)                    # 1,3,14,14,16,16
tokens = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, -1, 3 * P * P)
print("tokens:", tuple(tokens.shape), "-> 196 patches of 768 values")
conv = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16)
lin = torch.nn.Linear(768, 768)
lin.weight.data = conv.weight.data.reshape(768, -1); lin.bias.data = conv.bias.data
print("patch embedding == stride-16 conv:", torch.allclose(lin(tokens), conv(img).flatten(2).transpose(1, 2), atol=1e-4))
for p in [32, 16, 8]:
    n = (224 // p) ** 2
    print(f"patch {p:2d}: {n:4d} tokens, attention matrix {n*n:>9,} entries")

# --- NT-Xent (SimCLR) loss ------------------------------------------------------------------
def nt_xent(z1, z2, tau=0.5):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))                           # a view is not its own negative
    N = len(z1)
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)])
    return F.cross_entropy(sim, targets)
base = torch.randn(256, 64)
print(f"loss, views nearly identical: {nt_xent(base + 0.05*torch.randn_like(base), base + 0.05*torch.randn_like(base)):.3f}")
print(f"loss, views unrelated       : {nt_xent(torch.randn(256, 64), torch.randn(256, 64)):.3f}  (chance = ln(2N-1) = {math.log(511):.3f})")
```

```python
# --- The colour shortcut: matching views by colour histogram alone ---------------------------------
# Each "image" has a random global colour cast. Two crops share it unless colour jitter re-randomizes it.
N = 256
cast = torch.rand(N, 3)                                          # per-image colour
def view(jitter):
    c = cast + (0.5 * torch.randn(N, 3) if jitter else 0.02 * torch.randn(N, 3))
    return c                                                     # the "colour histogram" feature of the view
for jitter in [False, True]:
    v1, v2 = view(jitter), view(jitter)
    acc = ((torch.cdist(v1, v2).argmin(1) == torch.arange(N)).float().mean())
    print(f"colour jitter={jitter}: matching views by colour alone succeeds {acc:.0%} of the time")
print("-> without jitter, the pretext task is solvable by colour: the loss drops, but no semantics are learned")

# --- MAE-style random masking keeps 25% of the patches ------------------------------------------------
L, ratio = 196, 0.75
keep = torch.rand(L).argsort()[: int(L * (1 - ratio))]
print(f"MAE: the encoder sees {len(keep)} of {L} patches ({len(keep)/L:.0%})")
```

---

## Pitfalls & misconceptions

- **Training a ViT from scratch on a small dataset.** Use a pretrained one, or a CNN.
- **Weak augmentations in contrastive SSL.** The model learns shortcuts.
- **Judging SSL by its loss value.** Evaluate with linear probes or k-NN.
- **Comparing linear-probe and fine-tune numbers** as if they measured the same thing.
- **Ignoring resolution changes:** ViT position embeddings must be interpolated when the input size changes.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| ViT tokens | $(H/P)\cdot(W/P)$ patches + [CLS]; patch embedding = a stride-$P$ conv |
| Attention cost | $O(N^2)$ in the number of patches |
| NT-Xent | $-\log\frac{e^{s_{ij}/\tau}}{\sum_{k\ne i}e^{s_{ik}/\tau}}$ over $2N$ views |
| Anti-collapse (BYOL/DINO) | EMA teacher + stop-gradient (+ predictor; + centring and sharpening) |
| MAE | mask about 75%; the encoder sees only the visible patches; MSE on the masked ones |
| Evaluation | linear probe / k-NN (frozen) vs fine-tuning (starting point) |

## Answer sketches for the lesson's self-check

<details>
<summary>1. CNN inductive biases missing in ViTs, and the implication.</summary>

Locality, translation equivariance through weight sharing, and a hierarchical receptive field. ViTs must learn these from data, so they need much more data, strong augmentation, distillation, or SSL pretraining. At scale, their weaker bias stops being a limitation and they scale better.
</details>

<details>
<summary>2. SimCLR positives, negatives, and why batch size matters.</summary>

Positives: the two augmented views of the same image. Negatives: all the other $2N-2$ views in the batch. A larger batch gives more negatives, which makes a harder, more informative contrastive task (MoCo's queue achieves this without huge batches).
</details>

<details>
<summary>3. Why don't BYOL/DINO collapse without negatives?</summary>

The architecture is asymmetric: the student predicts a slowly moving EMA teacher, with no gradient flowing through the teacher (plus a predictor head in BYOL, and centring with sharpening in DINO). These dynamics keep the targets informative and make the constant solution unstable, so it isn't reached in practice.
</details>

<details>
<summary>4. Why mask 75% of the patches, versus BERT's 15%?</summary>

Images are spatially redundant: lightly masked patches can be inferred by local interpolation. Heavy masking forces holistic, semantic understanding. Text is information-dense, so 15% is already challenging. The heavy masking also makes the encoder about 3–4× cheaper.
</details>

<details>
<summary>5. What does a linear probe measure that fine-tuning doesn't?</summary>

The quality and linear separability of the frozen features themselves. Fine-tuning measures how good a starting point the whole network is, and can compensate for weak frozen features.
</details>

<details>
<summary>6. The SimCLR loss drops fast, but the linear probe stays near chance.</summary>

The pretext task is being solved by a low-level shortcut, most likely a missing colour jitter (views matched by colour histogram), or crops that are too weak or too similar. Strengthen and diversify the augmentations (the demo shows colour-only matching succeeding without jitter).
</details>

## Where this leads

Next: [CV-03 notes](03-clip-vision-language-models.md). Contrastive learning matched two *image* views. CLIP matches an image with its *caption*, which gives a shared space for vision and language, zero-shot classification, and the foundation for vision-language models.
