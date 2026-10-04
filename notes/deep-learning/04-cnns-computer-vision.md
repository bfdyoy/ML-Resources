# DL-04 notes: Convolutional Networks & Computer Vision

[← Lesson DL-04](../../lessons/deep-learning/04-cnns-computer-vision.md) · [All notes](../README.md) · [← DL-03 notes](03-training-deep-networks.md) · Next: [DL-05 notes →](05-embeddings-sequences-attention.md)

> **Reading time** ≈ 55 min. **You need:** [DL-03 notes](03-training-deep-networks.md) (residual connections, BatchNorm, augmentation).

---

## Where we are

An MLP treats a 224×224 image as 150,528 unrelated numbers. It doesn't know that neighbouring pixels belong together, or that a cat in the corner is still a cat.
Convolutional networks build two facts about images **into the architecture**:

- **locality:** nearby pixels matter together;
- **translation equivariance:** the same pattern means the same thing wherever it appears.

That's the first example of an **inductive bias**, an idea that runs through every architecture lesson that follows.

---

## 1. The convolution operation

A conv layer slides a small **kernel** $K$ (say 3×3) over the input and takes a weighted sum at each position. Deep-learning "convolution" is technically **cross-correlation** (the kernel isn't flipped), which is fine because the kernel is learned anyway:

```math
Y[i, j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K[u, v]\; X[i + u,\ j + v] + b .
```

**With channels:** an input with $C_\text{in}$ channels and $C_\text{out}$ output channels uses $C_\text{out}$ kernels, each of size $k\times k\times C_\text{in}$. Each output channel is a **feature map**: "where in the image does this pattern appear?"

### 1.1 Output size

With input size $n$, kernel $k$, padding $p$, and stride $s$:

```math
n_\text{out} = \Big\lfloor \frac{n + 2p - k}{s} \Big\rfloor + 1 .
```

*Examples:* $n = 32$, $k = 3$, $p = 1$, $s = 1$ gives 32 ("same" padding). With $s = 2$ it gives 16 (downsampling). Without padding, $n = 32$, $k = 5$ gives 28.

### 1.2 Why so few parameters

A conv layer with a $k\times k$ kernel, $C_\text{in}\to C_\text{out}$, has $k^2 C_\text{in} C_\text{out} + C_\text{out}$ parameters, **independent of the image size**, because the same kernel is reused at every position (**weight sharing**).

*Worked example (the lesson's question 1).* A 3×3 conv from 64 to 128 channels has $9\cdot64\cdot128 + 128 = 73{,}856$ parameters. A dense layer mapping a 32×32×64 input (65,536 values) to an output of the same size as the conv's (32×32×128 = 131,072 values) has $65{,}536\times131{,}072 \approx 8.6$ **billion** weights.
That's about 100,000× more, and it would have to learn the same edge detector separately at every location.

### 1.3 Equivariance and invariance

Shift the input and a conv layer's output shifts the same way. That's **equivariance**, and it comes from weight sharing. **Pooling** (max or average over small windows) and, at the end, **global average pooling** turn it into approximate **invariance**: "is there a cat anywhere?"
The invariance is only approximate, because strided layers break exact shift-equivariance.

---

## 2. Receptive fields and the hierarchy

The **receptive field** of a unit is the input region that can influence it. Stacking layers grows it. With stride-1 $k\times k$ convs, each layer adds $k - 1$; strides multiply the growth of every later layer:

```math
r_l = r_{l-1} + (k_l - 1)\prod_{i \lt l} s_i,\qquad r_0 = 1 .
```

**Why stacked 3×3 convs beat one big kernel:** two 3×3 layers see a 5×5 region, and three see 7×7. Compare the costs for $C$ channels in and out:

| Option | Receptive field | Parameters | Non-linearities |
|---|---|---|---|
| One 7×7 conv | 7×7 | $49C^2$ | 1 |
| Three 3×3 convs | 7×7 | $27C^2$ | 3 |

The same view for 45% fewer parameters, *and* more non-linearities, so more expressive power. That was the VGG insight.

**The feature hierarchy.** Early layers detect edges and colours, middle layers textures and parts, and late layers whole objects. Visualization studies (Distill's *Feature Visualization*) show this directly.
It's also why **transfer learning** works: early features are generic, and only the late ones are task-specific.

---

## 3. The architecture lineage, in one table

| Model | Year | Key idea |
|---|---|---|
| LeNet | 1998 | conv → pool → conv → pool → dense, for digits |
| AlexNet | 2012 | Deep CNN + ReLU + dropout + GPUs; won ImageNet by a wide margin |
| VGG | 2014 | Only 3×3 convs, very deep and uniform |
| GoogLeNet/Inception | 2014 | Parallel branches; 1×1 convs to reduce channels cheaply |
| **ResNet** | 2015 | Residual connections (DL-03 §1.2): 152 layers trainable |
| EfficientNet / ConvNeXt | 2019 / 2022 | Principled scaling; modernized CNNs that match ViTs |
| ViT | 2020 | No convolutions: image patches as tokens to a transformer ([CV-02 notes](../vision/02-vision-transformers-self-supervised.md)) |

A **1×1 convolution** is a dense layer applied at every pixel. It mixes channels without looking at neighbours, and it's used everywhere to change the channel count cheaply.

---

## 4. Fine-tuning a pretrained model

The recipe:

1. Replace the classifier head.
2. **Freeze** the backbone and train the head.
3. Unfreeze, and fine-tune everything with a **smaller learning rate** (often a smaller one still for the early layers: *discriminative learning rates*).

Why each step:

- **Freeze first:** a random head produces large, noisy gradients that would wreck the pretrained features before the head learns anything sensible.
- **Small learning rate for pretrained weights:** they're already near a good solution, and big steps would erase what they learned from millions of images (*catastrophic forgetting*).
- **Augmentation** fills in for the data you don't have. Make it match the variation you'll see in deployment.

**Domain shift** (the lesson's debug question): 98% on validation images that look like the training set, but failure on users' phone photos (different cameras, lighting, blur, framing). Test the guess by collecting a small labeled set *of real phone photos* and comparing.
Then fix it with matching augmentations, by adding real phone photos to training, and by monitoring production inputs (PROD-03).

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

# --- conv2d with explicit loops vs F.conv2d -----------------------------------------
X = torch.randn(1, 2, 6, 6)                   # batch, channels, height, width
K = torch.randn(3, 2, 3, 3)                   # out_ch, in_ch, kh, kw
b = torch.randn(3)
Y = torch.zeros(1, 3, 4, 4)
for o in range(3):
    for i in range(4):
        for j in range(4):
            Y[0, o, i, j] = (K[o] * X[0, :, i:i + 3, j:j + 3]).sum() + b[o]
print("loops == F.conv2d:", torch.allclose(Y, F.conv2d(X, K, b), atol=1e-5))

# --- Output-size formula and parameter counts ----------------------------------------
out = lambda n, k, p, s: (n + 2 * p - k) // s + 1
print("sizes:", out(32, 3, 1, 1), out(32, 3, 1, 2), out(32, 5, 0, 1))
conv = nn.Conv2d(64, 128, 3, padding=1)
print("3x3 conv 64->128 params:", sum(p.numel() for p in conv.parameters()))
print("dense 32*32*64 -> 32*32*128 weights:", f"{(32*32*64) * (32*32*128):,}")
print("one 7x7 vs three 3x3 (C=64):", 49 * 64 * 64, "vs", 3 * 9 * 64 * 64)

# --- Receptive field of a small stack ----------------------------------------------------
layers = [(3, 1), (3, 1), (3, 2), (3, 1), (3, 2), (3, 1)]   # (kernel, stride)
r, jump = 1, 1
for k, s in layers:
    r += (k - 1) * jump; jump *= s
print("receptive field after the stack:", r)

# --- Translation equivariance: shift in -> shift out -------------------------------------
img = torch.zeros(1, 1, 12, 12); img[0, 0, 3, 3] = 1
shifted = torch.roll(img, shifts=(2, 2), dims=(2, 3))
k = torch.randn(1, 1, 3, 3)
o1, o2 = F.conv2d(img, k, padding=1), F.conv2d(shifted, k, padding=1)
print("conv(shift(x)) == shift(conv(x)):", torch.allclose(torch.roll(o1, (2, 2), (2, 3)), o2, atol=1e-6))
```

---

## Pitfalls & misconceptions

- **Mixing up channels-first and channels-last.** PyTorch is `N, C, H, W`. Images loaded with PIL or NumPy are `H, W, C`.
- **Forgetting ImageNet normalization** (mean and std) when fine-tuning a model pretrained with it.
- **Augmenting the validation set.** Validation needs deterministic transforms only.
- **Fine-tuning everything at a high learning rate from step 1.** It destroys the pretrained features.
- **"CNNs are translation-invariant."** They're equivariant by design, and only approximately invariant after pooling and striding.

## Cheat sheet

| Item | Formula |
|---|---|
| Output size | $\lfloor (n + 2p - k)/s\rfloor + 1$ |
| Conv parameters | $k^2C_\text{in}C_\text{out} + C_\text{out}$ |
| Receptive field | $r_l = r_{l-1} + (k_l-1)\prod_{i \lt l}s_i$ |
| Stacked 3×3 | $n$ layers → $(2n+1)\times(2n+1)$ field, $9nC^2$ parameters |
| Fine-tuning | freeze → train head → unfreeze at a small learning rate |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Parameters of a 3×3 conv 64→128 vs a dense layer.</summary>

$9\cdot64\cdot128 + 128 = 73{,}856$. A dense layer from 32×32×64 to an output of the same size has about $8.6\times10^9$ weights. Weight sharing and locality make the difference.
</details>

<details>
<summary>2. Why stacked 3×3 beats one 7×7.</summary>

The same 7×7 receptive field with $27C^2$ instead of $49C^2$ parameters, plus three non-linearities instead of one, so it's more expressive and cheaper.
</details>

<details>
<summary>3. What do residual connections solve?</summary>

Vanishing gradients and degradation in very deep networks. The identity path makes the block Jacobian $I + \partial F/\partial x$, so gradients flow back undiminished, and unneeded blocks can learn to be the identity (DL-03 §1.2).
</details>

<details>
<summary>4. Why freeze early layers first and use a smaller learning rate?</summary>

The random new head would send large, noisy gradients into good pretrained features. Freezing protects them while the head learns. Later, small learning rates adapt the features gently without catastrophic forgetting. Early layers are generic, so they need the least change.
</details>

<details>
<summary>5. 98% on validation, fails on users' phone photos.</summary>

Domain shift: validation came from the training distribution, and phone photos differ (lighting, blur, angle, compression). Test it by labeling a small set of real phone photos and evaluating on them. Fix it by adding such data, using matching augmentation, and monitoring the inputs.
</details>

## Where this leads

Next: [DL-05 notes](05-embeddings-sequences-attention.md). CNNs exploit the grid structure of images. Text is a **sequence of discrete symbols**, so we need two new ideas:
**embeddings**, to turn symbols into vectors, and a way to handle **order and long-range context**. That second need leads straight to attention.
