# Playbook 3: Deep learning tricks

[← Playbook](README.md) · Previous: [Tabular tricks](02-tabular-tricks.md) · Next: [LLM & retrieval tricks](04-llm-and-retrieval-tricks.md)

> Collected from [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187), Karpathy's [recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/),
> Leslie Smith's [cyclical learning rates](https://arxiv.org/abs/1506.01186) and [hyperparameter report](https://arxiv.org/abs/1803.09820), and the papers linked in each section.
> The demos are tiny CPU runs that show the *mechanism*. **You need:** DL-02, DL-03.

---

## 1. Find the learning rate with a range test

**When:** a new model, a new dataset, or a new batch size. The learning rate is the hyperparameter that matters most.
**Why:** in a few hundred steps you can see the whole range, from "too small to move" through "fast" to "diverging".
**How:** start at a tiny LR and multiply it by a constant every step. Record the smoothed loss, and stop when it blows up.
Pick an LR about **10× below the point where the loss is lowest** (still on the steep downhill part).

```python
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(2048, 20)
w_true = torch.randn(20, 3)
y = (X @ w_true + 0.5 * torch.randn(2048, 3)).argmax(1)

def make_model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))

def lr_range_test(lr_min=1e-5, lr_max=10.0, steps=150, beta=0.9):
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
    mult, lr, avg, out = (lr_max / lr_min) ** (1 / steps), lr_min, 0.0, []
    g = torch.Generator().manual_seed(0)
    for t in range(1, steps + 1):
        idx = torch.randint(0, 2048, (64,), generator=g)
        loss = F.cross_entropy(model(X[idx]), y[idx])
        avg = beta * avg + (1 - beta) * loss.item()
        smooth = avg / (1 - beta ** t)                        # bias-corrected moving average
        if t > 10 and smooth > 4 * min(s for _, s in out):
            break
        out.append((lr, smooth))
        opt.zero_grad(); loss.backward(); opt.step()
        lr *= mult
        for gr in opt.param_groups:
            gr["lr"] = lr
    return out

curve = lr_range_test()
best_lr = min(curve, key=lambda c: c[1])[0]
print(f"loss is lowest at lr = {best_lr:.3g}; stopped at lr = {curve[-1][0]:.3g}; suggested lr = {best_lr / 10:.2g}")
```

The smoothed loss is lowest around lr ≈ **0.44**, and the test stops itself at lr ≈ **1.45** when the loss explodes. So the suggestion is lr ≈ **0.044**.
That takes 150 steps, a fraction of one training run.

**The trap:** the test is batch-size specific. Change the batch size and you must re-run it (roughly, the LR scales with the batch size for SGD, as in §2).

## 2. Warm up, then decay (one-cycle or cosine)

**When:** essentially always, and especially with Adam(W), large batches, transformers, or BatchNorm-free nets.
**Why:** at initialization the gradients are large and the optimizer's statistics (Adam's second moment) are unreliable. A short **warmup** (1–5% of steps) avoids early blow-ups.
A **decay** to near zero at the end lets the weights settle into a minimum. The large-minibatch paper ([Goyal et al.](https://arxiv.org/abs/1706.02677)) combines a gradual warmup
with the **linear scaling rule**: multiply the batch size by k, multiply the LR by k.
**How:** `torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=..., total_steps=...)`, or a linear warmup followed by cosine decay.

**The trap:** stepping the scheduler once per *epoch* when it was built for *steps*, or vice versa. Plot the LR you actually used.

## 3. Initialize the last-layer bias to the log prior

**When:** imbalanced classification, or regression with a large target mean.
**Why:** a freshly initialized net predicts roughly uniform probabilities. With 1% positives, the first hundreds of steps are spent learning just
"positives are rare", while big gradients shake up the rest of the network. Setting the output bias to `log(p / (1 − p))` starts the model at the base rate.
The focal-loss paper ([RetinaNet](https://arxiv.org/abs/1708.02002)) uses exactly this prior initialization. Karpathy's recipe calls it "init well".

```python
p_pos = 0.01
Xi = torch.randn(4000, 10)
yi = (torch.rand(4000) < p_pos).float()

def first_losses(bias_init, steps=100):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    with torch.no_grad():
        m[2].bias.fill_(bias_init)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(m(Xi).squeeze(1), yi)
        losses.append(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    return losses

zero, prior = first_losses(0.0), first_losses(math.log(p_pos / (1 - p_pos)))
print(f"bias 0:        loss at step 0 = {zero[0]:.3f}, after 100 steps = {zero[-1]:.3f}")
print(f"bias log-prior: loss at step 0 = {prior[0]:.3f}, after 100 steps = {prior[-1]:.3f}")
print(f"entropy of the base rate = {-(p_pos * math.log(p_pos) + (1 - p_pos) * math.log(1 - p_pos)):.3f}")
```

With a zero bias, the first loss is **0.724** (≈ ln 2), and after 100 Adam steps it's still **0.266**: the net spends its first steps learning the base rate.
With the log-prior bias, it starts at **0.065**, already near the base-rate entropy of 0.056, and it's at **0.061** after 100 steps. (On this random data there's nothing more to learn, so the base rate *is* the best possible loss.)
The same idea applies to regression: set the output bias to the target mean (or standardize the target).

**Sanity check that comes free with it:** the initial loss of a K-class classifier should be ≈ ln K. If it's far off, something is wrong before training even starts ([debugging playbook](06-debugging-playbook.md#b-deep-learning-training)).

## 4. Zero-initialize the last layer of each residual branch

**When:** ResNets and transformers (any `x + f(x)` block).
**Why:** if the last BatchNorm γ (or the last linear layer) of each residual branch starts at zero, every block starts as the identity, so a 100-layer net trains like a shallow one at first.
Both [Bag of Tricks](https://arxiv.org/abs/1812.01187) and [Goyal et al.](https://arxiv.org/abs/1706.02677) report it helps. GPT-2-style models apply a related scaling (they shrink the init of residual output projections by 1/√(number of layers)).
**How:** `nn.init.zeros_(block.bn2.weight)`. In torchvision ResNets, it's `resnet50(zero_init_residual=True)`.

## 5. Label smoothing

**When:** classification with many classes and clean-ish labels, where you want better calibration and slightly better accuracy.
**How:** `F.cross_entropy(logits, y, label_smoothing=0.1)`. The target becomes `1 − ε + ε/K` for the true class and `ε/K` for every other class.
**Why:** with hard targets, the loss keeps pushing logits apart forever. With smoothed targets, the optimal logit gap between the true class and the others is *finite*:
`log((1 − ε + ε/K) / (ε/K))`.

```python
for K, eps in [(10, 0.1), (1000, 0.1)]:
    gap = math.log((1 - eps + eps / K) / (eps / K))
    print(f"K={K:5d}, eps={eps}: the optimal logit gap is {gap:.2f}, so the true-class probability is {1 - eps + eps / K:.4f}")
```

For 10 classes and ε = 0.1, the optimum is a logit gap of **4.51** (a true-class probability of 0.91), not infinity. The net can stop inflating its weights.

**The trap:** [Müller, Kornblith & Hinton](https://arxiv.org/abs/1906.02629) show that label-smoothed *teachers* make **worse teachers for distillation**: smoothing erases the "dark knowledge" in the wrong-class probabilities.
Don't smooth a model you plan to distill from.

## 6. mixup and CutMix: train on blends

**When:** image classification (and, with care, audio and tabular data), when the model overfits or is overconfident.
**How:** [mixup](https://arxiv.org/abs/1710.09412) trains on `λ·x_i + (1 − λ)·x_j` with the same blend of the labels, where λ ~ Beta(α, α).
[CutMix](https://arxiv.org/abs/1905.04899) pastes a rectangle from one image into another and mixes the labels by area. Both act as strong regularizers that smooth the decision boundaries.

```python
def mixup_batch(x, y, num_classes, alpha=0.2, gen=None):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.shape[0], generator=gen)
    y1 = F.one_hot(y, num_classes).float()
    return lam * x + (1 - lam) * x[perm], lam * y1 + (1 - lam) * y1[perm]

torch.manual_seed(0)
xb, yb = torch.randn(8, 3, 4, 4), torch.randint(0, 5, (8,))
xm, ym = mixup_batch(xb, yb, 5, gen=torch.Generator().manual_seed(0))
loss = torch.sum(-ym * F.log_softmax(torch.randn(8, 5), dim=1), dim=1).mean()   # soft-target cross-entropy
print(xm.shape, ym.sum(1))
```

Every mixed label row still sums to 1. The loss is ordinary cross-entropy with soft targets (`F.cross_entropy` accepts probabilities as targets too).

**The trap:** mixup slows convergence. Train longer, and turn it off for the last few epochs if accuracy on clean images matters most.

## 7. Average the weights: EMA, SWA, model soups

**When:** almost free accuracy and stability at the end of training, or when you've fine-tuned several runs from the same pretrained model.
**Why:** SGD with a non-zero LR bounces around a minimum. The **average** of the iterates sits closer to the center than any single iterate.
Variants: an **EMA** of the weights during training, [**SWA**](https://arxiv.org/abs/1803.05407) (average checkpoints from the tail of training, which finds wider optima),
and [**model soups**](https://arxiv.org/abs/2203.05482) (average the weights of several fine-tunes that share an initialization, at no extra inference cost).

```python
rng = np.random.default_rng(0)
A = rng.normal(size=(5000, 10))
w_star = rng.normal(size=10)
b_vec = A @ w_star + rng.normal(scale=1.0, size=5000)
A_te = rng.normal(size=(2000, 10))

w, iterates = np.zeros(10), []
for t in range(4000):                                   # plain SGD with a constant step: it never settles
    i = rng.integers(0, 5000, 16)
    w -= 0.05 * 2 * A[i].T @ (A[i] @ w - b_vec[i]) / 16
    if t >= 2000:
        iterates.append(w.copy())
excess = lambda v: np.mean((A_te @ (v - w_star)) ** 2)  # test error above the irreducible noise
print(f"excess test MSE  last iterate: {excess(w):.4f}   average of the last 2000 iterates: {excess(np.mean(iterates, axis=0)):.4f}")
```

The final SGD iterate's *excess* test error (above the irreducible noise) is **0.0516**. The average of the last 2,000 iterates gets **0.0029**, about 18× smaller, with no extra training.
In PyTorch: `torch.optim.swa_utils.AveragedModel` (with `get_ema_multi_avg_fn` for EMA), and remember to recompute BatchNorm statistics after averaging (`update_bn`).

**The trap:** averaging only works within one "basin". Two models trained from *different* random inits usually can't be averaged (their hidden units are permuted). Soups need a shared pretrained start.

## 8. Test-time augmentation (TTA)

**When:** image (or audio) models where the prediction should be invariant to flips, small crops, or shifts.
**How:** predict on the original *and* on a few augmented copies (horizontal flip, 2–5 crops), then average the probabilities. It typically gains a little accuracy and calibration, at k× the inference cost.
**The trap:** use only augmentations the label is invariant to (don't flip digits or text), and measure the gain on validation data. It's not guaranteed.

## 9. Progressive resizing and gradient accumulation

**Progressive resizing** (fast.ai): train the early epochs on small images (e.g. 128 px), then finish at full size. The early epochs are much faster, and the size change acts as augmentation.

**Gradient accumulation:** when the batch you want doesn't fit in memory, run k micro-batches, call `backward()` on each (gradients add up in `.grad`), and step once.
Divide each micro-batch loss by k, so that the sum matches the mean over the large batch.

```python
torch.manual_seed(0)
Xa, ya = torch.randn(64, 8), torch.randint(0, 3, (64,))

def grads(model, chunks):
    model.zero_grad()
    for xc, yc in zip(Xa.chunk(chunks), ya.chunk(chunks)):
        (F.cross_entropy(model(xc), yc) / chunks).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])

torch.manual_seed(1); plain = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
torch.manual_seed(1); with_bn = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))
for name, m in [("no BatchNorm", plain), ("with BatchNorm", with_bn)]:
    diff = (grads(m, 1) - grads(m, 4)).abs().max().item()
    print(f"{name:15s} max |full-batch grad - 4x accumulated grad| = {diff:.2e}")
```

Without BatchNorm, four accumulated micro-batches reproduce the full-batch gradient exactly (difference **2e-8**, which is float32 round-off).
**With BatchNorm, they don't** (difference **0.04**), because each micro-batch is normalized with its own statistics. Accumulation gives you the large batch's *gradient*, not its *BatchNorm*.
Use GroupNorm or LayerNorm if you rely on accumulation with tiny micro-batches.

## 10. Discriminative learning rates when fine-tuning

**When:** fine-tuning a pretrained network on a new task.
**Why:** early layers hold general features (edges, syntax) that need little change. The new head is random and needs a lot.
**How:** parameter groups, `torch.optim.AdamW([{"params": backbone.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}])`.
Or, fast.ai-style: first train only the head with the backbone frozen, then unfreeze everything at a lower LR.

## 11. Sharpness-aware minimization (SAM)

**When:** you've already tuned the basics and want a bit more generalization, and you can afford about 2× the compute per step.
**Why:** [SAM](https://arxiv.org/abs/2010.01412) takes the gradient at the *worst* nearby point (a small step uphill) and descends from there. That favours flat minima, which tend to generalize better.
**The trap:** it doubles the cost of every step. Compare against simply training 2× longer.

## 12. Distill a big model into a small one

**When:** the accurate model is too slow or too big to deploy.
**How:** train the small model on a mix of the true labels and the teacher's **softened** probabilities (a temperature T > 1 on both models' logits; [Hinton et al.](https://arxiv.org/abs/1503.02531)).
The soft targets carry the teacher's knowledge of which wrong answers are *almost* right. And see §5's trap: don't distill from a label-smoothed teacher.

---

## Cheat sheet

| Situation | Trick |
|---|---|
| New model or data | LR range test (§1) → one-cycle or warmup + cosine (§2) |
| Imbalanced classes, or a slow start | Output bias = log prior (§3); check the initial loss ≈ ln K |
| Very deep residual net | Zero-init the last γ in each branch (§4) |
| Overconfident or overfitting classifier | Label smoothing (§5), mixup/CutMix (§6) |
| Free gains at the end | EMA/SWA, soups for fine-tunes (§7), TTA (§8) |
| Batch doesn't fit in memory | Gradient accumulation, and beware of BatchNorm (§9) |
| Fine-tuning | Discriminative LRs, or freeze then unfreeze (§10) |
| Deployment is too slow | Distillation (§12) |
