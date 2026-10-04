# DL-02 notes: PyTorch Fluency

[← Lesson DL-02](../../lessons/deep-learning/02-pytorch-fluency.md) · [All notes](../README.md) · [← DL-01 notes](01-neural-networks-from-scratch.md) · Next: [DL-03 notes →](03-training-deep-networks.md)

> **Reading time** ≈ 45 min. **You need:** [DL-01 notes](01-neural-networks-from-scratch.md) (autograd, the training step). Run every code cell, because fluency comes from your fingers.

---

## Where we are

DL-01 built autograd from scratch on scalars. PyTorch is the same idea on **tensors** (n-dimensional arrays) that can live on a GPU. This note covers the mental model behind the API:

- how tensors are stored;
- how broadcasting lines up shapes;
- what autograd records;
- the one training loop you'll write a thousand times.

Most "my model doesn't learn" bugs are plumbing bugs from this list.

---

## 1. Tensors: shape, strides, and memory

A tensor is a **flat block of memory**, plus a **shape**, plus **strides** (how many elements to jump to move one step along each dimension). A $3\times4$ row-major tensor has strides $(4, 1)$.

- **`transpose`/`permute`** don't move any data. They swap the strides. The result is **non-contiguous**: its elements are no longer laid out in row order.
- **`view`** reinterprets the same memory with a new shape. That only works if the new shape is compatible with the current strides, so it **fails on many non-contiguous tensors**.
- **`reshape`** returns a view when it can, and **copies** when it must. Use `reshape` unless you specifically need a guarantee that no copy is made.
- **`contiguous()`** makes a row-ordered copy.

**dtype and device.** `float32` is the default for parameters. Labels for `CrossEntropyLoss` must be `int64` (`long`). Every tensor taking part in an operation must be on the **same device**: a GPU model with a CPU batch raises a device-mismatch error.
The fix is `x = x.to(device)` (and remember that `.to` returns a new tensor; it doesn't modify `x` in place).

## 2. Broadcasting: the rule, and its classic bug

To combine two tensors, align their shapes **from the right**. Each pair of dimensions must be **equal, or one of them must be 1** (missing dimensions count as 1). Size-1 dimensions are stretched virtually, without copying.

| a.shape | b.shape | Result | Why |
|---|---|---|---|
| (32, 10) | (10,) | (32, 10) | Bias added to every row |
| (32, 1) | (1, 10) | (32, 10) | Outer-product-like grid |
| (32, 1) | (32,) | **(32, 32)** | Silent bug! |

**The silent bug.** A regression model outputs shape `(N, 1)` and the targets have shape `(N,)`. Then `pred - y` has shape `(N, N)`: every prediction minus every target. The MSE is computed over $N^2$ pairs, the model learns roughly "predict the mean",
**and nothing crashes**. (`nn.MSELoss` warns about it; a hand-written loss doesn't.) Fix it with `pred.squeeze(-1)` or `y.unsqueeze(-1)`, and assert shapes at the boundaries.

## 3. Autograd, the PyTorch way

- **Leaf tensors** with `requires_grad=True` (parameters) receive `.grad`. Every operation on them records a node in a graph, exactly like `Value` in DL-01.
- `loss.backward()` walks the graph in reverse and **adds** into each leaf's `.grad`. It's the `+=` again. Hence **`optimizer.zero_grad()` every step**, or gradients from earlier batches pile up.
  (Accumulating on purpose over several micro-batches is *gradient accumulation*, a way to simulate a big batch.)
- After `backward()`, the graph is freed. Calling `backward` twice needs `retain_graph=True`, which you almost never want.
- **`torch.no_grad()`** turns recording off, which saves memory and time. Use it for evaluation, and inside manual weight updates.
- **`detach()`** cuts a tensor out of the graph. Use it when you log a loss, or for targets that shouldn't receive gradient.

## 4. `nn.Module`, train/eval modes, and the loop

Assigning an `nn.Parameter` or a sub-`nn.Module` as an attribute **registers** it, so `model.parameters()` finds it and `model.to(device)` moves it.
(A plain Python list of layers is *not* registered. Use `nn.ModuleList`.)

**Two independent switches. You usually want both at inference time:**

| Switch | Affects | Why |
|---|---|---|
| `model.eval()` / `model.train()` | **Layer behaviour**: dropout is off in eval mode; BatchNorm uses its running statistics instead of the batch's statistics | Deterministic, consistent predictions |
| `torch.no_grad()` | **Autograd recording** | Memory and speed |

`eval()` doesn't stop gradients, and `no_grad()` doesn't switch dropout off. That's why you need both.

**The canonical loop** (memorize this):

```
for epoch in range(E):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)      # raw logits into CrossEntropyLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        ... compute validation metrics ...
```

**Data:**

- A `Dataset` implements `__len__` and `__getitem__(i)`, returning one example.
- A `DataLoader` batches, shuffles (training data only), and loads in parallel with `num_workers`.
- Transforms with randomness (augmentation) belong in the *training* dataset only.

**Reproducibility and checkpoints.** Set seeds (`torch.manual_seed`, plus NumPy and `random`). Save the `state_dict` of both the model **and** the optimizer, so you can resume training.

```python
import torch, torch.nn as nn
torch.manual_seed(0)

# --- view vs reshape on a non-contiguous tensor ------------------------------
a = torch.arange(12).reshape(3, 4)
t = a.T                                         # transpose: same memory, swapped strides
print("strides a:", a.stride(), " strides a.T:", t.stride(), " contiguous?", t.is_contiguous())
try:
    t.view(12)
except RuntimeError as e:
    print("view failed:", str(e).split(".")[0])
print("reshape works:", t.reshape(12)[:6].tolist(), "(it copied)")

# --- The (N,1) vs (N,) broadcasting bug -------------------------------------------
N = 5
pred, y = torch.randn(N, 1), torch.randn(N)
print("pred - y has shape", tuple((pred - y).shape), "<- should have been", (N,))

# --- train() vs eval(): dropout ---------------------------------------------------
drop = nn.Dropout(p=0.5); x = torch.ones(8)
drop.train(); print("train mode:", drop(x).tolist())     # kept units are scaled by 1/(1-p) = 2, so the expected value stays 1
drop.eval();  print("eval mode :", drop(x).tolist())
```

```python
# --- A complete, idiomatic training loop on synthetic data -----------------------------
from torch.utils.data import Dataset, DataLoader

class Blobs(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.y = torch.randint(0, 3, (n,), generator=g)
        centers = torch.tensor([[0., 0.], [3., 0.], [0., 3.]])
        self.x = centers[self.y] + torch.randn(n, 2, generator=g)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dl = DataLoader(Blobs(2000, 0), batch_size=64, shuffle=True)
val_dl = DataLoader(Blobs(500, 1), batch_size=256)

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 3)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            correct += (model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
    print(f"epoch {epoch}: last train loss {loss.item():.3f}, val acc {correct / len(val_dl.dataset):.3f}")

torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, "/tmp/ckpt.pt")
```

---

## Pitfalls & misconceptions

- **Applying softmax before `CrossEntropyLoss`.** It expects raw logits, so applying softmax first double-squashes them. Training is then slow or stalls.
- **Labels with the wrong dtype or shape:** float labels, or one-hot labels where class indices are expected.
- **Forgetting `model.eval()`** during validation. Dropout and BatchNorm then change your metrics.
- **Shuffled features but unshuffled labels**: a custom dataset that shuffles `x` and `y` separately. Validation accuracy sits at chance.
- **Evaluating on training-mode BatchNorm with batch size 1.**
- **Accumulating the loss tensor itself** in a list for logging. That keeps every graph alive and leaks memory. Use `loss.item()`.

## Cheat sheet

| Need | Code |
|---|---|
| Safe reshape | `x.reshape(...)` |
| Device move | `x = x.to(device)` (returns a new tensor) |
| Step | `zero_grad()` → `backward()` → `step()` |
| Inference | `model.eval()` + `with torch.no_grad():` |
| Log a scalar | `loss.item()` |
| Classification loss | `nn.CrossEntropyLoss()(logits, long_labels)` |
| Checkpoint | `torch.save({"model": m.state_dict(), "opt": o.state_dict()}, path)` |

## Answer sketches for the lesson's self-check

<details>
<summary>1. view vs reshape: when does view fail?</summary>

`view` reinterprets the existing memory without copying, so it needs a stride-compatible layout. It fails on non-contiguous tensors (after `transpose`/`permute`, for example). `reshape` falls back to a copy when needed.
</details>

<details>
<summary>2. Why zero_grad every step?</summary>

`backward()` adds into `.grad`, so without zeroing, each step uses the sum of the current and all previous gradients. It's only correct to skip it deliberately, for gradient accumulation.
</details>

<details>
<summary>3. model.eval() vs torch.no_grad().</summary>

`eval()` changes layer behaviour (dropout off, BatchNorm uses running statistics). `no_grad()` stops autograd recording (it saves memory and time). They're independent, and inference wants both.
</details>

<details>
<summary>4. Model on GPU, batch on CPU?</summary>

A `RuntimeError` about tensors being on different devices. Move each batch with `.to(device)`, and remember that `.to` returns a new tensor.
</details>

<details>
<summary>5. Training loss drops, validation stays at chance: three PyTorch bugs.</summary>

(1) Labels misaligned with inputs (separate shuffling, or a wrong index in `__getitem__`). (2) Validation preprocessing differs from training (different normalization, missing transform), or validation runs in train mode with BatchNorm and tiny batches. (3) Evaluating the wrong thing: argmax over the wrong dimension, comparing against the wrong label tensor, or a stale or uninitialized model. Also check for the `(N,1)` vs `(N,)` broadcasting bug in a custom loss or metric.
</details>

## Where this leads

Next: [DL-03 notes](03-training-deep-networks.md). Now that the plumbing is solid, the hard part is making *deep* networks train well: initialization, normalization, optimizers, schedules, and regularization,
each with the math explaining why it works.
