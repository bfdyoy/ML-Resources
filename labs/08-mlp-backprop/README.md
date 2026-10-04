# Lab 08: MLP backprop with a gradient check

[← Labs](../README.md) · Lessons: [DL-01](../../lessons/deep-learning/01-neural-networks-from-scratch.md) → [DL-02](../../lessons/deep-learning/02-pytorch-fluency.md) · Notes: [DL-01 notes](../../notes/deep-learning/01-neural-networks-from-scratch.md)

**Time** ≈ 2 h · **You'll practise:** matrix-shaped backprop ("the gradient has the shape of the thing"), the `softmax − one_hot` gradient, and the CS231n habit of checking every backward pass numerically.

| Function | Checked against |
|---|---|
| `linear_backward`, `relu_backward` | numeric gradients, shapes |
| `softmax_cross_entropy` | `torch.nn.functional.cross_entropy`; stable at logit 1000 |
| `mlp_loss_and_grads` | numeric gradient check (relative error < 1e-6) **and** PyTorch autograd |
| `numeric_grad` | restores its input |

```bash
pytest labs/08-mlp-backprop
```

**Bonus:** run the gradient check in `float32` instead of `float64`. What relative error do you get now, and why does `eps = 1e-6` stop being a good choice?
(Round-off error grows like `machine-eps / eps`, while truncation error shrinks like `eps²`. Find the best `eps` for each precision.)
