# Lab 07: micrograd, a scalar autograd engine

[← Labs](../README.md) · Lesson: [DL-01 Neural networks from scratch](../../lessons/deep-learning/01-neural-networks-from-scratch.md) · Notes: [DL-01 notes](../../notes/deep-learning/01-neural-networks-from-scratch.md) · Original: [Karpathy's nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

**Time** ≈ 2 h · **You'll practise:** local derivatives, the chain rule as "multiply by the upstream gradient", gradient accumulation, and topological sort.

The constructor, `__repr__`, `__add__` (a worked example of the pattern), and the derived operators (`-`, `/`, reflected ops) are given. You write
`__mul__`, `__pow__`, `exp`, `log`, `tanh`, `relu` and `backward`.

| Test | What it catches |
|---|---|
| `test_add_mul_backward` | the basic chain rule |
| `test_gradients_accumulate_when_reused` | `=` instead of `+=` in a `_backward` |
| `test_against_torch` | every op, compared with PyTorch autograd in float64 |
| `test_train_a_neuron` | the whole loop works end to end |

```bash
pytest labs/07-micrograd
```

**Bonus:** add `sigmoid` as one op with its own local derivative, `s(1 - s)`, and check it against `1 / (1 + (-x).exp())` built from primitives.
Which version builds a smaller graph, and why do frameworks fuse operations like this?
