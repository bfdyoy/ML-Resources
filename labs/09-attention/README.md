# Lab 09: Attention, masks & multi-head self-attention

[← Labs](../README.md) · Lesson: [DL-06 Transformers](../../lessons/deep-learning/06-transformers.md) · Notes: [DL-06 notes](../../notes/deep-learning/06-transformers.md)

**Time** ≈ 1.5 h · **You'll practise:** the attention formula with real tensor shapes, masking with `-inf` before the softmax, and the reshape/transpose dance for heads.

| Function | Checked against |
|---|---|
| `causal_mask` | shape, dtype, and lower-triangular entries |
| `scaled_dot_product_attention` | `torch.nn.functional.scaled_dot_product_attention`, with and without `is_causal` |
| `split_heads`, `merge_heads` | round trip; head 1 is channels 4–7 |
| `MultiHeadSelfAttention.forward` | a reference built from PyTorch's fused kernel; **changing future tokens never changes past outputs** |

```bash
pytest labs/09-attention
```

**Bonus:**
1. Add a padding mask (B, T) that combines with the causal mask. What happens to a row whose keys are *all* masked, and how do real implementations avoid NaNs there?
2. Count the FLOPs and the memory of the (T, T) score matrix for T = 8,192 and 32 heads. That count is the reason FlashAttention exists ([DL-07](../../lessons/deep-learning/07-performance-gpus-mixed-precision.md)).
