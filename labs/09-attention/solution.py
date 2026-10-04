"""Lab 09: scaled dot-product attention, causal masking, and multi-head self-attention in PyTorch.

Shapes: q, k, v are (..., T, d). Masks are boolean, True = "may attend". x for the module is (B, T, C).
"""
import math

import torch
from torch import nn


def causal_mask(T, device=None):
    """(T, T) boolean mask, True on and below the diagonal: position i may attend to j <= i."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


def scaled_dot_product_attention(q, k, v, mask=None):
    """Return (out, weights) with weights = softmax(q k^T / sqrt(d) with masked positions at -inf) and out = weights v.

    Subgoals: 1. scores = q @ k.transpose(-2, -1) / sqrt(d)   2. masked_fill(~mask, -inf)
              3. softmax over the LAST dim (the keys)          4. weights @ v
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = scores.softmax(dim=-1)
    return w @ v, w


def split_heads(x, n_heads):
    """(B, T, C) -> (B, n_heads, T, C // n_heads). Hint: view, then transpose(1, 2)."""
    B, T, C = x.shape
    return x.view(B, T, n_heads, C // n_heads).transpose(1, 2)


def merge_heads(x):
    """(B, h, T, d) -> (B, T, h * d), the inverse of split_heads. Hint: transpose, then .contiguous().view."""
    B, h, T, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, h * d)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """(given) One fused projection for q, k, v, and an output projection."""
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=True):
        """Subgoals: 1. q, k, v = self.qkv(x).chunk(3, dim=-1)   2. split heads
                     3. attention (with causal_mask if causal)    4. merge heads, then self.proj"""
        q, k, v = (split_heads(t, self.n_heads) for t in self.qkv(x).chunk(3, dim=-1))
        mask = causal_mask(x.shape[1], x.device) if causal else None
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        return self.proj(merge_heads(out))
