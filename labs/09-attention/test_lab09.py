import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(0)


def test_causal_mask(impl):
    m = impl.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m[2, 2] and m[3, 0] and not m[0, 1]
    assert m.sum() == 10


def test_attention_matches_pytorch(impl):
    q, k, v = (torch.randn(2, 3, 5, 8, dtype=torch.float64) for _ in range(3))
    out, w = impl.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 3, 5, dtype=torch.float64))
    out_c, w_c = impl.scaled_dot_product_attention(q, k, v, impl.causal_mask(5))
    torch.testing.assert_close(out_c, F.scaled_dot_product_attention(q, k, v, is_causal=True))
    assert torch.all(w_c.triu(1) == 0)


def test_split_merge_roundtrip(impl):
    x = torch.randn(2, 5, 12)
    h = impl.split_heads(x, 3)
    assert h.shape == (2, 3, 5, 4)
    torch.testing.assert_close(h[:, 1], x[..., 4:8])
    torch.testing.assert_close(impl.merge_heads(h), x)


def test_mhsa_matches_reference(impl):
    m = impl.MultiHeadSelfAttention(16, 4).double()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    q, k, v = m.qkv(x).chunk(3, -1)
    q, k, v = (t.view(2, 6, 4, 4).transpose(1, 2) for t in (q, k, v))
    ref = m.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 6, 16))
    torch.testing.assert_close(m(x), ref)


def test_causality(impl):
    m = impl.MultiHeadSelfAttention(16, 4).double()
    x = torch.randn(1, 6, 16, dtype=torch.float64)
    y = m(x)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 16, dtype=torch.float64)       # change the future
    torch.testing.assert_close(m(x2)[:, :4], y[:, :4])           # the past must not move
    assert not torch.allclose(m(x2, causal=False)[:, :4], m(x, causal=False)[:, :4])
