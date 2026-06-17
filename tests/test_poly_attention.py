import pytest
import torch
from poly_attention import PolyAttention, NPolyAttention, Order2PolyAttention
from rotary_embedding_torch import RotaryEmbedding

param = pytest.mark.parametrize

@param('causal', (False, True))
@param('kv_heads', (1, 2, 4))
def test_poly_attention(causal, kv_heads):
    attn = PolyAttention(dim = 128, heads = 4, kv_heads = kv_heads, dim_head = 32, causal = causal)
    x = torch.randn(2, 32, 128)

    rotary_emb = RotaryEmbedding(32)
    rotary_pos_emb = rotary_emb(torch.arange(32))

    out = attn(x, rotary_pos_emb = rotary_pos_emb)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

@param('kv_heads', (1, 2, 4))
def test_poly_attention_mask(kv_heads):
    attn = PolyAttention(dim = 128, heads = 4, kv_heads = kv_heads, dim_head = 32)
    x = torch.randn(2, 32, 128)

    rotary_emb = RotaryEmbedding(32)
    rotary_pos_emb = rotary_emb(torch.arange(32))

    mask = torch.ones(2, 32).bool()
    mask[:, 16:] = False

    out = attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

@param('causal', (False, True))
@param('kv_heads', (1, 2, 4))
def test_generalized_poly_attention_equivalence(causal, kv_heads):
    dim = 128
    heads = 4
    dim_head = 32

    model_order2 = Order2PolyAttention(dim = dim, heads = heads, kv_heads = kv_heads, dim_head = dim_head, causal = causal)
    model_n = NPolyAttention(dim = dim, order = 2, heads = heads, kv_heads = kv_heads, dim_head = dim_head, causal = causal)

    state_dict = model_order2.state_dict()
    state_dict['q_norms.0.weight'] = state_dict.pop('q1_norm.weight')
    state_dict['q_norms.1.weight'] = state_dict.pop('q2_norm.weight')
    state_dict['q_norms.2.weight'] = state_dict.pop('q3_norm.weight')

    model_n.load_state_dict(state_dict, strict = False)

    x = torch.randn(2, 32, 128)

    rotary_emb = RotaryEmbedding(dim_head)
    rotary_pos_emb = rotary_emb(torch.arange(32))

    mask = torch.ones(2, 32).bool()
    mask[:, 16:] = False

    out_order2 = model_order2(x, rotary_pos_emb = rotary_pos_emb)
    out_n = model_n(x, rotary_pos_emb = rotary_pos_emb)

    assert torch.allclose(out_order2, out_n, atol = 1e-5)

    out_order2_masked = model_order2(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
    out_n_masked = model_n(x, mask = mask, rotary_pos_emb = rotary_pos_emb)

    assert torch.allclose(out_order2_masked, out_n_masked, atol = 1e-5)

@param('causal', (False, True))
@param('kv_heads', (1, 2, 4))
@param('order', (2, 3, 4))
def test_n_poly_attention(causal, kv_heads, order):
    attn = NPolyAttention(dim = 128, order = order, heads = 4, kv_heads = kv_heads, dim_head = 32, causal = causal)
    x = torch.randn(2, 32, 128)

    rotary_emb = RotaryEmbedding(32)
    rotary_pos_emb = rotary_emb(torch.arange(32))

    out = attn(x, rotary_pos_emb = rotary_pos_emb)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

def test_poly_attention_kv_cache():
    attn = PolyAttention(dim = 128, heads = 4, dim_head = 32, causal = True)

    x = torch.randn(2, 5, 128)

    out_parallel = attn(x)

    cache = None
    out_stepwise = []

    for i in range(5):
        out, cache = attn(x[:, i:i+1], cache = cache, return_cache = True)
        out_stepwise.append(out)

    out_stepwise = torch.cat(out_stepwise, dim = -2)

    assert torch.allclose(out_parallel, out_stepwise, atol = 1e-5)

def test_poly_attention_kv_cache_with_rotary():
    attn = PolyAttention(dim = 128, heads = 4, dim_head = 32, causal = True, use_rotary_embed = True)

    x = torch.randn(2, 5, 128)

    out_parallel = attn(x)

    cache = None
    out_stepwise = []

    for i in range(5):
        out, cache = attn(x[:, i:i+1], cache = cache, return_cache = True)
        out_stepwise.append(out)

    out_stepwise = torch.cat(out_stepwise, dim = -2)

    assert torch.allclose(out_parallel, out_stepwise, atol = 1e-5)
