import pytest
import torch

from poly_attention import PolyAttention, NPolyAttention, Order2PolyAttention
from poly_attention.poly_vit import PolyViT

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
    softclamp_value = 20.

    model_order2 = Order2PolyAttention(dim = dim, heads = heads, kv_heads = kv_heads, dim_head = dim_head, causal = causal, softclamp_value = softclamp_value)
    model_n = NPolyAttention(dim = dim, order = 2, heads = heads, kv_heads = kv_heads, dim_head = dim_head, causal = causal, softclamp_value = softclamp_value)

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

@param('order', (2, 3, 4))
def test_prenorm(order):
    if order == 2:
        attn = PolyAttention(dim = 128, heads = 4, dim_head = 32, prenorm = True)
    else:
        attn = NPolyAttention(dim = 128, order = order, heads = 4, dim_head = 32, prenorm = True)

    x = torch.randn(2, 32, 128)
    out = attn(x)

    assert out.shape == (2, 32, 128)
    assert not torch.isnan(out).any()

@param('order', (2, 3))
def test_poly_vit(order):
    vit = PolyViT(
        image_size = 32,
        patch_size = 8,
        num_classes = 10,
        dim = 64,
        depth = 2,
        heads = 4,
        mlp_dim = 128,
        order = order
    )

    img = torch.randn(2, 3, 32, 32)
    preds = vit(img)

    assert preds.shape == (2, 10)
    assert not torch.isnan(preds).any()

from poly_attention.poly_attention import reference_poly_attention

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'cuda required')
@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('softclamp_val', (None, 20.0))
@pytest.mark.parametrize('seq_len', (31, 32, 127, 128))
def test_fused_poly_attention(causal, softclamp_val, seq_len):
    from poly_attention.fused_poly_attention import fused_poly_attention

    torch.manual_seed(42)
    shape = (2, 4, seq_len, 64)

    tensors = [torch.randn(shape, requires_grad = True, device = 'cuda') for _ in range(4)]
    pt_tensors = [t.clone().detach().requires_grad_(True) for t in tensors]

    q1, q2, q3, v3 = pt_tensors

    out_pt, _, _ = reference_poly_attention(q1, q2, q2, q3, v3, softclamp_val = softclamp_val, causal = causal)
    dout = torch.randn_like(out_pt)
    out_pt.backward(dout)

    out_tr = fused_poly_attention(*tensors, softclamp_val = softclamp_val, is_causal = causal)
    out_tr.backward(dout)

    assert torch.allclose(out_pt, out_tr, atol = 1e-2), f'fwd max diff: {(out_pt - out_tr).abs().max().item()}'

    for i, (pt_t, tr_t) in enumerate(zip(pt_tensors, tensors)):
        assert torch.allclose(pt_t.grad, tr_t.grad, atol = 1e-2), f'grad {i} max diff: {(pt_t.grad - tr_t.grad).abs().max().item()}'

@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'cuda required')
@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('softclamp_val', (None, 20.0))
@pytest.mark.parametrize('seq_len', (31, 64))
def test_poly_attention_e2e(causal, softclamp_val, seq_len):
    from poly_attention.poly_attention import Order2PolyAttention

    torch.manual_seed(42)
    dim = 64
    heads = 4
    x = torch.randn(2, seq_len, dim, requires_grad = True, device = 'cuda')
    x_pt = x.clone().detach().requires_grad_(True)

    module_pt = Order2PolyAttention(
        dim = dim, heads = heads, causal = causal, softclamp_value = softclamp_val, use_fused_kernel = False
    ).cuda()

    module_tr = Order2PolyAttention(
        dim = dim, heads = heads, causal = causal, softclamp_value = softclamp_val, use_fused_kernel = True
    ).cuda()

    module_tr.load_state_dict(module_pt.state_dict())

    out_pt = module_pt(x_pt)
    dout = torch.randn_like(out_pt)
    out_pt.backward(dout)

    out_tr = module_tr(x)
    out_tr.backward(dout)

    assert torch.allclose(out_pt, out_tr, atol = 1e-2), f'fwd max diff: {(out_pt - out_tr).abs().max().item()}'
    assert torch.allclose(x_pt.grad, x.grad, atol = 1e-2), f'grad max diff: {(x_pt.grad - x.grad).abs().max().item()}'
