# Chakrabarti et al. https://arxiv.org/abs/2602.02422

from __future__ import annotations
from functools import partial

import torch
from torch import nn, einsum, stack, cat, Tensor
from torch.nn import Module, RMSNorm

import einx
from einops import repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

try:
    from poly_attention.flash_poly_attention import flash_poly_attention
except ImportError:
    flash_poly_attention = None

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def softclamp(x, clamp_value):
    return clamp_value * torch.tanh(x / clamp_value)

# attention

def reference_poly_attention(
    q1, q2_pass1, q2_pass2, q3, v3,
    mask = None,
    softclamp_value = None,
    causal = False,
    cache = None
):
    scale = q1.shape[-1] ** -0.5
    has_cache = exists(cache)

    # pass 1 - q2 attends to q3

    sim23 = einsum('b h i d, b h j d -> b h i j', q2_pass1, q3) * scale

    if exists(softclamp_value):
        sim23 = softclamp(sim23, softclamp_value)

    mask_value = -torch.finfo(sim23.dtype).max

    if causal and not has_cache:
        i, j = sim23.shape[-2:]
        causal_mask = torch.ones((i, j), device = sim23.device, dtype = torch.bool).triu(1)
        sim23 = sim23.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim23 = einx.where('b j, b h i j, -> b h i j', mask, sim23, mask_value)

    # pass 1 attention and aggregation

    lse23_step = sim23.logsumexp(dim = -1)
    attn23 = sim23.softmax(dim = -1)
    msg_step = einsum('b h i j, b h j d -> b h i d', attn23, v3)

    if has_cache:
        _, _, _, clse23, cmsg = cache
        lse23 = cat((clse23, lse23_step), dim = -1)
        msg = cat((cmsg, msg_step), dim = -2)
    else:
        lse23 = lse23_step
        msg = msg_step

    # pass 2 - q1 attends to q2

    sim12 = einsum('b h i d, b h j d -> b h i j', q1, q2_pass2) * scale

    if exists(softclamp_value):
        sim12 = softclamp(sim12, softclamp_value)

    if causal and not has_cache:
        i, j = sim12.shape[-2:]
        causal_mask = torch.ones((i, j), device = sim12.device, dtype = torch.bool).triu(1)
        sim12 = sim12.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim12 = einx.where('b j, b h i j, -> b h i j', mask, sim12, mask_value)

    # add logsumexp from pass 1 as bias to pass 2

    sim12 = einx.add('b h j, b h i j -> b h i j', lse23, sim12)

    # pass 2 attention and aggregation

    attn12 = sim12.softmax(dim = -1)
    out = einsum('b h i j, b h j d -> b h i d', attn12, msg)

    return out, lse23_step, msg_step

# poly attention

class Order2PolyAttention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        kv_heads = None,
        dim_head = 64,
        causal = False,
        shared_kv = False,
        softclamp_value = None,
        use_rotary_embed = False,
        prenorm = False,
        separate_context_norms = False,
        use_root_value_as_attn_gate = True,
        eps = 1e-9,
        use_flash_kernel = None,
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.context_norms = None
        if separate_context_norms and prenorm:
            self.context_norms = nn.ModuleList([
                RMSNorm(dim),
                RMSNorm(dim)
            ])

        self.use_flash_kernel = default(use_flash_kernel, exists(flash_poly_attention))
        assert not (self.use_flash_kernel and not exists(flash_poly_attention)), 'fused poly attention is not available'

        self.maybe_softclamp = partial(softclamp, c = softclamp_value) if exists(softclamp_value) else nn.Identity()

        self.eps = eps
        self.scale = dim_head ** -0.5

        kv_heads = default(kv_heads, heads)
        assert divisible_by(heads, kv_heads), 'heads must be divisible by kv_heads'

        self.heads = heads
        self.kv_heads = kv_heads

        dim_inner = dim_head * heads
        dim_inner_kv = dim_head * kv_heads

        self.causal = causal
        self.shared_kv = shared_kv
        self.softclamp_value = softclamp_value
        self.use_root_value_as_attn_gate = use_root_value_as_attn_gate

        self.is_gqa = heads != kv_heads

        self.split_q = Rearrange('b n (h d) -> b h n d', h = self.heads)
        kv_split = 1 if self.shared_kv else 2
        self.split_kv = Rearrange('b n (split h d) -> split b h n d', split = kv_split, h = self.kv_heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        if self.is_gqa:
            self.num_rep = heads // kv_heads

        self.to_q = LinearNoBias(dim, dim_inner)

        kv_mult = 1 if shared_kv else 2
        self.to_kvs = nn.ModuleList([
            LinearNoBias(dim, dim_inner_kv * kv_mult),
            LinearNoBias(dim, dim_inner_kv * kv_mult)
        ])

        self.q1_norm = RMSNorm(dim_head)
        self.q2_norm = RMSNorm(dim_head)
        self.q3_norm = RMSNorm(dim_head)

        self.rotary_emb = RotaryEmbedding(dim_head) if use_rotary_embed else None

        self.to_out = nn.Linear(dim_inner, dim)

    def forward(
        self,
        x,
        context: Tensor | tuple[Tensor, ...] | None = None,
        mask = None,
        rotary_pos_emb = None,
        cache = None,
        return_cache = False
    ):
        seq_len, device = x.shape[-2], x.device

        has_cache = exists(cache)

        if has_cache:
            assert seq_len == 1, 'sequence length must be 1 when using kv cache'

        orig_x = x
        x = self.norm(x)

        q1 = self.split_q(self.to_q(x))

        # contexts

        has_context = exists(context)
        context = default(context, orig_x)
        context = cast_tuple(context, 2)
        assert len(context) == 2

        if exists(self.context_norms):
            ctx1, ctx2 = (norm(c) for c, norm in zip(context, self.context_norms))
        else:
            ctx1, ctx2 = context if has_context else (x, x)

        # kvs

        kv1, kv2 = [self.split_kv(to_kv(c)) for c, to_kv in zip((ctx1, ctx2), self.to_kvs)]

        if self.shared_kv:
            (q2,), (q3,) = kv1, kv2
            v2, v3 = q2, q3
        else:
            q2, v2 = kv1
            q3, v3 = kv2

        # qk rmsnorm

        q1 = self.q1_norm(q1)
        q2 = self.q2_norm(q2)
        q3 = self.q3_norm(q3)

        # rotary external

        if exists(rotary_pos_emb):
            q1, q2, q3 = [apply_rotary_emb(rotary_pos_emb, q) for q in (q1, q2, q3)]


        # handle cache

        if has_cache:
            cq2, cq3, cv3, clse23, cmsg = cache

            q2_cache = cat((cq2, q2), dim = -2)
            q3_cache = cat((cq3, q3), dim = -2)
            v3_cache = cat((cv3, v3), dim = -2)
        else:
            q2_cache, q3_cache, v3_cache = q2, q3, v3

        if self.is_gqa:

            q2, q3, v2, v3 = (repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in (q2, q3, v2, v3))

            q2_full, q3_full, v3_full = (repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in (q2_cache, q3_cache, v3_cache))
        else:
            q2_full, q3_full, v3_full = q2_cache, q3_cache, v3_cache

        q_left = stack((q1, q2))
        q_right = stack((q2_full, q3_full))

        # rotary within module

        if not exists(rotary_pos_emb) and exists(self.rotary_emb):
            q_left, q_right = self.rotary_emb.rotate_queries_with_cached_keys(q_left, q_right)

        q1, q2_left = q_left[0], q_left[1]
        q2_right, q3 = q_right[0], q_right[1]

        # try to dispatch to fused kernel


        can_use_flash = (
            self.use_flash_kernel and
            exists(flash_poly_attention) and
            not has_cache and
            not return_cache and
            q1.is_cuda
        )

        if can_use_flash:
            out = flash_poly_attention(
                q1, q2_right, q3, v3_full,
                mask = mask,
                softclamp_value = self.softclamp_value,
                is_causal = self.causal
            )
            lse23_step, msg_step = None, None
        else:
            out, lse23_step, msg_step = reference_poly_attention(
                q1, q2_left, q2_right, q3_full, v3_full,
                mask = mask,
                softclamp_value = self.softclamp_value,
                causal = self.causal,
                cache = cache
            )

        # elementwise multiply root values

        if self.use_root_value_as_attn_gate:
            v2 = v2.sigmoid()

        out = v2 * out

        # combine heads

        out = self.to_out(self.merge_heads(out))

        if not return_cache:
            return out

        lse23_full = cat((clse23, lse23_step), dim=-1) if has_cache else lse23_step
        msg_full = cat((cmsg, msg_step), dim=-2) if has_cache else msg_step

        new_cache = (q2_cache, q3_cache, v3_cache, lse23_full, msg_full)

        return out, new_cache
