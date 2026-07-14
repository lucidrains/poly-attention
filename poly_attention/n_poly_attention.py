from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum, stack, Tensor
from torch.nn import Module, RMSNorm

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

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

def softclamp(x, c):
    return c * torch.tanh(x / c)

class NPolyAttention(Module):
    def __init__(
        self,
        dim,
        order = 3,
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
        eps = 1e-9
    ):
        super().__init__()
        assert order > 1, 'order must be greater than 1'
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.context_norms = None
        if separate_context_norms and prenorm:
            self.context_norms = nn.ModuleList([RMSNorm(dim) for _ in range(order)])

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
        self.split_kv = Rearrange('b n (split h d) -> split b h n d', split = 1 if shared_kv else 2, h = self.kv_heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        if self.is_gqa:
            self.num_rep = heads // kv_heads

        self.order = order

        self.to_q = LinearNoBias(dim, dim_inner)

        kv_mult = 1 if shared_kv else 2
        self.to_kvs = nn.ModuleList([LinearNoBias(dim, dim_inner_kv * kv_mult) for _ in range(order)])

        self.q_norms = nn.ModuleList([RMSNorm(dim_head) for _ in range(order + 1)])

        self.rotary_emb = RotaryEmbedding(dim_head) if use_rotary_embed else None

        self.to_out = nn.Linear(dim_inner, dim)

    def forward(
        self,
        x,
        context: Tensor | tuple[Tensor, ...] | None = None,
        mask = None,
        rotary_pos_emb = None
    ):
        device = x.device

        orig_x = x
        x = self.norm(x)

        q1 = self.split_q(self.to_q(x))

        # contexts

        has_context = exists(context)
        context = default(context, orig_x)
        context = cast_tuple(context, self.order)
        assert len(context) == self.order

        if exists(self.context_norms):
            context = tuple(norm(c) for c, norm in zip(context, self.context_norms))
        else:
            context = context if has_context else ((x,) * self.order)

        # kvs

        kvs = [self.split_kv(to_kv(c)) for c, to_kv in zip(context, self.to_kvs)]

        if self.shared_kv:
            q_rest = stack([kv[0] for kv in kvs])
            v_rest = q_rest
        else:
            q_rest, v_rest = map(stack, zip(*kvs))

        qs = (q1, *q_rest)
        vs = tuple(v_rest)

        qs = tuple(norm(q) for norm, q in zip(self.q_norms, qs))

        if exists(rotary_pos_emb):
            qs = tuple(apply_rotary_emb(rotary_pos_emb, q) for q in qs)

        if self.is_gqa:
            qs = (qs[0], *(repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in qs[1:]))
            vs = tuple(repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in vs)

        q_left = stack(qs[:-1])
        q_right = stack(qs[1:])

        if not exists(rotary_pos_emb) and exists(self.rotary_emb):
            q_left, q_right = self.rotary_emb.rotate_queries_with_cached_keys(q_left, q_right)

        # scores

        scores = einsum('... i d, ... j d -> ... i j', q_left, q_right) * self.scale
        if exists(self.softclamp_value):
            scores = softclamp(scores, self.softclamp_value)

        mask_value = -torch.finfo(scores.dtype).max

        # causal masking

        if self.causal:
            i, j = scores.shape[-2:]
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, mask_value)

        # padding masking

        if exists(mask):
            scores = einx.where('b j, c b h i j, -> c b h i j', mask, scores, mask_value)

        # aggregate from right to left

        root_value, *_, out = vs

        current_scores_k = scores[-1]
        scores12 = scores[0]

        for k in range(self.order - 1, 0, -1):
            lse_k = torch.logsumexp(current_scores_k, dim = -1)
            attn_k = current_scores_k.softmax(dim = -1)

            msg = einsum('b h j k, b h k d -> b h j d', attn_k, out)

            if k > 1:
                out = vs[k - 1] * msg
                current_scores_k = scores[k - 1] + rearrange(lse_k, 'b h j -> b h 1 j')
            else:
                out = msg
                scores12 = scores[0] + rearrange(lse_k, 'b h j -> b h 1 j')

        # final combine

        attn12 = scores12.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn12, out)

        # elementwise multiply root values

        if self.use_root_value_as_attn_gate:
            root_value = root_value.sigmoid()

        out = root_value * out

        # combine heads

        return self.to_out(self.merge_heads(out))
