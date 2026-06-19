from __future__ import annotations

import torch
from torch import nn, einsum, stack
from torch.nn import Module, RMSNorm
import torch.nn.functional as F

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from functools import partial

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
        softclamp_value = 20.,
        use_rotary_embed = False,
        prenorm = False,
        eps = 1e-9
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.eps = eps
        self.scale = dim_head ** -0.5

        kv_heads = default(kv_heads, heads)
        assert divisible_by(heads, kv_heads), 'heads must be divisible by kv_heads'

        self.heads = heads
        self.kv_heads = kv_heads

        dim_inner = dim_head * heads
        dim_inner_kv = dim_head * kv_heads

        self.causal = causal
        self.softclamp_value = softclamp_value

        self.is_gqa = heads != kv_heads

        self.split_q_gates = Rearrange('b n (split h d) -> split b h n d', split = 2, h = self.heads)
        self.split_kv = Rearrange('b n (split h d) -> split b h n d', split = order * 2, h = self.kv_heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        if self.is_gqa:
            self.num_rep = heads // kv_heads

        self.order = order

        self.to_q_gates = LinearNoBias(dim, dim_inner * 2)
        self.to_kv = LinearNoBias(dim, dim_inner_kv * (order * 2))

        self.q_norms = nn.ModuleList([RMSNorm(dim_head) for _ in range(order + 1)])

        self.rotary_emb = RotaryEmbedding(dim_head) if use_rotary_embed else None

        self.to_out = nn.Linear(dim_inner, dim)

    def forward(
        self,
        x,
        mask = None,
        rotary_pos_emb = None
    ):
        device = x.device

        x = self.norm(x)

        q1, gates = self.split_q_gates(self.to_q_gates(x))
        kv_chunks = self.split_kv(self.to_kv(x))
        q_rest, v_rest = kv_chunks.chunk(2, dim = 0)

        qs = [q1, *q_rest]
        vs = [*v_rest]

        qs = [norm(q) for norm, q in zip(self.q_norms, qs)]

        if exists(rotary_pos_emb):
            qs = [apply_rotary_emb(rotary_pos_emb, q) for q in qs]

        if self.is_gqa:
            qs[1:] = [repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in qs[1:]]
            vs = [repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in vs]

        q_left = stack(qs[:-1])
        q_right = stack(qs[1:])

        if not exists(rotary_pos_emb) and exists(self.rotary_emb):
            q_left, q_right = self.rotary_emb.rotate_queries_with_cached_keys(q_left, q_right)

        # scores

        scores = einsum('... i d, ... j d -> ... i j', q_left, q_right) * self.scale
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

        v_bar = vs[-1]

        current_scores_k = scores[-1]
        scores12 = scores[0]

        for k in range(self.order - 1, 0, -1):
            lse_k = torch.logsumexp(current_scores_k, dim = -1)
            attn_k = current_scores_k.softmax(dim = -1)

            msg = einsum('b h j k, b h k d -> b h j d', attn_k, v_bar)

            if k > 1:
                v_bar = vs[k - 1] * msg
                current_scores_k = scores[k - 1] + rearrange(lse_k, 'b h j -> b h 1 j')
            else:
                v_bar = msg
                scores12 = scores[0] + rearrange(lse_k, 'b h j -> b h 1 j')

        # final combine

        attn12 = scores12.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn12, v_bar)

        # elementwise multiply root values

        if self.order > 1:
            v2 = vs[0]
            out = v2 * out

        # gate

        out = out * gates.sigmoid()

        # combine heads

        return self.to_out(self.merge_heads(out))
