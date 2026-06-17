# Chakrabarti et al. https://arxiv.org/abs/2602.02422

from functools import partial

import torch
from torch import nn, einsum, stack, cat
from torch.nn import Module, RMSNorm
import torch.nn.functional as F

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rotary_embedding_torch import apply_rotary_emb

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
        softclamp_value = 20.,
        eps = 1e-9
    ):
        super().__init__()
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

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.is_gqa = heads != kv_heads

        if self.is_gqa:
            self.num_rep = heads // kv_heads

        self.to_q_gates = LinearNoBias(dim, dim_inner * 2)

        kv_mult = 2 if shared_kv else 4
        self.to_kv = LinearNoBias(dim, dim_inner_kv * kv_mult)

        self.q1_norm = RMSNorm(dim_head)
        self.q2_norm = RMSNorm(dim_head)
        self.q3_norm = RMSNorm(dim_head)

        self.to_out = nn.Linear(dim_inner, dim)

    def forward(self, x, mask = None, rotary_pos_emb = None, cache = None, return_cache = False):
        seq_len, device = x.shape[-2], x.device

        has_cache = exists(cache)

        if has_cache:
            assert seq_len == 1, 'sequence length must be 1 when using kv cache'

        q1, gates = [self.split_heads(t) for t in self.to_q_gates(x).chunk(2, dim = -1)]

        if self.shared_kv:
            q2, q3 = [self.split_heads(t) for t in self.to_kv(x).chunk(2, dim = -1)]
            v2, v3 = q2, q3
        else:
            q2, q3, v2, v3 = [self.split_heads(t) for t in self.to_kv(x).chunk(4, dim = -1)]

        q1 = self.q1_norm(q1)
        q2 = self.q2_norm(q2)
        q3 = self.q3_norm(q3)

        if exists(rotary_pos_emb):
            q1, q2, q3 = [apply_rotary_emb(rotary_pos_emb, q) for q in (q1, q2, q3)]

        if self.is_gqa:
            q2, q3, v2, v3 = (repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in (q2, q3, v2, v3))

        if has_cache:
            cq2, cq3, cv3, clse23, cmsg = cache
            q2_full = cat((cq2, q2), dim = -2)
            q3_full = cat((cq3, q3), dim = -2)
            v3_full = cat((cv3, v3), dim = -2)
        else:
            q2_full, q3_full, v3_full = q2, q3, v3

        q_left = stack((q1, q2))
        q_right = stack((q2_full, q3_full))

        scores = einsum('... i d, ... j d -> ... i j', q_left, q_right) * self.scale
        scores = softclamp(scores, self.softclamp_value)

        mask_value = -torch.finfo(scores.dtype).max

        # causal masking

        if self.causal and not has_cache:
            i, j = scores.shape[-2:]
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask, mask_value)

        # padding masking

        if exists(mask):
            scores = einx.where('b j, c b h i j, -> c b h i j', mask, scores, mask_value)

        scores12, scores23 = scores

        # aggregate

        lse23_step = torch.logsumexp(scores23, dim = -1)
        attn23 = scores23.softmax(dim = -1)

        msg_step = einsum('b h i j, b h j d -> b h i d', attn23, v3_full)

        if has_cache:
            lse23 = cat((clse23, lse23_step), dim = -1)
            msg = cat((cmsg, msg_step), dim = -2)
        else:
            lse23, msg = lse23_step, msg_step

        scores12 = scores12 + rearrange(lse23, 'b h j -> b h 1 j')

        attn12 = scores12.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn12, msg)

        # elementwise multiply root values

        out = v2 * out

        # gate

        out = out * gates.sigmoid()

        # combine heads

        out = self.to_out(self.merge_heads(out))

        if not return_cache:
            return out

        new_cache = (q2_full, q3_full, v3_full, lse23, msg)
        return out, new_cache
