from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum, stack, Tensor
from torch.nn import Module, RMSNorm

import einx
from einops import repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from torch_einops_utils import maybe, safe_cat

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
        multiply_root_value = False,
        use_root_value_as_attn_gate = False,
        attn_gate = False
    ):
        super().__init__()
        assert order > 1, 'order must be greater than 1'
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.context_norms = None
        if separate_context_norms and prenorm:
            self.context_norms = nn.ModuleList([RMSNorm(dim) for _ in range(order)])

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

        self.multiply_root_value = multiply_root_value
        self.use_root_value_as_attn_gate = use_root_value_as_attn_gate
        self.attn_gate = attn_gate

        self.is_gqa = heads != kv_heads

        assert not (use_root_value_as_attn_gate and (self.is_gqa or self.shared_kv)), 'cannot use root value as attention gate if using GQA or shared KV'

        q_split = 2 if attn_gate else 1
        self.split_q = Rearrange('b n (split h d) -> split b h n d', split = q_split, h = self.heads)

        self.has_root_v = multiply_root_value and not shared_kv
        kv1_split = 2 if self.has_root_v else 1
        self.split_kv1 = Rearrange('b n (split h d) -> split b h n d', split = kv1_split, h = self.kv_heads)

        kv_split = 1 if shared_kv else 2
        self.split_kv = Rearrange('b n (split h d) -> split b h n d', split = kv_split, h = self.kv_heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        if self.is_gqa:
            self.num_rep = heads // kv_heads

        self.order = order

        self.to_q = LinearNoBias(dim, dim_inner * q_split)

        kv_mult = 1 if shared_kv else 2
        self.to_kvs = nn.ModuleList([
            LinearNoBias(dim, dim_inner_kv * kv1_split),
            *[LinearNoBias(dim, dim_inner_kv * kv_mult) for _ in range(order - 1)]
        ])

        self.q_norms = nn.ModuleList([RMSNorm(dim_head) for _ in range(order + 1)])

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
        device = x.device
        seq_len = x.shape[-2]

        has_cache = exists(cache)
        if has_cache:
            assert seq_len == 1, 'sequence length must be 1 when using kv cache'

        orig_x = x
        x = self.norm(x)

        q_and_maybe_gates = self.split_q(self.to_q(x))

        if self.attn_gate:
            q1, gates = q_and_maybe_gates
        else:
            q1 = q_and_maybe_gates[0]

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
        kv1 = self.split_kv1(self.to_kvs[0](context[0]))
        kvs_rest = [self.split_kv(to_kv(c)) for c, to_kv in zip(context[1:], self.to_kvs[1:])]

        if self.shared_kv:
            qs_rest = (kv1[0], *[kv[0] for kv in kvs_rest])
            root_value = qs_rest[0]
            vs_for_aggregation = qs_rest[1:]
        else:
            qs_rest = (kv1[0], *[kv[0] for kv in kvs_rest])
            vs_for_aggregation = tuple(kv[1] for kv in kvs_rest)
            root_value = kv1[1] if self.has_root_v else None

        qs = (q1, *qs_rest)

        qs = tuple(norm(q) for norm, q in zip(self.q_norms, qs))

        if exists(rotary_pos_emb):
            qs = tuple(apply_rotary_emb(rotary_pos_emb, q) for q in qs)

        q1 = qs[0]
        qs_rest = qs[1:]

        if has_cache:
            c_qs, c_vs, c_lses, c_msgs = cache
        else:
            c_qs = (None,) * self.order
            c_vs = (None,) * (self.order - 1)
            c_lses = (None,) * (self.order - 1)
            c_msgs = (None,) * (self.order - 1)

        # Connect history
        qs_rest_cache = tuple(safe_cat((cq, q), dim = -2) for cq, q in zip(c_qs, qs_rest))
        vs_cache = tuple(safe_cat((cv, v), dim = -2) for cv, v in zip(c_vs, vs_for_aggregation))

        if self.is_gqa:
            match_kv = maybe(lambda t: repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep))
            qs_rest_left = tuple(match_kv(t) for t in qs_rest)
            qs_rest_right = tuple(match_kv(t) for t in qs_rest_cache)
            vs_full = tuple(match_kv(t) for t in vs_cache)
            if exists(root_value):
                root_value = match_kv(root_value)
        else:
            qs_rest_left = qs_rest
            qs_rest_right = qs_rest_cache
            vs_full = vs_cache

        q_left = stack((q1, *qs_rest_left[:-1]))
        q_right = stack(qs_rest_right)

        if not exists(rotary_pos_emb) and exists(self.rotary_emb):
            q_left, q_right = self.rotary_emb.rotate_queries_with_cached_keys(q_left, q_right)

        scores = einsum('... i d, ... j d -> ... i j', q_left, q_right) * self.scale

        if exists(self.softclamp_value):
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

        # aggregate from right to left
        out = vs_full[-1]
        current_scores_k = scores[-1]

        new_cache_lses = []
        new_cache_msgs = []

        for k in range(self.order - 1, 1, -1):
            lse_k_step = torch.logsumexp(current_scores_k, dim = -1)
            attn_k = current_scores_k.softmax(dim = -1)

            msg_step = einsum('b h i j, b h j d -> b h i d', attn_k, out)

            idx = (self.order - 1) - k
            clse = c_lses[idx]
            cmsg = c_msgs[idx]

            lse_k_full = safe_cat((clse, lse_k_step), dim = -1)
            msg_full = safe_cat((cmsg, msg_step), dim = -2)

            new_cache_msgs.append(msg_full)
            new_cache_lses.append(lse_k_full)

            out = vs_full[k - 2] * msg_full
            current_scores_k = einx.add('b h j, b h i j -> b h i j', lse_k_full, scores[k - 1])

        # final step (k = 1)
        lse_1_step = torch.logsumexp(current_scores_k, dim = -1)
        attn_1 = current_scores_k.softmax(dim = -1)

        msg_1_step = einsum('b h i j, b h j d -> b h i d', attn_1, out)

        idx = self.order - 2
        clse = c_lses[idx]
        cmsg = c_msgs[idx]

        lse_1_full = safe_cat((clse, lse_1_step), dim = -1)
        msg_1_full = safe_cat((cmsg, msg_1_step), dim = -2)

        new_cache_msgs.append(msg_1_full)
        new_cache_lses.append(lse_1_full)

        scores12 = einx.add('b h j, b h i j -> b h i j', lse_1_full, scores[0])

        # final combine
        attn12 = scores12.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn12, msg_1_full)

        # elementwise multiply root values
        if self.multiply_root_value:
            if self.use_root_value_as_attn_gate:
                root_value = root_value.sigmoid()

            out = root_value * out

        # attention gate
        if self.attn_gate:
            out = out * gates.sigmoid()

        # combine heads
        out = self.to_out(self.merge_heads(out))

        if not return_cache:
            return out

        new_c_lses = tuple(new_cache_lses)
        new_c_msgs = tuple(new_cache_msgs)

        new_cache = (qs_rest_cache, vs_cache, new_c_lses, new_c_msgs)

        return out, new_cache
