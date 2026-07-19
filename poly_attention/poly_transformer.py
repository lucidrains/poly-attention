from __future__ import annotations

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, RMSNorm, ModuleList

from torch_einops_utils import pack_with_inverse
from poly_attention.poly_attention import Order2PolyAttention as PolyAttention

# helpers

def exists(val):
    return val is not None

# classes

class AttentionResidual(Module):
    def __init__(
        self,
        dim,
        num_pseudo_queries = 1
    ):
        super().__init__()
        self.norm_keys = RMSNorm(dim)
        self.scale = dim ** -0.5

        self.pseudo_queries = nn.Parameter(torch.randn(num_pseudo_queries, dim) * 0.02)

    def forward(
        self,
        hiddens: list[Tensor] | tuple[Tensor, ...]
    ):
        if isinstance(hiddens, (list, tuple)):
            hiddens = torch.stack(hiddens)

        hiddens, inverse_pack = pack_with_inverse(hiddens, 'l * d')

        # cross attention

        values = hiddens
        keys = self.norm_keys(values)

        sim = einsum('n d, l m d -> n m l', self.pseudo_queries, keys) * self.scale

        # attention and aggregate

        attn = sim.softmax(dim = -1)

        out = einsum('n m l, l m d -> n m d', attn, values)

        # unbind and unpack

        out = inverse_pack(out, 'n * d')
        out = out.unbind(dim = 0)

        return out[0] if len(out) == 1 else out

# feedforward

def FeedForward(
    dim,
    mult = 4
):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# main class

class PolyTransformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        causal = False,
        deep_cross_attention = False,
        norm_out = False
    ):
        super().__init__()
        self.depth = depth
        self.deep_cross_attention = deep_cross_attention
        self.layers = ModuleList([])
        self.norm_out = RMSNorm(dim) if norm_out else nn.Identity()

        for _ in range(depth):
            self.layers.append(ModuleList([
                RMSNorm(dim),
                ModuleList([RMSNorm(dim), RMSNorm(dim)]) if deep_cross_attention else None,
                PolyAttention(dim = dim, heads = heads, dim_head = dim_head, causal = causal),
                FeedForward(dim = dim)
            ]))

        num_pseudo_queries = 3 if deep_cross_attention else 1

        self.attn_residuals = ModuleList([
            AttentionResidual(dim = dim, num_pseudo_queries = 1 if ind == (depth - 1) else num_pseudo_queries)
            for ind in range(depth)
        ])

    def forward(
        self,
        x,
        mask = None,
        return_layer_hiddens = False
    ):
        hiddens = []
        context = None

        for (attn_norm, context_norms, attn, ff), attn_residual in zip(self.layers, self.attn_residuals):

            # attention

            res = x

            if exists(context):
                context = tuple(norm(c) for norm, c in zip(context_norms, context))
                attn_kwargs = dict(context = context)
            else:
                attn_kwargs = dict()

            x = attn(attn_norm(x), mask = mask, **attn_kwargs) + res

            # feedforward

            x = ff(x) + x

            hiddens.append(x)

            attn_res_out = attn_residual(hiddens)

            is_last = len(hiddens) == self.depth

            if self.deep_cross_attention and not is_last:
                x, *context = attn_res_out
                context = tuple(context)
            else:
                x = attn_res_out

        out = self.norm_out(x)

        if not return_layer_hiddens:
            return out

        return out, hiddens
