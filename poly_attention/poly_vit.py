import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from poly_attention import PolyAttention
from poly_attention.n_poly_attention import NPolyAttention

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    assert divisible_by(dim, 4), "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = 'ij')

    y, x = (rearrange(t, 'h w -> (h w) 1') * omega for t in (y, x))

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = -1)
    return pe.type(dtype)

# classes

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, order = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            if order == 2:
                attn = PolyAttention(dim = dim, heads = heads, dim_head = dim_head, causal = False, prenorm = True)
            else:
                attn = NPolyAttention(dim = dim, order = order, heads = heads, dim_head = dim_head, causal = False, prenorm = True)

            ff = FeedForward(dim, mlp_dim)

            self.layers.append(ModuleList([attn, ff]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class PolyViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, order = 2):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, order = order)

        self.linear_head = nn.Linear(dim, num_classes, bias = False)

    def forward(self, img):
        device, dtype = img.device, img.dtype

        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(device, dtype = dtype)

        x = self.transformer(x)
        x = reduce(x, 'b n d -> b d', 'mean')

        return self.linear_head(x)
