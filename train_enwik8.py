from __future__ import annotations

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "tqdm",
#     "fire",
#     "accelerate",
#     "x-transformers",
#     "rotary-embedding-torch",
#     "einops"
# ]
# ///

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import math
import gzip
import random
import tqdm
import numpy as np
from collections import namedtuple

import torch
import fire
from torch import nn
from torch.nn import Module, ModuleList, RMSNorm
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange

from poly_attention import PolyAttention, NPolyAttention
from x_transformers import FeedForward

# helpers

Cache = namedtuple('Cache', ['layer_caches', 'seq_len'])

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None
    curr_token = out

    for _ in range(sample_num_times):
        logits, cache = net(curr_token, return_loss = False, cache = cache, return_cache = True)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)
        curr_token = sample

    return out[..., prompt_seq_len:]

# model

class Block(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, order = 2, use_flash_kernel = False):
        super().__init__()
        if order == 2:
            self.attn = PolyAttention(dim, heads = heads, dim_head = dim_head, causal = True, use_flash_kernel = use_flash_kernel)
        else:
            self.attn = NPolyAttention(dim, order = order, heads = heads, dim_head = dim_head, causal = True)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x, rotary_pos_emb = None, cache = None):
        attn_out, new_cache = self.attn(self.norm1(x), rotary_pos_emb = rotary_pos_emb, cache = cache, return_cache = True)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache

class PolyLM(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        seq_len,
        heads = 8,
        dim_head = 64,
        order = 2,
        use_flash_kernel = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.layers = ModuleList([Block(dim, heads=heads, dim_head=dim_head, order=order, use_flash_kernel=use_flash_kernel) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(self, x, return_loss = False, cache = None, return_cache = False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        seq_len, device = x.shape[1], x.device

        x = self.token_emb(x)

        past_seq_len = 0
        if exists(cache):
            cache, past_seq_len = cache.layer_caches, cache.seq_len

        pos = torch.arange(seq_len, device = device) + past_seq_len
        rotary_pos_emb = self.rotary_emb(pos)

        new_caches = []
        cache = default(cache, [None] * len(self.layers))

        for layer, layer_cache in zip(self.layers, cache):
            x, new_layer_cache = layer(x, rotary_pos_emb = rotary_pos_emb, cache = layer_cache)
            new_caches.append(new_layer_cache)

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            return (logits, Cache(new_caches, past_seq_len + seq_len)) if return_cache else logits

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
        return loss

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

# main

def main(
    num_batches: int = 10_000,
    batch_size: int = 4,
    grad_accum_every: int = 4,
    learning_rate: float = 1e-4,
    validate_every: int = 100,
    generate_every: int = 500,
    seq_len: int = 256,
    prime_length: int | None = None,
    generate_length: int | None = None,
    dim: int = 512,
    depth: int = 6,
    heads: int = 8,
    dim_head: int = 64,
    order: int = 2,
    use_flash_kernel: bool = False
):
    generate_length = default(generate_length, seq_len)
    prime_length = default(prime_length, int(generate_length * 0.25))

    # accelerators

    accelerator = Accelerator()

    model = PolyLM(
        num_tokens = 256,
        dim = dim,
        depth = depth,
        seq_len = seq_len,
        heads = heads,
        dim_head = dim_head,
        order = order,
        use_flash_kernel = use_flash_kernel
    )

    # prepare enwik8 data

    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    pbar = tqdm.tqdm(range(num_batches + 1), mininterval = 1.0, desc = "training", ncols=120)
    for i in pbar:
        model.train()

        for _ in range(grad_accum_every):
            data = next(train_loader)

            loss = model(data, return_loss = True)

            accelerator.backward(loss / grad_accum_every)

        pbar.set_postfix(loss = loss.item())

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)

                valid_loss = model(valid_data, return_loss = True)
                pbar.set_postfix(loss = loss.item(), val_loss = valid_loss.item())

        if i > 0 and divisible_by(i, generate_every):
            model.eval()

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp)
            accelerator.print(f"INPUT: {prime}")

            prompt = inp[None, ...]

            with torch.no_grad():
                sampled = base_decoding(model, prompt, generate_length)

            base_decode_output = decode_tokens(sampled[0])

            accelerator.print(f"\nOUTPUT: {base_decode_output}")

    accelerator.print("Training finished.")

if __name__ == '__main__':
    fire.Fire(main)
