# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "wandb",
#     "fire",
#     "accelerate",
#     "rotary-embedding-torch",
#     "x-transformers"
# ]
# ///

from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList, RMSNorm
import torch.nn.functional as F
from torch.optim import Adam

import wandb
import fire
from einops import rearrange
from accelerate import Accelerator

from poly_attention import PolyAttention, NPolyAttention
from x_transformers import Attention, FeedForward

# helpers

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

# data

def function_composition(
    seq_len,
    batch_size = 32,
    num_classes = 10,
    x = None,
    composition_depth = 2,
    device = 'cpu'
):
    inputs = torch.zeros((batch_size, seq_len, 2), dtype = torch.long, device = device)

    x_vals = torch.full((batch_size,), x, device = device) if exists(x) else torch.randint(0, num_classes, (batch_size,), device = device)
    funcs = torch.randint(0, num_classes, (batch_size, composition_depth, num_classes), device = device)

    targets = x_vals.clone()
    for step in range(composition_depth):
        targets = funcs[torch.arange(batch_size, device = device), step, targets]

    total_pos = min(composition_depth * num_classes, seq_len)

    if total_pos > 0:
        step_idx = torch.arange(total_pos, device = device) // num_classes
        pos_idx = torch.arange(total_pos, device = device) % num_classes

        inputs[:, :total_pos, 0] = funcs[torch.arange(batch_size, device = device)[:, None], step_idx, pos_idx]
        inputs[:, :total_pos, 1] = pos_idx

    if seq_len > 0:
        inputs[:, -1, 0] = x_vals
        inputs[:, -1, 1] = seq_len - 1

    return inputs, targets

# models

class Block(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_poly = False, order = 2, shared_kv = False):
        super().__init__()

        if use_poly:
            if order == 2:
                self.attn = PolyAttention(dim, heads = heads, dim_head = dim_head, shared_kv = shared_kv)
            else:
                self.attn = NPolyAttention(dim, order = order, heads = heads, dim_head = dim_head)
        else:
            self.attn = Attention(dim, heads = heads, dim_head = dim_head, qk_norm = True)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Model(Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        dim = 128,
        heads = 4,
        dim_head = 32,
        layers = 6,
        use_poly = False,
        order = 2,
        shared_kv = False
    ):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, dim)
        self.pos_enc = nn.Embedding(seq_len, dim)

        self.blocks = ModuleList([
            Block(dim, heads = heads, dim_head = dim_head, use_poly = use_poly, order = order, shared_kv = shared_kv)
            for _ in range(layers)
        ])

        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x_one_hot):
        pos = torch.arange(x_one_hot.shape[1], device = x_one_hot.device)
        x = self.embedding(x_one_hot) + self.pos_enc(pos)

        for block in self.blocks:
            x = block(x)

        return self.to_logits(x)

# training

def train_model(
    model_name: str,
    use_poly: bool,
    accelerator: Accelerator,
    layers: int,
    dim: int,
    heads: int,
    dim_head: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_classes: int = 10,
    composition_depth: int = 2,
    seed: int = 42,
    order: int = 2,
    shared_kv: bool = False
):
    torch.manual_seed(seed)

    seq_len = composition_depth * num_classes + 1
    vocab_size = 3 + num_classes + num_classes

    if accelerator.is_main_process:
        wandb.init(
            project = "poly-attention-toy-task",
            name = f"{model_name}-Order{order}-{layers}Layers" if use_poly else f"{model_name}-{layers}Layers",
            config = dict(
                model_type = model_name,
                use_poly = use_poly,
                epochs = epochs,
                batch_size = batch_size,
                learning_rate = lr,
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                layers = layers,
                order = order,
            )
        )

    model = Model(
        vocab_size = vocab_size,
        seq_len = seq_len,
        dim = dim,
        heads = heads,
        dim_head = dim_head,
        layers = layers,
        use_poly = use_poly,
        order = order,
        shared_kv = shared_kv
    )

    optimizer = Adam(model.parameters(), lr = lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    accelerator.print(f"Testing {model_name} with {layers} layers...")

    best_acc = 0.0
    for epoch in range(epochs):
        inputs, targets = function_composition(
            seq_len,
            batch_size = batch_size,
            num_classes = num_classes,
            composition_depth = composition_depth,
            device = accelerator.device
        )

        # one hot encode and add positional indicators

        b_in = rearrange(F.one_hot(inputs, num_classes = num_classes), 'b n f c -> b n (f c)').float()

        b_in = F.pad(b_in, (3, 0))
        b_in[:, :num_classes, 0] = 1.
        b_in[:, num_classes:-1, 1] = 1.
        b_in[:, -1, 2] = 1.

        # train step

        optimizer.zero_grad()

        logits = model(b_in)
        loss = F.cross_entropy(logits[:, -1], targets)

        accelerator.backward(loss)
        optimizer.step()

        # metrics

        acc = (logits[:, -1].argmax(dim = -1) == targets).float().mean().item()
        best_acc = max(best_acc, acc)

        if accelerator.is_main_process:
            wandb.log(dict(
                epoch = epoch + 1,
                train_loss = loss.item(),
                train_accuracy = acc,
            ))

            if divisible_by(epoch + 1, 10):
                accelerator.print(f"  Epoch {epoch+1:04d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

        if acc >= 1.0:
            accelerator.print(f"Accuracy reached 1.0 at epoch {epoch + 1}, stopping early.")
            break

    if accelerator.is_main_process:
        wandb.finish()

    return best_acc

def main(
    layers: int = 2,
    dim: int = 128,
    heads: int = 4,
    dim_head: int = 32,
    epochs: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_classes: int = 10,
    composition_depth: int = 2,
    seed: int = 42,
    order: int = 2,
    shared_kv: bool = False
):
    accelerator = Accelerator()

    accelerator.print(f"Running on {accelerator.device} with {layers} layers.")

    train_fn = partial(
        train_model,
        accelerator = accelerator,
        layers = layers,
        dim = dim,
        heads = heads,
        dim_head = dim_head,
        epochs = epochs,
        batch_size = batch_size,
        lr = lr,
        num_classes = num_classes,
        composition_depth = composition_depth,
        seed = seed,
        order = order
    )

    acc_poly = train_fn("PolyAttention", use_poly = True, shared_kv = shared_kv)
    acc_base = train_fn("BaseSelfAttention", use_poly = False)

    accelerator.print(f"Final Best Accuracies ({layers} Layers, Order {order}) -> Base: {acc_base:.4f} | Poly: {acc_poly:.4f}")

if __name__ == '__main__':
    fire.Fire(main)
