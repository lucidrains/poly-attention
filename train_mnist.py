# /// script
# dependencies = [
#   "accelerate",
#   "torchvision",
#   "wandb",
#   "einx",
#   "einops",
#   "rotary-embedding-torch",
#   "fire"
# ]
# ///

import fire
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

import torchvision.transforms as T
from torchvision.datasets import MNIST

import wandb
from accelerate import Accelerator

from poly_attention.poly_vit import PolyViT

# helpers

def divisible_by(num, den):
    return (num % den) == 0

# main

def train(
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    epochs: int = 10,
    dim: int = 128,
    depth: int = 6,
    heads: int = 8,
    dim_head: int = 64,
    mlp_dim: int = 512,
    patch_size: int = 7,
    image_size: int = 28,
    num_classes: int = 10,
    channels: int = 1,
    order: int = 2,
    project_name: str = 'poly-vit',
    run_name: str = 'baseline',
    refresh_every: int = 50,
    track_experiment_online: bool = False,
):
    # data

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    dataset = MNIST(
        root = 'data',
        download = True,
        train = True,
        transform = transform
    )

    val_dataset = MNIST(
        root = 'data',
        download = True,
        train = False,
        transform = transform
    )

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # model

    vit = PolyViT(
        dim = dim,
        num_classes = num_classes,
        image_size = image_size,
        patch_size = patch_size,
        depth = depth,
        heads = heads,
        dim_head = dim_head,
        mlp_dim = mlp_dim,
        channels = channels,
        order = order,
    )

    # optim

    optim = Adam(vit.parameters(), lr = learning_rate)

    # prepare

    accelerator = Accelerator()

    vit, optim, dataloader, val_dataloader = accelerator.prepare(vit, optim, dataloader, val_dataloader)

    # experiment

    wandb.init(
        project = project_name,
        mode = 'disabled' if not track_experiment_online else 'online'
    )

    wandb.run.name = run_name

    # loop

    val_acc = 0.0

    for epoch in range(epochs):
        vit.train()

        pbar = tqdm(dataloader)

        for images, labels in pbar:
            pbar.set_description(f'epoch {epoch}')

            logits = vit(images)
            loss = F.cross_entropy(logits, labels)

            wandb.log(dict(loss = loss.item()))

            pbar.set_postfix(loss = f'{loss.item():.3f}', val_acc = f'{val_acc:.4f}')

            accelerator.backward(loss)
            optim.step()
            optim.zero_grad()

        # validation

        vit.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_dataloader:
                batch = labels.shape[0]

                logits = vit(images)
                preds = logits.argmax(dim = -1)

                correct = correct + (preds == labels).sum().item()
                total = total + batch

        val_acc = correct / total

        wandb.log(dict(val_acc = val_acc))

if __name__ == '__main__':
    fire.Fire(train)
