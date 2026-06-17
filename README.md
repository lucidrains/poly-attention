
## Poly Attention (wip)

Implementation of <a href="https://arxiv.org/abs/2602.02422">Poly-Attention</a>, a general scheme for higher-order self-attention

## Install

```bash
$ pip install poly-attention
```

## Usage

```python
import torch
from poly_attention import PolyAttention

attn = PolyAttention(
    dim = 512,
    heads = 8,
    dim_head = 64,
    causal = False
)

x = torch.randn(1, 1024, 512)
out = attn(x) # (1, 1024, 512)
```

## Appreciation

- [@dillfrescott](https://github.com/dillfrescott) for submitting a stability fix

## Citations

```bibtex
@inproceedings{chakrabarti2026poly,
    title   = {Poly-attention: a general scheme for higher-order self-attention},
    author  = {Chakrabarti, Sayak and Pitassi, Toniann and Alman, Josh},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year    = {2026}
}
```
