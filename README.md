
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

tokens = torch.randn(1, 1024, 512)

out = attn(tokens) # (1, 1024, 512)
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

```bibtex
@misc{kayyam2026transformersneedprojectionssystematic,
    title   = {Do Transformers Need Three Projections? Systematic Study of QKV Variants},
    author  = {Ali Kayyam and Anusha Madan Gopal and M Anthony Lewis},
    year    = {2026},
    eprint  = {2606.04032},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2606.04032},
}
```
