
## Poly Attention

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

A Vision Transformer based on Poly-Attention

```python
import torch
from poly_attention import PolyViT

vit = PolyViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    order = 2 # standard poly attention order 2
)

images = torch.randn(1, 3, 256, 256)

preds = vit(images) # (1, 1000)
```

## Quick test

```bash
python train_function_composition.py --poly_layers=1 --base_layers=2
```

## Appreciation

- [@dillfrescott](https://github.com/dillfrescott) for submitting a stability fix

- [@pranoyr](https://github.com/pranoyr) for adding key value caching for n-order poly attention as well as more efficient GQA caching for poly attention!

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

```bibtex
@misc{kimiteam2026attentionresiduals,
    title   = {Attention Residuals},
    author  = {Kimi Team and Guangyu Chen and Yu Zhang and Jianlin Su and Weixin Xu and Siyuan Pan and Yaoyu Wang and Yucheng Wang and Guanduo Chen and Bohong Yin and Yutian Chen and Junjie Yan and Ming Wei and Y. Zhang and Fanqing Meng and Chao Hong and Xiaotong Xie and Shaowei Liu and Enzhe Lu and Yunpeng Tai and Yanru Chen and Xin Men and Haiqing Guo and Y. Charles and Haoyu Lu and Lin Sui and Jinguo Zhu and Zaida Zhou and Weiran He and Weixiao Huang and Xinran Xu and Yuzhi Wang and Guokun Lai and Yulun Du and Yuxin Wu and Zhilin Yang and Xinyu Zhou},
    year    = {2026},
    eprint  = {2603.15031},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2603.15031},
}
```

```bibtex
@misc{heddes2025deepcrossattentionsuperchargingtransformerresidual,
    title   = {DeepCrossAttention: Supercharging Transformer Residual Connections},
    author  = {Mike Heddes and Adel Javanmard and Kyriakos Axiotis and Gang Fu and MohammadHossein Bateni and Vahab Mirrokni},
    year    = {2025},
    eprint  = {2502.06785},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2502.06785},
}
```
