import torch

from poly_attention.flash_poly_attention import flash_poly_attention
from poly_attention.poly_attention import reference_poly_attention

# helper

def peak_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

# main

def verify(seq_len):
    print(f"\nseq_len: {seq_len}")

    tensors = [torch.randn(1, 4, seq_len, 64, device='cuda', dtype=torch.float16, requires_grad=True) for _ in range(4)]

    # fused

    torch.cuda.reset_peak_memory_stats()

    out = flash_poly_attention(*tensors, is_causal=True)
    out.backward(torch.randn_like(out))

    print(f"fused: {peak_memory_mb():.1f} MB")

    # reference

    torch.cuda.reset_peak_memory_stats()

    try:
        q1, q2, q3, v3 = tensors
        out, _, _ = reference_poly_attention(q1, q2, q2, q3, v3, causal=True)
        out.backward(torch.randn_like(out))

        print(f"ref:   {peak_memory_mb():.1f} MB")
    except RuntimeError:
        print("ref:   OOM")

if __name__ == '__main__':
    for seq_len in (8192, 16384, 32768, 65536):
        verify(seq_len)
