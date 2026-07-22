import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

# helpers

def exists(val):
    return val is not None

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX_Q', 'N_CTX_K', 'BLOCK_DMODEL']
)
@triton.jit
def _poly_flash_fwd_kernel(
    Q, K, V, sm_scale,
    Bias, Mask,
    Out, Lse,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_bz, stride_bh, stride_bn,
    stride_mz, stride_mn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SOFTCLAMP_VALUE: tl.constexpr,
):
    # get program ids

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    if HAS_BIAS:
        b_offset = off_z * stride_bz + off_h * stride_bh
    else:
        b_offset = 0

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize block pointers

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    # initialize stats

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load query

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)

    if IS_CAUSAL:
        end_n = tl.minimum((start_m + 1) * BLOCK_M, N_CTX_K)
    else:
        end_n = N_CTX_K

    # loop over key/value blocks

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        # load keys

        k = tl.load(k_ptrs, mask=offs_n_curr[None, :] < N_CTX_K, other=0.0)

        # compute attention scores

        sim = (tl.dot(q, k, allow_tf32=False) * sm_scale).to(tl.float32)

        # softclamp if needed

        if SOFTCLAMP_VALUE is not None:
            sim = SOFTCLAMP_VALUE * libdevice.tanh(sim / SOFTCLAMP_VALUE)

        # apply bias

        if HAS_BIAS:
            bias_ptrs = Bias + b_offset + offs_n_curr * stride_bn
            bias = tl.load(bias_ptrs, mask=offs_n_curr < N_CTX_K, other=0.0)
            sim += bias[None, :]

        # mask causal and sequence lengths

        if HAS_MASK:
            mask_ptrs = Mask + off_z * stride_mz + offs_n_curr * stride_mn
            mask_val = tl.load(mask_ptrs, mask=offs_n_curr < N_CTX_K, other=0.0)
            sim = tl.where(mask_val[None, :], sim, float("-inf"))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            sim = tl.where(causal_mask, sim, float("-inf"))

        sim = tl.where(offs_n_curr[None, :] < N_CTX_K, sim, float("-inf"))

        # stable softmax math

        m_ij = tl.maximum(m_i, tl.max(sim, 1))
        p = tl.math.exp(sim - m_ij[:, None])

        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        # load values and accumulate

        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX_K, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v, allow_tf32=False)

        m_i = m_ij

        # advance pointers

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back output and logsumexp

    acc = acc / l_i[:, None]

    out_offset = off_z * stride_oz + off_h * stride_oh
    out_ptrs = Out + out_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(out_ptrs, acc.to(q.dtype), mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL))

    l_ptrs = Lse + off_hz * N_CTX_Q + offs_m
    tl.store(l_ptrs, m_i + tl.math.log(l_i), mask=offs_m < N_CTX_Q)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
    ],
    key=['N_CTX_Q', 'N_CTX_K', 'BLOCK_DMODEL'],
    reset_to_zero=['DQ']
)
@triton.jit
def _poly_flash_bwd_kernel(
    Q, K, V, sm_scale,
    Out, DO, Lse, DLse,
    Bias, DBias, Mask,
    DQ, DK, DV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_bz, stride_bh, stride_bn,
    stride_mz, stride_mn,
    Z, H, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DLSE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SOFTCLAMP_VALUE: tl.constexpr,
):
    # get program ids

    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    if HAS_BIAS:
        b_offset = off_z * stride_bz + off_h * stride_bh
    else:
        b_offset = 0

    # block pointers for k and v

    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    dv_ptrs = DV + v_offset + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    dk_ptrs = DK + k_offset + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk

    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0.0)

    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dbias = tl.zeros([BLOCK_N], dtype=tl.float32)

    if IS_CAUSAL:
        start_m = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        start_m = 0

    offs_m_init = start_m + tl.arange(0, BLOCK_M)
    q_ptrs = Q + q_offset + offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk
    do_ptrs = DO + q_offset + offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk
    out_ptrs = Out + q_offset + offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk
    dq_ptrs = DQ + q_offset + offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk
    lse_ptrs = Lse + off_hz * N_CTX_Q + offs_m_init
    if HAS_DLSE:
        dlse_ptrs = DLse + off_hz * N_CTX_Q + offs_m_init

    # loop over queries

    for m in range(start_m, N_CTX_Q, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)

        # load chunk of queries and intermediates

        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
        out = tl.load(out_ptrs, mask=offs_m[:, None] < N_CTX_Q, other=0.0)
        lse = tl.load(lse_ptrs, mask=offs_m < N_CTX_Q, other=0.0)
        if HAS_DLSE:
            dlse = tl.load(dlse_ptrs, mask=offs_m < N_CTX_Q, other=0.0)

        # compute attention scores

        sim = (tl.dot(q, tl.trans(k), allow_tf32=False) * sm_scale).to(tl.float32)

        # softclamp if needed

        sim_tanh = sim
        if SOFTCLAMP_VALUE is not None:
            sim_tanh = SOFTCLAMP_VALUE * libdevice.tanh(sim / SOFTCLAMP_VALUE)

        sim_soft = sim_tanh
        if HAS_BIAS:
            bias_ptrs = Bias + b_offset + offs_n * stride_bn
            bias = tl.load(bias_ptrs, mask=offs_n < N_CTX_K, other=0.0)
            sim_soft += bias[None, :]

        if HAS_MASK:
            mask_ptrs = Mask + off_z * stride_mz + offs_n * stride_mn
            mask_val = tl.load(mask_ptrs, mask=offs_n < N_CTX_K, other=0.0)
            sim_soft = tl.where(mask_val[None, :], sim_soft, float("-inf"))

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            sim_soft = tl.where(causal_mask, sim_soft, float("-inf"))

        sim_soft = tl.where(offs_n[None, :] < N_CTX_K, sim_soft, float("-inf"))
        sim_soft = tl.where(offs_m[:, None] < N_CTX_Q, sim_soft, float("-inf"))

        # compute attention weights

        p = tl.math.exp(sim_soft - lse[:, None])

        # backward math

        dp = tl.dot(do, tl.trans(v), allow_tf32=False)
        Di = tl.sum(do * out, axis=1)
        ds = p * (dp - Di[:, None])

        if HAS_DLSE:
            ds += p * dlse[:, None]

        if HAS_BIAS:
            dbias += tl.sum(ds, axis=0)

        if SOFTCLAMP_VALUE is not None:
            ds = ds * (1.0 - (sim_tanh / SOFTCLAMP_VALUE) * (sim_tanh / SOFTCLAMP_VALUE))

        # accumulate gradients

        dv += tl.dot(tl.trans(p.to(v.dtype)), do, allow_tf32=False)
        dk += (tl.dot(tl.trans(ds.to(q.dtype)), q, allow_tf32=False) * sm_scale).to(tl.float32)

        dq_chunk = (tl.dot(ds.to(k.dtype), k, allow_tf32=False) * sm_scale).to(tl.float32)

        tl.atomic_add(dq_ptrs, dq_chunk, mask=(offs_m[:, None] < N_CTX_Q) & (offs_k[None, :] < BLOCK_DMODEL))

        # advance pointers

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_qm
        out_ptrs += BLOCK_M * stride_qm
        dq_ptrs += BLOCK_M * stride_qm
        lse_ptrs += BLOCK_M
        if HAS_DLSE:
            dlse_ptrs += BLOCK_M

    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL))
    tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX_K) & (offs_k[None, :] < BLOCK_DMODEL))

    if HAS_BIAS:
        dbias_ptrs = DBias + b_offset + offs_n * stride_bn
        tl.store(dbias_ptrs, dbias, mask=offs_n < N_CTX_K)


class PolyFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q1, q2, q3, v3, mask=None, softclamp_value=None, is_causal=True):
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        q3 = q3.contiguous()
        v3 = v3.contiguous()

        has_mask = mask is not None
        if has_mask:
            mask = mask.contiguous()
            stride_mz, stride_mn = mask.stride(0), mask.stride(1)
        else:
            stride_mz, stride_mn = 0, 0

        batch, heads, seq_len, dim = q1.shape
        sm_scale = dim ** -0.5

        msg = torch.empty_like(v3)
        lse23 = torch.empty((batch, heads, seq_len), device=q2.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(seq_len, META['BLOCK_M']),
            batch * heads,
            1
        )

        # Pass 1
        _poly_flash_fwd_kernel[grid](
            q2, q3, v3, sm_scale,
            None, mask, msg, lse23,
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            q3.stride(0), q3.stride(1), q3.stride(2), q3.stride(3),
            v3.stride(0), v3.stride(1), v3.stride(2), v3.stride(3),
            0, 0, 0,
            stride_mz, stride_mn,
            msg.stride(0), msg.stride(1), msg.stride(2), msg.stride(3),
            batch, heads, seq_len, seq_len,
            BLOCK_DMODEL=dim,
            IS_CAUSAL=is_causal, HAS_BIAS=False, HAS_MASK=has_mask, SOFTCLAMP_VALUE=softclamp_value
        )

        # Pass 2
        out = torch.empty_like(q1)
        lse12 = torch.empty((batch, heads, seq_len), device=q1.device, dtype=torch.float32)

        _poly_flash_fwd_kernel[grid](
            q1, q2, msg, sm_scale,
            lse23, mask, out, lse12,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            msg.stride(0), msg.stride(1), msg.stride(2), msg.stride(3),
            lse23.stride(0), lse23.stride(1), lse23.stride(2),
            stride_mz, stride_mn,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch, heads, seq_len, seq_len,
            BLOCK_DMODEL=dim,
            IS_CAUSAL=is_causal, HAS_BIAS=True, HAS_MASK=has_mask, SOFTCLAMP_VALUE=softclamp_value
        )

        ctx.saved_lse12 = lse12
        ctx.save_for_backward(q1, q2, q3, v3, msg, lse23, out, lse12, mask)
        ctx.sm_scale = sm_scale
        ctx.softclamp_value = softclamp_value
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        q1, q2, q3, v3, msg, lse23, out, lse12, mask = ctx.saved_tensors
        batch, heads, seq_len, dim = q2.shape
        sm_scale = ctx.sm_scale
        softclamp_value = ctx.softclamp_value
        is_causal = ctx.is_causal

        has_mask = mask is not None
        if has_mask:
            stride_mz, stride_mn = mask.stride(0), mask.stride(1)
        else:
            stride_mz, stride_mn = 0, 0

        dq1 = torch.zeros_like(q1)
        dq2_part2 = torch.zeros_like(q2)
        dmsg = torch.zeros_like(msg)
        dlse23 = torch.zeros_like(lse23)

        grid_bwd = lambda META: (triton.cdiv(seq_len, META['BLOCK_N']), batch * heads, 1)

        # Pass 2 Backward
        _poly_flash_bwd_kernel[grid_bwd](
            q1, q2, msg, sm_scale,
            out, dout, lse12, None,
            lse23, dlse23, mask,
            dq1, dq2_part2, dmsg,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            msg.stride(0), msg.stride(1), msg.stride(2), msg.stride(3),
            lse23.stride(0), lse23.stride(1), lse23.stride(2),
            stride_mz, stride_mn,
            batch, heads, seq_len, seq_len,
            BLOCK_DMODEL=dim,
            IS_CAUSAL=is_causal, HAS_BIAS=True, HAS_DLSE=False, HAS_MASK=has_mask, SOFTCLAMP_VALUE=softclamp_value
        )

        # Pass 1 Backward
        dq2_part1 = torch.zeros_like(q2)
        dq3 = torch.zeros_like(q3)
        dv3 = torch.zeros_like(v3)

        _poly_flash_bwd_kernel[grid_bwd](
            q2, q3, v3, sm_scale,
            msg, dmsg, lse23, dlse23,
            None, None, mask,
            dq2_part1, dq3, dv3,
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            q3.stride(0), q3.stride(1), q3.stride(2), q3.stride(3),
            v3.stride(0), v3.stride(1), v3.stride(2), v3.stride(3),
            0, 0, 0,
            stride_mz, stride_mn,
            batch, heads, seq_len, seq_len,
            BLOCK_DMODEL=dim,
            IS_CAUSAL=is_causal, HAS_BIAS=False, HAS_DLSE=True, HAS_MASK=has_mask, SOFTCLAMP_VALUE=softclamp_value
        )

        dq2 = dq2_part1 + dq2_part2

        return dq1, dq2, dq3, dv3, None, None, None

def flash_poly_attention(
    q1, q2, q3, v3,
    mask = None,
    context_mask = None,
    softclamp_value = None,
    is_causal = True
):
    assert not exists(context_mask), 'context_mask is not supported in flash_poly_attention'
    return PolyFlashAttention.apply(q1, q2, q3, v3, mask, softclamp_value, is_causal)
