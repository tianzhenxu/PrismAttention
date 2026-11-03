import torch
import triton
import triton.language as tl
import time
import os

os.environ["TRITON_PTXAS_EXTRA_ARGS"] = "-q"

@triton.jit
def cvt_e4m3x2_to_f16x2_ptx(x_u16):
    """Single PTX instruction to convert two e4m3 values to one f16x2 (int32)."""
    return tl.inline_asm_elementwise(
        "cvt.rn.f16x2.e4m3x2 $0, $1;",
        "=r,h",
        [x_u16],
        dtype=tl.int32,
        is_pure=True,
        pack=1
    )

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SEQ': 128}, num_warps=4),
#         triton.Config({'BLOCK_SEQ': 128}, num_warps=8),
#         triton.Config({'BLOCK_SEQ': 256}, num_warps=4),
#         triton.Config({'BLOCK_SEQ': 256}, num_warps=8),
#         triton.Config({'BLOCK_SEQ': 512}, num_warps=8),
#     ],
#     key=['seq_len', 'k_heads', 'batch'],
# )
@triton.jit
def int2_attention_score_kernel(
    Q_ptr,           # [B, num_q_heads, 1, HEAD_DIM]         bfloat16
    K_q_ptr,         # [B, K_HEADS, SEQ_LEN, HEAD_DIM//4]    uint8
    K_sz_ptr,        # [B, K_HEADS, SEQ_LEN, NUM_SCALES*2]   bfloat16
    Out_ptr,         # [B, K_HEADS, 1, SEQ_LEN]              bfloat16
    batch,
    seq_len,
    k_heads,
    stride_qb,  stride_qh,  stride_qd,
    stride_kb,  stride_kh,  stride_ks,  stride_kd,
    stride_szb, stride_szh, stride_szs, stride_szd,
    stride_ob,  stride_oh,  stride_os,
    sink_size,       # number of initial tokens to preserve
    recent_size,     # number of local tokens to preserve
    BLOCK_SEQ:  tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    GROUPSIZE:  tl.constexpr,
    Q_PER_KV:   tl.constexpr,
    PACKED_DIM: tl.constexpr,
    NUM_SCALES: tl.constexpr
):
    pid_seq = tl.program_id(0)
    pid_bh  = tl.program_id(1)

    bz = pid_bh // k_heads
    bh = pid_bh % k_heads
    seq_start = pid_seq * BLOCK_SEQ

    q_start_idx = bh * Q_PER_KV
    offs_q   = tl.arange(0, 16)
    offs_dpk = tl.arange(0, PACKED_DIM)

    q_base  = Q_ptr + bz * stride_qb + (q_start_idx + offs_q[:, None]) * stride_qh
    mask_q  = offs_q[:, None] < Q_PER_KV

    Q_0 = tl.load(q_base + (offs_dpk[None, :] * 4 + 0) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)
    Q_1 = tl.load(q_base + (offs_dpk[None, :] * 4 + 1) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)
    Q_2 = tl.load(q_base + (offs_dpk[None, :] * 4 + 2) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)
    Q_3 = tl.load(q_base + (offs_dpk[None, :] * 4 + 3) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)

    offs_seq = seq_start + tl.arange(0, BLOCK_SEQ)
    is_valid_seq = offs_seq < seq_len
    k_q_ptrs = K_q_ptr + bz * stride_kb + bh * stride_kh + offs_seq[:, None] * stride_ks + offs_dpk[None, :] * stride_kd
    K_q = tl.load(k_q_ptrs, mask=is_valid_seq[:, None], other=0).to(tl.uint32)

    offs_scales = tl.arange(0, NUM_SCALES)
    scale = tl.load(K_sz_ptr + bz * stride_szb + bh * stride_szh + offs_seq[:, None] * stride_szs + (offs_scales[None, :] * 2 + 0) * stride_szd).to(tl.bfloat16)
    zp    = tl.load(K_sz_ptr + bz * stride_szb + bh * stride_szh + offs_seq[:, None] * stride_szs + (offs_scales[None, :] * 2 + 1) * stride_szd).to(tl.bfloat16)

    v0, v1, v2, v3 = K_q & 0x03, (K_q >> 2) & 0x03, (K_q >> 4) & 0x03, (K_q >> 6) & 0x03
    lo, hi = (v1 << 8) | v0, (v3 << 8) | v2

    f16x2_lo = cvt_e4m3x2_to_f16x2_ptx(lo.to(tl.uint16)).to(tl.uint32)
    f16x2_hi = cvt_e4m3x2_to_f16x2_ptx(hi.to(tl.uint16)).to(tl.uint32)

    val0 = (f16x2_lo & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
    val1 = ((f16x2_lo >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
    val2 = (f16x2_hi & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
    val3 = ((f16x2_hi >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)

    scale_r, zp_r = scale[:, :, None], zp[:, :, None]

    deq0 = tl.reshape(tl.reshape(val0, (BLOCK_SEQ, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_SEQ, PACKED_DIM))
    deq1 = tl.reshape(tl.reshape(val1, (BLOCK_SEQ, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_SEQ, PACKED_DIM))
    deq2 = tl.reshape(tl.reshape(val2, (BLOCK_SEQ, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_SEQ, PACKED_DIM))
    deq3 = tl.reshape(tl.reshape(val3, (BLOCK_SEQ, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_SEQ, PACKED_DIM))

    acc = tl.zeros((16, BLOCK_SEQ), dtype=tl.float32)
    acc = tl.dot(Q_0, tl.trans(deq0), acc)
    acc = tl.dot(Q_1, tl.trans(deq1), acc)
    acc = tl.dot(Q_2, tl.trans(deq2), acc)
    acc = tl.dot(Q_3, tl.trans(deq3), acc)

    acc = tl.where(offs_q[:, None] < Q_PER_KV, acc, -1e20)
    max_scores = tl.max(acc, axis=0)

    # Override scores for Sink and Recent tokens to guarantee preservation
    # (bfloat16 max is ~3.38e38, so 1.0e10 is absolutely safe)
    is_sink = offs_seq < sink_size
    is_recent = offs_seq >= (seq_len - recent_size)
    force_1e10 = is_valid_seq & (is_sink | is_recent)
    
    max_scores = tl.where(force_1e10, 1.0e10, max_scores)

    offs_out = bz * stride_ob + bh * stride_oh + offs_seq * stride_os
    tl.store(Out_ptr + offs_out, max_scores.to(tl.bfloat16), mask=is_valid_seq)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def int2_attention_score(
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_sz: torch.Tensor,
    sink_size: int = 64,
    recent_size: int = 192,
    group_size: int = 32
) -> torch.Tensor:
    """
    Computes attention scores for INT2 quantized KV cache with Sink/Recent token preservation.

    Parameters
    ----------
    q           : [B, num_q_heads, 1, HEAD_DIM]               bfloat16
    k_q         : [B, K_HEADS, SEQ_LEN, HEAD_DIM//4]          uint8 (2-bit packed)
    k_sz        : [B, K_HEADS, SEQ_LEN, (HEAD_DIM//group)*2]  bfloat16 (scales & zero-points)
    sink_size   : number of initial tokens to preserve
    recent_size : number of local tokens to preserve
    group_size  : quantization group size along head_dim
    """
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert k_q.dtype == torch.uint8, "k_q must be uint8 (2-bit packed)"
    assert k_sz.dtype == torch.bfloat16, "k_sz must be bfloat16"

    B, num_q_heads, seq_len_q, HEAD_DIM = q.shape
    _, K_HEADS, SEQ_LEN, PACKED_DIM = k_q.shape

    assert seq_len_q == 1, f"Kernel optimized for decoding (seq_len=1), got {seq_len_q}"
    assert num_q_heads % K_HEADS == 0, f"Q heads ({num_q_heads}) must be divisible by K heads ({K_HEADS})"
    assert HEAD_DIM // 4 == PACKED_DIM, "k_q last dim must be HEAD_DIM // 4 (packed uint8)"

    Q_PER_KV = num_q_heads // K_HEADS
    NUM_SCALES = HEAD_DIM // group_size

    out_scores = torch.empty((B, K_HEADS, 1, SEQ_LEN), dtype=torch.bfloat16, device=q.device)

    BLOCK_SEQ = 128
    num_warps = 4

    grid = (triton.cdiv(SEQ_LEN, BLOCK_SEQ), B * K_HEADS)

    with torch.cuda.device(q.device):
        int2_attention_score_kernel[grid](
            q, k_q, k_sz, out_scores,
            B, SEQ_LEN, K_HEADS,
            q.stride(0), q.stride(1), q.stride(3),
            k_q.stride(0), k_q.stride(1), k_q.stride(2), k_q.stride(3),
            k_sz.stride(0), k_sz.stride(1), k_sz.stride(2), k_sz.stride(3),
            out_scores.stride(0), out_scores.stride(1), out_scores.stride(3),
            sink_size, recent_size,
            HEAD_DIM=HEAD_DIM, GROUPSIZE=group_size,
            Q_PER_KV=Q_PER_KV, PACKED_DIM=PACKED_DIM, NUM_SCALES=NUM_SCALES,
            BLOCK_SEQ=BLOCK_SEQ, num_warps=num_warps
        )

    return out_scores


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_E': 128}, num_warps=4),
#         triton.Config({'BLOCK_E': 128}, num_warps=8),
#         triton.Config({'BLOCK_E': 256}, num_warps=4),
#         triton.Config({'BLOCK_E': 256}, num_warps=8),
#         triton.Config({'BLOCK_E': 512}, num_warps=8),
#     ],
#     key=['E', 'k_heads', 'batch'],
# )
@triton.jit
def int2_attention_score_indexed_kernel(
    Q_ptr,           # [B, num_q_heads, 1, HEAD_DIM]         bfloat16
    K_q_ptr,         # [B, K_HEADS, MAX_SEQ, HEAD_DIM//4]    uint8
    K_sz_ptr,        # [B, K_HEADS, MAX_SEQ, NUM_SCALES*2]   bfloat16
    Idx_ptr,         # [B, K_HEADS, E]                       int32 / int64
    Out_ptr,         # [B, K_HEADS, 1, E]                    bfloat16
    batch,
    seq_len,
    E,               # Number of candidate indices
    k_heads,
    stride_qb,  stride_qh,  stride_qd,
    stride_kb,  stride_kh,  stride_ks,  stride_kd,
    stride_szb, stride_szh, stride_szs, stride_szd,
    stride_idxb,stride_idxh,stride_idxe,
    stride_ob,  stride_oh,  stride_oe,
    sink_size,       # number of initial tokens to preserve
    recent_size,     # number of local tokens to preserve    
    BLOCK_E:    tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    GROUPSIZE:  tl.constexpr,
    Q_PER_KV:   tl.constexpr,
    PACKED_DIM: tl.constexpr,
    NUM_SCALES: tl.constexpr
):
    pid_e  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bz = pid_bh // k_heads
    bh = pid_bh % k_heads
    e_start = pid_e * BLOCK_E

    q_start_idx = bh * Q_PER_KV
    offs_q   = tl.arange(0, 16)
    offs_dpk = tl.arange(0, PACKED_DIM)

    q_base  = Q_ptr + bz * stride_qb + (q_start_idx + offs_q[:, None]) * stride_qh
    mask_q  = offs_q[:, None] < Q_PER_KV

    Q_0 = tl.load(q_base + (offs_dpk[None, :] * 4 + 0) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)
    Q_1 = tl.load(q_base + (offs_dpk[None, :] * 4 + 1) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)
    Q_2 = tl.load(q_base + (offs_dpk[None, :] * 4 + 2) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)
    Q_3 = tl.load(q_base + (offs_dpk[None, :] * 4 + 3) * stride_qd, mask=mask_q, other=0.0).to(tl.bfloat16)

    offs_e = e_start + tl.arange(0, BLOCK_E)
    is_valid_e = offs_e < E

    # Load indices for this block
    idx = tl.load(Idx_ptr + bz * stride_idxb + bh * stride_idxh + offs_e * stride_idxe, mask=is_valid_e, other=0)

    # Gather K and Scale/ZP based on loaded indices
    k_q_ptrs = K_q_ptr + bz * stride_kb + bh * stride_kh + idx[:, None] * stride_ks + offs_dpk[None, :] * stride_kd
    K_q = tl.load(k_q_ptrs, mask=is_valid_e[:, None], other=0).to(tl.uint32)

    offs_scales = tl.arange(0, NUM_SCALES)
    scale = tl.load(K_sz_ptr + bz * stride_szb + bh * stride_szh + idx[:, None] * stride_szs + (offs_scales[None, :] * 2 + 0) * stride_szd, mask=is_valid_e[:, None], other=1.0).to(tl.bfloat16)
    zp    = tl.load(K_sz_ptr + bz * stride_szb + bh * stride_szh + idx[:, None] * stride_szs + (offs_scales[None, :] * 2 + 1) * stride_szd, mask=is_valid_e[:, None], other=0.0).to(tl.bfloat16)

    v0, v1, v2, v3 = K_q & 0x03, (K_q >> 2) & 0x03, (K_q >> 4) & 0x03, (K_q >> 6) & 0x03
    lo, hi = (v1 << 8) | v0, (v3 << 8) | v2

    f16x2_lo = cvt_e4m3x2_to_f16x2_ptx(lo.to(tl.uint16)).to(tl.uint32)
    f16x2_hi = cvt_e4m3x2_to_f16x2_ptx(hi.to(tl.uint16)).to(tl.uint32)

    val0 = (f16x2_lo & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
    val1 = ((f16x2_lo >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
    val2 = (f16x2_hi & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
    val3 = ((f16x2_hi >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)

    scale_r, zp_r = scale[:, :, None], zp[:, :, None]

    deq0 = tl.reshape(tl.reshape(val0, (BLOCK_E, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_E, PACKED_DIM))
    deq1 = tl.reshape(tl.reshape(val1, (BLOCK_E, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_E, PACKED_DIM))
    deq2 = tl.reshape(tl.reshape(val2, (BLOCK_E, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_E, PACKED_DIM))
    deq3 = tl.reshape(tl.reshape(val3, (BLOCK_E, NUM_SCALES, 8)) * scale_r - zp_r, (BLOCK_E, PACKED_DIM))

    acc = tl.zeros((16, BLOCK_E), dtype=tl.float32)
    acc = tl.dot(Q_0, tl.trans(deq0), acc)
    acc = tl.dot(Q_1, tl.trans(deq1), acc)
    acc = tl.dot(Q_2, tl.trans(deq2), acc)
    acc = tl.dot(Q_3, tl.trans(deq3), acc)

    acc = tl.where(offs_q[:, None] < Q_PER_KV, acc, -1e20)
    max_scores = tl.max(acc, axis=0)

    is_sink = idx < sink_size
    is_recent = idx >= (seq_len - recent_size)
    force_1e10 = is_valid_e & (is_sink | is_recent)
    
    max_scores = tl.where(force_1e10, 1.0e10, max_scores)

    offs_out = bz * stride_ob + bh * stride_oh + offs_e * stride_oe
    tl.store(Out_ptr + offs_out, max_scores.to(tl.bfloat16), mask=is_valid_e)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def int2_attention_score_indexed(
    q: torch.Tensor,
    k_q: torch.Tensor,
    k_sz: torch.Tensor,
    indices: torch.Tensor,
    comp: int,
    sink_size: int = 64,
    recent_size: int = 192,
    group_size: int = 32
) -> torch.Tensor:
    """
    Computes attention scores for INT2 quantized KV cache using specific indices (Top-K/Sparse).

    Parameters
    ----------
    q           : [B, num_q_heads, 1, HEAD_DIM]               bfloat16
    k_q         : [B, K_HEADS, MAX_SEQ, HEAD_DIM//4]          uint8 (2-bit packed)
    k_sz        : [B, K_HEADS, MAX_SEQ, (HEAD_DIM//group)*2]  bfloat16 (scales & zero-points)
    indices     : [B, K_HEADS, E]                             int32 or int64 (gather indices)
    comp        : int                                         number of valid columns in k_q
    sink_size   : number of initial tokens to preserve
    recent_size : number of local tokens to preserve    
    group_size  : quantization group size along head_dim
    """
    assert q.dtype == torch.bfloat16, "q must be bfloat16"
    assert k_q.dtype == torch.uint8, "k_q must be uint8 (2-bit packed)"
    assert k_sz.dtype == torch.bfloat16, "k_sz must be bfloat16"
    assert indices.dtype in (torch.int32, torch.int64), "indices must be integer type"

    B, num_q_heads, seq_len_q, HEAD_DIM = q.shape
    _, K_HEADS, MAX_SEQ, PACKED_DIM = k_q.shape
    _, _, E = indices.shape
    S = comp

    assert seq_len_q == 1, f"Kernel optimized for decoding (seq_len=1), got {seq_len_q}"
    assert num_q_heads % K_HEADS == 0, f"Q heads ({num_q_heads}) must be divisible by K heads ({K_HEADS})"
    assert HEAD_DIM // 4 == PACKED_DIM, "k_q last dim must be HEAD_DIM // 4 (packed uint8)"

    Q_PER_KV = num_q_heads // K_HEADS
    NUM_SCALES = HEAD_DIM // group_size

    out_scores = torch.empty((B, K_HEADS, 1, E), dtype=torch.bfloat16, device=q.device)

    BLOCK_E = 128
    num_warps = 4

    grid = (triton.cdiv(E, BLOCK_E), B * K_HEADS)

    with torch.cuda.device(q.device):
        int2_attention_score_indexed_kernel[grid](
            q, k_q, k_sz, indices, out_scores,
            B, S, E, K_HEADS,
            q.stride(0), q.stride(1), q.stride(3),
            k_q.stride(0), k_q.stride(1), k_q.stride(2), k_q.stride(3),
            k_sz.stride(0), k_sz.stride(1), k_sz.stride(2), k_sz.stride(3),
            indices.stride(0), indices.stride(1), indices.stride(2),
            out_scores.stride(0), out_scores.stride(1), out_scores.stride(3),
            sink_size, recent_size,
            HEAD_DIM=HEAD_DIM, GROUPSIZE=group_size,
            Q_PER_KV=Q_PER_KV, PACKED_DIM=PACKED_DIM, NUM_SCALES=NUM_SCALES,
            BLOCK_E=BLOCK_E,
            num_warps=num_warps
        )

    return out_scores