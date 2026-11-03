import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SEQ': 64}, num_warps=4),
#         triton.Config({'BLOCK_SEQ': 128}, num_warps=4),
#         triton.Config({'BLOCK_SEQ': 128}, num_warps=8),
#         triton.Config({'BLOCK_SEQ': 256}, num_warps=8),
#     ],
#     key=['seq_len', 'k_heads', 'batch'],
# )
@triton.jit
def int2_quantize_kernel(
    K_ptr,           # [B, SEQ_LEN, K_HEADS, HEAD_DIM]         bfloat16
    K_q_ptr,         # [B, K_HEADS, MAX_SEQ, PACKED_DIM]       uint8
    K_sz_ptr,        # [B, K_HEADS, MAX_SEQ, NUM_SCALES*2]     bfloat16
    batch,
    seq_len,
    k_heads,
    col,
    stride_kb,  stride_ks,  stride_kh,  stride_kd,
    stride_kqb, stride_kqh, stride_kqs, stride_kqd,
    stride_szb, stride_szh, stride_szs, stride_szd,
    eps,
    BLOCK_SEQ:  tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    GROUPSIZE:  tl.constexpr,
    NUM_SCALES: tl.constexpr,
    QS:         tl.constexpr,
    PACKED_DIM: tl.constexpr
):
    pid_seq = tl.program_id(0)
    pid_bh  = tl.program_id(1)

    bz = pid_bh // k_heads
    bh = pid_bh % k_heads
    seq_start = pid_seq * BLOCK_SEQ

    offs_seq = seq_start + tl.arange(0, BLOCK_SEQ)
    mask_seq = offs_seq < seq_len

    # 1. Fully coalesced memory read
    offs_d = tl.arange(0, HEAD_DIM)
    k_ptrs = K_ptr + bz * stride_kb + offs_seq[:, None] * stride_ks + bh * stride_kh + offs_d[None, :] * stride_kd
    x = tl.load(k_ptrs, mask=mask_seq[:, None], other=0.0).to(tl.float32)

    # 2. Min/Max and Scaling calculation
    x_3d = tl.reshape(x, (BLOCK_SEQ, NUM_SCALES, GROUPSIZE))
    x_min = tl.min(x_3d, axis=2)
    x_max = tl.max(x_3d, axis=2)

    scale = tl.maximum(x_max - x_min, eps) / 3.0
    inv_scale = 1.0 / scale

    q_3d = tl.clamp(tl.floor((x_3d - x_min[:, :, None]) * inv_scale[:, :, None] + 0.5), 0.0, 3.0).to(tl.int32)

    store_scale = scale * 512.0
    store_zp = -x_min

    # 3. Workaround for Triton slicing bug: tensor reduction for bit-packing
    q_4d = tl.reshape(q_3d, (BLOCK_SEQ, NUM_SCALES, QS, 4))
    
    shift = tl.arange(0, 4) * 2
    ones = tl.full((4,), 1, dtype=tl.int32)
    powers = ones << shift
    powers_4d = tl.reshape(powers, (1, 1, 1, 4))
    
    q_shifted = q_4d * powers_4d
    packed_3d = tl.sum(q_shifted, axis=3)
    packed_flat = tl.reshape(packed_3d, (BLOCK_SEQ, PACKED_DIM)).to(tl.uint8)

    # 4. Continuous KQ write
    offs_packed = tl.arange(0, PACKED_DIM)
    kq_ptrs = K_q_ptr + bz * stride_kqb + bh * stride_kqh + (col + offs_seq[:, None]) * stride_kqs + offs_packed[None, :] * stride_kqd
    tl.store(kq_ptrs, packed_flat, mask=mask_seq[:, None])

    # 5. Continuous interleaved KSZ write (scales and mins)
    offs_scales = tl.arange(0, NUM_SCALES)
    ksz_ptrs_scale = K_sz_ptr + bz * stride_szb + bh * stride_szh + (col + offs_seq[:, None]) * stride_szs + (offs_scales[None, :] * 2 + 0) * stride_szd
    ksz_ptrs_min   = K_sz_ptr + bz * stride_szb + bh * stride_szh + (col + offs_seq[:, None]) * stride_szs + (offs_scales[None, :] * 2 + 1) * stride_szd
    
    tl.store(ksz_ptrs_scale, store_scale.to(tl.bfloat16), mask=mask_seq[:, None])
    tl.store(ksz_ptrs_min, store_zp.to(tl.bfloat16), mask=mask_seq[:, None])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def int2_quantize(
    k: torch.Tensor,
    k_q_buf: torch.Tensor,
    k_sz_buf: torch.Tensor,
    col: int,
    group_size: int = 32,
    eps: float = 1e-5
) -> None:
    """
    Quantizes key tensors to INT2 (2-bit packed) and writes directly to KV cache buffers.

    Parameters
    ----------
    k          : [B, SEQ_LEN, K_HEADS, HEAD_DIM]             bfloat16
    k_q_buf    : [B, K_HEADS, MAX_SEQ, HEAD_DIM//4]          uint8 (2-bit packed)
    k_sz_buf   : [B, K_HEADS, MAX_SEQ, (HEAD_DIM//group)*2]  bfloat16 (scales & mins)
    col        : starting sequence column offset in the cache
    group_size : quantization group size along head_dim
    eps        : epsilon for numerical stability during scaling
    """
    assert k.dtype == torch.bfloat16, "k must be bfloat16"
    assert k_q_buf.dtype == torch.uint8, "k_q_buf must be uint8 (2-bit packed)"
    assert k_sz_buf.dtype == torch.bfloat16, "k_sz_buf must be bfloat16"

    B, SEQ_LEN, K_HEADS, HEAD_DIM = k.shape
    _, _, MAX_SEQ, PACKED_DIM = k_q_buf.shape

    assert HEAD_DIM % group_size == 0, f"HEAD_DIM ({HEAD_DIM}) must be divisible by group_size ({group_size})"
    assert HEAD_DIM // 4 == PACKED_DIM, f"k_q_buf last dim must be HEAD_DIM // 4 (packed uint8)"

    NUM_SCALES = HEAD_DIM // group_size
    QS = group_size // 4

    BLOCK_SEQ = 64
    num_warps = 4

    grid = (triton.cdiv(SEQ_LEN, BLOCK_SEQ), B * K_HEADS)

    with torch.cuda.device(k.device):
        int2_quantize_kernel[grid](
            k, k_q_buf, k_sz_buf,
            B, SEQ_LEN, K_HEADS, col,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            k_q_buf.stride(0), k_q_buf.stride(1), k_q_buf.stride(2), k_q_buf.stride(3),
            k_sz_buf.stride(0), k_sz_buf.stride(1), k_sz_buf.stride(2), k_sz_buf.stride(3),
            eps,
            HEAD_DIM=HEAD_DIM,
            GROUPSIZE=group_size,
            NUM_SCALES=NUM_SCALES,
            QS=QS,
            PACKED_DIM=PACKED_DIM,
            BLOCK_SEQ=BLOCK_SEQ,
            num_warps=num_warps
        )