import torch
import triton
import triton.language as tl

@triton.jit
def sparse_attention_kernel_optimized(
    # Pointers
    Q_ptr, K_ptr, V_ptr, Indices_ptr, Mask_ptr, Out_ptr,
    # Strides
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kl, stride_kh, stride_kd,
    stride_vb, stride_vl, stride_vh, stride_vd,
    stride_ib, stride_ih, stride_ik,
    stride_mb, stride_ml,
    stride_ob, stride_oh, stride_od,
    # Constants
    sm_scale,
    kv_head_num,
    group_size,
    topk,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_q_heads = kv_head_num * group_size

    q_h_idx = pid % num_q_heads
    b_idx = pid // num_q_heads
    kv_h_idx = q_h_idx // group_size

    # Load Query [BLOCK_D]
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = Q_ptr + (b_idx * stride_qb) + (q_h_idx * stride_qh) + (offs_d * stride_qd)
    q = tl.load(q_ptrs)

    # Initialize Online Softmax accumulators
    m_i = -1e10  # running max
    l_i = 0.0    # running sum of exponentials
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over topk in blocks of BLOCK_N
    for start_n in range(0, topk, BLOCK_N):
        # --- Load Indices & Mask ---
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk  # Boundary check

        # Load indices for this block
        i_base_ptr = Indices_ptr + (b_idx * stride_ib) + (kv_h_idx * stride_ih)
        i_ptrs = i_base_ptr + (offs_n * stride_ik)
        indices = tl.load(i_ptrs, mask=mask_n, other=0)

        # Load Attention Mask (Global)
        m_ptrs = Mask_ptr + (b_idx * stride_mb) + (indices * stride_ml)
        mask_vals = tl.load(m_ptrs, mask=mask_n, other=0)

        # --- Load K and Compute Score ---
        k_base = K_ptr + (b_idx * stride_kb) + (kv_h_idx * stride_kh)
        k_ptrs = k_base + (indices[:, None] * stride_kl) + (offs_d[None, :] * stride_kd)

        # Load K block
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute QK^T
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale

        # Apply Mask: set invalid positions to -inf
        is_valid = mask_n & (mask_vals == 1)
        qk = tl.where(is_valid, qk, -1e10)

        # --- Online Softmax Update ---
        m_curr = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        # --- Load V block ---
        v_base = V_ptr + (b_idx * stride_vb) + (kv_h_idx * stride_vh)
        v_ptrs = v_base + (indices[:, None] * stride_vl) + (offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # --- Update accumulators ---
        l_i = l_i * alpha + tl.sum(p, 0)
        weight_v = tl.sum(p[:, None] * v, axis=0)
        acc = acc * alpha + weight_v
        m_i = m_new

    # Final normalization
    l_i = l_i + 1e-6
    acc = acc / l_i

    # Store output
    out_ptrs = Out_ptr + (b_idx * stride_ob) + (q_h_idx * stride_oh) + (offs_d * stride_od)
    tl.store(out_ptrs, acc.to(Q_ptr.dtype.element_ty))


def token_sparse_flash_decode_optimized(query, key, value, indices, mask):
    """
    Optimized token sparse flash decode with fixed BLOCK_N=128
    Uses moderate block size to balance vectorization and register pressure
    """
    # Prepare inputs
    q = query.squeeze(1).contiguous()
    k = key.contiguous()
    v = value.contiguous()
    idx = indices.contiguous()
    m = mask.contiguous()

    bsz, q_head, head_dim = q.shape
    _, kv_len, kv_head, _ = k.shape
    _, _, topk = idx.shape

    group_size = q_head // kv_head
    sm_scale = head_dim ** -0.5

    # Fixed BLOCK_N=128 for balanced performance
    # Small enough to avoid register pressure, large enough for vectorization
    BLOCK_N = 128

    # Output buffer
    o = torch.empty_like(q)

    # Grid: one program per query head
    grid = (bsz * q_head,)

    with torch.cuda.device(q.device):
        sparse_attention_kernel_optimized[grid](
            q, k, v, idx, m, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            idx.stride(0), idx.stride(1), idx.stride(2),
            m.stride(0), m.stride(1),
            o.stride(0), o.stride(1), o.stride(2),
            sm_scale=sm_scale,
            kv_head_num=kv_head,
            group_size=group_size,
            topk=topk,
            BLOCK_D=head_dim,
            BLOCK_N=BLOCK_N
        )

    return o.unsqueeze(1)
