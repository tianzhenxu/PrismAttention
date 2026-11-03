"""
IndexCache Training-Free Offline Calibration

Implements the greedy search algorithm from the IndexCache paper
(Section 3.1.2 - Layer Selection Algorithm):

1. Start from all-F baseline (every layer runs its own indexer / full attention).
2. Greedy loop for K steps (K = target number of S layers):
   - For each currently-F layer (except layer 0), tentatively flip it to S.
   - Run forward pass on calibration set, compute LM loss.
   - Commit the flip that yields the lowest loss.
3. Save the resulting F/S configuration for inference.

S layers reuse the nearest *preceding* F layer's top-k attention indices,
performing sparse attention on only those k positions.

Two search modes are provided:
  --search_mode loss    : paper's primary method (accurate, slower)
  --search_mode similarity : similarity-based proxy (fast approximation)
"""

import torch
import torch.nn.functional as Func
import torch.distributed as dist
import triton
import triton.language as tl
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import transformers
from transformers.models.qwen3.modeling_qwen3 import (
    repeat_kv,
    apply_rotary_pos_emb,
)
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
import numpy as np
from tqdm import tqdm
import json
import os
import pickle
import logging
import time
import subprocess
from datetime import datetime
from typing import Optional, Tuple, List, Set
import random


# ==============================================================
# Logging setup
# ==============================================================
log = logging.getLogger("indexcache_calibration")
log.setLevel(logging.DEBUG)
_log_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
# Console handler (INFO level)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_log_formatter)
log.addHandler(_ch)


def _add_file_handler(output_dir: str, cfg_tag: str = "", ts: str = ""):
    """Add a file handler once output_dir is known.

    cfg_tag : short config string appended after the timestamp,
              e.g. "n64_k512_r25_beam2_frac50_w32"
    ts      : timestamp string (e.g. from shell TIMESTAMP var);
              if empty, generated from current time.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not ts:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{cfg_tag}" if cfg_tag else ""
    log_path = os.path.join(output_dir, f"calibration_{ts}{suffix}.log")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_log_formatter)
    log.addHandler(fh)
    log.info(f"Log file -> {log_path}")
    return log_path


# ==============================================================
# Distributed helpers
# ==============================================================
def _is_dist() -> bool:
    """Return True if running in a distributed context."""
    return dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _is_main() -> bool:
    return _rank() == 0


def _barrier():
    if _is_dist():
        dist.barrier()


def _setup_dist():
    """Initialize distributed process group (called once per worker)."""
    if "RANK" not in os.environ:
        return  # single-GPU fallback, no distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Only main process prints INFO; workers print WARNING+
    if rank != 0:
        log.setLevel(logging.WARNING)
        for h in log.handlers:
            h.setLevel(logging.WARNING)


def _cleanup_dist():
    if _is_dist():
        dist.destroy_process_group()


def _gpu_mem_str() -> str:
    """Return GPU memory usage for all visible GPUs via nvidia-smi.

    nvidia-smi reports physical memory used by ALL processes, so in
    distributed mode rank 0 can see every rank's actual GPU usage without
    cross-process communication.  Falls back to torch allocator stats if
    nvidia-smi is unavailable.

    Output format: "GPU0 7.5/79.1GB | GPU1 7.5/79.1GB | ..."
      used  = physical memory currently in use (from nvidia-smi)
      total = GPU capacity
    """
    if not torch.cuda.is_available():
        return "GPU: N/A"
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        parts = []
        for line in result.stdout.strip().splitlines():
            idx, used, total = line.split(", ")
            parts.append(f"GPU{idx.strip()} {int(used)/1024:.1f}/{int(total)/1024:.1f}GB")
        return " | ".join(parts)
    except Exception:
        # Fallback: current rank only (torch allocator view)
        dev   = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        resrv = torch.cuda.memory_reserved(dev)  / (1024 ** 3)
        return f"GPU{dev} {alloc:.1f}/{resrv:.1f}GB (rank {_rank()}, alloc/reserved)"


# ==============================================================
# Global state shared with the monkey-patched attention forward
# ==============================================================
_state = {
    # --- mode switch ---
    "mode": "full",            # "full" | "indexcache" | "calibrate_topk"

    # --- indexcache config ---
    "layer_config": {},        # layer_idx -> "F" or "S"
    "k": 128,                  # top-k budget: int for fixed, "auto" for seq_len//4
    "current_f_topk": None,    # [B, Q, k] from latest F layer

    # --- calibrate_topk collection ---
    "topk_indices": {},        # layer_idx -> list of [B, Q, k]

    # --- loss window: only compute loss on the last fraction of tokens ---
    # 1.0 = all tokens (default); 0.5 = last half; useful because sparse
    # attention diverges from dense mostly on later tokens in the sequence.
    "loss_last_frac": 1.0,
}


def _resolve_k(S: int) -> int:
    """Resolve actual top-k value given current sequence length S."""
    k = _state["k"]
    if k == "auto":
        return max(1, S // 4)
    return min(int(k), S)


@triton.jit
def _sparse_attn_prefill_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, Indices_ptr, Mask_ptr, Out_ptr,
    # Q strides: [B, H, Q, D]
    stride_qb, stride_qh, stride_qq, stride_qd,
    # K strides: [B, num_kv_heads, S, D] (unexpanded)
    stride_kb, stride_kh, stride_ks, stride_kd,
    # V strides: [B, num_kv_heads, S, D] (unexpanded)
    stride_vb, stride_vh, stride_vs, stride_vd,
    # Indices strides: [B, num_kv_heads, Q, k]
    stride_ib, stride_ih, stride_iq, stride_ik,
    # Mask strides: [B, Q, S] (squeezed from [B, 1, Q, S])
    stride_mb, stride_mq, stride_ms,
    # Out strides: [B, H, Q, D]
    stride_ob, stride_oh, stride_oq, stride_od,
    # Scalars
    sm_scale,
    kv_head_num,
    group_size,
    seq_len_q,
    topk,
    # Compile-time constants
    HAS_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Triton kernel for prefill-phase sparse attention.

    Grid: (B * num_q_heads * Q,)
    Each program computes attention output for one (batch, head, query_pos).
    It loops over top-k KV indices in tiles of BLOCK_N, using online softmax
    (Flash-Attention style) for numerical stability.
    """
    pid = tl.program_id(0)
    num_q_heads = kv_head_num * group_size

    # Decompose pid -> (b_idx, q_h_idx, q_idx)
    q_idx = pid % seq_len_q
    tmp = pid // seq_len_q
    q_h_idx = tmp % num_q_heads
    b_idx = tmp // num_q_heads
    kv_h_idx = q_h_idx // group_size

    # 1. Load Query [BLOCK_D]
    offs_d = tl.arange(0, BLOCK_D)
    q_ptrs = (Q_ptr
              + b_idx * stride_qb
              + q_h_idx * stride_qh
              + q_idx * stride_qq
              + offs_d * stride_qd)
    q = tl.load(q_ptrs)

    # 2. Online-softmax accumulators
    m_i = -1e10
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # 3. Loop over top-k indices in tiles of BLOCK_N
    for start_n in range(0, topk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < topk

        # --- A. Load top-k indices [BLOCK_N] ---
        i_ptrs = (Indices_ptr
                  + b_idx * stride_ib
                  + kv_h_idx * stride_ih
                  + q_idx * stride_iq
                  + offs_n * stride_ik)
        indices = tl.load(i_ptrs, mask=mask_n, other=0)

        # --- B. Load K at selected positions [BLOCK_N, BLOCK_D] ---
        k_base = K_ptr + b_idx * stride_kb + kv_h_idx * stride_kh
        k_ptrs = k_base + indices[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # --- C. QK^T -> scores [BLOCK_N] ---
        qk = tl.sum(q[None, :] * k, axis=1)
        qk *= sm_scale

        # --- D. Apply additive attention mask (causal / padding) ---
        if HAS_MASK:
            m_ptrs = (Mask_ptr
                      + b_idx * stride_mb
                      + q_idx * stride_mq
                      + indices * stride_ms)
            mask_vals = tl.load(m_ptrs, mask=mask_n, other=-1e10)
            qk = qk + mask_vals

        # Mask out-of-bounds positions
        qk = tl.where(mask_n, qk, -1e10)

        # --- E. Online softmax update ---
        m_curr = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        # --- F. Load V at selected positions [BLOCK_N, BLOCK_D] ---
        v_base = V_ptr + b_idx * stride_vb + kv_h_idx * stride_vh
        v_ptrs = v_base + indices[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # --- G. Accumulate ---
        l_i = l_i * alpha + tl.sum(p, 0)
        weight_v = tl.sum(p[:, None] * v, axis=0)
        acc = acc * alpha + weight_v
        m_i = m_new

    # 4. Final normalisation
    l_i = l_i + 1e-6
    acc = acc / l_i

    # 5. Store output
    out_ptrs = (Out_ptr
                + b_idx * stride_ob
                + q_h_idx * stride_oh
                + q_idx * stride_oq
                + offs_d * stride_od)
    tl.store(out_ptrs, acc.to(Q_ptr.dtype.element_ty))


def _sparse_attn_placeholder(
    query_states: torch.Tensor,     # [B, H, Q, D]
    key_states: torch.Tensor,       # [B, num_kv_heads, S, D]  (unexpanded)
    value_states: torch.Tensor,     # [B, num_kv_heads, S, D]  (unexpanded)
    topk_idx: torch.Tensor,         # [B, num_kv_heads, Q, k]
    scaling: float,
    attention_mask: Optional[torch.Tensor] = None,
    num_kv_heads: int = 1,
) -> torch.Tensor:
    """Sparse attention using only top-k KV positions per query.

    Uses a fused Triton kernel with online softmax (Flash-Attention style)
    for the prefill phase.  Each query position attends only to its top-k
    KV positions as given by ``topk_idx``.

    The kernel maps query heads to KV heads internally via
    ``kv_h_idx = q_h_idx // group_size``, so K/V should be passed
    **without** GQA expansion to save memory.

    Args:
        query_states:  [B, H, Q, D]  (all query heads)
        key_states:    [B, num_kv_heads, S, D]  (unexpanded)
        value_states:  [B, num_kv_heads, S, D]  (unexpanded)
        topk_idx:      [B, num_kv_heads, Q, k]  — per kv_head top-k indices
        scaling:       softmax temperature (typically head_dim ** -0.5)
        attention_mask: [B, 1, Q, S]  additive mask (0 = keep, large neg = mask)
        num_kv_heads:  number of KV heads (before GQA expansion)

    Returns:
        attn_output: [B, H, Q, D]
    """
    B, H, Q, D = query_states.shape
    k = topk_idx.shape[-1]

    q = query_states.contiguous()
    k_states = key_states.contiguous()
    v_states = value_states.contiguous()
    idx = topk_idx.contiguous()          # [B, num_kv_heads, Q, k]

    o = torch.empty_like(q)

    has_mask = attention_mask is not None and attention_mask.dim() == 4
    if has_mask:
        # [B, 1, Q, S] -> [B, Q, S]  (head dim is broadcast, squeeze it)
        mask = attention_mask.squeeze(1).contiguous()
    else:
        # Dummy tensor — won't be read because HAS_MASK=False
        mask = q

    group_size = H // num_kv_heads
    grid = (B * H * Q,)
    BLOCK_N = 64

    with torch.cuda.device(q.device):
        _sparse_attn_prefill_kernel[grid](
            q, k_states, v_states, idx, mask, o,
            # Q strides
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            # K strides
            k_states.stride(0), k_states.stride(1), k_states.stride(2), k_states.stride(3),
            # V strides
            v_states.stride(0), v_states.stride(1), v_states.stride(2), v_states.stride(3),
            # Indices strides
            idx.stride(0), idx.stride(1), idx.stride(2), idx.stride(3),
            # Mask strides
            mask.stride(0), mask.stride(1) if has_mask else 0, mask.stride(2) if has_mask else 0,
            # Out strides
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            sm_scale=scaling,
            kv_head_num=num_kv_heads,
            group_size=group_size,
            seq_len_q=Q,
            topk=k,
            HAS_MASK=has_mask,
            BLOCK_D=D,
            BLOCK_N=BLOCK_N,
        )

    return o


# ==============================================================
# Unified attention forward (replaces Qwen3Attention.forward)
# ==============================================================
def indexcache_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    B, H, Q, S = query_states.shape[0], query_states.shape[1], query_states.shape[2], key_states.shape[2]
    num_kv_heads = key_states.shape[1]                       # before GQA expansion
    num_q_per_kv = H // num_kv_heads

    mode = _state["mode"]
    lid  = self.layer_idx

    # --------------------------------------------------------
    # MODE: indexcache  (F/S sparse inference / calibration)
    #   Uses unexpanded K/V [B, num_kv_heads, S, D] — the Triton
    #   kernel maps q_head -> kv_head internally, saving group_size×
    #   memory vs repeat_kv.
    # --------------------------------------------------------
    if mode == "indexcache":
        layer_type = _state["layer_config"].get(lid, "F")
        actual_k = _resolve_k(S)

        if layer_type == "F":
            # F layer: compute lightweight compressed scores for top-k selection
            # q: [B, H, Q, D] -> [B, num_kv_heads, num_q_per_kv, Q, D] -> mean -> [B, num_kv_heads, Q, D]
            q_compressed = query_states.reshape(B, num_kv_heads, num_q_per_kv, Q, self.head_dim).mean(dim=2)
            # compressed_scores: [B, num_kv_heads, Q, S]
            compressed_scores = torch.matmul(q_compressed, key_states.transpose(-2, -1)) * self.scaling

            # Apply causal mask BEFORE top-k so future positions are never selected
            if attention_mask is not None:
                compressed_scores = compressed_scores + attention_mask  # [B,1,Q,S] broadcasts

            # each kv_head selects its own top-k (only from causally visible positions)
            _, topk_idx = compressed_scores.float().topk(actual_k, dim=-1)  # [B, num_kv_heads, Q, k]
            _state["current_f_topk"] = topk_idx

        else:
            # S layer: reuse preceding F layer's top-k indices
            topk_idx = _state["current_f_topk"]                 # [B, num_kv_heads, Q, k']

        # ----------------------------------------------------------
        # Split early / late positions for causal correctness:
        #   - Early positions (q < actual_k): visible tokens (q+1) <= k,
        #     do dense (full) causal attention — see everything visible.
        #   - Late positions (q >= actual_k): visible tokens > k,
        #     do sparse top-k attention using topk_idx.
        # ----------------------------------------------------------
        split_pos = min(actual_k, Q)   # boundary: first `split_pos` queries are "early"

        if split_pos >= Q:
            # All positions are early — full dense attention for the whole sequence
            key_states_exp   = repeat_kv(key_states,   num_q_per_kv)
            value_states_exp = repeat_kv(value_states, num_q_per_kv)
            attn_weights = torch.matmul(
                query_states, key_states_exp.transpose(-2, -1)
            ) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_probs = Func.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_probs, value_states_exp)
        elif split_pos <= 0:
            # All positions are late — pure sparse attention
            attn_output = _sparse_attn_placeholder(
                query_states, key_states, value_states,
                topk_idx, self.scaling, attention_mask,
                num_kv_heads=num_kv_heads,
            )
        else:
            # Mixed: early positions use dense, late positions use sparse
            # --- Early part: dense causal attention on q[:split_pos] ---
            q_early = query_states[:, :, :split_pos, :]          # [B, H, split_pos, D]
            key_states_exp   = repeat_kv(key_states,   num_q_per_kv)
            value_states_exp = repeat_kv(value_states, num_q_per_kv)
            attn_w_early = torch.matmul(
                q_early, key_states_exp.transpose(-2, -1)
            ) * self.scaling                                      # [B, H, split_pos, S]
            if attention_mask is not None:
                attn_w_early = attn_w_early + attention_mask[:, :, :split_pos, :]
            attn_p_early = Func.softmax(attn_w_early, dim=-1, dtype=torch.float32).to(query_states.dtype)
            out_early = torch.matmul(attn_p_early, value_states_exp)  # [B, H, split_pos, D]

            # --- Late part: sparse top-k attention on q[split_pos:] ---
            q_late   = query_states[:, :, split_pos:, :]         # [B, H, Q-split_pos, D]
            idx_late = topk_idx[:, :, split_pos:, :]             # [B, num_kv_heads, Q-split_pos, k]
            mask_late = attention_mask[:, :, split_pos:, :] if attention_mask is not None else None
            out_late = _sparse_attn_placeholder(
                q_late, key_states, value_states,
                idx_late, self.scaling, mask_late,
                num_kv_heads=num_kv_heads,
            )                                                     # [B, H, Q-split_pos, D]

            attn_output = torch.cat([out_early, out_late], dim=2) # [B, H, Q, D]

    # --------------------------------------------------------
    # MODE: full  (standard dense attention, baseline)
    #   Needs GQA-expanded K/V for the dense matmul.
    # --------------------------------------------------------
    else:
        key_states_exp   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states_exp = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states_exp.transpose(-2, -1)
        ) * self.scaling                                         # [B, H, Q, S]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_probs = Func.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states_exp)

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _apply_loss_window(shift_h: torch.Tensor,
                       shift_labels: torch.Tensor) -> tuple:
    """Trim shift_h / shift_labels to the last loss_last_frac fraction.

    shift_h      : [S-1, H]  (hidden states after norm, before lm_head)
    shift_labels : [S-1]     (token ids to predict)

    Returns the (possibly trimmed) pair.  When loss_last_frac == 1.0 the
    inputs are returned unchanged.
    """
    frac = _state["loss_last_frac"]
    if frac >= 1.0:
        return shift_h, shift_labels
    seq = shift_h.shape[0]
    start = int(seq * (1.0 - frac))
    return shift_h[start:], shift_labels[start:]


# ==============================================================
# Helper: compute LM loss on calibration set with a given config
# ==============================================================
@torch.no_grad()
def eval_loss(model, input_ids_list, attention_mask_list, layer_config):
    """
    Run model on LOCAL shard of calibration data with given F/S configuration.
    When distributed, all_reduce to aggregate loss/token counts across GPUs.
    Returns average per-token cross-entropy loss (global).
    """
    _state["mode"] = "indexcache"
    _state["layer_config"] = layer_config
    _state["current_f_topk"] = None

    total_loss = 0.0
    total_tokens = 0

    for idx, (input_ids, attn_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        _state["current_f_topk"] = None          # reset per sample
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous().squeeze(0)   # [S-1, V]
        shift_labels = input_ids[:, 1:].contiguous().squeeze(0)    # [S-1]
        shift_logits, shift_labels = _apply_loss_window(shift_logits, shift_labels)
        loss = Func.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
        log.debug(f"  eval_loss sample {idx}: seq_len={input_ids.shape[1]}, "
                  f"batch_loss={loss.item():.4f}, tokens={shift_labels.numel()}")

    # --- Distributed aggregation ---
    if _is_dist():
        stats = torch.tensor([total_loss, total_tokens], dtype=torch.float64,
                             device=torch.cuda.current_device())
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_tokens = int(stats[1].item())

    avg_loss = total_loss / max(total_tokens, 1)
    _state["mode"] = "full"
    return avg_loss


# ==============================================================
# Layer-wise hidden-state cache for fast greedy search
# ==============================================================

def _pin(t: torch.Tensor) -> torch.Tensor:
    """Return a pinned-memory CPU copy of `t` for fast non-blocking D2H transfer."""
    cpu = torch.empty_like(t, device="cpu", pin_memory=True)
    cpu.copy_(t, non_blocking=True)
    return cpu


@torch.no_grad()
def _build_layer_cache(model, input_ids_list, attention_mask_list, layer_config,
                       return_loss: bool = False):
    """Run one full forward pass per sample under `layer_config` and return
    the intermediate hidden states at every layer boundary.

    Parameters
    ----------
    return_loss : bool
        If True, also compute cross-entropy loss (combining baseline forward
        with cache building into a single pass) and return (cache, avg_loss).
        If False (default), return cache only.

    Returns
    -------
    cache : list[dict]  (or (cache, float) when return_loss=True)
        One dict per sample.  cache[i]["h"][l] is the hidden state **before**
        layer l (i.e. after layer l-1), shape [1, S_i, H], stored on pinned CPU.
        cache[i]["f_topk_before"][l] is the value of _state["current_f_topk"]
        just before layer l ran (needed to seed S-layers in partial forwards).
        cache[i]["pos_emb"]  : (cos, sin) precomputed for that sample, on CPU.
        cache[i]["mask_4d"]  : 4-D additive causal+padding mask on CPU (or None).
        cache[i]["input_ids"]: original [1, S_i] token ids.
        cache[i]["attn_mask"]: original [1, S_i] 2-D attention mask.
    """
    _state["mode"] = "indexcache"
    _state["layer_config"] = layer_config

    backbone   = model.model
    num_layers = len(backbone.layers)
    CHUNK_LM   = 4096
    cache      = []
    total_loss   = 0.0
    total_tokens = 0

    for input_ids, attn_mask in zip(input_ids_list, attention_mask_list):
        _state["current_f_topk"] = None
        device = input_ids.device

        # ---- Prepare inputs (mirrors Qwen3Model.forward prologue) ----
        inputs_embeds = backbone.embed_tokens(input_ids)           # [1, S, H]
        cache_position = torch.arange(inputs_embeds.shape[1], device=device)
        position_ids   = cache_position.unsqueeze(0)
        mask_4d        = backbone._update_causal_mask(
            attn_mask, inputs_embeds, cache_position, None, False
        )
        pos_emb = backbone.rotary_emb(inputs_embeds, position_ids) # (cos, sin)

        # ---- Walk through layers, snapshotting h and f_topk before each layer ----
        h = inputs_embeds
        h_snapshots: List[torch.Tensor] = []
        f_topk_snapshots: List[Optional[torch.Tensor]] = []

        for layer in backbone.layers[:num_layers]:
            # Snapshot BEFORE this layer runs (non-blocking D2H to pinned memory)
            h_snapshots.append(_pin(h.detach()))
            cur = _state["current_f_topk"]
            f_topk_snapshots.append(_pin(cur.detach()) if cur is not None else None)

            layer_out = layer(
                h,
                attention_mask=mask_4d,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=pos_emb,
            )
            h = layer_out[0]

        cache.append({
            "h":              h_snapshots,
            "f_topk_before":  f_topk_snapshots,
            "pos_emb":        (_pin(pos_emb[0]), _pin(pos_emb[1])),
            "mask_4d":        _pin(mask_4d) if mask_4d is not None else None,
            "input_ids":      input_ids,
            "attn_mask":      attn_mask,
        })

        if return_loss:
            h = backbone.norm(h)                       # [1, S, H]
            shift_h      = h[:, :-1, :].squeeze(0)    # [S-1, H]
            shift_labels = input_ids[:, 1:].squeeze(0)
            shift_h, shift_labels = _apply_loss_window(shift_h, shift_labels)
            batch_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
            for c_start in range(0, shift_h.shape[0], CHUNK_LM):
                c_end    = min(c_start + CHUNK_LM, shift_h.shape[0])
                logits_c = model.lm_head(shift_h[c_start:c_end])
                batch_loss += Func.cross_entropy(
                    logits_c, shift_labels[c_start:c_end], reduction="sum"
                ).double()
            total_loss   += batch_loss.item()
            total_tokens += shift_labels.shape[0]

    # Ensure all non-blocking D2H copies are complete before returning
    torch.cuda.synchronize()

    _state["mode"] = "full"

    if return_loss:
        if _is_dist():
            stats = torch.tensor([total_loss, total_tokens], dtype=torch.float64,
                                 device=torch.cuda.current_device())
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss   = stats[0].item()
            total_tokens = int(stats[1].item())
        avg_loss = total_loss / max(total_tokens, 1)
        return cache, avg_loss

    return cache


@torch.no_grad()
def _update_layer_cache(model, layer_cache, input_ids_list, _attention_mask_list,
                        changed_layer: int, new_config: dict):
    """Incrementally update the layer cache after one layer is flipped F→S.

    Only layers ``changed_layer .. num_layers-1`` are recomputed per sample;
    h[0..changed_layer-1] and f_topk_before[0..changed_layer-1] remain valid.

    Note: h[changed_layer] itself is also unchanged (it is the output of the
    unchanged layer changed_layer-1), so we start the recompute FROM that layer
    (using the cached h[changed_layer] as input) and update h[changed_layer+1..].

    Modifies ``layer_cache`` in-place.
    """
    _state["mode"] = "indexcache"
    _state["layer_config"] = new_config

    backbone   = model.model
    num_layers = len(backbone.layers)

    for entry, input_ids in zip(layer_cache, input_ids_list):
        device       = input_ids.device
        # Restore the persistent positional/mask tensors from CPU
        cos_cpu, sin_cpu = entry["pos_emb"]
        pos_emb      = (cos_cpu.to(device), sin_cpu.to(device))
        mask_4d_cpu  = entry["mask_4d"]
        mask_4d      = mask_4d_cpu.to(device) if mask_4d_cpu is not None else None
        cache_pos    = torch.arange(input_ids.shape[1], device=device)
        position_ids = cache_pos.unsqueeze(0)

        # Load h[changed_layer] (unchanged) as starting point.
        # h[changed_layer] and f_topk_before[changed_layer] are already valid in
        # the cache (unchanged layers 0..changed_layer-1 produced the same values).
        # We start recomputing from changed_layer so the loop overwrites
        # h[changed_layer..L-1] and f_topk_before[changed_layer..L-1] correctly.
        h = entry["h"][changed_layer].to(device)
        _state["current_f_topk"] = (
            entry["f_topk_before"][changed_layer].to(device)
            if entry["f_topk_before"][changed_layer] is not None else None
        )

        for l_idx in range(changed_layer, num_layers):
            # Snapshot BEFORE this layer runs.
            # For l_idx == changed_layer the values are the same as before;
            # for l_idx > changed_layer they reflect the updated config.
            entry["h"][l_idx]             = _pin(h.detach())
            cur = _state["current_f_topk"]
            entry["f_topk_before"][l_idx] = _pin(cur.detach()) if cur is not None else None

            layer_out = backbone.layers[l_idx](
                h,
                attention_mask=mask_4d,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_pos,
                position_embeddings=pos_emb,
            )
            h = layer_out[0]

    torch.cuda.synchronize()
    _state["mode"] = "full"



@torch.no_grad()
def _eval_loss_candidates_batched(model, layer_cache, candidates, current_config):
    """Evaluate all candidate layers in one pass, with a SINGLE all_reduce.

    For each candidate c in `candidates`, tentatively sets layer c -> "S"
    and runs _eval_loss_from_cache locally (no all_reduce inside).
    After all candidates are evaluated, one all_reduce aggregates all
    local (loss, tokens) pairs simultaneously, reducing the per-candidate
    all_reduce calls from N down to 1.

    Returns
    -------
    dict[int -> float]  : global average per-token loss for each candidate
    """
    backbone   = model.model
    num_layers = len(backbone.layers)
    CHUNK_LM   = 4096

    # local_stats[i] = [total_loss, total_tokens] for candidate i
    local_stats = []

    for c in candidates:
        trial_config = dict(current_config)
        trial_config[c] = "S"

        _state["mode"] = "indexcache"
        _state["layer_config"] = trial_config

        tl = 0.0
        tt = 0
        for entry in layer_cache:
            input_ids = entry["input_ids"]
            device    = input_ids.device
            cos_cpu, sin_cpu = entry["pos_emb"]
            pos_emb   = (cos_cpu.to(device), sin_cpu.to(device))
            mask_4d_cpu = entry["mask_4d"]
            mask_4d   = mask_4d_cpu.to(device) if mask_4d_cpu is not None else None
            cache_pos = torch.arange(input_ids.shape[1], device=device)
            position_ids = cache_pos.unsqueeze(0)

            f_topk_cpu = entry["f_topk_before"][c]
            _state["current_f_topk"] = f_topk_cpu.to(device) if f_topk_cpu is not None else None

            h = entry["h"][c].to(device)
            for i in range(c, num_layers):
                layer_out = backbone.layers[i](
                    h,
                    attention_mask=mask_4d,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_pos,
                    position_embeddings=pos_emb,
                )
                h = layer_out[0]

            h = backbone.norm(h)
            shift_h      = h[:, :-1, :].squeeze(0)
            shift_labels = input_ids[:, 1:].squeeze(0)
            shift_h, shift_labels = _apply_loss_window(shift_h, shift_labels)

            batch_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
            for cs in range(0, shift_h.shape[0], CHUNK_LM):
                ce = min(cs + CHUNK_LM, shift_h.shape[0])
                logits_c = model.lm_head(shift_h[cs:ce])
                batch_loss += Func.cross_entropy(
                    logits_c, shift_labels[cs:ce], reduction="sum"
                ).double()
            tl += batch_loss.item()
            tt += shift_labels.shape[0]

            log.debug(f"  _eval_loss_batched candidate={c}: tokens={shift_labels.shape[0]}, loss={batch_loss.item():.4f}")

        local_stats.append([tl, tt])

    _state["mode"] = "full"

    # ---- Single all_reduce for ALL candidates at once ----
    n = len(candidates)
    if _is_dist() and n > 0:
        flat = torch.tensor(
            [x for pair in local_stats for x in pair],
            dtype=torch.float64,
            device=torch.cuda.current_device(),
        )  # shape: [2*n]
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        result = {}
        for i, c in enumerate(candidates):
            tl = flat[2 * i].item()
            tt = int(flat[2 * i + 1].item())
            result[c] = tl / max(tt, 1)
    else:
        result = {}
        for i, c in enumerate(candidates):
            tl, tt = local_stats[i]
            result[c] = tl / max(tt, 1)

    return result


@torch.no_grad()
def _eval_loss_from_cache(model, layer_cache, start_layer, layer_config,
                          update_cache: bool = False):
    """Compute per-token loss starting the forward pass from `start_layer`,
    using cached hidden states for layers 0..(start_layer-1).

    For each candidate flip (layer start_layer -> S), only layers
    start_layer..num_layers-1 are recomputed.

    Parameters
    ----------
    start_layer  : first layer to recompute (0-indexed)
    layer_config : full {layer_idx -> "F"/"S"} config for the trial
    update_cache : if True, write the computed h[start_layer..L-1] and
                   f_topk_before[start_layer..L-1] back into layer_cache
                   in-place (non-blocking D2H to pinned memory).
                   Use this for the winning candidate so that the next step
                   gets an up-to-date cache without a separate
                   _update_layer_cache call.
    """
    _state["mode"] = "indexcache"
    _state["layer_config"] = layer_config

    backbone   = model.model
    num_layers = len(backbone.layers)
    CHUNK_LM   = 4096

    total_loss   = 0.0
    total_tokens = 0

    for entry in layer_cache:
        input_ids   = entry["input_ids"]
        device      = input_ids.device
        cos_cpu, sin_cpu = entry["pos_emb"]
        pos_emb     = (cos_cpu.to(device), sin_cpu.to(device))
        mask_4d_cpu = entry["mask_4d"]
        mask_4d     = mask_4d_cpu.to(device) if mask_4d_cpu is not None else None
        cache_pos   = torch.arange(input_ids.shape[1], device=device)
        position_ids = cache_pos.unsqueeze(0)

        # Restore f_topk to the value that was active just before start_layer
        # in the cached forward (needed when start_layer is an S layer).
        f_topk_cpu = entry["f_topk_before"][start_layer]
        _state["current_f_topk"] = f_topk_cpu.to(device) if f_topk_cpu is not None else None

        # Start from the cached hidden state at start_layer
        h = entry["h"][start_layer].to(device)

        for i in range(start_layer, num_layers):
            if update_cache:
                # Snapshot BEFORE this layer: same logic as _build_layer_cache.
                entry["h"][i]             = _pin(h.detach())
                cur = _state["current_f_topk"]
                entry["f_topk_before"][i] = _pin(cur.detach()) if cur is not None else None

            layer_out = backbone.layers[i](
                h,
                attention_mask=mask_4d,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_pos,
                position_embeddings=pos_emb,
            )
            h = layer_out[0]

        h = backbone.norm(h)                         # [1, S, H]
        shift_h      = h[:, :-1, :].squeeze(0)       # [S-1, H]
        shift_labels = input_ids[:, 1:].squeeze(0)   # [S-1]

        if update_cache:
            # Only needed cache snapshots — skip lm_head / cross_entropy / all_reduce.
            continue

        shift_h, shift_labels = _apply_loss_window(shift_h, shift_labels)

        # lm_head in chunks to limit peak memory
        batch_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float64)
        for c_start in range(0, shift_h.shape[0], CHUNK_LM):
            c_end       = min(c_start + CHUNK_LM, shift_h.shape[0])
            logits_c    = model.lm_head(shift_h[c_start:c_end])
            batch_loss += Func.cross_entropy(
                logits_c, shift_labels[c_start:c_end], reduction="sum"
            ).double()

        total_loss   += batch_loss.item()
        total_tokens += shift_labels.shape[0]
        log.debug(f"  _eval_loss_from_cache start_layer={start_layer}: "
                  f"tokens={shift_labels.shape[0]}, loss={batch_loss.item():.4f}")

    if update_cache:
        torch.cuda.synchronize()   # ensure all non-blocking D2H copies are done
        _state["mode"] = "full"
        return 0.0                 # loss not needed; caller discards return value

    if _is_dist():
        stats = torch.tensor([total_loss, total_tokens], dtype=torch.float64,
                             device=torch.cuda.current_device())
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss   = stats[0].item()
        total_tokens = int(stats[1].item())

    _state["mode"] = "full"
    return total_loss / max(total_tokens, 1)


# ==============================================================
# Loss-based greedy search  
# ==============================================================
def greedy_search_loss(model, input_ids_list, attention_mask_list, num_layers,
                       retention_ratio, k):
    """
    Paper Algorithm 1:
      Start from all-F.  For K steps, flip one F->S that minimises LM loss.
      Layer 0 is always F.

    Optimisation (space-time trade-off):
      1. Baseline forward + initial cache are combined into ONE forward pass
         (return_loss=True).  No extra forward for the baseline.
      2. After each step, the cache is updated INCREMENTALLY: only layers
         best_layer..L-1 are recomputed (h[0..best_layer-1] are unchanged
         because those layers' configs did not change).
      3. D2H copies use non-blocking transfers to pinned memory to overlap
         with GPU compute.
    """
    target_f = max(1, int(round(num_layers * retention_ratio)))
    target_s = num_layers - target_f          # K steps to perform

    f_set: Set[int] = set(range(num_layers))  # start: all F
    _state["k"] = k

    # ---- Baseline (all F): build cache AND compute loss in one forward pass ----
    all_f_config = {i: "F" for i in range(num_layers)}
    t0 = time.time()
    layer_cache, base_loss = _build_layer_cache(
        model, input_ids_list, attention_mask_list, all_f_config, return_loss=True
    )
    log.info(f"  Baseline loss (all F): {base_loss:.6f}  ({time.time()-t0:.1f}s)")

    search_log = []
    search_t0  = time.time()

    for step in range(target_s):
        candidates = sorted(f_set - {0})       # layer 0 is never flipped
        log.info(f"  Step {step+1}/{target_s}: evaluating {len(candidates)} candidates ...")
        step_t0 = time.time()

        # Cache is already up-to-date under current_config — no separate build needed.
        current_config = {i: "F" if i in f_set else "S" for i in range(num_layers)}
        log.debug(f"    Layer cache ready ({_gpu_mem_str()})")

        # Evaluate all candidates with a single all_reduce (instead of one per candidate)
        candidate_losses = _eval_loss_candidates_batched(
            model, layer_cache, candidates, current_config
        )
        best_layer = min(candidate_losses, key=candidate_losses.__getitem__)
        best_loss  = candidate_losses[best_layer]

        step_elapsed_eval = time.time() - step_t0
        f_set.remove(best_layer)
        current_config[best_layer] = "S"

        for c in sorted(candidate_losses):
            marker = " <-- best" if c == best_layer else ""
            log.debug(f"    layer {c:>2d}: loss={candidate_losses[c]:.6f}{marker}")

        # ---- Incremental cache update: re-run best_layer's forward with
        #      update_cache=True so the already-computed hidden states are
        #      written back to the cache — no separate _update_layer_cache pass.
        upd_t0 = time.time()
        _eval_loss_from_cache(
            model, layer_cache, best_layer, current_config, update_cache=True
        )
        upd_elapsed = time.time() - upd_t0
        step_elapsed = time.time() - step_t0

        log.info(f"    -> flipped layer {best_layer:>2d} to S  |  loss={best_loss:.6f}  "
                 f"|  |F|={len(f_set)}  |  eval={step_elapsed_eval:.1f}s  "
                 f"upd={upd_elapsed:.1f}s  total={step_elapsed:.1f}s  |  {_gpu_mem_str()}")

        search_log.append({
            "step": step + 1,
            "removed_layer": best_layer,
            "loss": best_loss,
            "remaining_f": sorted(f_set),
            "candidate_losses": {str(k): v for k, v in candidate_losses.items()},
            "step_time_s": round(step_elapsed, 2),
        })

    total_elapsed = time.time() - search_t0
    log.info(f"  Greedy search done in {total_elapsed:.1f}s  ({total_elapsed/60:.1f}min)")
    _state["_search_log"] = search_log
    return sorted(f_set)


# ==============================================================
# Beam search helpers
# ==============================================================

def _clone_cache(layer_cache):
    """Shallow-clone a layer cache so each beam has an independent copy.

    We only copy the per-layer lists (h, f_topk_before), not the tensors
    themselves.  When _eval_loss_from_cache runs with update_cache=True it
    replaces list elements via assignment (_pin() returns a new tensor), so
    the parent beam's list is never mutated — the two caches diverge lazily
    at assignment time with zero extra tensor copies.
    """
    return [
        {
            "h":             list(e["h"]),
            "f_topk_before": list(e["f_topk_before"]),
            "pos_emb":   e["pos_emb"],    # read-only, safe to share
            "mask_4d":   e["mask_4d"],    # read-only
            "input_ids": e["input_ids"],  # read-only
            "attn_mask": e["attn_mask"],  # read-only
        }
        for e in layer_cache
    ]


@torch.no_grad()
def _eval_loss_multi_beam_batched(model, beams_data):
    """Evaluate all candidates for all beams, with a SINGLE all_reduce.

    Parameters
    ----------
    beams_data : list of (layer_cache, candidates, current_config)
        One tuple per active beam.

    Returns
    -------
    list of dict[int -> float]
        One loss dict per beam, mapping candidate layer -> global avg loss.
    """
    backbone   = model.model
    num_layers = len(backbone.layers)
    CHUNK_LM   = 4096

    # all_local_stats[beam_idx][cand_idx] = [tl, tt]
    all_local_stats = []

    for layer_cache, candidates, current_config in beams_data:
        beam_stats = []
        for c in candidates:
            trial_config = dict(current_config)
            trial_config[c] = "S"
            _state["mode"]        = "indexcache"
            _state["layer_config"] = trial_config

            tl, tt = 0.0, 0
            for entry in layer_cache:
                input_ids = entry["input_ids"]
                device    = input_ids.device
                cos_cpu, sin_cpu = entry["pos_emb"]
                pos_emb   = (cos_cpu.to(device), sin_cpu.to(device))
                mask_4d_cpu = entry["mask_4d"]
                mask_4d   = mask_4d_cpu.to(device) if mask_4d_cpu is not None else None
                cache_pos = torch.arange(input_ids.shape[1], device=device)
                position_ids = cache_pos.unsqueeze(0)

                f_topk_cpu = entry["f_topk_before"][c]
                _state["current_f_topk"] = f_topk_cpu.to(device) if f_topk_cpu is not None else None

                h = entry["h"][c].to(device)
                for i in range(c, num_layers):
                    layer_out = backbone.layers[i](
                        h,
                        attention_mask=mask_4d,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=cache_pos,
                        position_embeddings=pos_emb,
                    )
                    h = layer_out[0]

                h = backbone.norm(h)
                shift_h      = h[:, :-1, :].squeeze(0)
                shift_labels = input_ids[:, 1:].squeeze(0)
                shift_h, shift_labels = _apply_loss_window(shift_h, shift_labels)

                batch_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
                for cs in range(0, shift_h.shape[0], CHUNK_LM):
                    ce = min(cs + CHUNK_LM, shift_h.shape[0])
                    logits_c = model.lm_head(shift_h[cs:ce])
                    batch_loss += Func.cross_entropy(
                        logits_c, shift_labels[cs:ce], reduction="sum"
                    ).double()
                tl += batch_loss.item()
                tt += shift_labels.shape[0]

            beam_stats.append([tl, tt])
        all_local_stats.append(beam_stats)

    _state["mode"] = "full"

    # ---- Single all_reduce for ALL beams × ALL candidates ----
    flat_list = [x for beam_stats in all_local_stats for tl, tt in beam_stats for x in (tl, tt)]
    if _is_dist() and flat_list:
        flat = torch.tensor(flat_list, dtype=torch.float64,
                            device=torch.cuda.current_device())
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat = flat.tolist()
    else:
        flat = flat_list

    # ---- Unpack into per-beam dicts ----
    results = []
    idx = 0
    for (_, candidates, _) in beams_data:
        result = {}
        for c in candidates:
            tl = flat[idx];  tt = int(flat[idx + 1]);  idx += 2
            result[c] = tl / max(tt, 1)
        results.append(result)
    return results


def beam_search_loss(model, input_ids_list, attention_mask_list, num_layers,
                     retention_ratio, k, beam_size=4):
    """Beam search variant of greedy_search_loss.

    At each step, instead of keeping only the single best flip, we keep the
    top-`beam_size` (beam, layer) pairs by loss.  This explores a richer
    search space at the cost of beam_size× compute per step.

    All beams' candidates are evaluated with a SINGLE all_reduce per step
    (via _eval_loss_multi_beam_batched), so the communication overhead
    does not scale with beam_size.
    """
    target_f = max(1, int(round(num_layers * retention_ratio)))
    target_s = num_layers - target_f

    all_f_config = {i: "F" for i in range(num_layers)}
    _state["k"] = k

    # ---- Baseline: build initial cache and loss in one forward ----
    t0 = time.time()
    cache0, base_loss = _build_layer_cache(
        model, input_ids_list, attention_mask_list, all_f_config, return_loss=True
    )
    log.info(f"  Baseline loss (all F): {base_loss:.6f}  ({time.time()-t0:.1f}s)")

    # Each beam is a dict with keys: config, f_set, loss, cache, history
    beams = [{
        "config":  dict(all_f_config),
        "f_set":   set(range(num_layers)),
        "loss":    base_loss,
        "cache":   cache0,
        "history": [],
    }]

    search_log = []
    search_t0  = time.time()

    for step in range(target_s):
        # Build (cache, candidates, config) tuples for every active beam
        beams_data = [
            (b["cache"], sorted(b["f_set"] - {0}), b["config"])
            for b in beams
        ]
        total_cands = sum(len(c) for _, c, _ in beams_data)
        log.info(f"  Step {step+1}/{target_s}: {len(beams)} beam(s), "
                 f"{total_cands} total candidates ...")
        step_t0 = time.time()

        # Evaluate all candidates across all beams — one all_reduce
        all_cand_losses = _eval_loss_multi_beam_batched(model, beams_data)

        # Flatten to (loss, beam_idx, layer) and keep top beam_size
        flat = [
            (loss, b_idx, c)
            for b_idx, cand_losses in enumerate(all_cand_losses)
            for c, loss in cand_losses.items()
        ]
        flat.sort(key=lambda x: x[0])
        # Global top-beam_size selection; skip candidates that would produce
        # an f_set already chosen (dedup only, no per-parent diversity guarantee).
        seen_configs: set = set()
        top: list = []
        for loss, b_idx, c in flat:
            if len(top) >= beam_size:
                break
            new_f_set = beams[b_idx]["f_set"] - {c}
            key = frozenset(new_f_set)
            if key not in seen_configs:
                seen_configs.add(key)
                top.append((loss, b_idx, c))

        step_elapsed_eval = time.time() - step_t0

        # Log candidate summary (debug)
        top_set = {(bi, cc) for _, bi, cc in top}
        for b_idx, cand_losses in enumerate(all_cand_losses):
            for c in sorted(cand_losses):
                marker = " <-- kept" if (b_idx, c) in top_set else ""
                log.debug(f"    beam{b_idx} layer {c:>2d}: loss={cand_losses[c]:.6f}{marker}")

        # Build new beams: clone parent cache and update incrementally
        upd_t0 = time.time()
        new_beams = []
        for loss, b_idx, c in top:
            parent     = beams[b_idx]
            new_config = dict(parent["config"]);  new_config[c] = "S"
            new_f_set  = parent["f_set"] - {c}
            new_cache  = _clone_cache(parent["cache"])
            _eval_loss_from_cache(model, new_cache, c, new_config, update_cache=True)
            new_beams.append({
                "config":  new_config,
                "f_set":   new_f_set,
                "loss":    loss,
                "cache":   new_cache,
                "history": parent["history"] + [{"step": step + 1, "layer": c, "loss": loss}],
            })
        upd_elapsed  = time.time() - upd_t0
        step_elapsed = time.time() - step_t0

        log.info(f"    Top-{len(top)} beams:")
        for rank_i, (loss, b_idx, c) in enumerate(top):
            log.info(f"      [{rank_i}] loss={loss:.6f}  flip=layer{c:>2d}  parent=beam{b_idx}")
        log.info(f"    eval={step_elapsed_eval:.1f}s  upd={upd_elapsed:.1f}s  "
                 f"total={step_elapsed:.1f}s  |  {_gpu_mem_str()}")

        search_log.append({
            "step":         step + 1,
            "top_flips":    [{"beam": b_idx, "layer": c, "loss": loss} for loss, b_idx, c in top],
            "step_time_s":  round(step_elapsed, 2),
        })
        beams = new_beams

    total_elapsed = time.time() - search_t0
    best_beam = beams[0]   # already sorted by loss (top was sorted ascending)
    log.info(f"  Beam search done in {total_elapsed:.1f}s  ({total_elapsed/60:.1f}min)")
    log.info(f"  Best beam loss: {best_beam['loss']:.6f}")
    log.info(f"  Best beam history: "
             + " -> ".join(f"layer{h['layer']}(loss={h['loss']:.4f})" for h in best_beam["history"]))

    _state["_search_log"] = search_log
    return sorted(best_beam["f_set"])


def _nearest_preceding_f(s, sorted_f):
    prev = None
    for f in sorted_f:
        if f <= s:
            prev = f
        else:
            break
    return prev if prev is not None else sorted_f[0]



# ==============================================================
# Build S -> F mapping
# ==============================================================
def build_fs_mapping(f_layers: List[int], num_layers: int):
    f_set = set(f_layers)
    sorted_f = sorted(f_layers)
    mapping = {}
    for s in range(num_layers):
        if s in f_set:
            continue
        mapping[s] = _nearest_preceding_f(s, sorted_f)
    return mapping


# ==============================================================
# Prepare calibration data
# ==============================================================
def prepare_calibration_data(tokenizer, data_path, num_samples, max_seq_len, device):
    """Load and tokenise calibration data, then shard across distributed ranks."""
    log.info(f"Loading calibration data from {data_path} ...")

    if data_path.endswith(".jsonl"):
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if "content" in obj.keys():
                    texts.append(obj["content"])
                elif "completion" in obj.keys():
                    texts.append(obj["completion"])
                else:
                    raise ValueError(f"Invalid JSON object keys: {obj.keys()}")
    else:
        df = pd.read_parquet(data_path)
        texts = df["prompt"].tolist()

    # Shuffle and take num_samples
    random.shuffle(texts)
    texts = texts[:num_samples]

    # Tokenise all samples (on main rank for logging, but all ranks need data)
    all_input_ids = []
    all_attention_masks = []
    seq_lengths = []
    iter_texts = tqdm(texts, desc="Tokenising") if _is_main() else texts
    for text in iter_texts:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,       # chat template already adds them
        )
        all_input_ids.append(enc.input_ids.to(device))
        all_attention_masks.append(enc.attention_mask.to(device))
        seq_lengths.append(enc.input_ids.shape[1])

    log.info(f"  {len(all_input_ids)} samples tokenised (total)")
    log.info(f"  Seq lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
             f"mean={np.mean(seq_lengths):.0f}, median={np.median(seq_lengths):.0f}")
    log.debug(f"  First 10 seq lengths: {seq_lengths[:10]}")

    # --- Shard across ranks (load-balanced by total tokens) ---
    rank = _rank()
    world = _world_size()
    # Sort by length descending, then round-robin assign ranks.
    # This is "longest-first + round-robin" (serpentine assignment):
    # rank 0 gets the longest, rank 1 the 2nd longest, ..., then wraps back.
    # Result: each rank's total token count is nearly equal, minimising
    # the load-imbalance stall at the all_reduce barrier.
    sorted_indices = sorted(range(len(all_input_ids)),
                            key=lambda i: seq_lengths[i], reverse=True)
    local_indices = sorted_indices[rank::world]
    local_input_ids = [all_input_ids[i] for i in local_indices]
    local_attention_masks = [all_attention_masks[i] for i in local_indices]

    local_tokens = sum(seq_lengths[i] for i in local_indices)
    log.info(f"  Rank {rank}/{world}: {len(local_input_ids)} local samples "
             f"(indices: {local_indices[:5]}{'...' if len(local_indices) > 5 else ''}) "
             f"total_tokens={local_tokens}")

    return local_input_ids, local_attention_masks, seq_lengths


# ==============================================================
# Main
# ==============================================================
def parse_args():
    p = argparse.ArgumentParser(description="IndexCache offline calibration")
    p.add_argument("--model_name_or_path", type=str,
                   default="/root/paddlejob/share-storage/gpfs/system-public/tzx/model/Qwen/Qwen3-4B")
    p.add_argument("--data_path", type=str,
                   default="/root/paddlejob/share-storage/gpfs/system-public/tzx/model/dataset/"
                           "Kurt232/Sampled-Laser-Dataset-Qwen-3-4B-Thinking/data/"
                           "train-00000-of-00001.parquet")
    p.add_argument("--output_dir", type=str, default="./calibration_output")
    p.add_argument("--run_timestamp", type=str, default="",
                   help="Timestamp string injected by the launch shell script "
                        "(e.g. 20260414_160138). When provided, the Python log "
                        "file uses this timestamp so all files from the same run "
                        "share identical names.")
    p.add_argument("--num_samples", type=int, default=16,
                   help="Number of calibration samples (mini-batches of size 1)")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--k", type=str, default="128",
                   help="Top-k budget: integer for fixed k, or 'auto' for seq_len//4 per sample")
    p.add_argument("--retention_ratio", type=float, default=0.25,
                   help="Fraction of layers to keep as F (e.g. 0.25 = 1/4)")
    p.add_argument("--search_mode", type=str, default="loss",
                   choices=["loss"],
                   help="loss = paper's primary method;")
    p.add_argument("--beam_size", type=int, default=1,
                   help="Beam size for layer search (default 1 = greedy). "
                        "beam_size>1 keeps the top-N flips at each step and "
                        "explores beam_size paths in parallel, at the cost of "
                        "beam_size× compute and memory per step.")
    p.add_argument("--loss_last_frac", type=float, default=1.0,
                   help="Fraction of tokens (from sequence end) used to compute LM loss. "
                        "1.0 = all tokens; 0.5 = last half. "
                        "Sparse attention diverges from dense mainly on later tokens, "
                        "so a smaller value (e.g. 0.5) gives higher signal and faster eval.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def _set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    _set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Distributed init ----
    _setup_dist()
    rank = _rank()
    world = _world_size()

    # ---- Set up file logging (main rank only writes log file) ----
    log_path = None
    if _is_main():
        # e.g. n64_k512_r25_beam2_frac50_w32
        k_str   = str(args.k)
        r_str   = str(int(args.retention_ratio * 100))
        frac_str = str(int(args.loss_last_frac * 100))
        beam_str = str(args.beam_size)
        w_str   = str(_world_size())
        cfg_tag = f"n{args.num_samples}_k{k_str}_r{r_str}_beam{beam_str}_frac{frac_str}_w{w_str}"
        log_path = _add_file_handler(args.output_dir, cfg_tag, ts=args.run_timestamp)
    log.info("=" * 60)
    log.info("  IndexCache Offline Calibration")
    log.info("=" * 60)
    log.info(f"Arguments: {vars(args)}")
    log.info(f"Distributed: rank={rank}, world_size={world}")
    log.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        log.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log.info(f"  GPU{i}: {torch.cuda.get_device_name(i)}  "
                     f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
    global_t0 = time.time()

    # ---- Load model: each rank loads on its own GPU ----
    log.info("Loading model ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if _is_dist():
        # Each rank loads the full model on its own GPU (local_rank = GPU index on this node)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        model = transformers.Qwen3ForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
    else:
        # Single-GPU fallback
        model = transformers.Qwen3ForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        device = next(model.parameters()).device

    model.eval()
    num_layers = model.config.num_hidden_layers
    log.info(f"  Model loaded in {time.time()-t0:.1f}s: {num_layers} layers, "
             f"device={device}, rank={rank}, {_gpu_mem_str()}")

    # ---- Monkey-patch attention forward ----
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = (
        indexcache_attention_forward
    )
    _state["k"] = args.k if args.k == "auto" else int(args.k)
    _state["loss_last_frac"] = max(0.01, min(1.0, args.loss_last_frac))

    # ---- Prepare calibration data (sharded across ranks) ----
    _barrier()
    t0 = time.time()
    input_ids_list, attention_mask_list, all_seq_lengths = prepare_calibration_data(
        tokenizer, args.data_path, args.num_samples, args.max_seq_len, device
    )
    log.info(f"  Data prepared in {time.time()-t0:.1f}s")
    # ---- Log effective k (use all_seq_lengths from full dataset) ----
    seq_lengths = all_seq_lengths
    if _state["k"] == "auto":
        example_ks = [max(1, sl // 4) for sl in seq_lengths[:5]]
        log.info(f"  k=auto (seq_len//4 per sample). "
                 f"Seq lengths: min={min(seq_lengths)}, median={int(np.median(seq_lengths))}, max={max(seq_lengths)}. "
                 f"Example effective k: {example_ks}")
    else:
        fixed_k = _state["k"]
        min_seq = min(seq_lengths)
        if fixed_k >= min_seq:
            log.warning(f"  k={fixed_k} >= shortest seq_len={min_seq}, "
                        f"sparse mask will have NO effect on short samples. "
                        f"Consider using --k auto or a smaller value.")
        log.info(f"  k={fixed_k} (fixed). Seq lengths: min={min_seq}, "
                 f"median={int(np.median(seq_lengths))}, max={max(seq_lengths)}")

    # ---- Run search ----
    _barrier()
    if args.search_mode == "loss":
        log.info(f"\n{'='*60}")
        if args.beam_size > 1:
            log.info(f"Loss-based beam search  (beam_size={args.beam_size}, retention={args.retention_ratio})")
            log.info(f"{'='*60}")
            f_layers = beam_search_loss(
                model, input_ids_list, attention_mask_list,
                num_layers, args.retention_ratio, _state["k"],
                beam_size=args.beam_size,
            )
        else:
            log.info(f"Loss-based greedy search  (retention={args.retention_ratio})")
            log.info(f"{'='*60}")
            f_layers = greedy_search_loss(
                model, input_ids_list, attention_mask_list,
                num_layers, args.retention_ratio, _state["k"],
            )

    # ---- Build mapping & stats ----
    s_to_f = build_fs_mapping(f_layers, num_layers)
    s_layers = [i for i in range(num_layers) if i not in f_layers]

    # ---- Validate: measure final LM loss ----
    log.info("\nValidating final configuration ...")
    t0 = time.time()
    final_config = {i: "F" for i in f_layers}
    for s, f in s_to_f.items():
        final_config[s] = "S"
    final_loss = eval_loss(model, input_ids_list, attention_mask_list, final_config)

    all_f_config = {i: "F" for i in range(num_layers)}
    baseline_loss = eval_loss(model, input_ids_list, attention_mask_list, all_f_config)
    log.info(f"  Validation done in {time.time()-t0:.1f}s")

    # ---- Report & Save (main rank only) ----
    total_elapsed = time.time() - global_t0
    log.info(f"\n{'='*60}")
    log.info(f"  RESULTS")
    log.info(f"{'='*60}")
    log.info(f"  F layers ({len(f_layers):>2d}): {f_layers}")
    log.info(f"  S layers ({len(s_layers):>2d}): {s_layers}")
    log.info(f"  S -> F mapping: {s_to_f}")
    log.info(f"  Baseline loss (all F) : {baseline_loss:.6f}")
    log.info(f"  IndexCache loss       : {final_loss:.6f}")
    log.info(f"  Loss delta            : {final_loss - baseline_loss:+.6f}")
    log.info(f"  Total elapsed         : {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    log.info(f"  World size            : {world}")
    log.info(f"  {_gpu_mem_str()}")
    log.info(f"{'='*60}")

    if _is_main():
        # ---- Save configuration ----
        config = {
            "f_layers": f_layers,
            "s_layers": s_layers,
            "s_to_f_mapping": {int(k): int(v) for k, v in s_to_f.items()},
            "k": _state["k"],              # "auto" or int
            "retention_ratio": args.retention_ratio,
            "num_layers": num_layers,
            "search_mode": args.search_mode,
            "baseline_loss": baseline_loss,
            "indexcache_loss": final_loss,
            "loss_delta": final_loss - baseline_loss,
            "num_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "world_size": world,
            "total_time_s": round(total_elapsed, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save search log (per-step details)
        search_log = _state.get("_search_log", [])
        config["search_log"] = search_log

        out_json = os.path.join(args.output_dir, "indexcache_config.json")
        with open(out_json, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Config saved -> {out_json}")

        out_pkl = os.path.join(args.output_dir, "indexcache_config.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(config, f)
        log.info(f"Pickle saved -> {out_pkl}")
        log.info(f"Log saved   -> {log_path}")
        log.info("Done.")

    _cleanup_dist()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
