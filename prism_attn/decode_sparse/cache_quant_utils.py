"""
KC2CompressionCacheBuffered — true INT2 quantized KV cache

Utilizes the optimized Triton kernels for INT2 KV cache operations.
All layers now use a unified memory layout:
    k_q_buf  [B, H, max_S, D//4]         uint8      — per-token D//4 bytes contiguous
    k_sz_buf [B, H, max_S, (D//gs)*2]    bfloat16   — combined scales and zero-points

The `int2_attention_score` kernel natively handles GQA (Grouped-Query Attention) 
and internally manages the preservation of sink and recent tokens.
"""

from typing import Optional
import torch

from prism_attn.kernels.quant.int2_quant import int2_quantize
from prism_attn.kernels.quant.int2_qk_maxreduce import int2_attention_score, int2_attention_score_indexed


class KC2CompressionCacheBuffered:
    """
    KV cache with true INT2 quantization. All tokens are compressed.

    Parameters
    ----------
    num_layers    : number of transformer layers
    group_size    : int2 quantization group size along head_dim
    max_seq_len   : upper bound on sequence length (for buffer pre-allocation)
    """

    def __init__(
        self,
        num_layers: int,
        group_size: int = 32,
        max_seq_len: int = 32768,
    ) -> None:
        self.num_layers      = num_layers
        self._group_size     = group_size
        self.max_seq_len     = max_seq_len
        self.max_compressed_len = max_seq_len

        self.k_q_buf  = [None] * num_layers
        self.k_sz_buf = [None] * num_layers

        self.k_comp_len  = [0] * num_layers   # #tokens currently stored
        self._buf_allocated = [False] * num_layers

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _alloc(self, layer_idx: int, bsz: int, kv_heads: int, head_dim: int, device):
        D  = head_dim
        NG = D // self._group_size
        mc = self.max_compressed_len

        # [B, H, max_S, D//4]
        self.k_q_buf[layer_idx]  = torch.empty(
            (bsz, kv_heads, mc, D // 4), dtype=torch.uint8, device=device
        )
        
        # [B, H, max_S, (D//gs)*2] -> scales and mins interleaved
        self.k_sz_buf[layer_idx] = torch.empty(
            (bsz, kv_heads, mc, NG * 2), dtype=torch.bfloat16, device=device
        )

        self.k_comp_len[layer_idx] = 0
        self._buf_allocated[layer_idx] = True

    def _ensure_alloc(self, layer_idx, bsz, kv_heads, head_dim, device):
        if not self._buf_allocated[layer_idx]:
            self._alloc(layer_idx, bsz, kv_heads, head_dim, device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        k: Optional[torch.Tensor] = None,
        is_decode: bool = False,
    ) -> torch.Tensor:
        if k is not None and not k.is_contiguous():
            k = k.contiguous()

        bsz, _, kv_heads, head_dim = k.shape
        self._ensure_alloc(layer_idx, bsz, kv_heads, head_dim, k.device)

        if not is_decode:
            # Prefill: write from column 0, return original k
            self.k_comp_len[layer_idx] = self._quant_write(layer_idx, k, col=0)
            return k
        else:
            # Decode: append at current comp_len
            comp = self.k_comp_len[layer_idx]
            if comp < self.max_compressed_len:
                self.k_comp_len[layer_idx] = self._quant_write(layer_idx, k, col=comp)
            return None

    def batch_select_indices(self, indices: torch.Tensor):
        for layer in range(self.num_layers):
            if self._buf_allocated[layer]:
                self.k_q_buf[layer]  = self.k_q_buf[layer][indices]
                self.k_sz_buf[layer] = self.k_sz_buf[layer][indices]

    # ------------------------------------------------------------------
    # Fused attention score (decode hot-path)
    # ------------------------------------------------------------------

    def compute_attn_scores(
        self,
        layer_idx: int,
        q: torch.Tensor,
        sink_size: int = 64,
        recent_size: int = 192,
    ) -> torch.Tensor:
        """
        Compute attention scores directly from INT2 storage.

        Parameters
        ----------
        q           : [B, H_q, D] full query (all GQA heads). Will be cast to bfloat16.
        sink_size   : first sink_size positions forced to +1e10
        recent_size : last recent_size positions forced to +1e10

        Returns [B, H_kv, S] bfloat16.
        """
        comp = self.k_comp_len[layer_idx]
        assert comp > 0, "compute_attn_scores called before any tokens are stored"

        # The new kernel expects Q to be [B, H_q, 1, D] and handles GQA internally.
        # Ensure it is bfloat16 to match the kernel assertion.
        q_bf16 = q.to(torch.bfloat16).unsqueeze(2) 

        # Call the new kernel
        scores = int2_attention_score(
            q=q_bf16,
            k_q=self.k_q_buf[layer_idx][:, :, :comp, :],
            k_sz=self.k_sz_buf[layer_idx][:, :, :comp, :],
            sink_size=sink_size,
            recent_size=recent_size,
            group_size=self._group_size
        )

        # Output is [B, H_kv, 1, S], we squeeze to [B, H_kv, S] to maintain compatibility
        return scores.squeeze(2)

    def compute_attn_scores_indexed(
        self,
        layer_idx: int,
        q: torch.Tensor,
        indices: torch.Tensor,
        sink_size: int = 64,
        recent_size: int = 192,        
    ) -> torch.Tensor:
        """
        Compute scores for a specific set of candidate token positions.

        Parameters
        ----------
        q           : [B, H_q, D] full query (all GQA heads). Will be cast to bfloat16.
        indices shape: [B, H_kv, E]  —  Top-K indices into the cached KV.

        Returns [B, H_kv, S] bfloat16.        
        """
        comp = self.k_comp_len[layer_idx]
        assert comp > 0, "compute_attn_scores called before any tokens are stored"

        # The new kernel expects Q to be [B, H_q, 1, D] and handles GQA internally.
        # Ensure it is bfloat16 to match the kernel assertion.
        q_bf16 = q.to(torch.bfloat16).unsqueeze(2) 

        # Call the new kernel
        scores = int2_attention_score_indexed(
            q=q_bf16,
            k_q=self.k_q_buf[layer_idx],    # 可以直接传整块缓存，Kernel 依靠 indices 进行精准寻址
            k_sz=self.k_sz_buf[layer_idx],
            indices=indices,
            comp=comp,
            sink_size=sink_size,
            recent_size=recent_size,
            group_size=self._group_size
        )
        # Output is [B, H_kv, 1, S], we squeeze to [B, H_kv, S] to maintain compatibility
        return scores.squeeze(2)

    # ------------------------------------------------------------------
    # Quantize + write
    # ------------------------------------------------------------------

    def _quant_write(self, layer_idx: int, k_slice: torch.Tensor, col: int) -> int:
        """
        Quantize k_slice and write directly into the cache buffers.

        k_slice : [B, S, H, D]  (any float dtype, will be cast to bfloat16 inside quantize)
        col     : starting token index in the S dimension of the buffers
        Returns : col + S  (next free position)
        """
        # The int2_quantize function expects bfloat16 tensors
        k_slice = k_slice.to(torch.bfloat16).contiguous()
        
        int2_quantize(
            k=k_slice,
            k_q_buf=self.k_q_buf[layer_idx],
            k_sz_buf=self.k_sz_buf[layer_idx],
            col=col,
            group_size=self._group_size,
        )
        return col + k_slice.shape[1]

