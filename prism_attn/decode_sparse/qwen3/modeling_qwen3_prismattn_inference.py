# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.generation import GenerationMixin

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
# from transformers.processing_utils import Unpack
from transformers.utils import (
    logging,
)
from .configuration_qwen3_prismattn import PrismQwen3Config
from ...utils import BaseModelOutputWithPastAndCache, CausalLMOutputWithPastAndCache
from ..attn_gate_inf import C2AttnGate
import copy, math, os, socket
from einops import rearrange
from ..attention_forward_sparse import token_sparse_flash_attention_forward
from ..attention_forward_dense import dense_flash_attention_forward
from ...modules.layernorm import RMSNorm
from flash_attn.layers.rotary import apply_rotary_emb_func
from ...decode_sparse.cache_quant_utils import KC2CompressionCacheBuffered as KC2CompressionCacheOptimized
from ...modules.common import apply_rotary_pos_emb



logger = logging.get_logger(__name__)

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj



class Qwen3PrismAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PrismQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
                
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

        if self.config.attention_method in [
            "prism",
            "indexcache",
        ]:
            self.attn_gate = C2AttnGate(
                num_k_head=config.num_key_value_heads,
                num_q_head=config.num_attention_heads,
                head_dim=self.head_dim,
            )

        self.mask_loss_func = torch.nn.KLDivLoss()
        self.use_flash_rope = config.use_flash_rope
        self.attention_method = config.attention_method
        self.start_layer = config.start_layer

        self.sink_size = config.sink_size
        self.recent_size = config.recent_size



    def _select_topk_from_candidates(
        self,
        q: torch.Tensor,
        extended_indices: torch.Tensor,
        budget: int,
        k_compressed_cache=None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Select topk indices from an extended candidate set.

        Scores are computed using the quantised key cache when available
        (via a fused Triton kernel), falling back to full-precision K gather
        only when no compressed cache is provided.

        Args:
            q: post-RoPE query [B, 1, num_q_heads, head_dim].
            extended_indices: candidate token indices [B, num_k_heads, E].
            budget: number of indices to keep.
            k_compressed_cache: KC2CompressionCacheBuffered (or compatible).
            layer_idx: which layer's cache to use.

        Returns:
            selected_indices: [B, num_k_heads, budget]
        """
        extended_budget = extended_indices.shape[2]
        if extended_budget <= budget:
            return extended_indices

        if k_compressed_cache is not None and hasattr(k_compressed_cache, 'compute_attn_scores_indexed'):
            # Fast path: score only the E candidates using the Triton INT2 kernel.
            # Returns [B, H_k, E] float32 — no full-key materialisation.
            scores = k_compressed_cache.compute_attn_scores_indexed(
                layer_idx, q[:, 0], extended_indices,
                sink_size=self.sink_size,
                recent_size=self.recent_size
            )
        else:
            # Fallback: should not be reached in normal use; kept for correctness.
            raise RuntimeError(
                "_select_topk_from_candidates: k_compressed_cache must be provided "
                "and support compute_attn_scores_indexed(). Pass k_full explicitly "
                "if you cannot use the quantised cache."
            )

        # Pick topk within candidates, remap to global token indices
        topk_local = torch.topk(scores, k=budget, dim=-1)[1]   # [B, H_k, budget]
        selected_indices = extended_indices.gather(2, topk_local)
        return selected_indices

    def prism_estimate(
        self,
        q, k, v,
        k_compressed_cache,
        q_len,
        attention_mask,
        past_key_value,
        cos, sin,
        cache_position,
        extended_factor,
        reuse_token_sparse_indices,
        reuse_extended_sparse_indices
    ):
        extended_sparse_indices = None
        if reuse_extended_sparse_indices is not None:
            # Reuse layer: update KV cache, then score candidates using
            # the quantised compressed key cache (Triton INT2 kernel).
            # Update k_compressed_cache first so the new token is included.
            k_compressed_cache.update(layer_idx=self.layer_idx, k=k, is_decode=(q_len == 1))
            token_sparse_indices = self._select_topk_from_candidates(
                q, reuse_extended_sparse_indices, self.config.token_budget,
                k_compressed_cache=k_compressed_cache, layer_idx=self.layer_idx,
            )
        elif reuse_token_sparse_indices is not None:
            # Legacy direct reuse: skip proposal, use indices as-is
            token_sparse_indices = reuse_token_sparse_indices
        else:
            # proposal layer: compute indices (optionally with extended candidates)
            gate_result = self.attn_gate(
                k,
                self.layer_idx,
                k_compressed_cache,
                q,
                attention_mask=attention_mask,
                budget=self.config.token_budget,
                sink_size=self.sink_size,
                recent_size=self.recent_size,
                extended_factor=extended_factor,
            )
            if extended_factor > 1.0 and isinstance(gate_result, tuple):
                token_sparse_indices, extended_sparse_indices = gate_result
            else:
                token_sparse_indices = gate_result

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k.flatten(-2, -1), v.flatten(-2, -1), self.layer_idx, cache_kwargs)
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        return k, v, token_sparse_indices, extended_sparse_indices

    @torch.no_grad()
    def indexcache_estimate(
        self,
        q, k, v,
        past_key_value,
        cos, sin,
        cache_position,
        reuse_token_sparse_indices,
        sink_size=0,
        recent_size=0
    ):
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k.flatten(-2, -1), v.flatten(-2, -1), self.layer_idx, cache_kwargs)
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if q.shape[1] != 1:
            return k, v, None, None

        if reuse_token_sparse_indices is not None:
            token_sparse_indices = reuse_token_sparse_indices
        else:
            bsz, k_len, num_k_head, head_dim = k.shape
            if k_len <= self.config.token_budget:
                token_sparse_indices = torch.arange(0, k_len, device=k.device)[None, None].expand((bsz, num_k_head, -1)).contiguous()
            else:
                repeat_k = torch.repeat_interleave(k, self.num_key_value_groups, dim=2)
                attn_score = torch.einsum('bhd,bjhd->bhj', q[:, 0], repeat_k)
                attn_score = attn_score.reshape(bsz, num_k_head, self.num_key_value_groups, k_len).max(dim=-2)[0]
                if sink_size != 0:
                    attn_score[..., :sink_size] = 1e10
                if recent_size != 0:
                    attn_score[..., -recent_size:] = 1e10
                token_sparse_indices = torch.topk(attn_score, k=self.config.token_budget, dim=-1)[1]
            
        return k, v, token_sparse_indices, None


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        k_compressed_cache: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reuse_token_sparse_indices: Optional[torch.Tensor] = None,  # reused indices (direct reuse, legacy)
        reuse_extended_sparse_indices: Optional[torch.Tensor] = None,  # extended candidate indices to select topk from
        extended_factor: float = 1.0,  # for proposal layers: compute m*topk candidates; >1 activates extended mode
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            (attn_output, sparsity_info, token_sparse_indices, extended_sparse_indices)
            - token_sparse_indices: the topk indices used for this layer's sparse attention
            - extended_sparse_indices: m*topk candidate indices (non-None only for proposal layers
              when extended_factor > 1.0); refined layers use these to pick their own topk
        """

        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.shape[1]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings

        if self.use_flash_rope:
            q = apply_rotary_emb_func(q, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
            k = apply_rotary_emb_func(k, cos, sin, False, True, cu_seqlens=None, max_seqlen=q_len)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        if self.attention_method == "prism":
            k, v, token_sparse_indices, extended_sparse_indices = self.prism_estimate(
                q, k, v,
                k_compressed_cache,
                q_len,
                attention_mask,
                past_key_value,
                cos, sin,
                cache_position,
                extended_factor,
                reuse_token_sparse_indices,
                reuse_extended_sparse_indices
            )
        elif self.attention_method == "indexcache":
            k, v, token_sparse_indices, extended_sparse_indices = self.indexcache_estimate(
                q, k, v,
                past_key_value,
                cos, sin,
                cache_position,
                reuse_token_sparse_indices,
            )
        else:
            token_sparse_indices = None
            extended_sparse_indices = None

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_value.update(k.flatten(-2, -1), v.flatten(-2, -1), self.layer_idx, cache_kwargs)
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Sparse Attention
        if self.config.attention_method == "dense" or self.layer_idx < self.start_layer:
            attn_output = dense_flash_attention_forward(
                q,
                k,
                v,
                attention_mask=attention_mask,
                query_length=q_len,
                softmax_scale=self.scaling,
                cache_seqlens=cache_seqlens,
            )
        else:
            attn_output = token_sparse_flash_attention_forward(
                q,
                k,
                v,
                attention_mask=attention_mask,
                query_length=q_len,
                softmax_scale=self.scaling,
                cache_seqlens=cache_seqlens,
                token_sparse_indices=token_sparse_indices,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # return indices so the model loop can cache them for downstream reuse
        # extended_sparse_indices is non-None only for proposal layers with extended_factor > 1
        return attn_output, token_sparse_indices, extended_sparse_indices


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: PrismQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = Qwen3MLP(config)
        self.self_attn = Qwen3PrismAttention(config=config, layer_idx=layer_idx)
        self.fused_norm = config.fused_norm

        if config.fused_norm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        k_compressed_cache: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        reuse_token_sparse_indices: Optional[torch.Tensor] = None,  # reused indices from another layer (direct)
        reuse_extended_sparse_indices: Optional[torch.Tensor] = None,  # extended candidates to select topk from
        extended_factor: float = 1.0,  # for proposal layers: m*topk extended candidate factor
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention

        hidden_states, token_sparse_indices, extended_sparse_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            k_compressed_cache=k_compressed_cache,
            cache_position=cache_position,
            cache_seqlens=cache_seqlens,
            position_embeddings=position_embeddings,
            reuse_token_sparse_indices=reuse_token_sparse_indices,
            reuse_extended_sparse_indices=reuse_extended_sparse_indices,
            extended_factor=extended_factor,
            **kwargs,
        )
        if self.fused_norm:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, token_sparse_indices, extended_sparse_indices)
        return outputs



class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: PrismQwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            if self.config.use_flash_rope:
                emb = freqs ## to use rope func in flash attn
                cos = emb.cos().squeeze(0)
                sin = emb.sin().squeeze(0)
            else:
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = PrismQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: PrismQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.num_layers = config.num_hidden_layers

        # Index reuse map: {layer_idx -> source_layer_idx}
        # Layers absent from the map compute their own gate; layers in the map reuse.
        #
        # Priority (highest first):
        #   1. proposal_layers env var  – comma-separated layer indices, e.g. "0,4,8,12"
        #   2. reuse_map config – explicit {layer: source} dict, full control
        #   3. proposal_layers config     – list of compute layers; all others reuse
        #   4. (none of the above)             – every layer computes its own indices
        _env_proposal_layers = os.environ.get('PRISM_PROPOSAL_LAYERS', '').strip()
        if _env_proposal_layers:
            proposal_layers = [int(x) for x in _env_proposal_layers.split(',') if x.strip()]
            self.index_reuse_map = self._build_reuse_map_from_proposal_layers(
                config.num_hidden_layers, proposal_layers
            )
        elif getattr(config, 'reuse_map', None):
            self.index_reuse_map: dict = {
                int(k): int(v) for k, v in config.reuse_map.items()
            }
        elif getattr(config, 'proposal_layers', None):
            self.index_reuse_map = self._build_reuse_map_from_proposal_layers(
                config.num_hidden_layers, list(config.proposal_layers)
            )
        else:
            self.index_reuse_map = {}

        # Will be printed on the first forward() call when device is known
        self._device_info_printed = False

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _build_reuse_map_from_proposal_layers(num_layers: int, proposal_layers: list) -> dict:
        """Build reuse map from an explicit list of layers that run the attention gate.

        Every Refined layer reuses the indices from its nearest preceding proposal layer.
        Layers before the first proposal layer also compute their own indices (no reuse).

        Example:
            proposal_layers=[0, 4, 8], num_layers=12
            → {1:0, 2:0, 3:0, 5:4, 6:4, 7:4, 9:8, 10:8, 11:8}
        """
        proposal_set = set(proposal_layers)
        sorted_proposals = sorted(proposal_layers)
        reuse_map = {}
        for i in range(num_layers):
            if i in proposal_set:
                continue
            # find the largest proposal layer index that is < i
            src = None
            for g in reversed(sorted_proposals):
                if g < i:
                    src = g
                    break
            if src is not None:
                reuse_map[i] = src
        return reuse_map

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        k_compressed_cache: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCache]:
        # Print device info once on the first forward, when weights are already on the target device
        if not self._device_info_printed:
            try:
                ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                ip = "unknown"
            dev = next(self.parameters()).device
            if dev.type == "cuda":
                props = torch.cuda.get_device_properties(dev)
                device_info = f"{dev} ({props.name})"
            else:
                device_info = str(dev)
            print(
                f"[PrismAttn] index_reuse_map={self.index_reuse_map} | "
                f"ip={ip} | device={device_info}"
            )
            self._device_info_printed = True

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            if k_compressed_cache is None:
                if past_key_values.get_seq_length() == 0:
                    self.k_compressed_cache = KC2CompressionCacheOptimized(
                        self.num_layers,
                        group_size=self.config.int2_group_size,
                        max_seq_len=self.config.max_cache_length,
                    )
                k_compressed_cache = self.k_compressed_cache

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        cache_seqlens = torch.sum(attention_mask.to(torch.int32), dim=-1, dtype=torch.int32) 

        hidden_states = inputs_embeds


        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Temporary store for token_sparse_indices produced by "source" layers within this
        # forward step.  Keys are layer_idx integers; values are index tensors.
        # Only layers that serve as a reuse source for at least one other layer are stored.
        #
        # When candidate_factor > 1.0 we store the *extended* candidate
        # indices (m*topk) so that each reuse layer can select its own topk from them.
        _source_layers = set(self.index_reuse_map.values())
        _env_ext = os.environ.get('PRISM_CANDIDATE_FACTOR', '').strip()
        if _env_ext:
            _extended_factor = float(_env_ext)
        else:
            _extended_factor = getattr(self.config, 'candidate_factor', 1.0)
        _use_extended = _extended_factor > 1.0
        # Separate stores: extended candidates (for reuse layers) vs topk indices (legacy direct reuse)
        _extended_store: dict = {}   # layer_idx -> extended_sparse_indices (m*topk)
        _indices_store: dict = {}    # layer_idx -> token_sparse_indices (topk), kept for legacy path

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_idx = decoder_layer.self_attn.layer_idx

            # Determine whether to pass pre-computed indices for this layer
            reuse_src = self.index_reuse_map.get(layer_idx, None)

            if reuse_src is not None and _use_extended and reuse_src in _extended_store:
                # Extended mode: pass candidate set, layer selects its own topk
                reuse_indices = None
                reuse_extended_indices = _extended_store[reuse_src]
            elif reuse_src is not None and not _use_extended and reuse_src in _indices_store:
                # Legacy direct reuse
                reuse_indices = _indices_store[reuse_src]
                reuse_extended_indices = None
            else:
                reuse_indices = None
                reuse_extended_indices = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                k_compressed_cache=k_compressed_cache,
                cache_position=cache_position,
                cache_seqlens=cache_seqlens,
                position_embeddings=position_embeddings,
                reuse_token_sparse_indices=reuse_indices,
                reuse_extended_sparse_indices=reuse_extended_indices,
                extended_factor=_extended_factor if (layer_idx in _source_layers and _use_extended) else 1.0,
            )

            hidden_states = layer_outputs[0]

            # Cache indices if this layer is a reuse source for downstream layers
            if layer_idx in _source_layers:
                _indices_store[layer_idx] = layer_outputs[1]   # topk indices (always available)
                if _use_extended and layer_outputs[2] is not None:
                    _extended_store[layer_idx] = layer_outputs[2]  # extended candidates


        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastAndCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            k_compressed_cache=k_compressed_cache if use_cache else None,
        )

class PrismDecodingQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_layers = config.num_hidden_layers

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        k_compressed_cache: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastAndCache]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            k_compressed_cache=k_compressed_cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndCache(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            k_compressed_cache=outputs.k_compressed_cache,
        )

    def batch_exist_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_length: int = 100,
        do_sample: bool = False,
        **model_kwargs,
    ):
        """
        Modified generation loop that dynamically adjusts batch size by filtering finished sequences
        and reorganizing the KV cache.
        """
        # Initialize variables
        generation_config, model_kwargs = self._prepare_generation_config(None)
        generated = input_ids

        if isinstance(generation_config.eos_token_id, list):
            eos_token_id = generation_config.eos_token_id[0]
            eos_token_ids = torch.tensor(generation_config.eos_token_id, device=input_ids.device)
        else:
            eos_token_id = generation_config.eos_token_id
        initial_batch_size = input_ids.shape[0]

        device = input_ids.device
        finished = torch.zeros(initial_batch_size, dtype=torch.bool, device=device)
        current_kvcache = DynamicCache()
        
        current_kcompressed_cache = KC2CompressionCacheOptimized(
            self.num_layers,
            group_size=self.config.int2_group_size,
            max_seq_len=self.config.max_cache_length,
        )
        
        cur_input = generated
        cur_to_orig = torch.arange(initial_batch_size, device=device)

        if do_sample:
            top_p_warper = TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1)

        for step in range(max_length - generated.shape[1]):
            # Forward pass: get next token logits and updated past_key_values
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = self(
                    cur_input, 
                    attention_mask=attention_mask,
                    past_key_values=current_kvcache, 
                    k_compressed_cache=current_kcompressed_cache,
                    use_cache=True,
                    logits_to_keep=1,
            )
                

            logits = outputs.logits[:, -1, :].clone().float()
            logits = logits.to(input_ids.device)

            if do_sample:
                logits /= generation_config.temperature
                processed_logits = top_p_warper(cur_input, logits)
                probs = torch.softmax(processed_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Update the kv cache with the new keys and values.
            current_kvcache = outputs.past_key_values
            current_kcompressed_cache = outputs.k_compressed_cache

            new_tokens_full = torch.full((initial_batch_size, 1), eos_token_id,
                                        dtype=next_tokens.dtype, device=device)
            new_tokens_full[cur_to_orig] = next_tokens

            # Append the token to each sequence.
            generated = torch.cat([generated, new_tokens_full], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=1)


            # Update finished flags for the active sequences.
            if isinstance(generation_config.eos_token_id, list):
                finished[cur_to_orig] |= torch.isin(next_tokens.squeeze(1), eos_token_ids)
            else:
                finished[cur_to_orig] |= (next_tokens.squeeze(1) == eos_token_id)
            # finished[cur_to_orig] |= (next_tokens.squeeze(1) == eos_token_id)

            # If all sequences are finished, break.
            if finished.all():
                break

            # Determine which sequences in the current batch (cache) are still active.
            current_finished = finished[cur_to_orig]
            active_local = ~current_finished
            if active_local.sum().item() < cur_to_orig.shape[0]:
                active_indices_local = torch.nonzero(active_local, as_tuple=False).squeeze(-1)
                # Update the kv cache using indices relative to the current cache.
                print("active_local_batches", active_indices_local, "kvlen:", attention_mask.shape[1], flush=True)
                current_kvcache.batch_select_indices(active_indices_local)
                current_kcompressed_cache.batch_select_indices(active_indices_local)
                if attention_mask is not None:
                    attention_mask = attention_mask[active_indices_local]

                cur_to_orig = cur_to_orig[active_indices_local]

            # Prepare the next input tokens using the updated mapping.
            cur_input = generated[cur_to_orig, -1:].clone()

        # Pad finished sequences back to original batch size if needed
        return generated
