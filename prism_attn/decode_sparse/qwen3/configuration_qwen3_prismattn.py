# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# All rights reserved.
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

"""Qwen3 configuration extended with Prism and IndexCache runtime options."""

from typing import Dict, List, Literal, Optional

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


AttentionMethod = Literal["prism", "indexcache", "dense"]


class PrismQwen3Config(Qwen3Config):
    """Official Qwen3 config plus Prism/IndexCache inference options.

    All base-model architecture fields are handled by ``Qwen3Config``.
    This class only adds options required by Prism and IndexCache.

    Args:
        attention_method:
            ``"prism"`` enables proposal-refined index sharing.
            ``"indexcache"`` enables direct cross-layer index reuse.
            ``"dense"`` uses dense attention.
        token_budget:
            Number of final sparse KV indices selected per KV head.
        start_layer:
            Layers before this index use dense attention.
        candidate_factor:
            Prism proposal size divided by ``token_budget``.
            A factor of 2.0 means proposal layers produce 2K candidates
            and reuse layers refine them back to K.
            IndexCache must use 1.0.
        proposal_layers:
            Layers that compute fresh Prism proposals or IndexCache indices.
        reuse_map:
            Optional explicit mapping from reuse layer to proposal/full layer.
        int2_group_size:
            Quantization group size used by the INT2 K cache.
        max_cache_length:
            Maximum sequence length preallocated by the compressed K cache.
        sink_size:
            Number of initial sink tokens forced into selector scores.
        recent_size:
            Number of recent tokens forced into selector scores.
        fused_norm:
            Whether to use the optional fused RMSNorm implementation.
        use_flash_rope:
            Whether to use the FlashAttention rotary embedding kernel.
    """

    model_type = "qwen3"

    def __init__(
        self,
        attention_method: AttentionMethod = "prism",
        token_budget: int = 4096,
        start_layer: int = 0,
        candidate_factor: float = 2.0,
        proposal_layers: Optional[List[int]] = None,
        reuse_map: Optional[Dict[int, int]] = None,
        int2_group_size: int = 32,  
        max_cache_length: int = 32768,
        sink_size: int = 64,
        recent_size: int = 192,
        fused_norm: bool = False,
        use_flash_rope: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if attention_method not in {"prism", "indexcache", "dense"}:
            raise ValueError(
                "attention_method must be one of "
                "{'prism', 'indexcache', 'dense'}, "
                f"got {attention_method!r}"
            )

        if token_budget <= 0:
            raise ValueError(
                f"token_budget must be positive, got {token_budget}"
            )

        if start_layer < 0:
            raise ValueError(
                f"start_layer must be non-negative, got {start_layer}"
            )

        if candidate_factor < 1.0:
            raise ValueError(
                "candidate_factor must be at least 1.0, "
                f"got {candidate_factor}"
            )

        if attention_method == "indexcache" and candidate_factor != 1.0:
            raise ValueError(
                "IndexCache performs direct index reuse and requires "
                "candidate_factor=1.0"
            )

        if int2_group_size <= 0:
            raise ValueError(
                "int2_group_size must be positive, "
                f"got {int2_group_size}"
            )

        if max_cache_length <= 0:
            raise ValueError(
                "max_cache_length must be positive, "
                f"got {max_cache_length}"
            )

        if sink_size < 0 or recent_size < 0:
            raise ValueError(
                "sink_size and recent_size must be non-negative"
            )

        if proposal_layers is not None:
            proposal_layers = sorted(set(proposal_layers))
            if any(layer < 0 for layer in proposal_layers):
                raise ValueError(
                    "proposal_layers must contain non-negative layer indices"
                )

        if reuse_map is not None:
            reuse_map = {
                int(layer): int(source)
                for layer, source in reuse_map.items()
            }
            if any(
                layer < 0 or source < 0
                for layer, source in reuse_map.items()
            ):
                raise ValueError(
                    "reuse_map must contain non-negative layer indices"
                )

        self.attention_method = attention_method
        self.token_budget = token_budget
        self.start_layer = start_layer
        self.candidate_factor = candidate_factor
        self.proposal_layers = proposal_layers
        self.reuse_map = reuse_map

        self.int2_group_size = int2_group_size
        self.max_cache_length = max_cache_length
        self.sink_size = sink_size
        self.recent_size = recent_size

        self.fused_norm = fused_norm
        self.use_flash_rope = use_flash_rope


__all__ = ["PrismQwen3Config"]