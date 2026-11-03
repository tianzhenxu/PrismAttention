from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput


@dataclass
class BaseModelOutputWithPastAndCache(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    k_compressed_cache: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class CausalLMOutputWithPastAndCache(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    k_compressed_cache: Optional[Tuple[torch.FloatTensor, ...]] = None

