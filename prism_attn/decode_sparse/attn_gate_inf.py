import torch
import torch.nn as nn

class C2AttnGate(nn.Module):
    def __init__(self, 
                 num_k_head,
                 num_q_head,
                 head_dim,
                ):
        super(C2AttnGate, self).__init__()
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.head_dim = head_dim

    def forward(self,
            k, # [b, klen, k_head, head_dim]
            layer_idx,
            k_compressed_cache,
            q=None, #[b, 1, q_head, head_dim]
            attention_mask=None, # [b, 1, klen]
            budget=2048,
            sink_size=64,
            recent_size=192,
            extended_factor=1.0,
        ):
        """
        Args:
            extended_factor: when > 1.0, additionally returns an extended candidate index
                tensor of size (bsz, num_k_head, min(extended_budget, k_len)) alongside
                the normal topk indices.  The return value becomes a 2-tuple
                ``(token_sparse_indices, extended_sparse_indices)``; the first element
                contains the real ``budget`` indices used for this layer's sparse
                attention, while the second contains ``extended_budget = round(m * budget)``
                indices that downstream reuse-layers can further select from.
                When ``extended_factor <= 1.0`` (default) the return value is just
                ``token_sparse_indices`` (unchanged behaviour).
        """
        is_decode = k.shape[1] == 1
        # Update cache (appends k; returns exact k during prefill, ignored here)
        k_compressed_cache.update(layer_idx=layer_idx, k=k, is_decode=is_decode)

        if not is_decode:
            return None
        else:
            bsz = k.shape[0]
            k_len = k_compressed_cache.k_comp_len[layer_idx]
            if k_len <= budget:
                token_sparse_indices = torch.arange(0, k_len, device=k.device)[None, None].expand((bsz, self.num_k_head, -1)).contiguous()
                if extended_factor > 1.0:
                    return token_sparse_indices, token_sparse_indices
                return token_sparse_indices

            # q_compressed = q.reshape(
            #     (q.shape[0], self.num_k_head, self.num_q_head//self.num_k_head, self.head_dim)
            # ).mean(dim=-2)  # [B, H_k, D]

            # Fused: compute scores + apply sink/recent forcing + padding mask
            # — all inside one Triton kernel, no extra pass over output tensor.
            attn = k_compressed_cache.compute_attn_scores(
                layer_idx, q[:, 0],
                sink_size=min(sink_size, budget),
                recent_size=min(recent_size, budget),
            )
            token_sparse_indices = torch.topk(attn, k=budget, dim=-1)[1]

            if extended_factor > 1.0:
                extended_budget = min(int(round(extended_factor * budget)), k_len)
                extended_sparse_indices = torch.topk(attn, k=extended_budget, dim=-1)[1]
                return token_sparse_indices, extended_sparse_indices

            return token_sparse_indices


