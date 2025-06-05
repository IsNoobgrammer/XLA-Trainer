import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.cache_utils import Cache

try:
    from torch_xla.experimental.custom_kernel import flash_attention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    # Fallback for testing without XLA runtime
    def flash_attention(*args, **kwargs): 
        return torch.zeros(1)  # Dummy for CPU tests


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key and value states for grouped query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class XLAFlashAttentionWrapper(nn.Module):
    """Standard XLA Flash Attention wrapper without QK normalization."""
    
    def __init__(self, original_attention, mesh, partition_spec):
        super().__init__()
        self.original_attention = original_attention
        self.mesh = mesh
        self.partition_spec = partition_spec
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # Compute groups
        self.head_dim = original_attention.head_dim
        self.hidden_size = original_attention.config.hidden_size
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Compute Q, K, V
        query_states = self.original_attention.q_proj(hidden_states)
        key_states = self.original_attention.k_proj(hidden_states)
        value_states = self.original_attention.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat key and value states for GQA to match query heads
        if self.num_kv_groups > 1:
            key_states = repeat_kv(key_states, self.num_kv_groups)
            value_states = repeat_kv(value_states, self.num_kv_groups)

        # Ensure tensors are contiguous
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Apply XLA Flash Attention
        if FLASH_ATTENTION_AVAILABLE:
            attn_output = flash_attention(
                q=query_states,  # [bsz, num_heads, q_len, head_dim]
                k=key_states,    # [bsz, num_heads, kv_len, head_dim] after repeat
                v=value_states,  # [bsz, num_heads, kv_len, head_dim] after repeat
                causal=True,     # Use causal attention
                sm_scale=self.scaling,
                partition_spec=self.partition_spec,
                mesh=self.mesh
            )
        else:
            # Fallback for CPU testing - simple scaled dot-product attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )

        # Reshape output and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None


class XLAFlashAttentionQKNormWrapper(nn.Module):
    """XLA Flash Attention wrapper with QK normalization (Qwen3-style)."""
    
    def __init__(self, original_attention, mesh, partition_spec):
        super().__init__()
        self.original_attention = original_attention
        self.mesh = mesh
        self.partition_spec = partition_spec
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # Compute groups
        self.head_dim = original_attention.head_dim
        self.hidden_size = original_attention.config.hidden_size
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx

        # Add QK normalization layers (Qwen3-style)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Compute Q, K, V with QK normalization (Qwen3-style)
        # Apply normalization BEFORE transpose - this is the key difference
        query_states = self.q_norm(
            self.original_attention.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.original_attention.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.original_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary positional embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat key and value states for GQA to match query heads
        if self.num_kv_groups > 1:
            key_states = repeat_kv(key_states, self.num_kv_groups)
            value_states = repeat_kv(value_states, self.num_kv_groups)

        # Ensure tensors are contiguous
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Apply XLA Flash Attention
        if FLASH_ATTENTION_AVAILABLE:
            attn_output = flash_attention(
                q=query_states,  # [bsz, num_heads, q_len, head_dim]
                k=key_states,    # [bsz, num_heads, kv_len, head_dim] after repeat
                v=value_states,  # [bsz, num_heads, kv_len, head_dim] after repeat
                causal=True,     # Use causal attention
                sm_scale=self.scaling,
                partition_spec=self.partition_spec,
                mesh=self.mesh
            )
        else:
            # Fallback for CPU testing - simple scaled dot-product attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )

        # Reshape output using input_shape and apply output projection
        bsz, q_len = input_shape
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None