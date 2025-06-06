import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.cache_utils import Cache

try:
    from splash import SplashAttentionConfig, splash_attention
    SPLASH_ATTENTION_AVAILABLE = True
except ImportError:
    SPLASH_ATTENTION_AVAILABLE = False
    # Fallback for testing without TPU runtime
    class SplashAttentionConfig:
        def to_json(self): return "{}"
    def splash_attention(*args, **kwargs): 
        return torch.zeros(1)  # Dummy for CPU tests


class SplashAttentionWrapper(nn.Module):
    """Standard Splash Attention wrapper without QK normalization."""
    
    def __init__(
        self,
        original_attention: nn.Module,
        config: SplashAttentionConfig,
    ):
        """
        A wrapper to replace the original attention mechanism with Splash Attention.

        Args:
            original_attention: The original attention module (e.g., LlamaAttention).
            config: An instance of SplashAttentionConfig containing all necessary parameters.
        """
        super().__init__()
        self.original_attention = original_attention
        self.config = config

        # Extract attributes from original attention
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.head_dim = original_attention.head_dim
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

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Ensure tensors are contiguous
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Scale query states
        query_states = query_states * self.scaling

        # Call Splash Attention with the provided config
        if SPLASH_ATTENTION_AVAILABLE:
            attn_output = splash_attention(
                query_states,
                key_states,
                value_states,
                self.config.to_json(),
                decoder_segment_ids=None,
                attn_logits_soft_cap=None,
            )
        else:
            # Fallback for CPU testing - simple scaled dot-product attention
            # Expand KV heads if needed to match query heads
            if self.num_kv_heads != self.num_heads:
                if self.num_heads % self.num_kv_heads != 0:
                    raise ValueError(f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads ({self.num_kv_heads}) for expansion.")
                repeat_factor = self.num_heads // self.num_kv_heads
                key_states = key_states.repeat_interleave(repeat_factor, dim=1)
                value_states = value_states.repeat_interleave(repeat_factor, dim=1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )

        # Reshape output and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None


class SplashAttentionQKNormWrapper(nn.Module):
    """Splash Attention wrapper with QK normalization (Qwen3-style)."""
    
    def __init__(
        self,
        original_attention: nn.Module,
        config: SplashAttentionConfig,
    ):
        """
        A wrapper to replace the original attention mechanism with Splash Attention + QK normalization.

        Args:
            original_attention: The original attention module (e.g., LlamaAttention).
            config: An instance of SplashAttentionConfig containing all necessary parameters.
        """
        super().__init__()
        self.original_attention = original_attention
        self.config = config

        # Extract attributes from original attention
        self.num_heads = original_attention.config.num_attention_heads
        self.num_kv_heads = original_attention.config.num_key_value_heads
        self.head_dim = original_attention.head_dim
        self.scaling = original_attention.scaling
        self.layer_idx = original_attention.layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional["Cache"] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.original_attention.q_norm(self.original_attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.original_attention.k_norm(self.original_attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.original_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Ensure tensors are contiguous
        # query_states = query_states.contiguous()
        # key_states = key_states.contiguous()
        # value_states = value_states.contiguous()
        
        # if self.num_kv_groups > 1:
        #     key_states = repeat_kv(key_states, self.num_kv_groups)
        #     value_states = repeat_kv(value_states, self.num_kv_groups)

        query_states = query_states * self.scaling  ## query_states /= math.sqrt(self.head_dim)

        attn_output = splash_attention(
            query_states,
            key_states,
            value_states,
            self.config.to_json(),
            decoder_segment_ids=None,
            attn_logits_soft_cap=None,
        )

        # Reshape output and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.original_attention.o_proj(attn_output)

        return attn_output, None