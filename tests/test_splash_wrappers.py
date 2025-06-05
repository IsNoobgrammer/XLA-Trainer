import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from moneypatch_attention.splash_wrappers import (
    SplashAttentionWrapper,
    SplashAttentionQKNormWrapper,
    SPLASH_ATTENTION_AVAILABLE,
)


class MockConfig:
    """Mock configuration object for testing."""
    def __init__(self, num_attention_heads=32, num_key_value_heads=8, hidden_size=4096):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size


class MockOriginalAttention:
    """Mock original attention module for testing."""
    def __init__(self, config):
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = 1.0 / (self.head_dim ** 0.5)
        self.layer_idx = 0
        
        # Mock projection layers
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


class MockSplashAttentionConfig:
    """Mock Splash attention config for testing."""
    def to_json(self):
        return '{"sa_block_q": 512, "mesh": "test_mesh"}'


@pytest.fixture
def mock_config():
    return MockConfig()


@pytest.fixture
def mock_original_attention(mock_config):
    return MockOriginalAttention(mock_config)


@pytest.fixture
def mock_splash_config():
    return MockSplashAttentionConfig()


@pytest.fixture
def sample_inputs():
    """Generate sample inputs for testing."""
    batch_size, seq_len, hidden_size = 2, 128, 4096
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Mock position embeddings (cos, sin)
    cos = torch.randn(1, seq_len, 128)  # head_dim = 4096/32 = 128
    sin = torch.randn(1, seq_len, 128)
    position_embeddings = (cos, sin)
    
    return hidden_states, position_embeddings


class TestSplashAttentionWrapper:
    """Test cases for SplashAttentionWrapper."""
    
    def test_initialization(self, mock_original_attention, mock_splash_config):
        """Test wrapper initialization."""
        wrapper = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        
        assert wrapper.original_attention == mock_original_attention
        assert wrapper.config == mock_splash_config
        assert wrapper.num_heads == 32
        assert wrapper.num_kv_heads == 8
        assert wrapper.head_dim == 128
        
    def test_forward_output_shape(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test that forward pass produces correct output shapes."""
        wrapper = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        hidden_states, position_embeddings = sample_inputs
        
        with torch.no_grad():
            output, attn_weights = wrapper.forward(hidden_states, position_embeddings)
        
        # Check output shape matches input hidden_states shape
        assert output.shape == hidden_states.shape
        assert attn_weights is None  # Splash attention doesn't return weights
        
    def test_forward_with_cache(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test forward pass with KV cache."""
        wrapper = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        hidden_states, position_embeddings = sample_inputs
        
        # Mock cache object
        mock_cache = Mock()
        mock_cache.update.return_value = (
            torch.randn(2, 8, 128, 128),  # key_states
            torch.randn(2, 8, 128, 128)   # value_states
        )
        
        with torch.no_grad():
            output, _ = wrapper.forward(
                hidden_states,
                position_embeddings,
                past_key_value=mock_cache,
                cache_position=torch.arange(128)
            )
        
        assert output.shape == hidden_states.shape
        mock_cache.update.assert_called_once()

    @pytest.mark.skipif(SPLASH_ATTENTION_AVAILABLE, reason="Only test fallback when splash unavailable")
    def test_cpu_fallback(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test CPU fallback when splash attention is not available."""
        wrapper = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        hidden_states, position_embeddings = sample_inputs
        
        with torch.no_grad():
            output, _ = wrapper.forward(hidden_states, position_embeddings)
        
        # Should still produce correct output shape even with fallback
        assert output.shape == hidden_states.shape


class TestSplashAttentionQKNormWrapper:
    """Test cases for SplashAttentionQKNormWrapper."""
    
    def test_initialization(self, mock_original_attention, mock_splash_config):
        """Test wrapper initialization includes QK norm layers."""
        wrapper = SplashAttentionQKNormWrapper(mock_original_attention, mock_splash_config)
        
        assert wrapper.original_attention == mock_original_attention
        assert wrapper.config == mock_splash_config
        assert hasattr(wrapper, 'q_norm')
        assert hasattr(wrapper, 'k_norm')
        assert isinstance(wrapper.q_norm, nn.LayerNorm)
        assert isinstance(wrapper.k_norm, nn.LayerNorm)
        
    def test_qk_norm_dimensions(self, mock_original_attention, mock_splash_config):
        """Test QK norm layers have correct dimensions."""
        wrapper = SplashAttentionQKNormWrapper(mock_original_attention, mock_splash_config)
        
        # Both norms should be applied to head_dim
        assert wrapper.q_norm.normalized_shape == (128,)  # head_dim = 4096/32
        assert wrapper.k_norm.normalized_shape == (128,)
        
    def test_forward_output_shape(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test that forward pass with QK norm produces correct output shapes."""
        wrapper = SplashAttentionQKNormWrapper(mock_original_attention, mock_splash_config)
        hidden_states, position_embeddings = sample_inputs
        
        with torch.no_grad():
            output, attn_weights = wrapper.forward(hidden_states, position_embeddings)
        
        # Check output shape matches input hidden_states shape
        assert output.shape == hidden_states.shape
        assert attn_weights is None
        
    def test_qk_norm_applied(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test that QK normalization is actually applied."""
        wrapper = SplashAttentionQKNormWrapper(mock_original_attention, mock_splash_config)
        hidden_states, position_embeddings = sample_inputs
        
        # Patch the norm layers to verify they are called
        with patch.object(wrapper.q_norm, 'forward', wraps=wrapper.q_norm.forward) as mock_q_norm, \
             patch.object(wrapper.k_norm, 'forward', wraps=wrapper.k_norm.forward) as mock_k_norm:
            
            with torch.no_grad():
                wrapper.forward(hidden_states, position_embeddings)
            
            # Verify norm layers were called
            mock_q_norm.assert_called_once()
            mock_k_norm.assert_called_once()


class TestWrapperComparison:
    """Test comparing normal and QK-norm wrapper behaviors."""
    
    def test_different_outputs(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test that normal and QK-norm wrappers produce different outputs."""
        wrapper_normal = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        wrapper_qknorm = SplashAttentionQKNormWrapper(mock_original_attention, mock_splash_config)
        
        hidden_states, position_embeddings = sample_inputs
        
        with torch.no_grad():
            output_normal, _ = wrapper_normal.forward(hidden_states, position_embeddings)
            output_qknorm, _ = wrapper_qknorm.forward(hidden_states, position_embeddings)
        
        # Outputs should be different due to QK normalization
        assert not torch.allclose(output_normal, output_qknorm, atol=1e-6)
        
    def test_same_output_shapes(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test that both wrappers produce same output shapes."""
        wrapper_normal = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        wrapper_qknorm = SplashAttentionQKNormWrapper(mock_original_attention, mock_splash_config)
        
        hidden_states, position_embeddings = sample_inputs
        
        with torch.no_grad():
            output_normal, _ = wrapper_normal.forward(hidden_states, position_embeddings)
            output_qknorm, _ = wrapper_qknorm.forward(hidden_states, position_embeddings)
        
        assert output_normal.shape == output_qknorm.shape


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_shapes(self, mock_original_attention, mock_splash_config):
        """Test handling of invalid input shapes."""
        wrapper = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        
        # Invalid hidden_states shape (missing sequence dimension)
        invalid_hidden_states = torch.randn(2, 4096)  # Missing seq_len
        cos = torch.randn(1, 128, 128)
        sin = torch.randn(1, 128, 128)
        position_embeddings = (cos, sin)
        
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            wrapper.forward(invalid_hidden_states, position_embeddings)
            
    def test_mismatched_position_embeddings(self, mock_original_attention, mock_splash_config, sample_inputs):
        """Test handling of mismatched position embedding shapes."""
        wrapper = SplashAttentionWrapper(mock_original_attention, mock_splash_config)
        hidden_states, _ = sample_inputs
        
        # Mismatched cos/sin shapes
        cos = torch.randn(1, 64, 128)  # Wrong seq_len
        sin = torch.randn(1, 128, 128)  # Correct seq_len
        invalid_position_embeddings = (cos, sin)
        
        with pytest.raises((RuntimeError, ValueError)):
            wrapper.forward(hidden_states, invalid_position_embeddings)


if __name__ == "__main__":
    pytest.main([__file__])