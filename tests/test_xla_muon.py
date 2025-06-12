"""
Tests for Muon + SyncFree AdamW hybrid optimizer.
"""

import pytest
import torch
import torch.nn as nn

# Test imports
try:
    from optimizers import MuonSyncFreeAdamW, create_muon_syncfree_adamw
    from optimizers.syncfree_adamw import SyncFreeAdamW
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(100, input_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, idx=None):
        if idx is not None:
            x = self.embedding(idx)
        x = torch.relu(self.linear1(x))
        x = self.layer_norm(x)
        return self.linear2(x)


@pytest.mark.skipif(not OPTIMIZERS_AVAILABLE, reason="Optimizers not available")
class TestMuonSyncFreeAdamW:
    """Test Muon + SyncFree AdamW hybrid optimizer."""
    
    def test_hybrid_optimizer_init(self):
        """Test hybrid optimizer initialization."""
        model = SimpleModel()
        
        # Separate parameters
        matrix_params = [p for n, p in model.named_parameters() 
                        if p.ndim >= 2 and 'embed' not in n]
        other_params = [p for n, p in model.named_parameters() 
                       if p.ndim < 2 or 'embed' in n]
        
        param_groups = [
            {'params': matrix_params, 'use_muon': True, 'lr': 0.02},
            {'params': other_params, 'use_muon': False, 'lr': 3e-4}
        ]
        
        optimizer = MuonSyncFreeAdamW(param_groups)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['use_muon'] is True
        assert optimizer.param_groups[1]['use_muon'] is False
    
    def test_hybrid_optimizer_step(self):
        """Test hybrid optimizer step."""
        model = SimpleModel()
        
        # Use factory function
        optimizer = create_muon_syncfree_adamw(model)
        
        # Forward pass
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
    
    def test_factory_function(self):
        """Test factory function for creating hybrid optimizer."""
        model = SimpleModel()
        
        optimizer = create_muon_syncfree_adamw(
            model,
            muon_lr=0.01,
            adamw_lr=1e-3,
            muon_weight_decay=0.1
        )
        
        assert isinstance(optimizer, MuonSyncFreeAdamW)
        
        # Check that parameters were separated correctly
        muon_groups = [g for g in optimizer.param_groups if g['use_muon']]
        adamw_groups = [g for g in optimizer.param_groups if not g['use_muon']]
        
        assert len(muon_groups) > 0  # Should have matrix parameters
        assert len(adamw_groups) > 0  # Should have embedding/bias parameters
    
    def test_parameter_separation(self):
        """Test that parameters are correctly separated."""
        model = SimpleModel()
        
        optimizer = create_muon_syncfree_adamw(model)
        
        # Count parameters in each group
        muon_param_count = 0
        adamw_param_count = 0
        
        for group in optimizer.param_groups:
            if group['use_muon']:
                muon_param_count += len(group['params'])
            else:
                adamw_param_count += len(group['params'])
        
        # Should have both types of parameters
        assert muon_param_count > 0
        assert adamw_param_count > 0
        
        # Total should match model parameters
        total_model_params = len(list(model.parameters()))
        assert muon_param_count + adamw_param_count == total_model_params
    
    def test_state_dict_save_load(self):
        """Test saving and loading optimizer state."""
        model = SimpleModel()
        optimizer = create_muon_syncfree_adamw(model)
        
        # Take a step to create state
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Save state dict
        state_dict = optimizer.state_dict()
        
        # Create new optimizer and load state
        new_optimizer = create_muon_syncfree_adamw(model)
        new_optimizer.load_state_dict(state_dict)
        
        # States should match
        assert len(new_optimizer.state) == len(optimizer.state)


@pytest.mark.skipif(not OPTIMIZERS_AVAILABLE, reason="Optimizers not available")
class TestNewtonSchulzOptimized:
    """Test optimized Newton-Schulz iteration."""
    
    def test_newton_schulz_square_matrix(self):
        """Test Newton-Schulz on square matrix."""
        from optimizers.raw_optimizers.muon_syncfree_adamw import zeropower_via_newtonschulz5_xla_optimized
        
        # Create a random matrix
        G = torch.randn(10, 10)
        result = zeropower_via_newtonschulz5_xla_optimized(G, steps=5)
        
        assert result.shape == G.shape
        assert result.dtype == torch.bfloat16
    
    def test_newton_schulz_rectangular_matrix(self):
        """Test Newton-Schulz on rectangular matrix."""
        from optimizers.raw_optimizers.muon_syncfree_adamw import zeropower_via_newtonschulz5_xla_optimized
        
        # Test tall matrix
        G = torch.randn(20, 10)
        result = zeropower_via_newtonschulz5_xla_optimized(G, steps=5)
        assert result.shape == G.shape
        
        # Test wide matrix  
        G = torch.randn(10, 20)
        result = zeropower_via_newtonschulz5_xla_optimized(G, steps=5)
        assert result.shape == G.shape
    
    def test_muon_update_optimized(self):
        """Test optimized Muon update function."""
        from optimizers.raw_optimizers.muon_syncfree_adamw import muon_update_xla_optimized
        
        # Test with 2D matrix
        grad = torch.randn(10, 20)
        momentum = torch.zeros_like(grad)
        
        update = muon_update_xla_optimized(grad, momentum, beta=0.95)
        assert update.shape == grad.shape
        
        # Test with 4D conv tensor
        grad_4d = torch.randn(32, 16, 3, 3)
        momentum_4d = torch.zeros_like(grad_4d)
        
        update_4d = muon_update_xla_optimized(grad_4d, momentum_4d, beta=0.95)
        assert update_4d.shape == grad_4d.shape


@pytest.mark.skipif(not OPTIMIZERS_AVAILABLE, reason="Optimizers not available")
class TestXLACompatibility:
    """Test XLA compatibility of the implementation."""
    
    def test_no_dynamic_shapes(self):
        """Test that implementation avoids dynamic shapes."""
        model = SimpleModel()
        optimizer = create_muon_syncfree_adamw(model)
        
        # Multiple forward passes with same batch size
        for _ in range(3):
            x = torch.randn(8, 10)  # Fixed batch size
            y = torch.randn(8, 5)
            
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    def test_no_distributed_calls(self):
        """Test that implementation doesn't use torch.distributed."""
        import sys
        from optimizers.raw_optimizers import muon_syncfree_adamw
        
        # Check that torch.distributed is not imported
        source_code = open(muon_syncfree_adamw.__file__).read()
        assert 'torch.distributed' not in source_code
        assert 'dist.get_world_size' not in source_code
        assert 'dist.get_rank' not in source_code
        assert 'dist.all_gather' not in source_code


if __name__ == "__main__":
    pytest.main([__file__])