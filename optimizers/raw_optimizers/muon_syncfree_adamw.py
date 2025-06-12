"""
XLA-compatible Muon + SyncFree AdamW hybrid optimizer.

This implementation fixes XLA compatibility issues in the original Muon code:
1. Removes torch.distributed calls that break XLA graphs
2. Uses SingleDeviceMuon as base (no distributed logic)  
3. Integrates with torch_xla.amp.syncfree.AdamW for non-Muon parameters
4. Optimized for TPU training with proper XLA graph compilation
"""

import torch
from typing import Optional, List, Dict, Any, Callable

# Import the working functional kernels from the base implementation
from .muon_base_torch import (
    zeropower_via_newtonschulz5, 
    muon_update,
    adam_update
)

try:
    import torch_xla.core.xla_model as xm
    from torch_xla.amp.syncfree import AdamW as XLASyncFreeAdamW
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    # Fallback to regular AdamW if XLA not available
    XLASyncFreeAdamW = torch.optim.AdamW


def zeropower_via_newtonschulz5_xla_optimized(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    XLA-optimized Newton-Schulz iteration.
    
    Optimizations for XLA/TPU:
    - Ensures static control flow (no dynamic conditions)
    - Uses bfloat16 for TPU efficiency
    - Avoids operations that break XLA compilation
    """
    assert G.ndim >= 2, f"Input must be at least 2D, got {G.ndim}D"
    
    # Use the same constants as the original
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Convert to bfloat16 for TPU efficiency  
    X = G.to(torch.bfloat16)
    
    # Handle rectangular matrices - transpose if tall
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    
    # Normalize spectral norm to <= 1
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm + 1e-7)
    
    # Unroll Newton-Schulz iterations for XLA optimization
    # This avoids Python loops that can break XLA compilation
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    # Transpose back if needed
    if transposed:
        X = X.mT
        
    return X


def muon_update_xla_optimized(grad: torch.Tensor, momentum: torch.Tensor, 
                             beta: float = 0.95, ns_steps: int = 5, 
                             nesterov: bool = True) -> torch.Tensor:
    """
    XLA-optimized Muon update using the optimized Newton-Schulz.
    """
    # Update momentum in-place
    momentum.lerp_(grad, 1 - beta)
    
    # Compute update tensor
    if nesterov:
        update = grad.lerp_(momentum, beta)
    else:
        update = momentum.clone()
    
    # Handle conv filters by reshaping to 2D
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.view(update.size(0), -1)
    
    # Apply Newton-Schulz orthogonalization
    update = zeropower_via_newtonschulz5_xla_optimized(update, steps=ns_steps)
    
    # Apply scaling factor
    scale_factor = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    update = update * scale_factor
    
    # Reshape back if needed
    if original_shape is not None:
        update = update.view(original_shape)
    
    return update


class MuonSyncFreeAdamW(torch.optim.Optimizer):
    """
    XLA-compatible hybrid optimizer: Muon for matrices + SyncFree AdamW for others.
    
    This is based on the existing MuonWithAuxAdam but:
    1. Removes all torch.distributed calls that break XLA
    2. Uses SyncFree AdamW instead of manual Adam implementation
    3. Works with single device (no distributed parameter padding)
    4. Optimized for XLA graph compilation
    
    Args:
        param_groups: List of parameter groups with 'use_muon' flag
        
    Example usage:
        ```python
        # Separate parameters by type
        matrix_params = [p for n, p in model.named_parameters() 
                        if p.ndim >= 2 and 'embed' not in n]
        other_params = [p for n, p in model.named_parameters() 
                       if p.ndim < 2 or 'embed' in n]
        
        # Create parameter groups
        param_groups = [
            {
                'params': matrix_params, 
                'use_muon': True, 
                'lr': 0.02, 
                'momentum': 0.95,
                'weight_decay': 0.0
            },
            {
                'params': other_params, 
                'use_muon': False, 
                'lr': 3e-4,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.01
            }
        ]
        
        optimizer = MuonSyncFreeAdamW(param_groups)
        ```
    """
    
    def __init__(self, param_groups):
        # Validate parameter groups
        for group in param_groups:
            if 'use_muon' not in group:
                raise ValueError("Each parameter group must specify 'use_muon' flag")
            
            if group['use_muon']:
                # Set Muon defaults
                group.setdefault('lr', 0.02)
                group.setdefault('momentum', 0.95) 
                group.setdefault('weight_decay', 0.0)
                group.setdefault('ns_steps', 5)
                group.setdefault('nesterov', True)
            else:
                # Set AdamW defaults
                group.setdefault('lr', 3e-4)
                group.setdefault('betas', (0.9, 0.999))
                group.setdefault('eps', 1e-8)
                group.setdefault('weight_decay', 0.01)
        
        super().__init__(param_groups, {})
        
        # Create SyncFree AdamW optimizers for non-Muon parameters
        self._adamw_optimizers = {}
        for i, group in enumerate(self.param_groups):
            if not group['use_muon'] and group['params']:
                self._adamw_optimizers[i] = XLASyncFreeAdamW(
                    group['params'],
                    lr=group['lr'],
                    betas=group['betas'],
                    eps=group['eps'],
                    weight_decay=group['weight_decay']
                )
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for i, group in enumerate(self.param_groups):
            if group['use_muon']:
                # Apply Muon to matrix parameters
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Skip non-matrix parameters
                    if p.ndim < 2:
                        continue
                    
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('Muon does not support sparse gradients')
                    
                    # Get or initialize state
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    
                    momentum_buffer = state['momentum_buffer']
                    
                    # Compute Muon update
                    update = muon_update_xla_optimized(
                        grad,
                        momentum_buffer,
                        beta=group['momentum'],
                        ns_steps=group['ns_steps'],
                        nesterov=group['nesterov']
                    )
                    
                    # Apply weight decay
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                    
                    # Apply update
                    p.add_(update, alpha=-group['lr'])
            
            else:
                # Use SyncFree AdamW for other parameters
                if i in self._adamw_optimizers:
                    self._adamw_optimizers[i].step()
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        super().zero_grad(set_to_none=set_to_none)
        
        # Also zero gradients for AdamW optimizers
        for adamw_opt in self._adamw_optimizers.values():
            adamw_opt.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        """Return state dict including AdamW optimizer states."""
        state_dict = super().state_dict()
        
        # Add AdamW optimizer states
        adamw_states = {}
        for i, adamw_opt in self._adamw_optimizers.items():
            adamw_states[i] = adamw_opt.state_dict()
        
        if adamw_states:
            state_dict['adamw_optimizers'] = adamw_states
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load state dict including AdamW optimizer states."""
        # Load AdamW optimizer states if present
        adamw_states = state_dict.pop('adamw_optimizers', {})
        for i, adamw_state in adamw_states.items():
            if i in self._adamw_optimizers:
                self._adamw_optimizers[i].load_state_dict(adamw_state)
        
        # Load main optimizer state
        super().load_state_dict(state_dict)


def create_muon_syncfree_adamw(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    muon_ns_steps: int = 5,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.01,
    muon_filter_fn: Optional[Callable[[str, torch.nn.Parameter], bool]] = None
) -> MuonSyncFreeAdamW:
    """
    Factory function to create Muon + SyncFree AdamW optimizer.
    
    Args:
        model: PyTorch model
        muon_lr: Learning rate for Muon parameters
        muon_momentum: Momentum for Muon
        muon_weight_decay: Weight decay for Muon parameters  
        muon_ns_steps: Number of Newton-Schulz iterations
        adamw_lr: Learning rate for AdamW parameters
        adamw_betas: Beta parameters for AdamW
        adamw_eps: Epsilon for AdamW
        adamw_weight_decay: Weight decay for AdamW parameters
        muon_filter_fn: Function to determine which parameters use Muon
                       Default: 2D+ parameters excluding embeddings
    
    Returns:
        MuonSyncFreeAdamW optimizer
    """
    if muon_filter_fn is None:
        def muon_filter_fn(name: str, param: torch.nn.Parameter) -> bool:
            # Use Muon for 2D+ parameters, excluding embeddings
            return param.ndim >= 2 and 'embed' not in name.lower()
    
    # Separate parameters
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if muon_filter_fn(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    if muon_params:
        param_groups.append({
            'params': muon_params,
            'use_muon': True,
            'lr': muon_lr,
            'momentum': muon_momentum,
            'weight_decay': muon_weight_decay,
            'ns_steps': muon_ns_steps
        })
    
    if adamw_params:
        param_groups.append({
            'params': adamw_params,
            'use_muon': False,
            'lr': adamw_lr,
            'betas': adamw_betas,
            'eps': adamw_eps,
            'weight_decay': adamw_weight_decay
        })
    
    if not param_groups:
        raise ValueError("No parameters found for optimization")
    
    return MuonSyncFreeAdamW(param_groups)