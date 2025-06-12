"""
XLA-optimized Muon optimizer for TPU training.

This module provides XLA-compatible implementations of the Muon optimizer
that work efficiently with torch-xla and TPU hardware.

Key XLA optimizations:
- Replaces torch.distributed with torch_xla collectives
- Removes dynamic world_size/rank calls that break graph compilation  
- Uses static shapes and XLA-friendly operations
- Optimized Newton-Schulz iteration for TPU bfloat16
"""

import torch
from typing import Optional, List, Dict, Any

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

from ..syncfree_adamw import SyncFreeAdamW


def zeropower_via_newtonschulz5_xla(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    XLA-optimized Newton-Schulz iteration for orthogonalization.
    
    This version is optimized for XLA graph compilation and TPU execution:
    - Uses static shapes and avoid dynamic operations
    - Optimized for bfloat16 computation on TPU
    - Avoids operations that break XLA graph compilation
    """
    assert G.ndim >= 2, f"Input must be at least 2D, got {G.ndim}D"
    
    # Constants for quintic iteration
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Convert to bfloat16 for TPU efficiency
    X = G.to(torch.bfloat16)
    
    # Handle tall matrices by transposing
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    
    # Normalize to ensure spectral norm <= 1
    # Use a small epsilon to avoid division by zero
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm + 1e-7)
    
    # Newton-Schulz iterations (unrolled for XLA optimization)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    
    # Transpose back if needed
    if transposed:
        X = X.mT
    
    return X


def muon_update_xla(grad: torch.Tensor, momentum: torch.Tensor, 
                   beta: float = 0.95, ns_steps: int = 5, 
                   nesterov: bool = True) -> torch.Tensor:
    """
    XLA-optimized Muon update function.
    
    Args:
        grad: Gradient tensor
        momentum: Momentum buffer
        beta: Momentum decay factor
        ns_steps: Number of Newton-Schulz iterations
        nesterov: Whether to use Nesterov momentum
    
    Returns:
        Update tensor
    """
    # Update momentum buffer in-place
    momentum.lerp_(grad, 1 - beta)
    
    # Choose update based on Nesterov flag
    if nesterov:
        update = grad.lerp_(momentum, beta)
    else:
        update = momentum
    
    # Handle 4D tensors (conv filters) by reshaping to 2D
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.view(update.size(0), -1)
    
    # Apply Newton-Schulz orthogonalization
    update = zeropower_via_newtonschulz5_xla(update, steps=ns_steps)
    
    # Apply scaling factor for rectangular matrices
    scale_factor = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    update = update * scale_factor
    
    # Reshape back to original shape if needed
    if original_shape is not None:
        update = update.view(original_shape)
    
    return update


class XLAMuon(torch.optim.Optimizer):
    """
    XLA-optimized Muon optimizer for TPU training.
    
    This version is designed to work efficiently with torch-xla and avoids
    operations that break XLA graph compilation.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        weight_decay: Weight decay coefficient (default: 0.0)
        momentum: Momentum factor (default: 0.95)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)
    """
    
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, 
                 momentum: float = 0.95, ns_steps: int = 5, nesterov: bool = True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Only apply Muon to 2D+ parameters
                if p.ndim < 2:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                # Get momentum buffer
                momentum_buffer = state['momentum_buffer']
                
                # Compute Muon update
                update = muon_update_xla(
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
        
        return loss


class XLAMuonWithSyncFreeAdamW(torch.optim.Optimizer):
    """
    XLA-optimized hybrid optimizer combining Muon and SyncFree AdamW.
    
    This optimizer applies Muon to 2D+ parameters (weight matrices) and
    SyncFree AdamW to other parameters (embeddings, biases, etc.).
    
    Args:
        param_groups: List of parameter groups with 'use_muon' flag
        
    Example:
        ```python
        # Separate parameters
        matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        
        # Create parameter groups
        param_groups = [
            {'params': matrix_params, 'use_muon': True, 'lr': 0.02},
            {'params': other_params, 'use_muon': False, 'lr': 3e-4}
        ]
        
        optimizer = XLAMuonWithSyncFreeAdamW(param_groups)
        ```
    """
    
    def __init__(self, param_groups):
        # Validate and set defaults for parameter groups
        for group in param_groups:
            if 'use_muon' not in group:
                raise ValueError("Each parameter group must have 'use_muon' flag")
            
            if group['use_muon']:
                # Muon defaults
                group.setdefault('lr', 0.02)
                group.setdefault('weight_decay', 0.0)
                group.setdefault('momentum', 0.95)
                group.setdefault('ns_steps', 5)
                group.setdefault('nesterov', True)
            else:
                # AdamW defaults
                group.setdefault('lr', 3e-4)
                group.setdefault('betas', (0.9, 0.999))
                group.setdefault('eps', 1e-8)
                group.setdefault('weight_decay', 0.01)
        
        super().__init__(param_groups, {})
        
        # Create separate AdamW optimizers for non-Muon parameters
        self._adamw_optimizers = {}
        for i, group in enumerate(self.param_groups):
            if not group['use_muon']:
                adamw_params = group['params']
                if adamw_params:  # Only create if there are parameters
                    self._adamw_optimizers[i] = SyncFreeAdamW(
                        adamw_params,
                        lr=group['lr'],
                        betas=group['betas'],
                        eps=group['eps'],
                        weight_decay=group['weight_decay']
                    )
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for i, group in enumerate(self.param_groups):
            if group['use_muon']:
                # Apply Muon updates
                for p in group['params']:
                    if p.grad is None or p.ndim < 2:
                        continue
                    
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('Muon does not support sparse gradients')
                    
                    state = self.state[p]
                    
                    # Initialize state
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    
                    # Get momentum buffer
                    momentum_buffer = state['momentum_buffer']
                    
                    # Compute Muon update
                    update = muon_update_xla(
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
                # Use SyncFree AdamW for non-Muon parameters
                if i in self._adamw_optimizers:
                    self._adamw_optimizers[i].step()
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        super().zero_grad(set_to_none=set_to_none)
        # Also zero gradients for AdamW optimizers
        for adamw_opt in self._adamw_optimizers.values():
            adamw_opt.zero_grad(set_to_none=set_to_none)


def create_xla_muon_with_syncfree_adamw(
    model: torch.nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.01,
    muon_filter_fn: Optional[callable] = None
) -> XLAMuonWithSyncFreeAdamW:
    """
    Factory function to create XLA Muon + SyncFree AdamW optimizer.
    
    Args:
        model: PyTorch model
        muon_lr: Learning rate for Muon parameters
        muon_momentum: Momentum for Muon
        muon_weight_decay: Weight decay for Muon parameters
        adamw_lr: Learning rate for AdamW parameters
        adamw_betas: Beta parameters for AdamW
        adamw_eps: Epsilon for AdamW
        adamw_weight_decay: Weight decay for AdamW parameters
        muon_filter_fn: Function to determine which parameters use Muon
                       (default: 2D+ parameters excluding embeddings)
    
    Returns:
        XLAMuonWithSyncFreeAdamW optimizer
    """
    if muon_filter_fn is None:
        def muon_filter_fn(name: str, param: torch.nn.Parameter) -> bool:
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
            'weight_decay': muon_weight_decay
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
    
    return XLAMuonWithSyncFreeAdamW(param_groups)