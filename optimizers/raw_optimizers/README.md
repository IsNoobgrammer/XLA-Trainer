# Raw Optimizers

This directory contains raw optimizer implementations that can be used directly or wrapped for specific backends.

## Files

### `muon_base_torch.py`
Original Muon implementation with PyTorch distributed support. Contains:
- `zeropower_via_newtonschulz5()` - Newton-Schulz orthogonalization
- `muon_update()` - Core Muon update function
- `Muon` - Distributed Muon optimizer
- `SingleDeviceMuon` - Single device Muon optimizer  
- `MuonWithAuxAdam` - Hybrid Muon + Adam optimizer
- `SingleDeviceMuonWithAuxAdam` - Single device hybrid optimizer

**XLA Compatibility Issues:**
- Uses `torch.distributed` calls that break XLA graph compilation
- Dynamic world_size/rank operations not compatible with TPU
- Complex distributed parameter handling

### `muon_base_optax.py`
JAX/Optax implementation of Muon optimizer for JAX-based training.

### `muon_syncfree_adamw.py` ⭐ **Recommended for TPU**
XLA-compatible hybrid optimizer combining Muon + SyncFree AdamW:
- **XLA Optimized**: No torch.distributed calls, static shapes, TPU-friendly
- **Hybrid Approach**: Muon for weight matrices, SyncFree AdamW for embeddings/biases
- **TPU Efficient**: Uses bfloat16, optimized Newton-Schulz iteration
- **Easy to Use**: Factory function `create_muon_syncfree_adamw(model)`

## Usage

### For TPU Training (Recommended)
```python
from optimizers import create_muon_syncfree_adamw

optimizer = create_muon_syncfree_adamw(
    model,
    muon_lr=0.02,           # Learning rate for weight matrices
    adamw_lr=3e-4,          # Learning rate for embeddings/biases
    muon_weight_decay=0.0,  # Weight decay for Muon parameters
    adamw_weight_decay=0.01 # Weight decay for AdamW parameters
)
```

### For CPU/GPU Training
```python
from optimizers.raw_optimizers.muon_base_torch import SingleDeviceMuonWithAuxAdam

# Manual parameter separation
matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'embed' not in n]
other_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'embed' in n]

param_groups = [
    {'params': matrix_params, 'use_muon': True, 'lr': 0.02},
    {'params': other_params, 'use_muon': False, 'lr': 3e-4}
]

optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
```

## Key Differences

| Feature | `muon_base_torch.py` | `muon_syncfree_adamw.py` |
|---------|---------------------|-------------------------|
| XLA Compatible | ❌ | ✅ |
| TPU Optimized | ❌ | ✅ |
| Distributed | ✅ | ❌ (use XLA collectives) |
| AdamW Backend | Manual implementation | SyncFree AdamW |
| Graph Compilation | Breaks | Works |
| Recommended For | CPU/GPU distributed | TPU training |