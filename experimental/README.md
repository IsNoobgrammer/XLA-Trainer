# Experimental Features

This directory contains experimental utilities for TPU-Trainer that are not yet considered stable across versions. Use them at your own discretion.

## Model Smoothing

The `model_smoothing` module provides scheduled parameter averaging to improve model generalization and training stability.

### Overview

Model smoothing (also known as weight EMA or Stochastic Weight Averaging) periodically blends current model parameters with a stored buffer of previous parameters. This can help:

- **Improve generalization** by reducing overfitting
- **Stabilize training** by smoothing parameter trajectories  
- **Find flatter minima** that generalize better

### Quick Start

```python
from experimental import ModelSmoother, SmoothingConfig

# Configure smoothing
config = SmoothingConfig(
    update_interval=1000,  # smooth every 1000 steps
    alpha=0.5             # simple average (50/50 blend)
)

# Initialize smoother
smoother = ModelSmoother(model, config)

# Training loop
for step in range(total_steps):
    optimizer.zero_grad()
    loss = forward_backward(model, batch)
    optimizer.step()
    
    # Perform smoothing if needed
    smoother.maybe_smooth(step)
```

### How It Works

Instead of classical exponential moving average (EMA):
```python
smoothed = decay * smoothed + (1 - decay) * params  # every step
```

We use **scheduled averaging** to reduce memory overhead:

1. **Buffer Creation**: Store a copy of parameters at initialization
2. **Scheduled Updates**: Every `update_interval` steps:
   - Blend current parameters with buffer: `new_params = (1-α) * params + α * buffer`
   - Update buffer with the blended result
3. **Memory Efficient**: Only one extra parameter copy vs. continuous EMA

### Configuration

#### `SmoothingConfig`

- **`update_interval`** (int, default=1000): Steps between smoothing operations
- **`alpha`** (float, default=0.5): Weight given to buffer when blending
  - `0.5` = simple average (recommended)
  - `0.25` = 75% current params, 25% buffer
  - `0.75` = 25% current params, 75% buffer

### TPU/XLA Considerations

The implementation is designed for TPU training with several optimizations:

#### ✅ **SPMD Sharding Safe**
- Preserves tensor parallelism sharding specifications
- Buffers maintain same sharding as original parameters
- Works with both data and model parallelism

#### ✅ **XLA Graph Friendly**
- Uses in-place operations (`mul_`, `add_`, `copy_`) to avoid new tensor creation
- Wraps updates with `xm.mark_step()` for proper graph compilation
- Minimal impact on XLA compilation time

#### ✅ **Memory Efficient**
- Single buffer copy vs. continuous EMA tracking
- Buffers allocated on same device as parameters
- No cross-device tensor movement

### Best Practices

#### **Hyperparameter Tuning**

```python
# Conservative smoothing (recommended starting point)
SmoothingConfig(update_interval=1000, alpha=0.5)

# More aggressive smoothing
SmoothingConfig(update_interval=500, alpha=0.7)

# Light smoothing for fine-tuning
SmoothingConfig(update_interval=2000, alpha=0.3)
```

#### **Training Loop Integration**

```python
# ✅ Correct order
optimizer.zero_grad()
loss = model(batch)
loss.backward()
optimizer.step()
smoother.maybe_smooth(step)  # After optimizer step

# ❌ Wrong order  
smoother.maybe_smooth(step)  # Before optimizer - creates gradient mismatch
optimizer.step()
```

#### **Checkpointing**

```python
# Save smoother state with model checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'smoother': smoother.state_dict(),  # Include buffer state
    'step': step
}

# Restore smoother state
smoother.load_state_dict(checkpoint['smoother'])
```

### Potential Drawbacks

#### **Memory Overhead**
- Requires storing full copy of model parameters
- For large models (>10B params), this could be 10s-100s GB extra memory
- May force smaller batch sizes on memory-constrained TPUs

#### **Training Dynamics**
- **Gradient-Parameter Mismatch**: Gradients computed on pre-smoothing parameters, but optimizer updates post-smoothing parameters
- **Periodic Jumps**: Parameter values change discontinuously at smoothing steps
- **Hyperparameter Sensitivity**: `alpha` and `update_interval` require tuning

#### **XLA Complexity**
- Dynamic control flow (`step % interval == 0`) may cause graph recompilation
- Additional `xm.mark_step()` calls increase synchronization overhead
- SPMD sharding preservation relies on evolving TPU APIs

#### **Task Dependence**
- May help pretraining generalization but hurt fine-tuning convergence
- Benefits vary significantly across model architectures and datasets
- Interaction with learning rate schedules not well understood

### When to Use

#### **Good Candidates**
- Large-scale pretraining runs (>100k steps)
- Models prone to overfitting
- Training with high learning rates
- Scenarios where generalization > convergence speed

#### **Avoid When**
- Memory-constrained environments
- Short training runs (<10k steps)
- Fine-tuning pretrained models
- Debugging/development (adds complexity)

### Implementation Details

#### **Sharding Preservation**
```python
# Buffer creation preserves SPMD sharding
for p in model.parameters():
    buf = p.detach().clone()
    if hasattr(p, '_sharding_spec'):
        buf = xs.mark_sharding(buf, mesh, p._sharding_spec)
    buffer.append(buf)
```

#### **In-Place Updates**
```python
# Avoids creating new tensors that lose sharding
p.mul_(1.0 - alpha).add_(buffer, alpha=alpha)  # In-place blend
buffer.copy_(p)  # In-place buffer refresh
```

#### **XLA Integration**
```python
xm.mark_step()  # Before smoothing operations
# ... smoothing operations ...
xm.mark_step()  # After smoothing operations
```

### Experimental Status

This feature is **experimental** and may change in future versions:

- API may evolve based on user feedback
- TPU/XLA compatibility tested on limited configurations
- Performance characteristics may vary across model sizes
- Interaction with other training techniques not fully explored

Use in production at your own risk and thoroughly validate on your specific use case.

### References

- [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407)
- [Exponential Moving Average for Deep Learning](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
- [TPU Training Best Practices](https://cloud.google.com/tpu/docs/training-best-practices)