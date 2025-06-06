# TPU-Trainer Metrics System

A modular, aesthetic, and efficient metrics collection system optimized for XLA/TPU training pipelines.

## 🎯 Features

- **Drag-and-drop integration** - Add comprehensive logging in 3 lines of code
- **XLA-optimized** - Minimal overhead, respects graph compilation boundaries  
- **Aesthetic console output** - Beautiful colored formatting with smart value display
- **Cost-aware W&B logging** - Built-in bandwidth limits and frequency controls
- **Modular design** - Use only what you need, extend easily
- **TPU-specific metrics** - MFU, throughput, memory utilization

## 🚀 Quick Start

### Minimal Integration (3 lines)

```python
from metrics import TrainingLogger

logger = TrainingLogger(project="my-project", run_name="experiment-1")

# In your training loop:
logger.log_step(step=step, loss=loss.item(), lr=lr, mfu=0.56)
```

### Full Integration

```python
from metrics import TrainingLogger, MetricsTracker
from losses import cross_entropy_with_entropy_loss

# Setup
logger = TrainingLogger(project="my-project")
tracker = MetricsTracker(model, sequence_length=2048, world_size=8)

# Training loop
for step, batch in enumerate(dataloader):
    # Compute loss with built-in entropy 
    ce_loss, entropy, total_loss = cross_entropy_with_entropy_loss(
        logits, labels, pad_id=tokenizer.pad_token_id, entropy_weight=0.001
    )
    
    # Update performance tracking
    tracker.update(local_batch_size=64)
    
    # Log everything
    logger.log_step(
        step=step,
        loss=total_loss.item(),
        ce_loss=ce_loss.item(), 
        entropy=entropy.item(),
        mfu=tracker.mfu,
        tokens_per_sec=tracker.tokens_per_second,
        lr=scheduler.get_last_lr()[0]
    )
```

## 📁 Module Structure

```
metrics/
├── __init__.py          # Clean public API
├── compute.py           # Core metrics computation (MFU, FLOP counting)
├── tracker.py           # Stateful metrics accumulation  
├── logger.py            # Aesthetic logging + W&B integration
├── example_usage.py     # Complete examples
└── README.md           # This file
```

**Note**: Loss functions are in the separate `losses/` module to avoid duplication.

## 🔧 Core Components

### `TrainingLogger` - Aesthetic Logging

Beautiful console output with optional W&B integration:

```python
logger = TrainingLogger(
    project="my-project",
    log_every=50,        # W&B frequency
    print_every=25,      # Console frequency  
    use_colors=True,     # ANSI colors
    max_wandb_mb=100     # Cost control
)

logger.log_step(loss=1.5, lr=3e-4, mfu=0.56)
# Output: [  12.3m] loss=1.500 │ lr=3.000e-04 │ mfu=0.560
```

### `MetricsTracker` - Performance Monitoring

Tracks throughput, MFU, and timing:

```python
tracker = MetricsTracker(
    model=model,
    sequence_length=2048,
    world_size=8,
    theoretical_peak_flops=2.2e15  # TPU v4-8
)

tracker.update(local_batch_size=64)
print(f"MFU: {tracker.mfu:.1%}")  # MFU: 56.0%
```

### Loss Functions (from `losses/` module)

Use the existing optimized loss functions:

```python
from losses import cross_entropy_with_entropy_loss

ce_loss, entropy, total_loss = cross_entropy_with_entropy_loss(
    logits, labels,
    pad_id=tokenizer.pad_token_id,
    entropy_weight=0.001
)
# Fused computation for efficiency
```

## ⚡ Performance Considerations

### XLA Compatibility
- All tensor→scalar conversions happen once per metric
- No dynamic shapes or control flow in hot paths
- Logging frequency controls prevent graph recompilation

### W&B Cost Control
```python
LogConfig(
    max_wandb_mb=100,      # Hard limit on data sent
    log_every=50,          # Reduce frequency for large runs
    enable_histograms=False # Disable expensive gradient tracking
)
```

### Memory Overhead
- `MetricsTracker`: ~1KB per instance
- `LossTracker`: ~10KB with default 100-step window
- `TrainingLogger`: ~5KB + W&B queue size

## 🎨 Console Output Examples

**Standard format:**
```
[  12.3m] step=1000 │ loss=1.500 │ lr=3.000e-04 │ mfu=56.0% │ tokens_per_sec=364,000
[  12.4m] [VAL] val_loss=1.200 │ val_perplexity=3.320
```

**Compact format:**
```
[12.3m] step=1000 | loss=1.500 | lr=3e-04 | mfu=0.56 | tok/s=364k
```

## 🔧 Configuration

### LogConfig Options

```python
LogConfig(
    # W&B settings
    project="my-project",
    run_name="experiment-1", 
    tags=["tpu", "llama"],
    
    # Frequency control
    log_every=50,           # W&B logging interval
    print_every=25,         # Console logging interval
    
    # Cost control  
    max_wandb_mb=100.0,     # Max data per run
    enable_histograms=False, # Gradient/weight histograms
    
    # Aesthetics
    use_colors=True,        # ANSI color codes
    compact_format=False,   # Compact vs. beautiful format
    
    # XLA/TPU
    master_only=True        # Only log from master process
)
```

## 📊 Available Metrics

### Core Performance
- `mfu` - Model FLOPs Utilization (0.0-1.0)
- `tokens_per_second` - Training throughput
- `recent_mfu` - MFU over recent window
- `steps` - Optimization steps completed
- `tokens` - Total tokens processed

### Loss & Training
- `ce_loss` - Cross-entropy loss
- `entropy` - Token entropy (for regularization)
- `total_loss` - Combined loss with regularization
- `grad_norm` - Gradient norm
- `lr` - Learning rate

### Validation
- `val_loss` - Validation loss
- `val_perplexity` - Validation perplexity
- Custom validation metrics

## 🧪 Testing

Run the example to test your setup:

```bash
cd metrics/
python example_usage.py
```

## 🛠️ Extending

### Custom Metrics

```python
from metrics.tracker import LossTracker

# Track custom metrics
custom_tracker = LossTracker()
custom_tracker.update(
    custom_metric=my_value,
    another_metric=other_value
)

# Get windowed averages
avg_custom = custom_tracker.get_average("custom_metric", window=20)
```

### Custom Loss Functions

```python
def my_custom_loss(logits, labels):
    # Your loss computation
    loss = F.cross_entropy(logits, labels)
    
    # Add any metrics you want to track
    confidence = F.softmax(logits, dim=-1).max(dim=-1)[0].mean()
    
    return loss, {"confidence": confidence}
```

## 🤝 Integration with Existing Code

The metrics system is designed to be **additive** - you can integrate it gradually:

1. **Start minimal**: Just add `TrainingLogger` for better console output
2. **Add performance tracking**: Include `MetricsTracker` for MFU monitoring  
3. **Optimize losses**: Replace loss functions with fused versions
4. **Full integration**: Use all components for comprehensive monitoring

No need to rewrite your existing training loop!

## 📈 Roadmap

- [ ] **Phase 1**: Core training metrics ✅
- [ ] **Phase 2**: Environment/pipeline metrics (TPU memory, data loading)
- [ ] **Phase 3**: Task-specific metrics (BLEU, Rouge for SFT)
- [ ] **Phase 4**: Advanced cost controls and volume management
- [ ] **Phase 5**: Documentation and examples ✅

---

*Built for the TPU-Trainer project with ❤️ for beautiful, efficient training.*