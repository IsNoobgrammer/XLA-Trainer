"""Example usage of the modular metrics system.

This demonstrates how to integrate the metrics system into a training loop
with minimal code changes and maximum aesthetic appeal.
"""

import torch
import torch.nn as nn
from .logger import TrainingLogger, LogConfig
from .tracker import MetricsTracker, LossTracker
# Use the existing losses module
from losses import cross_entropy_with_entropy_loss

# Mock model for demonstration
class SimpleModel(nn.Module):
    def __init__(self, vocab_size=50000, hidden_size=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.lm_head(x)

def example_training_loop():
    """Demonstrate drag-and-drop metrics integration."""
    
    # 1. Setup (one-time configuration)
    model = SimpleModel()
    
    # Configure logging
    log_config = LogConfig(
        project="tpu-trainer-demo",
        run_name="example-run",
        log_every=10,      # W&B every 10 steps
        print_every=5,     # Console every 5 steps
        use_colors=True,
        compact_format=False
    )
    
    # Initialize trackers
    logger = TrainingLogger(config=log_config)
    metrics_tracker = MetricsTracker(
        model=model,
        sequence_length=2048,
        world_size=8,  # TPU v4-8
        theoretical_peak_flops=2.2e15  # 8 * 275 TFLOP/s
    )
    loss_tracker = LossTracker(window_size=50)
    
    # 2. Training loop (minimal integration)
    with logger.log_section("Training"):
        for step in range(100):
            # Simulate training step
            batch_size = 64
            seq_len = 2048
            vocab_size = 50000
            
            # Mock forward pass
            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Compute loss with metrics (single function call)
            ce_loss, entropy, total_loss = cross_entropy_with_entropy_loss(
                logits, labels, 
                pad_id=tokenizer.pad_token_id if 'tokenizer' in locals() else 0,
                entropy_weight=0.001
            )
            
            # Update trackers (one line each)
            metrics_tracker.update(local_batch_size=batch_size)
            loss_tracker.update(
                ce_loss=ce_loss,
                entropy=entropy, 
                total_loss=total_loss
            )
            
            # Log everything (one function call)
            if step % 5 == 0:  # Every 5 steps
                # Get current metrics
                perf_metrics = metrics_tracker.as_dict()
                loss_metrics = loss_tracker.as_dict()
                
                # Single logging call with all metrics
                logger.log_step(
                    step=step,
                    # Loss metrics
                    loss=total_loss.item(),
                    ce_loss=ce_loss.item(),
                    entropy=entropy.item(),
                    # Performance metrics  
                    mfu=perf_metrics.get('mfu', 0),
                    tokens_per_sec=perf_metrics.get('tokens_per_second', 0),
                    # Learning rate (mock)
                    lr=3e-4 * (0.99 ** (step // 10)),
                    # Gradient norm (mock)
                    grad_norm=1.2 + 0.1 * torch.randn(1).item()
                )
            
            # Validation logging (every 20 steps)
            if step % 20 == 0 and step > 0:
                val_loss = ce_loss * 0.9  # Mock validation loss
                logger.log_validation(
                    step=step,
                    val_loss=val_loss.item(),
                    val_perplexity=torch.exp(val_loss).item()
                )
            
            # Reset interval tracking for recent throughput
            if step % 10 == 0:
                metrics_tracker.reset_interval()
    
    # 3. Cleanup (automatic with context manager)
    logger.finish()
    
    print("\n🎉 Training completed! Check your W&B dashboard for detailed metrics.")

def example_minimal_integration():
    """Show the absolute minimal integration for existing codebases."""
    
    # Just add these 3 lines to your existing training script:
    from metrics import TrainingLogger
    logger = TrainingLogger(project="my-project")
    
    # In your training loop, replace print statements with:
    # logger.log_step(step=step, loss=loss.item(), lr=lr, mfu=mfu)
    
    # That's it! You get beautiful console output + W&B logging
    
    for step in range(10):
        loss = torch.tensor(2.0 - step * 0.1)  # Mock decreasing loss
        logger.log_step(
            step=step,
            loss=loss.item(),
            lr=1e-4,
            tokens_per_sec=1000 + step * 50
        )

if __name__ == "__main__":
    print("🚀 Running metrics system example...")
    print("\n" + "="*60)
    print("FULL INTEGRATION EXAMPLE")
    print("="*60)
    example_training_loop()
    
    print("\n" + "="*60) 
    print("MINIMAL INTEGRATION EXAMPLE")
    print("="*60)
    example_minimal_integration()