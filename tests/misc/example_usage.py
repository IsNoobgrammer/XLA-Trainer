"""Lightweight example of the metrics system for CUDA.

This demonstrates how to integrate the metrics system into a training loop
with a small model that fits in 4GB VRAM.
"""

import torch
import torch.nn as nn
from metrics.logger import TrainingLogger, LogConfig
from metrics.tracker import MetricsTracker, LossTracker
# Use the existing losses module
from losses.cross_entropy import cross_entropy_with_entropy_loss

class TinyModel(nn.Module):
    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.lm_head(x)

def example_training_loop():
    """Demonstrate metrics integration with a small CUDA model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup (one-time configuration)
    vocab_size = 1000  # Reduced from 50k
    hidden_size = 128   # Reduced from 768
    batch_size = 8      # Reduced from 64
    seq_len = 256       # Reduced from 2048
    
    model = TinyModel(vocab_size=vocab_size, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Configure logging with W&B
    log_config = LogConfig(
        project="tpu-trainer-demo",  # W&B project name
        run_name="cuda-example",
        log_every=5,      # Log more frequently for short run
        print_every=2,     # Print more frequently for short run
        use_colors=True,
        compact_format=True
    )
    
    # Initialize trackers
    logger = TrainingLogger(config=log_config)
    metrics_tracker = MetricsTracker(
        model=model,
        sequence_length=seq_len,
        world_size=1,  # Single GPU
        theoretical_peak_flops=10e12  # Approximate for consumer GPU
    )
    loss_tracker = LossTracker(window_size=10)
    
    try:
        # 2. Training loop (minimal integration)
        with logger.log_section("Training"):
            for step in range(20):  # Reduced from 100 steps
                # Generate synthetic data on GPU with requires_grad
                logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
                labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                
                # Forward pass
                outputs = model(torch.zeros(batch_size, seq_len, dtype=torch.long, device=device))
                
                # Compute loss with metrics
                ce_loss, entropy, total_loss = cross_entropy_with_entropy_loss(
                    outputs,  # Use model outputs instead of random logits
                    labels,
                    pad_id=0,  # Simple padding token
                    entropy_weight=0.001
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Update trackers
                metrics_tracker.update(local_batch_size=batch_size)
                loss_tracker.update(
                    ce_loss=ce_loss.detach(),
                    entropy=entropy.detach(),
                    total_loss=total_loss.detach()
                )
                
                # Log metrics every few steps
                if step % 2 == 0:
                    perf_metrics = metrics_tracker.as_dict()
                    logger.log_step(
                        step=step,
                        loss=total_loss.item(),
                        ce_loss=ce_loss.item(),
                        entropy=entropy.item(),
                        mfu=perf_metrics.get('mfu', 0),
                        tokens_per_sec=perf_metrics.get('tokens_per_second', 0),
                        lr=3e-4 * (0.99 ** (step // 2)),
                        gpu_mem=f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
                    )
                
                # Validation (simplified)
                if step > 0 and step % 5 == 0:
                    with torch.no_grad():
                        val_loss = ce_loss * 0.9
                        logger.log_validation(
                            step=step,
                            val_loss=val_loss.item(),
                            val_perplexity=torch.exp(val_loss).item()
                        )
                
                # Clean up
                del logits, labels
                torch.cuda.empty_cache()
                
        print("\nTraining completed!")
        
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Ensure cleanup
        logger.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def example_minimal_integration():
    """Show the absolute minimal integration for existing codebases."""
    try:
        # Minimal setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger = TrainingLogger(
            project="tpu-trainer-demo",  # W&B project name
            run_name="minimal-example",
            print_every=1,
            compact_format=True
        )
        
        # Tiny model
        model = nn.Linear(64, 10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Minimal training loop
        for step in range(5):
            x = torch.randn(32, 64, device=device)
            y = torch.randint(0, 10, (32,), device=device)
            
            # Forward + backward
            outputs = model(x)
            loss = torch.nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Simple logging
            logger.log_step(
                step=step,
                loss=loss.item(),
                lr=1e-3,
                gpu_mem=f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
            )
            
    finally:
        logger.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Running lightweight CUDA example...")
    print("\n" + "="*60)
    print("FULL INTEGRATION EXAMPLE")
    print("="*60)
    example_training_loop()
    
    print("\n" + "="*60) 
    print("MINIMAL INTEGRATION EXAMPLE")
    print("="*60)
    example_minimal_integration()