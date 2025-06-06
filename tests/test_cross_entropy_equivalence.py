import os
import sys
import torch
import time
import gc
import psutil
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.cross_entropy import cross_entropy_loss, cross_entropy_with_entropy_loss

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device):
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Randomly set some labels to pad_id
    mask = torch.rand(batch_size, seq_len, device=device) < 0.1
    labels[mask] = pad_id
    return logits, labels

def get_memory_usage():
    """Returns memory usage in MB"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

def benchmark_fn(fn, *args, warmup=5, repeat=100, **kwargs):
    """Benchmark a function's execution time and memory usage"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    
    for _ in range(repeat):
        result = fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    
    avg_time = (end_time - start_time) * 1000 / repeat  # in ms
    mem_used = end_mem - start_mem  # in MB
    
    return result, avg_time, mem_used

def test_cross_entropy_equivalence():
    device = get_device()
    print(f"Using device: {device}")
    
    # Test with different sizes
    test_cases = [
        (4, 8, 16, 4),         # Tiny (for quick verification)
        (16, 64, 512, 4),     # Small
        (32, 128, 2048, 4),   # Medium
        (64, 256, 8192, 4),   # Large
    ]
    
    for batch_size, seq_len, vocab_size, pad_id in test_cases:
        print(f"\nTesting with batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
        logits, labels = random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device)
        
        # ===== EQUIVALENCE TESTING =====
        print("\n[EQUIVALENCE TESTING]")
        
        # Test 1: Basic cross-entropy vs cross_entropy_with_entropy with weight=0
        ce_loss = cross_entropy_loss(logits, labels, pad_id)
        ce_loss2, entropy, total_loss = cross_entropy_with_entropy_loss(
            logits, labels, pad_id, entropy_weight=0.0
        )
        
        # Print detailed comparison
        print(f"Basic CE loss: {ce_loss.item():.8f}")
        print(f"CE with entropy (w=0): {ce_loss2.item():.8f}")
        print(f"Total loss (should match above): {total_loss.item():.8f}")
        
        # Check numerical equivalence
        assert torch.allclose(ce_loss, ce_loss2, atol=1e-6, rtol=1e-4), \
            f"Loss mismatch: {ce_loss.item()} vs {ce_loss2.item()}"
        assert torch.allclose(total_loss, ce_loss2, atol=1e-6, rtol=1e-4), \
            f"Total loss mismatch: {total_loss.item()} vs {ce_loss2.item()}"
            
        # Test 2: Verify entropy calculation is non-zero when weight > 0
        _, entropy, _ = cross_entropy_with_entropy_loss(
            logits, labels, pad_id, entropy_weight=0.001
        )
        print(f"Entropy value: {entropy.item():.8f}")
        assert entropy > 0, "Entropy should be positive"
        
        print("[EQUIVALENCE TESTS PASSED]\n")
        
        # Benchmark cross_entropy_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _, ce_time, ce_mem = benchmark_fn(
            cross_entropy_loss, logits, labels, pad_id,
            warmup=5, repeat=100
        )
        
        # Benchmark cross_entropy_with_entropy_loss with entropy_weight=0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create a wrapper function with fixed entropy_weight
        def ce_with_entropy_wrapper(logits, labels, pad_id):
            return cross_entropy_with_entropy_loss(logits, labels, pad_id, entropy_weight=0.0)[0]
            
        _, cee_time, cee_mem = benchmark_fn(
            ce_with_entropy_wrapper,
            logits, labels, pad_id,
            warmup=5, repeat=100
        )
        
        print(f"cross_entropy_loss: {ce_time:.4f} ms, {ce_mem:.2f} MB")
        print(f"cross_entropy_with_entropy (w=0): {cee_time:.4f} ms, {cee_mem:.2f} MB")
        print(f"Speed ratio: {cee_time/ce_time:.2f}x")
        print(f"Memory ratio: {cee_mem/ce_mem:.2f}x")
        
        # Test with non-zero entropy weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        def ce_with_weighted_entropy_wrapper(logits, labels, pad_id):
            return cross_entropy_with_entropy_loss(logits, labels, pad_id, entropy_weight=0.001)
            
        _, cee_time_weighted, cee_mem_weighted = benchmark_fn(
            ce_with_weighted_entropy_wrapper,
            logits, labels, pad_id,
            warmup=5, repeat=100
        )
        print(f"cross_entropy_with_entropy (w=0.001): {cee_time_weighted:.4f} ms, {cee_mem_weighted:.2f} MB")
        print(f"Speed ratio (weighted): {cee_time_weighted/ce_time:.2f}x")
        print(f"Memory ratio (weighted): {cee_mem_weighted/ce_mem:.2f}x")

if __name__ == "__main__":
    test_cross_entropy_equivalence()
    print("\nAll tests passed.")
