import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import psutil
import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from losses.cross_entropy import cross_entropy_loss, cross_entropy_with_entropy_loss

def get_memory_usage():
    """Returns memory usage in MB"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

def random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device):
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Randomly set some labels to pad_id (25% chance)
    mask = torch.rand(batch_size, seq_len, device=device) < 0.25
    labels[mask] = pad_id
    return logits, labels

def benchmark(fn, *args, warmup=5, repeat=100, **kwargs):
    """Benchmark a function's execution time and memory usage"""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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

def test_equivalence():
    """Verify that both implementations produce the same results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_id = 0
    batch_size, seq_len, vocab_size = 4, 8, 16
    logits, labels = random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device)
    
    # Test standard CE
    ce_loss = cross_entropy_loss(logits, labels, pad_id)
    
    # Test fused CE + entropy with weight=0
    ce_loss2, _, _ = cross_entropy_with_entropy_loss(logits, labels, pad_id, entropy_weight=0.0)
    
    # Verify they're numerically equivalent
    assert torch.allclose(ce_loss, ce_loss2, atol=1e-6, rtol=1e-4), \
        f"Loss mismatch: {ce_loss.item()} vs {ce_loss2.item()}"
    
    print("✅ Equivalence test passed")

def run_benchmarks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_id = 0
    
    # Test cases: (batch_size, seq_len, vocab_size, repeat_count)
    test_cases = [
        (8, 32, 128, 500),     # Small
        (32, 128, 1024, 100),  # Medium
        (64, 256, 4096, 20),   # Large
    ]
    
    print(f"\n{'='*80}")
    print(f"{'Cross-Entropy Implementation Comparison':^80}")
    print(f"{'Device:':<15} {device}")
    print(f"{'='*80}")
    
    for batch_size, seq_len, vocab_size, repeat_count in test_cases:
        logits, labels = random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device)
        
        print(f"\n{'='*80}")
        print(f"Batch: {batch_size}, Seq: {seq_len}, Vocab: {vocab_size}")
        print(f"Logits shape: {tuple(logits.shape)}, Labels shape: {tuple(labels.shape)}")
        print(f"{'='*80}")
        
        # Standard CE
        _, t1, m1 = benchmark(
            cross_entropy_loss, logits, labels, pad_id,
            warmup=5, repeat=repeat_count
        )
        
        # Fused CE + entropy (with weight=0 for fair comparison)
        (_, _, _), t2, m2 = benchmark(
            lambda *a, **kw: cross_entropy_with_entropy_loss(*a, **kw, entropy_weight=0.0),
            logits, labels, pad_id,
            warmup=5, repeat=repeat_count
        )
        
        # Fused CE + entropy (with actual entropy computation)
        (_, entropy, _), t3, m3 = benchmark(
            cross_entropy_with_entropy_loss,
            logits, labels, pad_id, 0.001,
            warmup=5, repeat=repeat_count
        )
        
        # Print results
        print(f"{'Standard CE:':<35} {t1:>8.4f} ms, {m1:>8.4f} MB")
        print(f"{'Fused CE (w=0):':<35} {t2:>8.4f} ms, {m2:>8.4f} MB  ({(t2/t1-1)*100:+.1f}% time, {m2/m1:.2f}x memory)")
        print(f"{'Fused CE (w=0.001):':<35} {t3:>8.4f} ms, {m3:>8.4f} MB  ({(t3/t1-1)*100:+.1f}% time, {m3/m1:.2f}x memory)")
        print(f"{'Entropy value:':<35} {entropy.item():.6f}")

if __name__ == "__main__":
    test_equivalence()
    run_benchmarks()
