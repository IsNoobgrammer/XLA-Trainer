import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import psutil
import os


def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

def random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device):
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = torch.rand(batch_size, seq_len, device=device) < 0.25
    labels[mask] = pad_id
    return logits, labels

def entropy_all_positions(logits):
    # logits: (batch, seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq)
    entropy_mean = entropy.mean()
    return entropy_mean

def benchmark(fn, *args, repeat=100, **kwargs):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start_mem = get_memory_usage()
    start_time = time.perf_counter()
    for _ in range(repeat):
        result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    end_mem = get_memory_usage()
    avg_time = (end_time - start_time) * 1000 / repeat
    mem_used = end_mem - start_mem
    return result, avg_time, mem_used

def ce_f_entropy(logits, labels, pad_id):
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=pad_id
    )
    entropy = entropy_all_positions(logits)
    return ce, entropy

def ce_nn_entropy(logits, labels, pad_id):
    ce_module = nn.CrossEntropyLoss(ignore_index=pad_id)
    ce = ce_module(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    entropy = entropy_all_positions(logits)
    return ce, entropy

def ce_logprob_nll_entropy_fused(logits, labels, pad_id):
    # Compute log_probs once, use for both loss and entropy
    log_probs = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        labels.view(-1),
        ignore_index=pad_id
    )
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return ce, entropy

def run_benchmarks(batch_size, seq_len, vocab_size, pad_id, device, repeat=100):
    logits, labels = random_logits_labels(batch_size, seq_len, vocab_size, pad_id, device)
    print(f"\n=== Shape: batch={batch_size}, seq={seq_len}, vocab={vocab_size} ===")
    print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

    print("[F.cross_entropy + entropy]")
    (_, entropy1), t1, m1 = benchmark(ce_f_entropy, logits, labels, pad_id, repeat=repeat)
    print(f"Avg time: {t1:.6f} ms, Mem: {m1:.4f} MB, Entropy: {entropy1.item():.6f}")

    print("[nn.CrossEntropyLoss + entropy]")
    (_, entropy2), t2, m2 = benchmark(ce_nn_entropy, logits, labels, pad_id, repeat=repeat)
    print(f"Avg time: {t2:.6f} ms, Mem: {m2:.4f} MB, Entropy: {entropy2.item():.6f}")

    print("[log_softmax + nll_loss + entropy (fused)]")
    (_, entropy3), t3, m3 = benchmark(ce_logprob_nll_entropy_fused, logits, labels, pad_id, repeat=repeat)
    print(f"Avg time: {t3:.6f} ms, Mem: {m3:.4f} MB, Entropy: {entropy3.item():.6f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_id = 0
    print(f"Device: {device}")
    # Small
    run_benchmarks(batch_size=8, seq_len=32, vocab_size=128, pad_id=pad_id, device=device, repeat=200)
    # Medium
    run_benchmarks(batch_size=32, seq_len=128, vocab_size=1024, pad_id=pad_id, device=device, repeat=50)
    # Large
    run_benchmarks(batch_size=64, seq_len=256, vocab_size=4096, pad_id=pad_id, device=device, repeat=10)

if __name__ == "__main__":
    main()
