import time
import os
import psutil
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Local import
from losses.cut_cross_entropy import linear_cross_entropy


def _dummy_data(batch: int, seq: int, dim: int, vocab: int, device):
    """Create random embeddings / classifier / labels."""
    tokens = batch * seq
    embeddings = torch.randn(tokens, dim, device=device, requires_grad=True)
    classifier = torch.randn(vocab, dim, device=device, requires_grad=True)
    labels = torch.randint(0, vocab, (tokens,), device=device)
    return embeddings, classifier, labels


def _get_cpu_mem() -> int:
    """Get current process resident memory (RSS) in bytes."""
    return psutil.Process(os.getpid()).memory_info().rss

def _bench_standard(emb, cls, y):
    """Standard cross-entropy via materialised logits."""
    start_cuda = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    start_cpu = _get_cpu_mem()
    t0 = time.perf_counter()

    logits = emb @ cls.T  # (N, V)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    end_cpu = _get_cpu_mem()
    peak_cuda = (torch.cuda.max_memory_allocated() - start_cuda) if start_cuda is not None else None
    peak_cpu = end_cpu - start_cpu
    return t1 - t0, peak_cuda, peak_cpu


def _bench_cut_ce(emb, cls, y):
    start_cuda = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    start_cpu = _get_cpu_mem()
    t0 = time.perf_counter()

    loss = linear_cross_entropy(emb, cls, y)
    loss.backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    end_cpu = _get_cpu_mem()
    peak_cuda = (torch.cuda.max_memory_allocated() - start_cuda) if start_cuda is not None else None
    peak_cpu = end_cpu - start_cpu
    return t1 - t0, peak_cuda, peak_cpu


def run_benchmark(
    batch: int,
    seq: int,
    dim: int,
    vocab: int,
    device: str | torch.device | None = None,
    trials: int = 2
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*80}\nBenchmarking batch={batch}, seq={seq}, dim={dim}, vocab={vocab}, device={device}\n{'='*80}")

    results = []
    for trial in range(trials):
        torch.manual_seed(trial)
        torch.cuda.manual_seed_all(trial)
        import random
        random.seed(trial)

        # Generate data
        emb, cls, y = _dummy_data(batch, seq, dim, vocab, device)
        emb_std = emb.clone().detach().requires_grad_(True)
        cls_std = cls.clone().detach().requires_grad_(True)
        y_std = y.clone().detach()
        emb_cut = emb.clone().detach().requires_grad_(True)
        cls_cut = cls.clone().detach().requires_grad_(True)
        y_cut = y.clone().detach()

        # Benchmark Standard CE
        t0_std = time.perf_counter()
        logits = emb_std @ cls_std.T
        loss_std = F.cross_entropy(logits, y_std)
        loss_std.backward()
        t1_std = time.perf_counter()
        std_time = t1_std - t0_std
        std_cuda_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        std_cpu_mem = _get_cpu_mem()

        # Benchmark Cut CE
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0_cut = time.perf_counter()
        loss_cut = linear_cross_entropy(emb_cut, cls_cut, y_cut)
        loss_cut.backward()
        t1_cut = time.perf_counter()
        cut_time = t1_cut - t0_cut
        cut_cuda_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        cut_cpu_mem = _get_cpu_mem()

        # Compare numerical accuracy
        loss_diff = abs(loss_std.item() - loss_cut.item())
        grad_emb_diff = (emb_std.grad - emb_cut.grad).abs().max().item()
        grad_cls_diff = (cls_std.grad - cls_cut.grad).abs().max().item()

        # Store results
        result = {
            'trial': trial + 1,
            'batch': batch,
            'std_time': std_time,
            'cut_time': cut_time,
            'std_cuda_mem': std_cuda_mem / (1024 ** 2) if std_cuda_mem else None,
            'cut_cuda_mem': cut_cuda_mem / (1024 ** 2) if cut_cuda_mem else None,
            'std_cpu_mem': std_cpu_mem / (1024 ** 2),
            'cut_cpu_mem': cut_cpu_mem / (1024 ** 2),
            'loss_diff': loss_diff,
            'grad_emb_diff': grad_emb_diff,
            'grad_cls_diff': grad_cls_diff
        }
        results.append(result)

        # Print trial results
        print(f"\nTrial {trial + 1}:")
        print(f"  Standard CE: {std_time:.3f}s | "
              f"CUDA: {result['std_cuda_mem']:.2f} MB | "
              f"CPU: {result['std_cpu_mem']:.2f} MB")
        print(f"  Cut CE     : {cut_time:.3f}s | "
              f"CUDA: {result['cut_cuda_mem']:.2f} MB | "
              f"CPU: {result['cut_cpu_mem']:.2f} MB")
        print(f"  Speedup    : {std_time / cut_time:.2f}x")
        print(f"  Loss diff  : {loss_diff:.2e}")
        print(f"  Grad diffs : emb={grad_emb_diff:.2e}, cls={grad_cls_diff:.2e}")

    return results


def compare_loss_and_grads(batch, seq, dim, vocab, device, runs=5, threshold=3e-5):
    print(f"\nComparing Standard CE vs Cut CE for batch={batch}, seq={seq}, dim={dim}, vocab={vocab}, runs={runs}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for run in range(runs):
        torch.manual_seed(run)
        import random
        random.seed(run)
        # Generate same data for both
        emb, cls, y = _dummy_data(batch, seq, dim, vocab, device)
        emb_std = emb.clone().detach().requires_grad_(True)
        cls_std = cls.clone().detach().requires_grad_(True)
        y_std = y.clone().detach()
        emb_cut = emb.clone().detach().requires_grad_(True)
        cls_cut = cls.clone().detach().requires_grad_(True)
        y_cut = y.clone().detach()

        # Standard CE
        logits = emb_std @ cls_std.T
        loss_std = F.cross_entropy(logits, y_std)
        loss_std.backward()
        grad_emb_std = emb_std.grad.detach().cpu()
        grad_cls_std = cls_std.grad.detach().cpu()

        # Cut CE
        loss_cut = linear_cross_entropy(emb_cut, cls_cut, y_cut)
        loss_cut.backward()
        grad_emb_cut = emb_cut.grad.detach().cpu()
        grad_cls_cut = cls_cut.grad.detach().cpu()

        # Compare
        loss_diff = abs(loss_std.item() - loss_cut.item())
        grad_emb_diff = (grad_emb_std - grad_emb_cut).abs().max().item()
        grad_cls_diff = (grad_cls_std - grad_cls_cut).abs().max().item()
        print(f"Run {run+1}: loss_diff={loss_diff:.2e}, grad_emb_diff={grad_emb_diff:.2e}, grad_cls_diff={grad_cls_diff:.2e}")
        if loss_diff > threshold or grad_emb_diff > threshold or grad_cls_diff > threshold:
            print(f"  WARNING: Difference exceeds threshold {threshold}")
    print("Comparison complete.\n")

def print_summary(results):
    """Print a summary of benchmark results across all batch sizes."""
    print("\n" + "="*100)
    print(f"{'Batch':>6} | {'Trial':>6} | {'Std Time (s)':>12} | {'Cut Time (s)':>12} | "
          f"{'Std CUDA (MB)':>14} | {'Cut CUDA (MB)':>14} | "
          f"{'Std CPU (MB)':>12} | {'Cut CPU (MB)':>12} | "
          f"{'Speedup':>8} | {'Loss Diff':>10} | {'Grad Emb Diff':>14} | {'Grad Cls Diff':>14}")
    print("-" * 150)

    for result in results:
        for trial in result['trials']:
            print(f"{result['batch']:6d} | "
                  f"{trial['trial']:6d} | "
                  f"{trial['std_time']:12.3f} | "
                  f"{trial['cut_time']:12.3f} | "
                  f"{trial['std_cuda_mem']:14.2f} | "
                  f"{trial['cut_cuda_mem']:14.2f} | "
                  f"{trial['std_cpu_mem']:12.2f} | "
                  f"{trial['cut_cpu_mem']:12.2f} | "
                  f"{trial['std_time']/trial['cut_time']:8.2f}x | "
                  f"{trial['loss_diff']:10.2e} | "
                  f"{trial['grad_emb_diff']:14.2e} | "
                  f"{trial['grad_cls_diff']:14.2e}")
    print("="*100 + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Cut Cross-Entropy vs standard CE")
    parser.add_argument("--batch", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32, 64, 128],
                       help="Batch sizes to benchmark (space-separated)")
    parser.add_argument("--seq", type=int, default=128, help="Sequence length")
    parser.add_argument("--dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials per batch size")
    args = parser.parse_args()

    all_results = []
    for batch in args.batch:
        try:
            results = run_benchmark(batch, args.seq, args.dim, args.vocab, args.device, args.trials)
            all_results.append({
                'batch': batch,
                'trials': results
            })
        except Exception as e:
            print(f"Error benchmarking batch={batch}: {e}")
            continue

    # Print final summary
    print_summary(all_results)
