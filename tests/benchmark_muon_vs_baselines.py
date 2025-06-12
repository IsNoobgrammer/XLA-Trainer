"""
Benchmark SingleDeviceMuonWithAuxAdam (Muon+AdamW) vs. AdamW, SGD, RMSprop
on a larger model. Visualizes step time and CUDA memory usage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import sys
import traceback
from optimizers.raw_optimizers.muon_base_torch import SingleDeviceMuonWithAuxAdam
from optimizers.raw_optimizers.muon_base_torch_legacy import SingleDeviceMuonWithAuxAdamLegacy

def flush():
    sys.stdout.flush()
    sys.stderr.flush()

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.bias = nn.Parameter(torch.zeros(10))  # 1D param
    def forward(self, x):
        x = self.seq(x)
        return x + self.bias

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_param_groups(model):
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    return [
        {"params": muon_params, "use_muon": True},
        {"params": adamw_params, "use_muon": False}
    ]

def benchmark_optimizer(opt, model, device, n_steps=10):
    torch.cuda.empty_cache()
    model.train()
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    x = torch.randn(batch_size, 512, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    # Warmup
    for _ in range(2):
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(n_steps):
        opt.zero_grad()
        start = time.time()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize(device)
        times.append(time.time() - start)
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    return sum(times) * 1000 / len(times), max_mem

def main():
    print("[START] Running Muon (legacy vs optimized) vs Baselines benchmark...")
    flush()
    device = get_device()
    print(f"Using device: {device}")
    results = []
    names = []
    # Muon (legacy)
    model0 = LargeMLP().to(device)
    muon_opt_legacy = SingleDeviceMuonWithAuxAdamLegacy(get_param_groups(model0))
    t_muon_legacy, mem_muon_legacy = benchmark_optimizer(muon_opt_legacy, model0, device)
    results.append((t_muon_legacy, mem_muon_legacy))
    names.append("Muon (legacy)")
    # Muon (optimized)
    model1 = LargeMLP().to(device)
    muon_opt = SingleDeviceMuonWithAuxAdam(get_param_groups(model1))
    t_muon, mem_muon = benchmark_optimizer(muon_opt, model1, device)
    results.append((t_muon, mem_muon))
    names.append("Muon (opt)")
    # AdamW
    model2 = LargeMLP().to(device)
    adamw_opt = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    t_adamw, mem_adamw = benchmark_optimizer(adamw_opt, model2, device)
    results.append((t_adamw, mem_adamw))
    names.append("AdamW")
    # SGD
    model3 = LargeMLP().to(device)
    sgd_opt = torch.optim.SGD(model3.parameters(), lr=1e-2, momentum=0.9)
    t_sgd, mem_sgd = benchmark_optimizer(sgd_opt, model3, device)
    results.append((t_sgd, mem_sgd))
    names.append("SGD")
    # RMSprop
    model4 = LargeMLP().to(device)
    rms_opt = torch.optim.RMSprop(model4.parameters(), lr=1e-2)
    t_rms, mem_rms = benchmark_optimizer(rms_opt, model4, device)
    results.append((t_rms, mem_rms))
    names.append("RMSprop")
    # Print results
    print("\nBenchmark Results:")
    print(f"{'Optimizer':<15} {'Step Time (ms)':>15} {'Max CUDA Mem (MB)':>20}")
    for name, (t, mem) in zip(names, results):
        print(f"{name:<15} {t:>15.2f} {mem:>20.2f}")
    flush()
    # Plot
    fig, ax1 = plt.subplots()
    color1 = 'tab:blue'
    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Step Time (ms)', color=color1)
    ax1.bar(names, [r[0] for r in results], color=color1, alpha=0.6, label='Step Time (ms)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Max CUDA Mem (MB)', color=color2)
    ax2.plot(names, [r[1] for r in results], color=color2, marker='o', label='Max CUDA Mem (MB)')
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title('Optimizer Step Time and Memory Usage')
    fig.tight_layout()
    plt.savefig('optimizer_benchmark.png')
    plt.show()
    print("[END] Benchmark finished. Plot saved as optimizer_benchmark.png")
    flush()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("[ERROR] Exception during benchmark run:")
        traceback.print_exc()
        flush()
