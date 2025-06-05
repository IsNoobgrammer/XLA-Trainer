# TPU-Trainer

A modular, accelerator-agnostic deep learning training framework focused on PyTorch, XLA/TPU, and efficient transformer/NLP workflows.

---

### 🟡 In Progress / To Do
See the TODO section below for planned features and research directions.

---

## TODO

### 1. SPMD-Compatible Checkpointing
- **Goal:** Enable robust checkpointing that works with SPMD (Single Program, Multiple Data) distributed training on TPUs.
- **Reference:**  
  - [PyTorch/XLA SPMD Distributed Checkpointing Docs](https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html)
  - [GitHub: spmd_distributed_checkpoint.md](https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_distributed_checkpoint.md)
- **Notes:**  
  - Use `torch.distributed.checkpoint` with a CPU process group (e.g., `gloo`) for SPMD mode.
  - XLA backend is not supported for collectives in SPMD, so CPU group is needed.
  - On Cloud TPU, auto-checkpointing on preemption is supported.
- **Action Items:**
  - Integrate SPMD-compatible checkpointing logic.
  - Test save/load on multi-host TPU pod.

### 2. Cut Cross Entropy (Tilewise/Blockwise)
- **Goal:** Implement memory-efficient cross-entropy loss (tilewise/blockwise) for large-vocab models.
- **Reference:**  
  - [Apple ML: Cut Your Losses in Large-Vocabulary Language Models](https://machinelearning.apple.com/research/cut-your-losses)
  - [arXiv: Liger Kernels and CCE](https://arxiv.org/html/2411.09009v1)
  - [cut-cross-entropy PyPI](https://pypi.org/project/cut-cross-entropy/)
- **Notes:**  
  - Standard cross-entropy materializes a large logits matrix (batch, vocab), causing high memory use.
  - Cut Cross-Entropy (CCE) computes loss in tiles/chunks, reducing memory by up to 95%.
  - May have a tradeoff with compute or numerical stability.
- **Action Items:**
  - Prototype tilewise/blockwise cross-entropy (forward & backward).
  - Benchmark memory and speed vs. standard loss.

### 3. Better Logging with wandb & XLA Metrics
- **Goal:** Improve experiment tracking with Weights & Biases (wandb) and detailed metrics, while handling XLA device/host tensor movement safely.
- **Reference:**  
  - [PyTorch/XLA: Moving Tensors Between Host/Device](https://docs.pytorch.org/xla/release/r2.7/learn/pytorch-on-xla-devices.html)
- **Notes:**  
  - Use `.cpu()` or `.to('cpu')` to move XLA tensors to host before logging.
  - Avoid logging XLA tensors directly (can cause sync issues or errors).
  - For metrics, prefer `tensor.item()` after moving to CPU.
- **Action Items:**
  - Refactor logging to ensure all tensors are on CPU before wandb log.
  - Add more granular metrics (e.g., per-step, per-epoch, memory usage).

### 4. Better Optimizer: Newton-Schulz, AdamW, Lion, Muon
- **Goal:** Implement and benchmark advanced optimizers, including a robust Newton-Schulz update, AdamW, Lion, and Muon.
- **Reference:**  
  - [PyTorch Optimizer API](https://pytorch.org/docs/stable/optim.html)
- **Notes:**  
  - Newton-Schulz is useful for matrix inverse roots (e.g., in second-order methods).
  - Lion and Muon are recent optimizers for LLMs.
- **Action Items:**
  - Modularize optimizer code.
  - Add/test new optimizers.

### 5. Config File (YAML)
- **Goal:** Centralize all hyperparameters and settings in a YAML config.
- **Notes:**  
  - Use `omegaconf` or `pyyaml`.
  - Keep a main config and a separate YAML for Splash/FlashAttention.
- **Action Items:**
  - Implement config loading/parsing.
  - Allow referencing sub-configs (e.g., for attention modules).

### 6. Modularize Loss Functions
- **Goal:** Make loss functions pluggable and extensible.
- **Action Items:**
  - Refactor to support multiple loss types.
  - Add more loss options (e.g., label smoothing, focal, CCE).

### 7. Modularize Optimizer & Scheduler
- **Goal:** Self-contained, easily swappable optimizer and scheduler modules.
- **Action Items:**
  - Refactor optimizer/scheduler setup.
  - Support custom schedules.

### 8. Remainder Checkpointing for TPU
- **Goal:** Ensure checkpointing handles remainder/partial steps on TPU (important for fault tolerance).
- **Action Items:**
  - Add logic for checkpointing at non-epoch boundaries.
  - Test recovery from partial checkpoints.

---

## Directory Structure

```
TPU-Trainer/
├── data/
│   ├── conversation_dataset.py  # Modular chat dataset
│   ├── pretrain_dataset.py      # Modular pretrain dataset
│   └── ...
├── moneypatch_attention/
│   ├── splash.py                # Splash/FlashAttention logic
│   ├── flash_wrappers.py        # Wrappers for FlashAttention
│   └── splash_wrappers.py       # Wrappers for SplashAttention
├── tests/
│   ├── test_datasets.py         # Dataset tests
│   ├── test_flash_wrappers.py   # FlashAttention tests
│   └── test_splash_wrappers.py  # SplashAttention tests
├── example_usage.py             # Example usage script
├── script.txt                   # Notes, scripts, or logs
└── ...
```

---

## Contributing

Contributions and suggestions are welcome! Please open an issue or PR.

---

## References
- [PyTorch/XLA SPMD Distributed Checkpointing](https://docs.pytorch.org/xla/master/perf/spmd_distributed_checkpoint.html)
- [Cut Your Losses in Large-Vocabulary Language Models](https://machinelearning.apple.com/research/cut-your-losses)
- [PyTorch/XLA: Moving Tensors Between Host/Device](https://docs.pytorch.org/xla/release/r2.7/learn/pytorch-on-xla-devices.html)
- [PyTorch Optimizer API](https://pytorch.org/docs/stable/optim.html)

---

For more details and up-to-date progress, see the code and tests in this repository.
