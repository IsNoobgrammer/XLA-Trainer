"""Core metrics computation functions.

Stateless helper functions for computing performance metrics like MFU,
FLOP counts, and throughput. These functions are backend-agnostic and
work with any PyTorch model.
"""
from __future__ import annotations

import torch

__all__ = [
    "compute_model_flops_per_token",
    "compute_mfu",
    "compute_tokens_in_batch",
]


def compute_model_flops_per_token(
    model: torch.nn.Module,
    *,
    include_backward: bool = True,
    backward_trainable_only: bool = True,
) -> int:
    """Estimate FLOPs required to process **one** token.

    The estimate follows a simplified rule-of-thumb widely used for
    autoregressive Transformers:

    * Forward pass ≈ ``#parameters`` multiply-adds per token.
    * Backward pass ≈ ``#parameters`` multiply-adds per token *if* gradients
      are computed for that parameter.

    The caller can specify whether the backward contribution should include all
    parameters or **only** those that require gradients.

    Parameters
    ----------
    model: torch.nn.Module
        Model under consideration.
    include_backward: bool, default = True
        If *False* only the forward contribution is counted.
    backward_trainable_only: bool, default = True
        When ``include_backward`` is *True* this flag controls whether the
        backward contribution should be restricted to *trainable* parameters
        (*PyTorch ``requires_grad``*) or *all* parameters.

    Returns
    -------
    int
        Estimated FLOPs per token.
    """
    # Forward contribution always counts *all* parameters.
    n_params_total = sum(p.numel() for p in model.parameters())
    flops = n_params_total  # forward only

    if include_backward:
        if backward_trainable_only:
            n_params_backward = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            n_params_backward = n_params_total
        flops += n_params_backward

    return flops


def compute_mfu(
    model: torch.nn.Module,
    tokens_per_second: float,
    theoretical_peak_flops: float,
) -> float:
    """Compute **Model FLOPs Utilisation (MFU)**.

    MFU is defined as the achieved FLOPs divided by the theoretical peak FLOPs
    of the accelerator.  Values close to *1.0* indicate that the hardware is
    being used to near peak efficiency.

    Notes
    -----
    This is an *estimate*.  Precise FLOP counting is architecture dependent and
    outside the scope of this helper.

    Parameters
    ----------
    model: torch.nn.Module
        The model currently being trained.
    tokens_per_second: float
        Throughput measured *after* data loading and forward/backward passes
        have been overlapped.
    theoretical_peak_flops: float
        Device peak FLOPs **per second**.  For example, TPU-v4 ≈ 275 TFLOPs per
        core when using BF16.  Remember to convert *TFLOPs → FLOPs* (multiply
        by 1e12).

    Returns
    -------
    float
        MFU in the range ``[0.0, 1.0]``.
    """
    flops_per_token = compute_model_flops_per_token(model)
    achieved_flops = flops_per_token * tokens_per_second
    return achieved_flops / theoretical_peak_flops


def compute_tokens_in_batch(batch_size: int, sequence_length: int) -> int:
    """Helper to obtain *total* number of tokens for a batch."""
    return batch_size * sequence_length


def compute_throughput(
    total_tokens: int,
    elapsed_seconds: float,
) -> float:
    """Compute tokens per second throughput."""
    return total_tokens / max(elapsed_seconds, 1e-6)


def compute_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in megabytes."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assume 4 bytes per parameter (fp32) or 2 bytes (fp16/bf16)
    # Use 4 bytes as conservative estimate
    return (total_params * 4) / (1024 * 1024)


def compute_gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """Compute the gradient norm across all model parameters."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.0
    
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    return total_norm.item()