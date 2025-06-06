"""Model parameter smoothing on TPU/XLA.

This module provides *parameter smoothing* a.k.a. weight EMA (exponential
moving average) / SWA-like update that can be *scheduled* every `k` steps.

The classical EMA update is:

    smoothed = decay * smoothed + (1 - decay) * params

Instead, to keep the extra memory low we keep a single *buffer* copy of the
weights (captured at step multiples of ``update_interval``) and perform a
cheap in-place update every ``update_interval`` steps::

    if step % update_interval == 0:
        # blend current parameters with buffer — default is simple mean
        for p, b in zip(model.parameters(), buffer):
            p.mul_(0.5).add_(b, alpha=0.5)
            b.copy_(p)  # refresh buffer ← blended params

This behaviour matches the sketch given by the user:

    • At step 0 we store a copy of the weights ⇒ *buffer*
    • At step ``k`` we add ``buffer / 2`` to the live model/2
    • Now the buffer holds the *k-th* weights, so at step 2k we again add
      ``buffer / 2`` and so on.

We expose a minimal public API so callers just need to instantiate a
`ModelSmoother` **once** and call ``maybe_smooth(step)`` each training step.

The implementation is XLA-aware and SPMD-safe:

1. Parameters live on TPU devices → we allocate the buffer on the same device.
2. We wrap updates in ``torch.xla._XLAC._xla_mark_step`` to respect XLA graphs.
3. We preserve SPMD sharding specifications when cloning buffers to maintain
   tensor parallelism across TPU cores.
4. All operations are in-place to avoid creating new tensors that might lose
   sharding annotations.
5. We avoid moving tensors across shards; this works with both data and model
   parallelism.

Memory impact: one extra copy of parameters (the buffer). This is acceptable
in most scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Iterator, Optional, Union

import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    XLA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Fallback to CPU/GPU for unit-tests or local runs.
    XLA_AVAILABLE = False
    xm = None  # type: ignore
    xs = None  # type: ignore
    Mesh = None  # type: ignore


@dataclass
class SmoothingConfig:
    """Configuration for :class:`ModelSmoother`."""

    update_interval: int = 1_000  # perform smoothing every *k* optimizer steps
    alpha: float = 0.5  # weight given to *buffer* when averaging with live params

    def __post_init__(self) -> None:
        if self.update_interval <= 0:
            raise ValueError("update_interval must be > 0")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")


class ModelSmoother:
    """Light-weight weight-smoothing helper.

    Typical usage::

        smoother = ModelSmoother(model, SmoothingConfig(update_interval=500))
        for step in range(total_steps):
            loss = forward_backward_update(model, batch)
            smoother.maybe_smooth(step)
    """

    def __init__(self, model: torch.nn.Module, cfg: Optional[SmoothingConfig] = None):
        self.model = model
        self.cfg = cfg or SmoothingConfig()

        # Single buffer storing last captured parameters
        # CRITICAL: preserve SPMD sharding annotations when cloning
        self._buffer: List[torch.Tensor] = []
        for p in model.parameters():
            buf = p.detach().clone()
            buf.requires_grad_(False)
            
            # Preserve SPMD sharding spec if available
            if XLA_AVAILABLE and hasattr(p, '_sharding_spec') and p._sharding_spec is not None:
                # Copy the sharding specification to maintain tensor parallelism
                buf = xs.mark_sharding(buf, xs.get_1d_mesh(), p._sharding_spec)
            elif XLA_AVAILABLE and hasattr(p, 'sharding_spec'):
                # Alternative sharding spec attribute name
                buf = xs.mark_sharding(buf, xs.get_1d_mesh(), p.sharding_spec)
            
            self._buffer.append(buf)

    def _preserve_sharding(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Helper to preserve SPMD sharding from source to target tensor."""
        if not XLA_AVAILABLE:
            return target
            
        # Try different ways to access sharding spec (API may vary)
        sharding_spec = None
        for attr in ['_sharding_spec', 'sharding_spec', '_spec']:
            if hasattr(source, attr):
                sharding_spec = getattr(source, attr)
                if sharding_spec is not None:
                    break
        
        if sharding_spec is not None:
            try:
                # Try to apply the sharding spec to target
                return xs.mark_sharding(target, xs.get_1d_mesh(), sharding_spec)
            except Exception:
                # Fallback: return target as-is if sharding fails
                pass
        
        return target

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def maybe_smooth(self, step: int) -> None:
        """Perform smoothing if *step* is a multiple of ``update_interval``."""
        if step % self.cfg.update_interval != 0:
            return  # nothing to do

        # Ensure we update inside an XLA step marker when applicable
        if XLA_AVAILABLE:
            xm.mark_step()

        with torch.no_grad():
            buf_iter: Iterator[torch.Tensor] = iter(self._buffer)
            for p in self.model.parameters():
                buf = next(buf_iter)
                
                # Verify sharding consistency (debug check)
                if XLA_AVAILABLE and hasattr(p, '_sharding_spec') and hasattr(buf, '_sharding_spec'):
                    if p._sharding_spec != buf._sharding_spec:
                        raise RuntimeError(f"Sharding spec mismatch between param and buffer: "
                                         f"{p._sharding_spec} vs {buf._sharding_spec}")
                
                # new_p = (1 - alpha) * p + alpha * buffer  (element-wise mean when alpha=0.5)
                # Use in-place operations to preserve sharding
                p.mul_(1.0 - self.cfg.alpha).add_(buf, alpha=self.cfg.alpha)
                buf.copy_(p)  # refresh buffer with the blended value

        if XLA_AVAILABLE:
            # Mark another step so that the copy_ ops are materialised.
            xm.mark_step()

    # ------------------------------------------------------------------
    # checkpointing utilities (optional)
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, torch.Tensor]:  # compatible with torch api
        return {f"buffer_{i}": t.clone() for i, t in enumerate(self._buffer)}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if len(state) != len(self._buffer):
            raise ValueError("State dict size mismatch for ModelSmoother")
        for i, t in enumerate(self._buffer):
            key = f"buffer_{i}"
            if key not in state:
                raise KeyError(f"Missing {key} in state dict")
            t.copy_(state[key])
