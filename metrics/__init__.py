"""Modular metrics system for training monitoring.

This package provides a comprehensive metrics collection and logging system
designed for modern ML training pipelines. It's optimized for XLA/TPU workloads
with minimal overhead and aesthetic logging.

Quick Start:
    >>> from metrics import TrainingLogger, MetricsTracker
    >>> logger = TrainingLogger(project="my-project", run_name="experiment-1")
    >>> tracker = MetricsTracker(model, sequence_length=2048, world_size=8)
    >>> 
    >>> # In training loop:
    >>> tracker.update(batch_size=64)
    >>> logger.log_step(loss=loss.item(), lr=scheduler.get_last_lr()[0])
"""

# Core metrics computation
from .compute import (
    compute_model_flops_per_token,
    compute_mfu,
    compute_tokens_in_batch,
)

# Tracking and accumulation
from .tracker import MetricsTracker, LossTracker

# Logging and visualization
from .logger import TrainingLogger, LogConfig

__version__ = "0.1.0"

__all__ = [
    # Core computation
    "compute_model_flops_per_token",
    "compute_mfu", 
    "compute_tokens_in_batch",
    
    # Tracking
    "MetricsTracker",
    "LossTracker",
    
    # Logging
    "TrainingLogger",
    "LogConfig",
]