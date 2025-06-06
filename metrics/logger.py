"""Aesthetic logging and visualization for training metrics.

Provides a drag-and-drop logging interface that handles both console output
and W&B logging with minimal overhead and beautiful formatting.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

__all__ = ["TrainingLogger", "LogConfig", "format_metrics"]


@dataclass
class LogConfig:
    """Configuration for logging behavior."""
    
    # W&B settings
    project: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Logging frequency
    log_every: int = 50  # Log to W&B every N steps
    print_every: int = 25  # Print to console every N steps
    
    # Cost control
    max_wandb_mb: float = 100.0  # Max MB per run
    enable_histograms: bool = False  # Expensive gradient/weight histograms
    
    # Console formatting
    use_colors: bool = True
    compact_format: bool = False
    
    # XLA-specific
    master_only: bool = True  # Only log from master process


class TrainingLogger:
    """Aesthetic training logger with W&B integration.
    
    Provides a simple interface for logging training metrics with beautiful
    console output and optional W&B tracking. Optimized for XLA/TPU workloads.
    
    Example:
        >>> logger = TrainingLogger(project="my-project", run_name="exp-1")
        >>> logger.log_step(step=100, loss=1.5, lr=3e-4, mfu=0.56)
        >>> logger.log_validation(step=100, val_loss=1.2, val_acc=0.85)
    """
    
    def __init__(
        self,
        config: Optional[LogConfig] = None,
        **kwargs
    ):
        """Initialize the logger.
        
        Parameters
        ----------
        config: Optional[LogConfig]
            Logging configuration. If None, uses defaults.
        **kwargs
            Override config parameters directly
        """
        self.config = config or LogConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.wandb_run = None
        self._step_count = 0
        self._wandb_bytes = 0
        self._start_time = time.time()
        
        # Initialize W&B if available and configured
        if WANDB_AVAILABLE and self.config.project:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not self._should_log():
            return
            
        try:
            self.wandb_run = wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                tags=self.config.tags,
                notes=self.config.notes,
                resume="allow"
            )
            
            # Define custom metrics
            if self.wandb_run:
                wandb.define_metric("train/step")
                wandb.define_metric("train/*", step_metric="train/step")
                wandb.define_metric("val/*", step_metric="train/step")
                wandb.define_metric("system/*", step_metric="train/step")
                
        except Exception as e:
            print(f"⚠️  Failed to initialize W&B: {e}")
            self.wandb_run = None
    
    def _should_log(self) -> bool:
        """Check if this process should log (master only for XLA)."""
        if not self.config.master_only:
            return True
            
        if XLA_AVAILABLE:
            return xm.is_master_ordinal()
        return True
    
    def _format_value(self, value: Any) -> str:
        """Format a value for pretty printing."""
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:,.0f}"
            elif abs(value) >= 1:
                return f"{value:.3f}"
            elif abs(value) >= 0.001:
                return f"{value:.4f}"
            else:
                return f"{value:.2e}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)
    
    def _get_color_code(self, metric_name: str) -> str:
        """Get ANSI color code for a metric."""
        if not self.config.use_colors:
            return ""
            
        color_map = {
            "loss": "\033[91m",      # Red
            "lr": "\033[94m",        # Blue  
            "mfu": "\033[92m",       # Green
            "tokens": "\033[93m",    # Yellow
            "grad": "\033[95m",      # Magenta
            "val": "\033[96m",       # Cyan
        }
        
        for key, color in color_map.items():
            if key in metric_name.lower():
                return color
        return "\033[97m"  # White
    
    def _format_console_line(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for console output."""
        if self.config.compact_format:
            items = [f"{k}={self._format_value(v)}" for k, v in metrics.items()]
            return " | ".join(items)
        
        # Beautiful formatting with colors
        formatted_items = []
        for key, value in metrics.items():
            color = self._get_color_code(key)
            reset = "\033[0m" if self.config.use_colors else ""
            formatted_value = self._format_value(value)
            formatted_items.append(f"{color}{key}{reset}={formatted_value}")
        
        return " │ ".join(formatted_items)
    
    def log_step(
        self,
        step: Optional[int] = None,
        **metrics: Union[float, int, torch.Tensor]
    ) -> None:
        """Log training step metrics.
        
        Parameters
        ----------
        step: Optional[int]
            Training step number. If None, uses internal counter.
        **metrics
            Named metrics to log (e.g., loss=1.5, lr=3e-4)
        """
        if not self._should_log():
            return
            
        step = step or self._step_count
        self._step_count = max(self._step_count, step + 1)
        
        # Convert tensors to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):  # torch.Tensor
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value
        
        # Add step to metrics
        processed_metrics["step"] = step
        
        # Console logging
        if step % self.config.print_every == 0:
            elapsed = time.time() - self._start_time
            timestamp = f"[{elapsed/60:6.1f}m]"
            
            if self.config.use_colors:
                timestamp = f"\033[90m{timestamp}\033[0m"  # Gray
            
            metrics_str = self._format_console_line(processed_metrics)
            print(f"{timestamp} {metrics_str}")
        
        # W&B logging
        if (self.wandb_run and 
            step % self.config.log_every == 0 and
            self._wandb_bytes < self.config.max_wandb_mb * 1024 * 1024):
            
            # Prefix metrics with 'train/'
            wandb_metrics = {f"train/{k}": v for k, v in processed_metrics.items()}
            wandb_metrics["train/step"] = step
            
            try:
                self.wandb_run.log(wandb_metrics, step=step)
                # Rough estimate of bytes logged
                self._wandb_bytes += len(str(wandb_metrics)) * 2
            except Exception as e:
                print(f"⚠️  W&B logging failed: {e}")
    
    def log_validation(
        self,
        step: int,
        **metrics: Union[float, int, torch.Tensor]
    ) -> None:
        """Log validation metrics.
        
        Parameters
        ----------
        step: int
            Training step number
        **metrics
            Named validation metrics (e.g., val_loss=1.2, val_acc=0.85)
        """
        if not self._should_log():
            return
        
        # Convert tensors to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = value
        
        # Console logging with validation prefix
        elapsed = time.time() - self._start_time
        timestamp = f"[{elapsed/60:6.1f}m]"
        
        if self.config.use_colors:
            timestamp = f"\033[90m{timestamp}\033[0m"
            val_prefix = "\033[96m[VAL]\033[0m"  # Cyan
        else:
            val_prefix = "[VAL]"
        
        metrics_str = self._format_console_line(processed_metrics)
        print(f"{timestamp} {val_prefix} {metrics_str}")
        
        # W&B logging
        if self.wandb_run and self._wandb_bytes < self.config.max_wandb_mb * 1024 * 1024:
            wandb_metrics = {f"val/{k}": v for k, v in processed_metrics.items()}
            wandb_metrics["train/step"] = step
            
            try:
                self.wandb_run.log(wandb_metrics, step=step)
                self._wandb_bytes += len(str(wandb_metrics)) * 2
            except Exception as e:
                print(f"⚠️  W&B validation logging failed: {e}")
    
    def log_system(
        self,
        step: int,
        **metrics: Union[float, int]
    ) -> None:
        """Log system metrics (memory, throughput, etc.).
        
        Parameters
        ----------
        step: int
            Training step number
        **metrics
            System metrics (e.g., memory_mb=1024, gpu_util=0.95)
        """
        if not self._should_log():
            return
            
        if self.wandb_run and self._wandb_bytes < self.config.max_wandb_mb * 1024 * 1024:
            wandb_metrics = {f"system/{k}": v for k, v in metrics.items()}
            wandb_metrics["train/step"] = step
            
            try:
                self.wandb_run.log(wandb_metrics, step=step)
                self._wandb_bytes += len(str(wandb_metrics)) * 2
            except Exception as e:
                print(f"⚠️  W&B system logging failed: {e}")
    
    @contextmanager
    def log_section(self, section_name: str):
        """Context manager for logging a section of training."""
        if self.config.use_colors:
            print(f"\033[1m{'='*60}\033[0m")
            print(f"\033[1m{section_name.upper()}\033[0m")
            print(f"\033[1m{'='*60}\033[0m")
        else:
            print("=" * 60)
            print(section_name.upper())
            print("=" * 60)
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if self.config.use_colors:
                print(f"\033[90m{section_name} completed in {elapsed:.1f}s\033[0m")
            else:
                print(f"{section_name} completed in {elapsed:.1f}s")
    
    def finish(self) -> None:
        """Clean up and finish logging."""
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"⚠️  Error finishing W&B run: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def format_metrics(metrics: Dict[str, Any], precision: int = 3) -> str:
    """Standalone function to format metrics dictionary for display.
    
    Parameters
    ----------
    metrics: Dict[str, Any]
        Dictionary of metric names and values
    precision: int
        Number of decimal places for float values
        
    Returns
    -------
    str
        Formatted string representation
    """
    formatted_items = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) >= 1000:
                formatted_items.append(f"{key}={value:,.0f}")
            elif abs(value) >= 1:
                formatted_items.append(f"{key}={value:.{precision}f}")
            else:
                formatted_items.append(f"{key}={value:.{precision+1}f}")
        elif isinstance(value, int):
            formatted_items.append(f"{key}={value:,}")
        else:
            formatted_items.append(f"{key}={value}")
    
    return " | ".join(formatted_items)