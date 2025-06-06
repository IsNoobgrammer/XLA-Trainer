"""Metrics tracking and accumulation.

Stateful classes for tracking metrics over the course of training,
with efficient accumulation and periodic reporting.
"""
from __future__ import annotations

import time
import collections
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch

from .compute import compute_tokens_in_batch, compute_mfu, compute_model_flops_per_token

__all__ = ["MetricsTracker", "LossTracker"]


@dataclass
class MetricsTracker:
    """Tracks running metrics such as tokens processed and wall-clock speed.
    
    This is the main metrics accumulator for training loops. It tracks
    core performance metrics like MFU, throughput, and timing.
    """

    model: torch.nn.Module
    sequence_length: int
    world_size: int = 1
    theoretical_peak_flops: Optional[float] = None

    start_time: float = field(default_factory=time.time, init=False)
    _token_counter: int = field(default=0, init=False)
    _step_counter: int = field(default=0, init=False)
    _last_log_time: float = field(default_factory=time.time, init=False)
    _last_log_tokens: int = field(default=0, init=False)

    def update(self, local_batch_size: int) -> None:
        """Call once per optimization step.

        Parameters
        ----------
        local_batch_size: int
            The batch size *per device* **before** gradient accumulation.
            The tracker will take ``world_size`` into account to compute global counts.
        """
        self._step_counter += 1
        tokens_this_step = compute_tokens_in_batch(local_batch_size, self.sequence_length) * self.world_size
        self._token_counter += tokens_this_step

    def reset_interval(self) -> None:
        """Reset interval tracking for computing recent throughput."""
        self._last_log_time = time.time()
        self._last_log_tokens = self._token_counter

    @property
    def elapsed(self) -> float:
        """Seconds since tracker initialization."""
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Overall tokens per second since start."""
        return self._token_counter / max(self.elapsed, 1e-6)

    @property
    def recent_tokens_per_second(self) -> float:
        """Tokens per second since last reset_interval() call."""
        elapsed_since_log = time.time() - self._last_log_time
        tokens_since_log = self._token_counter - self._last_log_tokens
        return tokens_since_log / max(elapsed_since_log, 1e-6)

    @property
    def mfu(self) -> Optional[float]:
        """Model FLOPs Utilization based on overall throughput."""
        if self.theoretical_peak_flops is None:
            return None
        return compute_mfu(
            self.model,
            tokens_per_second=self.tokens_per_second,
            theoretical_peak_flops=self.theoretical_peak_flops,
        )

    @property
    def recent_mfu(self) -> Optional[float]:
        """Model FLOPs Utilization based on recent throughput."""
        if self.theoretical_peak_flops is None:
            return None
        return compute_mfu(
            self.model,
            tokens_per_second=self.recent_tokens_per_second,
            theoretical_peak_flops=self.theoretical_peak_flops,
        )

    @property
    def steps(self) -> int:
        """Number of optimization steps completed."""
        return self._step_counter

    @property
    def tokens(self) -> int:
        """Total number of tokens processed."""
        return self._token_counter

    def as_dict(self, include_recent: bool = True) -> Dict[str, Any]:
        """Export metrics as a dictionary for logging."""
        d = {
            "steps": self._step_counter,
            "tokens": self._token_counter,
            "tokens_per_second": self.tokens_per_second,
            "elapsed_seconds": self.elapsed,
        }
        
        if include_recent:
            d["recent_tokens_per_second"] = self.recent_tokens_per_second
            
        if self.theoretical_peak_flops is not None:
            d["mfu"] = self.mfu
            if include_recent:
                d["recent_mfu"] = self.recent_mfu
                
        return d

    def __str__(self) -> str:
        """Human-readable summary of current metrics."""
        stats = self.as_dict(include_recent=False)
        mfu_str = f", MFU={stats['mfu']*100:5.2f}%" if "mfu" in stats else ""
        return (
            f"steps={stats['steps']}, tokens={stats['tokens']:,}, "
            f"tok/s={stats['tokens_per_second']:.2f}{mfu_str}"
        )


@dataclass
class LossTracker:
    """Tracks loss values and related metrics with running averages.
    
    Accumulates loss values over multiple steps and provides smoothed
    averages for stable logging.
    """
    
    window_size: int = 100
    
    _values: Dict[str, collections.deque] = field(
        default_factory=lambda: collections.defaultdict(lambda: collections.deque(maxlen=100)),
        init=False
    )
    _totals: Dict[str, float] = field(
        default_factory=lambda: collections.defaultdict(float),
        init=False
    )
    _counts: Dict[str, int] = field(
        default_factory=lambda: collections.defaultdict(int),
        init=False
    )

    def __post_init__(self):
        # Update maxlen for existing deques if window_size changed
        for deque in self._values.values():
            deque.maxlen = self.window_size

    def update(self, **metrics: float) -> None:
        """Update tracked metrics with new values.
        
        Parameters
        ----------
        **metrics: float
            Named metric values to track (e.g., ce_loss=1.5, entropy=2.1)
        """
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            
            self._values[name].append(value)
            self._totals[name] += value
            self._counts[name] += 1

    def get_average(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get windowed average of a metric.
        
        Parameters
        ----------
        name: str
            Name of the metric
        window: Optional[int]
            Window size for average. If None, uses all available values.
            
        Returns
        -------
        Optional[float]
            Average value, or None if metric not found
        """
        if name not in self._values or not self._values[name]:
            return None
            
        values = list(self._values[name])
        if window is not None:
            values = values[-window:]
            
        return sum(values) / len(values)

    def get_total_average(self, name: str) -> Optional[float]:
        """Get average over all recorded values (not windowed)."""
        if name not in self._counts or self._counts[name] == 0:
            return None
        return self._totals[name] / self._counts[name]

    def get_recent(self, name: str, n: int = 1) -> Optional[float]:
        """Get the most recent value(s) of a metric."""
        if name not in self._values or not self._values[name]:
            return None
            
        if n == 1:
            return self._values[name][-1]
        else:
            return list(self._values[name])[-n:]

    def reset(self, metric_name: Optional[str] = None) -> None:
        """Reset tracking for a specific metric or all metrics."""
        if metric_name is not None:
            if metric_name in self._values:
                self._values[metric_name].clear()
                self._totals[metric_name] = 0.0
                self._counts[metric_name] = 0
        else:
            self._values.clear()
            self._totals.clear()
            self._counts.clear()

    def as_dict(self, include_totals: bool = False) -> Dict[str, float]:
        """Export current averages as a dictionary."""
        result = {}
        
        for name in self._values:
            avg = self.get_average(name)
            if avg is not None:
                result[f"{name}_avg"] = avg
                
            if include_totals:
                total_avg = self.get_total_average(name)
                if total_avg is not None:
                    result[f"{name}_total_avg"] = total_avg
                    
        return result

    @property
    def tracked_metrics(self) -> list[str]:
        """List of currently tracked metric names."""
        return list(self._values.keys())