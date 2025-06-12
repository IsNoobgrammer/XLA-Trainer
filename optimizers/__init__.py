"""Public optimizers API for TPU-Trainer.

This subpackage re-exports optimizers so users can simply do::

    from optimizers import SyncFreeAdamW

without worrying about the underlying backend implementation. For built-in
XLA optimizers (e.g. SyncFree AdamW) we provide a thin wrapper that forwards
all arguments to the original implementation in `torch_xla`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export SyncFree AdamW (XLA) -------------------------------------------------
# ---------------------------------------------------------------------------
# from .syncfree_adamw import SyncFreeAdamW, create_syncfree_adamw  # Disabled: torch_xla not available

# ---------------------------------------------------------------------------
# Re-export Muon + SyncFree AdamW hybrid optimizer
# ---------------------------------------------------------------------------
from .raw_optimizers.muon_syncfree_adamw import (
    MuonSyncFreeAdamW,
    create_muon_syncfree_adamw
)

# ---------------------------------------------------------------------------
# When you add a new custom optimizer (Lion, Muon, MARS, etc.), import it here
# and append its public name to __all__ so that it becomes available via
# `from optimizers import OptimizerName`.
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # "SyncFreeAdamW",
    # "create_syncfree_adamw",
    "MuonSyncFreeAdamW",
    "create_muon_syncfree_adamw",
]
