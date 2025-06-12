"""
XLA SyncFree AdamW optimizer wrapper.

This exposes the XLA-optimized AdamW optimizer from torch_xla.amp.syncfree
as a local import for convenience and consistency.

Usage:
    from optimizers.adamw import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-3)

See: https://github.com/pytorch/xla/blob/master/torch_xla/amp/syncfree/adamw.py
"""

from typing import Any, Iterable

try:
    from torch_xla.amp.syncfree import AdamW as _XlaAdamW
except ImportError as exc:
    raise ImportError(
        "torch_xla must be installed to use XLA SyncFree AdamW optimizer."
    ) from exc

class AdamW(_XlaAdamW):
    """SyncFree AdamW optimizer for XLA/TPU (thin wrapper)."""
    def __init__(self, params: Iterable[Any], **kwargs):
        super().__init__(params, **kwargs)
