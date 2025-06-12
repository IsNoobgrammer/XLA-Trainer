"""
XLA SyncFree Adam optimizer wrapper.

This exposes the XLA-optimized Adam optimizer from torch_xla.amp.syncfree
as a local import for convenience and consistency.

Usage:
    from optimizers.adam import Adam
    optimizer = Adam(model.parameters(), lr=1e-3)

See: https://github.com/pytorch/xla/blob/master/torch_xla/amp/syncfree/adam.py
"""

from typing import Any, Iterable

try:
    from torch_xla.amp.syncfree import Adam as _XlaAdam
except ImportError as exc:
    raise ImportError(
        "torch_xla must be installed to use XLA SyncFree Adam optimizer."
    ) from exc

class Adam(_XlaAdam):
    """SyncFree Adam optimizer for XLA/TPU (thin wrapper)."""
    def __init__(self, params: Iterable[Any], **kwargs):
        super().__init__(params, **kwargs)
