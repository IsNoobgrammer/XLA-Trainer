"""
XLA SyncFree SGD optimizer wrapper.

This exposes the XLA-optimized SGD optimizer from torch_xla.amp.syncfree
as a local import for convenience and consistency.

Usage:
    from optimizers.sgd import SGD
    optimizer = SGD(model.parameters(), lr=0.01)

See: https://github.com/pytorch/xla/blob/master/torch_xla/amp/syncfree/sgd.py
"""

from typing import Any, Iterable

try:
    from torch_xla.amp.syncfree import SGD as _XlaSGD
except ImportError as exc:
    raise ImportError(
        "torch_xla must be installed to use XLA SyncFree SGD optimizer."
    ) from exc

class SGD(_XlaSGD):
    """SyncFree SGD optimizer for XLA/TPU (thin wrapper)."""
    def __init__(self, params: Iterable[Any], **kwargs):
        super().__init__(params, **kwargs)
