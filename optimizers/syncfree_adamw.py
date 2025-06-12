"""
XLA SyncFree AdamW optimizer wrapper.

This exposes the XLA-optimized AdamW optimizer from torch_xla.amp.syncfree
as a local import for convenience and consistency.

Usage:
    from optimizers.syncfree_adamw import SyncFreeAdamW
    optimizer = SyncFreeAdamW(model.parameters(), lr=1e-3)

See: https://github.com/pytorch/xla/blob/master/torch_xla/amp/syncfree/adamw.py
"""

from typing import Any, Iterable

try:
    from torch_xla.amp.syncfree import AdamW as _XlaSyncFreeAdamW
except ImportError as exc:
    raise ImportError(
        "torch_xla must be installed to use XLA SyncFree AdamW optimizer."
    ) from exc


class SyncFreeAdamW(_XlaSyncFreeAdamW):
    """SyncFree AdamW optimizer for XLA/TPU (thin wrapper)."""
    def __init__(self, params: Iterable[Any], **kwargs):
        super().__init__(params, **kwargs)


def create_syncfree_adamw(params: Iterable[Any], **kwargs) -> SyncFreeAdamW:
    """Factory function to create SyncFree AdamW optimizer."""
    return SyncFreeAdamW(params, **kwargs)