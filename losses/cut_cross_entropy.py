"""Cut Cross-Entropy (CCE) implementation that works on TPUs with torch/xla.

This is *not* a drop-in replacement for Apple's CUDA/TRITON kernel – it is a pure
PyTorch implementation that incrementally streams the vocabulary dimension so
that the full `(B, V)` logits matrix never materialises in memory.

Key ideas
=========
1.  Let `X` be the embedding matrix with shape `(N, D)` where `N = batch × seq`.
2.  Let `W` be the classifier weight matrix with shape `(V, D)`.
3.  For the standard cross-entropy one needs for every sample `i` the scalar
    `s_i = x_i · w_{y_i}` **and** `lse_i = logsumexp_j x_i · w_j`.
4.  We can compute `s_i` with an inexpensive gather.
5.  `lse_i` is accumulated in *chunks* over the vocabulary:
        lse = logaddexp(lse, logsumexp(chunk_logits, dim=1))
   Initial `lse` is `-inf` (i.e. `torch.full((N,), float('-inf'))`).  This is
   mathematically equivalent to the exact log-sum-exp but stores only the
   current chunk of logits in memory.

Because the algorithm uses only standard PyTorch ops, XLA will compile it into
an efficient TPU program.  The work is compute-bound but memory footprint is
≈ `max(N × chunk, N)` regardless of the vocabulary size.

Usage example
-------------
    import torch
    import torch_xla.core.xla_model as xm
    from losses.cut_cross_entropy import linear_cross_entropy

    embeddings = torch.randn(batch * seq, hidden, device=xm.xla_device())
    classifier = torch.randn(vocab, hidden, device=embeddings.device)
    target = torch.randint(0, vocab, (batch * seq,), device=embeddings.device)

    loss = linear_cross_entropy(embeddings, classifier, target)
    loss.backward()

Notes
-----
* The implementation supports an `ignore_index` (default `-100`) exactly like
  `torch.nn.functional.cross_entropy`.
* `chunk_size` can be tuned; a few thousand usually balances kernel launch
  overhead and memory use.  For TPUs large chunks (e.g. 32k) are fine because
  HBM is plentiful.
* A `shift` argument is provided as convenience for causal LM training so that
  users can pass the *unshifted* `(B, T, D)` embeddings and `(B, T)` labels.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def _logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable element-wise logaddexp for 1-D tensors.

    Args:
        a (torch.Tensor): Input tensor.
        b (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Element-wise logaddexp result.
    """
    return torch.logaddexp(a, b)

def _flatten_shift(
    embeddings: torch.Tensor, labels: torch.Tensor, shift: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply causal shift and flatten batch+time dimensions.

    Args:
        embeddings (torch.Tensor): (B, T, D) or (B*T, D)
        labels (torch.Tensor): (B, T) or (B*T,)
        shift (int): Shift amount; 1 for next-token prediction.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Flattened embeddings and labels.

    Raises:
        ValueError: If shift > 0 and embeddings has fewer than 3 dims.
    """
    if shift == 0:
        x = embeddings.reshape(-1, embeddings.size(-1))
        y = labels.reshape(-1)
        return x, y
    if embeddings.dim() < 3:
        raise ValueError("Shift requires embeddings to have shape (B, T, D)")
    x = embeddings[..., :-shift, :].contiguous().reshape(-1, embeddings.size(-1))
    y = labels[..., shift:].contiguous().reshape(-1)
    return x, y

def _chunked_logsumexp(
    x: torch.Tensor, classifier: torch.Tensor, bias: Optional[torch.Tensor], chunk_size: int
) -> torch.Tensor:
    """
    Compute logsumexp over the vocabulary dimension in chunks for memory efficiency.

    Args:
        x (torch.Tensor): (N, D) input embeddings.
        classifier (torch.Tensor): (V, D) vocab projection weights.
        bias (Optional[torch.Tensor]): (V,) optional bias.
        chunk_size (int): Number of vocab entries per chunk.

    Returns:
        torch.Tensor: (N,) logsumexp for each input.
    """
    N, D = x.shape
    V = classifier.size(0)
    device = x.device
    dtype = x.dtype
    lse = torch.full((N,), float("-inf"), device=device, dtype=dtype)
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        w_chunk = classifier[start:end]  # (C, D)
        logits_chunk = x @ w_chunk.T  # (N, C)
        if bias is not None:
            logits_chunk += bias[start:end].unsqueeze(0)
        lse_chunk = logits_chunk.logsumexp(dim=1)  # (N,)
        lse = _logaddexp(lse, lse_chunk)
    return lse

def linear_cross_entropy(
    embeddings: torch.Tensor,
    classifier: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    shift: int = 0,
    chunk_size: int = 8192,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Memory-efficient cross-entropy for linear vocab projections.

    Computes the loss:
        loss_i = - ( x_i · w_{y_i} + b_{y_i}  -  logsumexp_j ( x_i · w_j + b_j ) )
    without instantiating the full logits matrix.

    Args:
        embeddings (torch.Tensor): (N, D) or (B, T, D) token embeddings.
        classifier (torch.Tensor): (V, D) vocab projection weights.
        labels (torch.Tensor): (N,) or (B, T) target token ids.
        bias (Optional[torch.Tensor]): (V,) optional per-token bias.
        shift (int): If >0, perform causal shift (see _flatten_shift).
        chunk_size (int): Number of vocab entries processed at once.
        reduction (str): 'none', 'sum', or 'mean'.
        ignore_index (int): Target value to ignore.

    Returns:
        torch.Tensor: Loss (scalar or tensor, depending on reduction).

    Raises:
        ValueError: For invalid shapes or reduction argument.
    """
    if embeddings.dim() not in (2, 3):
        raise ValueError("Embeddings must have shape (N, D) or (B, T, D)")
    if classifier.dim() != 2:
        raise ValueError("Classifier must have shape (V, D)")
    if embeddings.size(-1) != classifier.size(-1):
        raise ValueError("Embedding dim D must match classifier dim D")
    device = embeddings.device
    dtype = embeddings.dtype
    x, y = _flatten_shift(embeddings, labels, shift)
    N, D = x.shape
    V = classifier.size(0)
    if bias is not None and bias.shape != (V,):
        raise ValueError("Bias must have shape (V,)")
    valid = y != ignore_index
    if valid.sum() == 0:
        # All positions ignored; return zero loss of correct type.
        return torch.zeros((), device=device, dtype=dtype)
    # Gather numerator dot products for correct classes.
    w_y = classifier[y.clamp(min=0)]  # clamp avoids negative index for ignore
    num = (x * w_y).sum(dim=1)
    if bias is not None:
        num += bias[y.clamp(min=0)]
    # Log-sum-exp over vocabulary in chunks.
    lse = _chunked_logsumexp(x, classifier, bias, chunk_size)
    # Cross-entropy.
    loss = lse - num  # (N,)
    # Mask ignored positions.
    loss = loss[valid]
    if reduction == "none":
        # Return same shape as input labels (after shift).
        full = torch.zeros_like(y, dtype=dtype)
        full[valid] = loss
        return full
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError("Invalid reduction: choose from 'none', 'sum', 'mean'")
