import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Basic cross entropy loss for next token prediction using nn.CrossEntropyLoss.
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        labels: Tensor of shape [batch_size, seq_len]
        pad_id: Padding token id to ignore in loss calculation
        
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Shift for next token prediction
    shifted_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[..., 1:].contiguous().view(-1)
    
    # Use nn.CrossEntropyLoss for better performance
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    return ce_loss_fn(shifted_logits, shifted_labels)


def cross_entropy_with_entropy_loss(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    pad_id: int, 
    entropy_weight: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cross entropy loss with entropy regularization using fused operations.
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        labels: Tensor of shape [batch_size, seq_len]
        pad_id: Padding token id to ignore in loss calculation
        entropy_weight: Weight for the entropy regularization term
        
    Returns:
        Tuple containing:
            - ce_loss: Cross-entropy loss
            - entropy: Mean entropy across all positions
            - total_loss: ce_loss + entropy_weight * entropy
    """
    # Shift for next token prediction
    shifted_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[..., 1:].contiguous().view(-1)
    
    # Compute log probabilities once and reuse
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    
    # Cross entropy loss (NLL of log probs)
    ce_loss = F.nll_loss(
        log_probs, 
        shifted_labels, 
        ignore_index=pad_id
    )
    
    # Compute entropy for all positions
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    # Total loss with entropy regularization
    total_loss = ce_loss + entropy_weight * entropy
    
    return ce_loss, entropy, total_loss