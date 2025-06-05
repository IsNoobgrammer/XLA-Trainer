import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, labels, pad_id):
    """Basic cross entropy loss for next token prediction."""
    # Shift for next token prediction
    shifted_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[..., 1:].contiguous().view(-1)
    
    return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=pad_id)


def cross_entropy_with_entropy_loss(logits, labels, pad_id, entropy_weight=0.001):
    """Cross entropy loss with entropy regularization."""
    # Shift for next token prediction
    shifted_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[..., 1:].contiguous().view(-1)
    
    # Compute log probabilities
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    
    # Cross entropy loss
    ce_loss = F.nll_loss(log_probs, shifted_labels, ignore_index=pad_id)
    
    # Entropy (no masking)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    return ce_loss, entropy, ce_loss + entropy_weight * entropy