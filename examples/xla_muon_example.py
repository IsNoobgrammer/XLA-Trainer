"""
Example usage of XLA-optimized Muon + SyncFree AdamW optimizer for TPU training.

This example demonstrates how to use the hybrid optimizer that applies:
- Muon to weight matrices (2D+ parameters)  
- SyncFree AdamW to embeddings, biases, and other parameters

The optimizer is designed for efficient TPU training with torch-xla.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import XLA utilities (optional, for TPU-specific features)
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
    print("XLA available - can run on TPU")
except ImportError:
    XLA_AVAILABLE = False
    print("XLA not available - running on CPU/GPU")

# Import our optimizers
from optimizers import create_muon_syncfree_adamw


class SimpleTransformerBlock(nn.Module):
    """Simple transformer-like model for demonstration."""
    
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)
        
        # These will use Muon (2D weight matrices)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        
        # These will use AdamW (1D biases, embeddings)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.output_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x, positions=None):
        batch_size, seq_len = x.shape
        
        # Embeddings
        x_embed = self.embedding(x)
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed = self.pos_embedding(positions)
        
        x = x_embed + pos_embed
        x = self.layer_norm1(x)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.layer_norm2(x)
        
        # Feed-forward
        ff_out = self.linear2(torch.relu(self.linear1(x)))
        x = x + self.dropout(ff_out)
        
        # Output
        return self.output_head(x)


def create_dummy_data(batch_size=32, seq_len=64, vocab_size=1000, num_batches=100):
    """Create dummy training data."""
    inputs = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_step(model, optimizer, inputs, targets, device):
    """Single training step."""
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Forward pass
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Optimization step
    if XLA_AVAILABLE:
        # Use XLA-specific optimizer step
        xm.optimizer_step(optimizer, barrier=False)
    else:
        optimizer.step()
    
    optimizer.zero_grad()
    
    return loss.item()


def main():
    """Main training loop."""
    # Set device
    if XLA_AVAILABLE:
        device = xm.xla_device()
        print(f"Using XLA device: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Create model
    model = SimpleTransformerBlock(vocab_size=1000, embed_dim=128, hidden_dim=512)
    model = model.to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer using factory function
    optimizer = create_muon_syncfree_adamw(
        model,
        muon_lr=0.02,           # Learning rate for weight matrices (Muon)
        muon_momentum=0.95,     # Momentum for Muon
        muon_weight_decay=0.0,  # Weight decay for Muon parameters
        adamw_lr=3e-4,          # Learning rate for other parameters (AdamW)
        adamw_betas=(0.9, 0.999), # Beta parameters for AdamW
        adamw_weight_decay=0.01   # Weight decay for AdamW parameters
    )
    
    # Print optimizer info
    print(f"Optimizer has {len(optimizer.param_groups)} parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        opt_type = "Muon" if group['use_muon'] else "SyncFree AdamW"
        print(f"  Group {i}: {opt_type}, {num_params} parameters, lr={group['lr']}")
    
    # Create data
    dataloader = create_dummy_data(batch_size=16, seq_len=32, num_batches=50)
    
    if XLA_AVAILABLE:
        # Wrap dataloader for XLA
        dataloader = pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
    
    # Training loop
    model.train()
    total_loss = 0.0
    num_steps = 0
    
    print("Starting training...")
    for epoch in range(2):  # Train for 2 epochs
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            loss = train_step(model, optimizer, inputs, targets, device)
            
            total_loss += loss
            epoch_loss += loss
            num_steps += 1
            epoch_steps += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            # XLA-specific: mark step for graph compilation
            if XLA_AVAILABLE:
                xm.mark_step()
        
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
    
    avg_loss = total_loss / num_steps
    print(f"Training completed. Average loss: {avg_loss:.4f}")
    
    # Test inference
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, 1000, (1, 32)).to(device)
        test_output = model(test_input)
        print(f"Test output shape: {test_output.shape}")


if __name__ == "__main__":
    main()