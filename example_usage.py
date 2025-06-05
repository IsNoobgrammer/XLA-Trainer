"""
Example usage of the simplified dataset classes.

This shows how to use ConversationDataset and PretrainDataset
in a way that's compatible with the original notebook.
"""

from datasets import Dataset
from transformers import AutoTokenizer
from data import ConversationDataset, PretrainDataset

# Example tokenizer (replace with your model's tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Example 1: ConversationDataset
print("=== ConversationDataset Example ===")
conversation_data = Dataset.from_dict({
    "messages": [
        [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ],
        [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have access to current weather data."}
        ]
    ]
})

conv_dataset = ConversationDataset(tokenizer, conversation_data, max_length=128)
print(f"Conversation dataset length: {len(conv_dataset)}")
print(f"Detected column: {conv_dataset.conversation_column}")

# Get a sample
sample = conv_dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Input IDs shape: {sample['input_ids'].shape}")
print()

# Example 2: PretrainDataset  
print("=== PretrainDataset Example ===")
pretrain_data = Dataset.from_dict({
    "text": [
        "This is some sample text for pretraining a language model.",
        "Another example of text that could be used for training.",
        "More text data for the model to learn from."
    ]
})

pretrain_dataset = PretrainDataset(tokenizer, pretrain_data, max_length=128)
print(f"Pretrain dataset length: {len(pretrain_dataset)}")
print(f"Detected column: {pretrain_dataset.text_column}")

# Get a sample
sample = pretrain_dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Input IDs shape: {sample['input_ids'].shape}")
print()

# Example 3: Using with DataLoader (like in the notebook)
print("=== DataLoader Example ===")
import torch
from torch.utils.data import DataLoader

# Create DataLoaders
conv_loader = DataLoader(conv_dataset, batch_size=2, shuffle=True)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=2, shuffle=True)

# Test a batch from conversation dataset
for batch in conv_loader:
    print("Conversation batch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    break

# Test a batch from pretrain dataset
for batch in pretrain_loader:
    print("Pretrain batch:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    break

print("\n✅ All examples completed successfully!")