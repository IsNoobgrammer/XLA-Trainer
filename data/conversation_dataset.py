import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, Any, Optional


class ConversationDataset(TorchDataset):
    """
    Simple dataset for conversation/chat data.
    
    Priority columns: 1st - "messages", 2nd - "conversations"
    """
    
    def __init__(self, tokenizer, dataset=None, max_length=2048, conversation_column=None):
        """
        Initialize the conversation dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            dataset: HuggingFace dataset or list of conversation data
            max_length: Maximum sequence length
            conversation_column: Specific column name to use (auto-detected if None)
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.conversation_column = conversation_column
        
        # Auto-detect conversation column if not specified
        if self.conversation_column is None:
            self.conversation_column = self._detect_conversation_column()
    
    def _detect_conversation_column(self) -> str:
        """Auto-detect conversation column. Priority: messages -> conversations"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        sample_item = self.dataset[0]
        
        # Priority 1: "messages"
        if "messages" in sample_item:
            return "messages"
        
        # Priority 2: "conversations"  
        if "conversations" in sample_item:
            return "conversations"
        
        raise ValueError("Could not detect conversation column. Expected 'messages' or 'conversations'")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get conversation from dataset
        conversation = self.dataset[idx][self.conversation_column]
        
        # Apply chat template if available, otherwise manual format
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            try:
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
            except:
                text = self._manual_format(conversation)
        else:
            text = self._manual_format(conversation)

        # Tokenize
        input_ids = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_ids["input_ids"].squeeze(0),
            "labels": input_ids["input_ids"].squeeze(0),  # For language modeling
            "attention_mask": input_ids["attention_mask"].squeeze(0),
        }
    
    def _manual_format(self, conversation):
        """Simple manual conversation formatting"""
        formatted = []
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)