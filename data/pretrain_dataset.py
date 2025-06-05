import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, Any, Optional


class PretrainDataset(TorchDataset):
    """
    Simple dataset for pretraining on raw text data.
    
    Priority columns: 1st - "text", 2nd - "data"
    """
    
    def __init__(self, tokenizer, dataset=None, max_length=2048, text_column=None):
        """
        Initialize the pretrain dataset.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            dataset: HuggingFace dataset or list of text data
            max_length: Maximum sequence length
            text_column: Specific column name to use (auto-detected if None)
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.text_column = text_column
        
        # Auto-detect text column if not specified
        if self.text_column is None:
            self.text_column = self._detect_text_column()
    
    def _detect_text_column(self) -> str:
        """Auto-detect text column. Priority: text -> data"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        sample_item = self.dataset[0]
        
        # Priority 1: "text"
        if "text" in sample_item:
            return "text"
        
        # Priority 2: "data"
        if "data" in sample_item:
            return "data"
        
        raise ValueError("Could not detect text column. Expected 'text' or 'data'")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text from dataset
        text = self.dataset[idx][self.text_column]
        
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