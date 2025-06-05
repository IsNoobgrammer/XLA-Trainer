"""
Simple data module for TPU-Trainer

Provides two essential dataset classes:
- ConversationDataset: For chat/conversation data (priority: messages -> conversations)
- PretrainDataset: For raw text pretraining (priority: text -> data)
"""

from .conversation_dataset import ConversationDataset
from .pretrain_dataset import PretrainDataset

__all__ = [
    "ConversationDataset",
    "PretrainDataset",
]

__version__ = "0.1.0"