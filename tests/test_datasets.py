import pytest
import torch
from unittest.mock import Mock
from datasets import Dataset

# Import our simple dataset classes
from data import ConversationDataset, PretrainDataset


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<|endoftext|>"
        self.chat_template = None
        
    def __call__(self, text, **kwargs):
        # Simple mock tokenization
        tokens = text.split()[:kwargs.get('max_length', 10)]
        input_ids = list(range(len(tokens)))
        
        # Pad to max_length if needed
        max_len = kwargs.get('max_length', len(input_ids))
        if kwargs.get('padding') == 'max_length':
            input_ids.extend([self.pad_token_id] * (max_len - len(input_ids)))
        
        result = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([[1] * len(tokens) + [0] * (max_len - len(tokens))])
        }
        return result
    
    def __len__(self):
        return 1000
    
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        # Mock chat template
        formatted = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def conversation_data():
    return Dataset.from_dict({
        "messages": [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"}
            ]
        ]
    })


@pytest.fixture
def pretrain_data():
    return Dataset.from_dict({
        "text": [
            "This is some sample text for pretraining.",
            "Another piece of text for the model to learn from."
        ]
    })


class TestConversationDataset:
    """Test cases for ConversationDataset."""
    
    def test_initialization(self, mock_tokenizer, conversation_data):
        """Test dataset initialization."""
        dataset = ConversationDataset(mock_tokenizer, conversation_data, max_length=128)
        
        assert dataset.conversation_column == "messages"
        assert len(dataset) == 2
        
    def test_conversation_column_detection(self, mock_tokenizer):
        """Test auto-detection of conversation column."""
        # Test with "conversations" column
        data = Dataset.from_dict({
            "conversations": [
                [{"role": "user", "content": "test"}]
            ]
        })
        dataset = ConversationDataset(mock_tokenizer, data)
        assert dataset.conversation_column == "conversations"
        
    def test_getitem(self, mock_tokenizer, conversation_data):
        """Test getting items from dataset."""
        dataset = ConversationDataset(mock_tokenizer, conversation_data, max_length=10)
        
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (10,)
        
    def test_manual_conversation_format(self, mock_tokenizer, conversation_data):
        """Test manual conversation formatting."""
        dataset = ConversationDataset(mock_tokenizer, conversation_data)
        
        # Test the manual formatting directly
        conversation = conversation_data[0]["messages"]
        formatted = dataset._manual_format(conversation)
        assert "user:" in formatted.lower()
        assert "assistant:" in formatted.lower()


class TestPretrainDataset:
    """Test cases for PretrainDataset."""
    
    def test_initialization(self, mock_tokenizer, pretrain_data):
        """Test dataset initialization."""
        dataset = PretrainDataset(mock_tokenizer, pretrain_data, max_length=128)
        
        assert dataset.text_column == "text"
        assert len(dataset) == 2
        
    def test_text_column_detection(self, mock_tokenizer):
        """Test auto-detection of text column."""
        # Test with "data" column
        data = Dataset.from_dict({
            "data": ["Some text content"]
        })
        dataset = PretrainDataset(mock_tokenizer, data)
        assert dataset.text_column == "data"
        
    def test_getitem(self, mock_tokenizer, pretrain_data):
        """Test getting items from dataset."""
        dataset = PretrainDataset(mock_tokenizer, pretrain_data, max_length=10)
        
        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (10,)


class TestErrorHandling:
    """Test error handling across datasets."""
    
    def test_empty_dataset_error(self, mock_tokenizer):
        """Test error handling with empty datasets."""
        empty_data = Dataset.from_dict({"messages": []})
        
        with pytest.raises(ValueError):
            ConversationDataset(mock_tokenizer, empty_data)
            
    def test_missing_text_column(self, mock_tokenizer):
        """Test error handling when text column is missing."""
        data = Dataset.from_dict({
            "other_field": ["some data"]
        })
        
        with pytest.raises(ValueError):
            PretrainDataset(mock_tokenizer, data)
            
    def test_missing_conversation_column(self, mock_tokenizer):
        """Test error handling when conversation column is missing."""
        data = Dataset.from_dict({
            "other_field": ["some data"]
        })
        
        with pytest.raises(ValueError):
            ConversationDataset(mock_tokenizer, data)


class TestDatasetCompatibility:
    """Test compatibility with original notebook usage."""
    
    def test_conversation_like_notebook(self, mock_tokenizer):
        """Test ConversationDataset works like the original notebook example."""
        # Simulate the original notebook usage
        data = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4, 5]]  # This would be tokenized data
        })
        
        # The original used tokenizer.decode, but our new version expects messages
        # So we test with proper conversation format
        conv_data = Dataset.from_dict({
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            ]
        })
        
        dataset = ConversationDataset(mock_tokenizer, conv_data, max_length=2048)
        item = dataset[0]
        
        # Should have the same structure as original
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == item["labels"].shape
        
    def test_pretrain_simple_usage(self, mock_tokenizer):
        """Test PretrainDataset for simple text data."""
        data = Dataset.from_dict({
            "text": ["Simple text for pretraining"]
        })
        
        dataset = PretrainDataset(mock_tokenizer, data, max_length=128)
        item = dataset[0]
        
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item


if __name__ == "__main__":
    pytest.main([__file__])