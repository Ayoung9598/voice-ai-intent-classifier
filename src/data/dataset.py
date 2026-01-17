"""
PyTorch Dataset for Voice AI Intent Classification

Handles data loading, tokenization, and batching for the intent classifier.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .preprocessor import TextPreprocessor, create_preprocessor


class IntentDataset(Dataset):
    """
    PyTorch Dataset for intent classification.
    
    Handles:
    - Text preprocessing
    - Tokenization with transformer tokenizer
    - Label encoding
    - Optional metadata (language, ASR confidence, etc.)
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
        preprocessor: Optional[TextPreprocessor] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of utterance texts
            labels: List of intent label indices (None for inference)
            tokenizer: HuggingFace tokenizer (will load default if None)
            max_length: Maximum sequence length
            preprocessor: Text preprocessor (will create default if None)
            metadata: Optional DataFrame with additional features
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.metadata = metadata
        
        # Initialize preprocessor
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Initialize tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        else:
            self.tokenizer = tokenizer
        
        # Preprocess all texts
        self.processed_texts = self.preprocessor.batch_preprocess(texts)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary with input_ids, attention_mask, and optionally labels
        """
        text = self.processed_texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        # Add label if available (training mode)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item
    
    def get_original_text(self, idx: int) -> str:
        """Get the original (unprocessed) text at index."""
        return self.texts[idx]
    
    def get_processed_text(self, idx: int) -> str:
        """Get the preprocessed text at index."""
        return self.processed_texts[idx]


class IntentLabelEncoder:
    """
    Encode and decode intent labels.
    
    Provides consistent label <-> index mapping across the pipeline.
    """
    
    # Canonical intent order (from dataset)
    INTENTS: List[str] = [
        "check_application_status",
        "start_new_application",
        "requirements_information",
        "fees_information",
        "appointment_booking",
        "payment_help",
        "reset_password_login_help",
        "speak_to_agent",
        "cancel_or_reschedule_appointment",
        "update_application_details",
        "document_upload_help",
        "service_eligibility",
        "complaint_or_support_ticket",
    ]
    
    def __init__(self, intents: Optional[List[str]] = None):
        """
        Initialize the label encoder.
        
        Args:
            intents: List of intent names. Uses default if None.
        """
        self.intents = intents or self.INTENTS
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(self.intents)}
        self.idx_to_intent = {idx: intent for idx, intent in enumerate(self.intents)}
    
    def encode(self, intent: str) -> int:
        """Convert intent name to index."""
        return self.intent_to_idx[intent]
    
    def decode(self, idx: int) -> str:
        """Convert index to intent name."""
        return self.idx_to_intent[idx]
    
    def encode_batch(self, intents: List[str]) -> List[int]:
        """Convert list of intent names to indices."""
        return [self.encode(intent) for intent in intents]
    
    def decode_batch(self, indices: List[int]) -> List[str]:
        """Convert list of indices to intent names."""
        return [self.decode(idx) for idx in indices]
    
    @property
    def num_labels(self) -> int:
        """Number of intent classes."""
        return len(self.intents)


def load_data(
    data_path: Union[str, Path],
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 128,
    preprocessor: Optional[TextPreprocessor] = None,
) -> Tuple[IntentDataset, IntentLabelEncoder, pd.DataFrame]:
    """
    Load data from CSV and create dataset.
    
    Args:
        data_path: Path to CSV file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        preprocessor: Text preprocessor
        
    Returns:
        Tuple of (IntentDataset, IntentLabelEncoder, raw DataFrame)
    """
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Initialize label encoder
    label_encoder = IntentLabelEncoder()
    
    # Extract texts and labels
    texts = df["utterance_text"].tolist()
    labels = label_encoder.encode_batch(df["intent"].tolist())
    
    # Create dataset
    dataset = IntentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        preprocessor=preprocessor,
        metadata=df,
    )
    
    return dataset, label_encoder, df


def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and optionally test sets.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV (optional)
        batch_size: Batch size for all loaders
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    """
    # Load shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    preprocessor = TextPreprocessor()
    
    # Load datasets
    train_dataset, label_encoder, _ = load_data(
        train_path, tokenizer, max_length, preprocessor
    )
    val_dataset, _, _ = load_data(
        val_path, tokenizer, max_length, preprocessor
    )
    
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    
    if test_path:
        test_dataset, _, _ = load_data(
            test_path, tokenizer, max_length, preprocessor
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    
    return dataloaders, label_encoder


# Quick test
if __name__ == "__main__":
    # Test label encoder
    encoder = IntentLabelEncoder()
    print(f"Number of intents: {encoder.num_labels}")
    print(f"Intent 0: {encoder.decode(0)}")
    print(f"'speak_to_agent' -> {encoder.encode('speak_to_agent')}")
    
    # Test with sample data
    sample_texts = [
        "Ndashaka kureba status ya application yanjye.",
        "What are the requirements for passport?",
    ]
    sample_labels = [0, 2]  # check_application_status, requirements_information
    
    dataset = IntentDataset(
        texts=sample_texts,
        labels=sample_labels,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample item keys: {dataset[0].keys()}")
    print(f"Input IDs shape: {dataset[0]['input_ids'].shape}")
