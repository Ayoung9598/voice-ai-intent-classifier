"""
Training Pipeline for Voice AI Intent Classifier

Handles training loop, validation, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import json
import time

from .intent_classifier import IntentClassifier


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    save_best_only: bool = True
    output_dir: str = "outputs/models"
    log_steps: int = 50


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation metric and stops training if no improvement
    for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "max",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class IntentTrainer:
    """
    Trainer for the intent classifier.
    
    Features:
    - Mixed precision training support
    - Gradient accumulation
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint saving
    - Training history logging
    """
    
    def __init__(
        self,
        model: IntentClassifier,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: IntentClassifier model
            config: Training configuration
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.config = config or TrainingConfig()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rate": [],
        }
        self.best_val_metric = 0.0
        self.current_epoch = 0
    
    def setup_optimizer(self, num_training_steps: int):
        """
        Set up optimizer and learning rate scheduler.
        
        Args:
            num_training_steps: Total number of training steps
        """
        # Separate encoder and classifier parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )
        
        # Linear warmup then linear decay
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, labels)
            
            total_loss += outputs["loss"].item()
            
            # Get predictions
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        
        # Calculate macro F1 (simplified version)
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        
        return {
            "val_loss": total_loss / len(val_loader),
            "val_accuracy": accuracy,
            "val_f1": macro_f1,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        compute_metrics: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            compute_metrics: Optional custom metrics function
            
        Returns:
            Training history
        """
        # Calculate total training steps
        num_training_steps = (
            len(train_loader) // self.config.gradient_accumulation_steps
        ) * self.config.num_epochs
        
        # Setup optimizer and scheduler
        self.setup_optimizer(num_training_steps)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode="max",  # Maximize F1 score
        )
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Training Intent Classifier")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["val_accuracy"])
            self.history["val_f1"].append(val_metrics["val_f1"])
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['val_f1']:.4f}")
            
            # Save best model
            if val_metrics["val_f1"] > self.best_val_metric:
                self.best_val_metric = val_metrics["val_f1"]
                self.save_checkpoint(output_dir / "best_model.pt")
                print(f"  ✓ New best model saved (F1: {self.best_val_metric:.4f})")
            
            # Check early stopping
            if early_stopping(val_metrics["val_f1"]):
                print(f"\n[!] Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint(output_dir / "final_model.pt")
        
        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best Val F1: {self.best_val_metric:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_metric": self.best_val_metric,
            "config": self.model.config,
            "history": self.history,
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.history = checkpoint.get("history", self.history)
        
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


# Quick test
if __name__ == "__main__":
    from intent_classifier import create_model
    
    model = create_model()
    config = TrainingConfig(num_epochs=2)
    trainer = IntentTrainer(model, config)
    
    print(f"Trainer initialized on device: {trainer.device}")
    print(f"Config: {trainer.config}")
