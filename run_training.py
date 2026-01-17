#!/usr/bin/env python
"""
Training Script for Voice AI Intent Classifier

Run this script to train the model from command line:
    python run_training.py

Or with custom config:
    python run_training.py --config configs/config.yaml --epochs 5
"""

import argparse
import yaml
from pathlib import Path
import torch

from src.data.dataset import create_dataloaders, IntentLabelEncoder
from src.models.intent_classifier import create_model
from src.models.trainer import IntentTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Voice AI Intent Classifier")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    
    output_dir = Path(args.output_dir or config["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Voice AI Intent Classifier - Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['name']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    print("Loading data...")
    dataloaders, label_encoder = create_dataloaders(
        train_path=config["data"]["train_path"],
        val_path=config["data"]["val_path"],
        test_path=config["data"]["test_path"],
        batch_size=config["training"]["batch_size"],
        max_length=config["model"]["max_length"],
    )
    
    print(f"  Train: {len(dataloaders['train'].dataset)} samples")
    print(f"  Val: {len(dataloaders['val'].dataset)} samples")
    print(f"  Test: {len(dataloaders['test'].dataset)} samples")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        use_focal_loss=True,
        focal_gamma=config["focal_loss"]["gamma"],
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Training config
    training_config = TrainingConfig(
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        num_epochs=config["training"]["num_epochs"],
        warmup_ratio=config["training"]["warmup_ratio"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        output_dir=str(output_dir),
    )
    
    # Train
    trainer = IntentTrainer(model, training_config, device=str(device))
    history = trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.load_checkpoint(output_dir / "best_model.pt")
    test_metrics = trainer.validate(dataloaders["test"])
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy: {test_metrics['val_accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['val_f1']:.4f}")
    print(f"{'='*60}")
    
    print(f"\n✅ Training complete! Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
