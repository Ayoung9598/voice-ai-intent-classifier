"""
Intent Classifier Model for Voice AI

XLM-RoBERTa based classifier with Focal Loss for handling
class imbalance in multilingual intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the intent classifier model."""
    model_name: str = "xlm-roberta-base"
    num_labels: int = 13
    dropout: float = 0.1
    hidden_size: int = 768  # XLM-RoBERTa base hidden size
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: Optional[torch.Tensor] = None


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the probability of the correct class
    - gamma focuses on hard examples (higher gamma = more focus)
    - alpha balances class weights
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter (0 = standard CE, higher = focus on hard examples)
            alpha: Class weights tensor of shape (num_classes,)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # Get probability of correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class IntentClassifier(nn.Module):
    """
    XLM-RoBERTa based intent classifier.
    
    Architecture:
    - XLM-RoBERTa encoder
    - Dropout for regularization
    - Linear classification head
    
    Supports:
    - Standard cross-entropy loss
    - Focal loss for class imbalance
    - Confidence scoring via softmax
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # Load pretrained XLM-RoBERTa
        self.encoder = AutoModel.from_pretrained(self.config.model_name)
        
        # Classification head
        self.dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(
            self.config.hidden_size,
            self.config.num_labels,
        )
        
        # Loss function
        if self.config.use_focal_loss:
            self.loss_fn = FocalLoss(
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            labels: Optional ground truth labels of shape (batch_size,)
            
        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), 'probabilities'
        """
        # Encode with XLM-RoBERTa
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classify
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        # Compute probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        result = {
            "logits": logits,
            "probabilities": probabilities,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            result["loss"] = loss
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = outputs["probabilities"]
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probs, dim=-1)
            
        return predicted, confidence
    
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get attention weights for explainability.
        
        Returns the attention weights from the last layer.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            
            # Get last layer attention (batch, heads, seq, seq)
            last_layer_attention = outputs.attentions[-1]
            
            # Average over heads
            attention = last_layer_attention.mean(dim=1)
            
        return attention
    
    def freeze_encoder(self, freeze: bool = True):
        """
        Freeze or unfreeze the encoder parameters.
        
        Useful for:
        - Initial training with frozen encoder
        - Fine-tuning after classifier head is trained
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
    
    @classmethod
    def from_pretrained(cls, path: str) -> "IntentClassifier":
        """
        Load a saved model.
        
        Args:
            path: Path to saved model checkpoint
            
        Returns:
            Loaded IntentClassifier
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", ModelConfig())
        
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def save_pretrained(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }, path)


def create_model(
    model_name: str = "xlm-roberta-base",
    num_labels: int = 13,
    dropout: float = 0.1,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> IntentClassifier:
    """
    Factory function to create an intent classifier.
    
    Args:
        model_name: Pretrained model name
        num_labels: Number of intent classes
        dropout: Dropout probability
        use_focal_loss: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        class_weights: Optional class weights for focal loss
        
    Returns:
        Configured IntentClassifier
    """
    config = ModelConfig(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=class_weights,
    )
    
    return IntentClassifier(config)


# Quick test
if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created: {model.config.model_name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_length = 32
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)
    dummy_labels = torch.tensor([0, 5])
    
    outputs = model(dummy_input_ids, dummy_attention_mask, dummy_labels)
    print(f"\nLogits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test prediction
    preds, confs = model.predict(dummy_input_ids, dummy_attention_mask)
    print(f"\nPredictions: {preds}")
    print(f"Confidences: {confs}")
