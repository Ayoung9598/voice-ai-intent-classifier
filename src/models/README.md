# Models Module Documentation

This module contains the model architecture and training pipeline for intent classification.

---

## intent_classifier.py

### Purpose
Defines the XLM-RoBERTa based intent classifier with Focal Loss for handling class imbalance.

### Classes

#### `ModelConfig`
Dataclass for model configuration.

```python
@dataclass
class ModelConfig:
    model_name: str = "xlm-roberta-base"  # Pretrained model
    num_labels: int = 13                   # Number of intent classes
    dropout: float = 0.1                   # Dropout probability
    hidden_size: int = 768                 # XLM-RoBERTa hidden dimension
    use_focal_loss: bool = True            # Use Focal Loss vs Cross-Entropy
    focal_gamma: float = 2.0               # Focal Loss focusing parameter
    focal_alpha: torch.Tensor = None       # Optional class weights
```

**Why these defaults?**
- `dropout=0.1`: Standard for fine-tuning transformers (prevents overfitting)
- `focal_gamma=2.0`: Common default, increases focus on hard examples
- `hidden_size=768`: Fixed by xlm-roberta-base architecture

---

#### `FocalLoss(nn.Module)`
Custom loss function for handling class imbalance.

##### The Math
Standard Cross-Entropy:
```
CE(p_t) = -log(p_t)
```

Focal Loss:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` = probability of correct class
- `γ` (gamma) = focusing parameter
- `α_t` = optional class weight

##### `__init__(gamma, alpha, reduction)`
```python
def __init__(
    gamma: float = 2.0,           # Focusing parameter
    alpha: torch.Tensor = None,   # Class weights [num_classes]
    reduction: str = "mean"       # 'mean', 'sum', or 'none'
):
```

**Why gamma=2.0?**
- `γ=0`: Equivalent to standard cross-entropy
- `γ=2`: Well-studied default that works well in practice
- Higher γ: More focus on hard examples (can be unstable)

##### `forward(inputs, targets) -> torch.Tensor`
**Purpose**: Compute focal loss between predictions and ground truth.

**Implementation**:
```python
def forward(self, inputs, targets):
    # 1. Get softmax probabilities
    probs = F.softmax(inputs, dim=-1)
    
    # 2. Standard cross-entropy (per sample)
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    
    # 3. Get probability of correct class
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    
    # 4. Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** self.gamma
    
    # 5. Apply focal weight
    focal_loss = focal_weight * ce_loss
    
    # 6. Optional class weights
    if self.alpha is not None:
        alpha_t = self.alpha.gather(0, targets)
        focal_loss = alpha_t * focal_loss
    
    return focal_loss.mean()  # or sum/none
```

**Why Focal Loss for this problem?**
1. Class imbalance: Some intents have 2x more samples than others
2. Label noise (~3%): Focal loss down-weights confident wrong predictions
3. Hard examples: Focuses learning on ambiguous utterances

---

#### `IntentClassifier(nn.Module)`
Main classification model.

##### Architecture
```
Input IDs -> XLM-RoBERTa Encoder -> [CLS] Token -> Dropout -> Linear -> Logits
                                     |
                              768-dim vector
                                     |
                               13-dim output
```

##### `__init__(config)`
**Initialization**:
```python
def __init__(self, config):
    # 1. Load pretrained XLM-RoBERTa
    self.encoder = AutoModel.from_pretrained(config.model_name)
    
    # 2. Classification head
    self.dropout = nn.Dropout(config.dropout)
    self.classifier = nn.Linear(768, 13)  # hidden_size -> num_labels
    
    # 3. Loss function
    if config.use_focal_loss:
        self.loss_fn = FocalLoss(gamma=config.focal_gamma)
    else:
        self.loss_fn = nn.CrossEntropyLoss()
```

**Why use [CLS] token?**
- XLM-RoBERTa adds special [CLS] token at position 0
- After encoding, [CLS] aggregates sequence-level information
- Standard approach for sequence classification

##### `forward(input_ids, attention_mask, labels=None)`
**Purpose**: Forward pass through the model.

**Flow**:
```python
def forward(self, input_ids, attention_mask, labels=None):
    # 1. Encode with XLM-RoBERTa
    outputs = self.encoder(input_ids, attention_mask)
    
    # 2. Extract [CLS] token (position 0)
    cls_output = outputs.last_hidden_state[:, 0, :]
    
    # 3. Apply dropout (regularization)
    cls_output = self.dropout(cls_output)
    
    # 4. Linear projection to logits
    logits = self.classifier(cls_output)
    
    # 5. Compute probabilities
    probabilities = F.softmax(logits, dim=-1)
    
    # 6. Compute loss if labels provided
    if labels is not None:
        loss = self.loss_fn(logits, labels)
        return {"logits": logits, "probabilities": probabilities, "loss": loss}
    
    return {"logits": logits, "probabilities": probabilities}
```

##### `predict(input_ids, attention_mask)`
**Purpose**: Get predicted class and confidence.

```python
def predict(self, input_ids, attention_mask):
    self.eval()  # Disable dropout
    with torch.no_grad():  # No gradient computation
        outputs = self.forward(input_ids, attention_mask)
        probs = outputs["probabilities"]
        
        # Get highest probability class
        confidence, predicted = torch.max(probs, dim=-1)
        
    return predicted, confidence
```

**Why torch.no_grad()?**
- Inference doesn't need gradients
- Saves memory and speeds up computation

##### `get_attention_weights(...)`
**Purpose**: Extract attention for explainability.

Returns attention from the last transformer layer, averaged across heads.
Useful for visualizing which words the model focuses on.

##### `freeze_encoder(freeze=True)`
**Purpose**: Freeze/unfreeze XLM-RoBERTa weights.

**Use case**:
- Phase 1: Freeze encoder, train only classifier head
- Phase 2: Unfreeze, fine-tune entire model with lower LR
- Saves training time and prevents catastrophic forgetting

---

### Functions

#### `create_model(...)`
Factory function to create IntentClassifier with custom config.

```python
def create_model(
    model_name="xlm-roberta-base",
    num_labels=13,
    dropout=0.1,
    use_focal_loss=True,
    focal_gamma=2.0,
    class_weights=None
):
    config = ModelConfig(...)
    return IntentClassifier(config)
```

---

## trainer.py

### Purpose
Handles the training loop, validation, early stopping, and checkpointing.

### Classes

#### `TrainingConfig`
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5           # Standard for fine-tuning BERT
    weight_decay: float = 0.01            # L2 regularization
    num_epochs: int = 10                  # Maximum epochs
    warmup_ratio: float = 0.1             # 10% warmup steps
    gradient_accumulation_steps: int = 1  # Effective batch size multiplier
    max_grad_norm: float = 1.0            # Gradient clipping threshold
    early_stopping_patience: int = 3      # Epochs without improvement
    save_best_only: bool = True
    output_dir: str = "outputs/models"
```

**Why learning_rate=2e-5?**
- Standard for fine-tuning pretrained transformers
- Higher LR (e.g., 1e-4) can destabilize pretrained weights
- Lower LR (e.g., 1e-6) trains too slowly

**Why warmup?**
- Prevents early large gradient updates that could harm pretrained weights
- Gradually increases LR from 0 to target over warmup period

---

#### `EarlyStopping`
Monitors validation metric and stops training if no improvement.

##### `__init__(patience, min_delta, mode)`
```python
def __init__(
    patience: int = 3,        # Epochs to wait
    min_delta: float = 0.001, # Minimum improvement
    mode: str = "max"         # "max" for F1, "min" for loss
):
```

##### `__call__(score) -> bool`
**Logic**:
```python
def __call__(self, score):
    if self.best_score is None:
        self.best_score = score
        return False
    
    # Check if improved
    if mode == "max":
        improved = score > self.best_score + min_delta
    else:
        improved = score < self.best_score - min_delta
    
    if improved:
        self.best_score = score
        self.counter = 0
    else:
        self.counter += 1
        if self.counter >= self.patience:
            return True  # Stop training
    
    return False
```

**Why early stopping?**
- Prevents overfitting to training data
- Saves training time
- Keeps best model based on validation performance

---

#### `IntentTrainer`

##### `__init__(model, config, device)`
**Initialization**:
- Move model to device (GPU/CPU)
- Initialize training state (history, best metric)
- Don't create optimizer yet (need to know total steps)

##### `setup_optimizer(num_training_steps)`
**Purpose**: Create optimizer and learning rate scheduler.

**Optimizer**: AdamW
```python
optimizer_grouped_parameters = [
    {
        # All parameters except biases and LayerNorm
        "params": [...],
        "weight_decay": 0.01,
    },
    {
        # Biases and LayerNorm (no weight decay)
        "params": [...],
        "weight_decay": 0.0,
    },
]
self.optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
```

**Why no weight decay on biases?**
- Weight decay is L2 regularization
- Biases don't benefit from regularization
- LayerNorm parameters are scale/shift, not weights

**Scheduler**: Linear warmup + linear decay
```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)
```

##### `train_epoch(train_loader) -> float`
**Purpose**: Train for one epoch, return average loss.

**Loop**:
```python
for batch in train_loader:
    # 1. Move data to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # 2. Forward pass
    outputs = model(input_ids, attention_mask, labels)
    loss = outputs["loss"]
    
    # 3. Scale for gradient accumulation
    loss = loss / gradient_accumulation_steps
    
    # 4. Backward pass
    loss.backward()
    
    # 5. Update weights (every N steps)
    if (step + 1) % gradient_accumulation_steps == 0:
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

**Why gradient clipping?**
- Prevents gradient explosion
- Stabilizes training, especially early on
- `max_grad_norm=1.0` is standard

##### `validate(val_loader) -> Dict`
**Purpose**: Evaluate model on validation set.

Returns:
```python
{
    "val_loss": average_loss,
    "val_accuracy": accuracy,
    "val_f1": macro_f1_score
}
```

**Why macro F1?**
- Treats all classes equally regardless of frequency
- Better than accuracy for imbalanced classes
- Standard metric for intent classification

##### `train(train_loader, val_loader) -> Dict`
**Purpose**: Full training loop with early stopping.

**Flow**:
```python
for epoch in range(num_epochs):
    # 1. Train one epoch
    train_loss = self.train_epoch(train_loader)
    
    # 2. Validate
    val_metrics = self.validate(val_loader)
    
    # 3. Log to history
    self.history["train_loss"].append(train_loss)
    self.history["val_f1"].append(val_metrics["val_f1"])
    
    # 4. Save best model
    if val_metrics["val_f1"] > self.best_val_metric:
        self.best_val_metric = val_metrics["val_f1"]
        self.save_checkpoint("best_model.pt")
    
    # 5. Check early stopping
    if early_stopping(val_metrics["val_f1"]):
        break
```

##### `save_checkpoint(path)` / `load_checkpoint(path)`
**Saves**:
- Model state dict (weights)
- Optimizer state dict (momentum, etc.)
- Scheduler state dict (current step)
- Training history
- Best validation metric
- Model config

**Why save optimizer state?**
- Allows resuming training from checkpoint
- Optimizer has momentum terms that are trained too

---

## Key Design Decisions

### Why XLM-RoBERTa?
1. **Multilingual**: Pretrained on 100+ languages including Kinyarwanda
2. **Code-switching**: Trained on mixed-language web data
3. **Size**: 278M params - good balance of performance and speed
4. **Community**: Well-supported, many resources available

### Why Fine-tune vs Feature Extraction?
- **Fine-tuning**: Update all weights, model adapts to task
- **Feature extraction**: Freeze encoder, only train classifier
- We fine-tune because intent classification needs task-specific representations

### Why Focal Loss vs Class Weights?
- Class weights: Scale loss by inverse frequency
- Focal Loss: Dynamically focuses on hard examples
- Focal Loss is more robust to label noise (~3% in our data)
