# Configuration Documentation

This folder contains YAML configuration files for the Voice AI Intent Classifier.

---

## config.yaml

The main configuration file with all hyperparameters and settings.

### Structure

```yaml
data:           # Data paths and settings
model:          # Model architecture settings
training:       # Training hyperparameters
focal_loss:     # Focal loss settings
inference:      # Inference/production settings
api:            # API server settings
```

---

## Section: `data`

```yaml
data:
  train_path: datasets/voiceai_intent_train.csv
  val_path: datasets/voiceai_intent_val.csv
  test_path: datasets/voiceai_intent_test.csv
  data_dictionary_path: datasets/voiceai_intent_data_dictionary.json
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `train_path` | Path to training CSV | `datasets/voiceai_intent_train.csv` |
| `val_path` | Path to validation CSV | `datasets/voiceai_intent_val.csv` |
| `test_path` | Path to test CSV | `datasets/voiceai_intent_test.csv` |
| `data_dictionary_path` | Path to data dictionary JSON | `datasets/voiceai_intent_data_dictionary.json` |

**Notes**:
- All paths are relative to project root
- val_path and test_path may be created from train data if not provided

---

## Section: `model`

```yaml
model:
  name: xlm-roberta-base
  num_labels: 13
  max_length: 128
  dropout: 0.1
```

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `name` | Pretrained model from HuggingFace | `xlm-roberta-base` | `xlm-roberta-large`, `bert-base-multilingual-cased` |
| `num_labels` | Number of intent classes | `13` | Dynamically updated by IntentLabelEncoder |
| `max_length` | Maximum sequence length | `128` | Up to 512 for XLM-RoBERTa |
| `dropout` | Dropout probability | `0.1` | 0.0-0.5 |

**Why these defaults?**

- **`xlm-roberta-base`**: Best balance of multilingual performance and speed
  - `xlm-roberta-large`: Better accuracy but 3x slower
  - `bert-base-multilingual-cased`: Older, generally worse

- **`max_length=128`**: Dataset max is ~70 chars, 128 tokens is sufficient
  - Longer sequences = slower training, more memory
  - 512 only needed for very long documents

- **`dropout=0.1`**: Standard for transformer fine-tuning
  - Higher (0.3): May hurt performance on small datasets
  - Lower (0.05): Risk of overfitting

---

## Section: `training`

```yaml
training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 3
```

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `batch_size` | Samples per gradient update | `16` | 8-32 (GPU memory dependent) |
| `learning_rate` | Initial learning rate | `2e-5` | 1e-5 to 5e-5 for transformers |
| `num_epochs` | Maximum training epochs | `10` | 3-20 |
| `weight_decay` | L2 regularization strength | `0.01` | 0.0-0.1 |
| `warmup_ratio` | Fraction of steps for LR warmup | `0.1` | 0.0-0.2 |
| `early_stopping_patience` | Epochs without improvement before stopping | `3` | 2-5 |

**Why these defaults?**

- **`learning_rate=2e-5`**: Standard for BERT-family fine-tuning
  - Higher (1e-4): Pretrained weights get destabilized
  - Lower (1e-6): Training too slow

- **`batch_size=16`**: Good for typical GPUs (8-16GB VRAM)
  - If GPU OOM: Reduce to 8
  - If more VRAM: Increase to 32 for faster training

- **`warmup_ratio=0.1`**: Prevents large early updates
  - First 10% of steps: LR increases from 0 to target
  - Prevents damaging pretrained weights early

- **`early_stopping_patience=3`**: Stop if no improvement for 3 epochs
  - Prevents overfitting
  - Usually converges in 5-7 epochs

---

## Section: `focal_loss`

```yaml
focal_loss:
  gamma: 2.0
  alpha: 0.25
```

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `gamma` | Focusing parameter | `2.0` | 0.0-5.0 |
| `alpha` | Class weight (if not computed dynamically) | `0.25` | 0.0-1.0 |

**Understanding gamma**:
- `γ = 0`: Standard cross-entropy (no focusing)
- `γ = 1`: Moderate focusing on hard examples
- `γ = 2`: Strong focusing (recommended)
- `γ > 2`: Very strong focusing (can be unstable)

**Formula**: `FL = -(1-p_t)^γ * log(p_t)`

When model is confident (high p_t):
- `(1 - 0.9)^2 = 0.01` → weight is low

When model is uncertain (low p_t):
- `(1 - 0.3)^2 = 0.49` → weight is high

---

## Section: `inference`

```yaml
inference:
  confidence_thresholds:
    high: 0.85
    medium: 0.60
```

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `high` | Threshold for auto-execution | `0.85` | Higher = fewer auto-executions, more confirmations |
| `medium` | Threshold for confirmation vs escalation | `0.60` | Lower = fewer escalations, more risk |

**Threshold tuning guide**:

| Scenario | Adjust |
|----------|--------|
| Too many escalations to humans | Lower `medium` threshold |
| Too many wrong auto-executions | Raise `high` threshold |
| User complaints about confirmations | Raise `high` (more auto-execute) |
| Critical errors in production | Raise both thresholds |

**Action mapping**:
- Confidence ≥ 0.85 → **Execute** automatically
- 0.60 ≤ Confidence < 0.85 → **Confirm** with user
- Confidence < 0.60 → **Escalate** to human agent

---

## Section: `api`

```yaml
api:
  host: 0.0.0.0
  port: 8000
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `host` | Network interface to bind | `0.0.0.0` (all interfaces) |
| `port` | Port number | `8000` |

**Host options**:
- `0.0.0.0`: Listen on all network interfaces (for containers/production)
- `127.0.0.1`: Listen only on localhost (for local development)

---

## Creating Custom Configurations

### For Experiments

```yaml
# configs/experiment_large_model.yaml
model:
  name: xlm-roberta-large
  dropout: 0.15

training:
  batch_size: 8  # Smaller batch for larger model
  learning_rate: 1e-5  # Lower LR for larger model
```

Usage:
```bash
python run_training.py --config configs/experiment_large_model.yaml
```

### For Production

```yaml
# configs/production.yaml
inference:
  confidence_thresholds:
    high: 0.90  # More conservative
    medium: 0.65

api:
  host: 0.0.0.0
  port: 80
```

---

## Environment-Specific Overrides

You can override config values with environment variables:

```bash
export VOICE_AI_LEARNING_RATE=3e-5
export VOICE_AI_BATCH_SIZE=32
python run_training.py
```

Or command-line arguments:
```bash
python run_training.py --learning-rate 3e-5 --batch-size 32
```

---

## Resource-Constrained Training Recommendations

For CPU-only or limited resource environments, consider these optimizations:

### CPU Training Configuration

```yaml
training:
  batch_size: 8                    # Smaller batch for memory efficiency
  num_epochs: 3                    # Fewer epochs, rely on early stopping
  gradient_accumulation_steps: 2   # Effective batch = 8 × 2 = 16
  early_stopping_patience: 2       # Stop faster when converged
```

**Rationale**:
- **Smaller batch size (8)**: Reduces memory footprint, allows CPU training without OOM
- **Gradient accumulation (2)**: Maintains effective batch size of 16 by accumulating gradients over 2 steps before updating weights
- **Fewer epochs (3)**: For prototyping/testing. XLM-RoBERTa typically converges in 3-5 epochs on this dataset
- **Early stopping patience (2)**: Prevents unnecessary training if validation loss plateaus

### Expected Training Times

| Environment | Batch Size | Epochs | Time per Epoch | Total Time |
|-------------|------------|--------|----------------|------------|
| **CPU (8-core)** | 8 | 3 | ~5-7 min | ~15-20 min |
| **CPU (4-core)** | 8 | 3 | ~8-12 min | ~25-35 min |
| **GPU (8GB)** | 16 | 10 | ~1-2 min | ~15-20 min |
| **GPU (16GB+)** | 32 | 10 | ~45 sec | ~8-10 min |

### Production Recommendations

For final production training with more resources:

```yaml
training:
  batch_size: 16               # Standard for GPU
  num_epochs: 10               # Allow full convergence
  gradient_accumulation_steps: 1
  early_stopping_patience: 3   # More patience for best model
```

---

## Configuration Loading Code

```python
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Access values
learning_rate = config["training"]["learning_rate"]
model_name = config["model"]["name"]
```
