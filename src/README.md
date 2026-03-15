# Source Code Documentation

This is the main source code directory for the Voice AI Intent Classifier.

---

## Module Overview

```
src/
├── data/               # Data loading and preprocessing
│   ├── preprocessor.py # Text cleaning for ASR transcripts
│   └── dataset.py      # PyTorch Dataset and DataLoader creation
│
├── models/             # Model architecture and training
│   ├── intent_classifier.py  # XLM-RoBERTa classifier with Focal Loss
│   └── trainer.py      # Training loop with early stopping
│
├── evaluation/         # Metrics and error analysis
│   ├── metrics.py      # F1, accuracy, ECE, confusion matrix
│   └── error_analysis.py     # Error categorization and patterns
│
├── inference/          # Production inference
│   ├── predictor.py    # IntentPredictor for making predictions
│   └── confidence.py   # Confidence-based action routing
│
└── api/                # REST API
    └── app.py          # FastAPI application
```

---

## Data Flow

```
Raw Text (ASR transcript)
    │
    ▼
TextPreprocessor (preprocessor.py)
  - Normalize Unicode
  - Fix repeated chars
  - Correct ASR errors
    │
    ▼
Tokenizer (XLM-RoBERTa)
  - Subword tokenization
  - Add special tokens
  - Pad/truncate to max_length
    │
    ▼
IntentClassifier (intent_classifier.py)
  - XLM-RoBERTa encoder
  - [CLS] extraction
  - Linear classifier
    │
    ▼
Probabilities (13 intents)
    │
    ▼
ConfidenceHandler (confidence.py)
  - Apply thresholds
  - Determine action
    │
    ▼
Response: {intent, confidence, action}
```

---

## Key Design Decisions

### 1. XLM-RoBERTa for Multilingual Support

**Why XLM-RoBERTa?**
- Pretrained on 100+ languages including Kinyarwanda
- Handles code-switching (mixing languages in one sentence)
- Strong baseline for low-resource languages
- 278M parameters - good balance of performance and speed

**Alternative considered**: mBERT
- Also multilingual but older architecture
- XLM-RoBERTa generally outperforms on cross-lingual tasks

---

### 2. Focal Loss for Class Imbalance

**Problem**: Some intents have 2x more samples than others

**Options considered**:
1. **Oversampling**: Duplicate minority samples
   - Risk: Overfitting to duplicates
2. **Class weights**: Scale loss by inverse frequency
   - Works but sensitive to frequency estimation
3. **Focal Loss**: Down-weight easy examples ✓
   - Dynamically focuses on hard cases
   - Robust to label noise

**Formula**: `FL = -(1-p_t)^γ * log(p_t)`
- `γ=2`: Strong focus on hard examples
- When p_t is high (easy), weight is low
- When p_t is low (hard), weight is high

---

### 3. Confidence-Based Fallback

**Problem**: Wrong predictions have real consequences in government services

**Solution**: Three-tier action system
| Confidence | Action | Reasoning |
|------------|--------|-----------|
| ≥ 85% | Execute | High confidence, proceed automatically |
| 60-85% | Confirm | Ask user to verify intent |
| < 60% | Escalate | Route to human agent |

**Why these thresholds?**
- 85%: Typical accuracy ~95% at this confidence level
- 60%: Below this, error rate rises sharply
- Configurable based on production metrics

---

### 4. Preprocessing for ASR Noise

**Challenges with voice transcripts**:
1. Repeated characters ("helllp" from stuttering)
2. Phonetic spelling ("aplikasiyo" for "application")
3. Code-switching (mixing Kinyarwanda/English)

**Our approach**:
- Conservative corrections (only clear errors)
- Preserve code-switching (model handles it)
- Unicode normalization (consistent diacritics)

**Why conservative?**
- Aggressive correction might break valid Kinyarwanda words
- XLM-RoBERTa is robust to minor noise

---

### 5. FastAPI for REST API

**Why FastAPI?**
- High performance (Starlette + Pydantic)
- Automatic API documentation (Swagger)
- Type validation with Pydantic models
- Easy to deploy (single Python file)

**Alternative considered**: Flask
- More widely used but slower
- No built-in validation
- FastAPI is modern standard for ML APIs

---

## Configuration

All configurable parameters are in `configs/config.yaml`:

```yaml
data:
  train_path: datasets/voiceai_intent_train.csv
  max_length: 128

model:
  name: xlm-roberta-base
  dropout: 0.1

training:
  learning_rate: 2e-5
  num_epochs: 10
  batch_size: 16
  early_stopping_patience: 3

focal_loss:
  gamma: 2.0

inference:
  confidence_thresholds:
    high: 0.85
    medium: 0.60
```

**Why YAML?**
- Human-readable
- Easy to version control
- Can load different configs for experiments

---

## Entry Points

### Training
```bash
# Script
python run_training.py --config configs/config.yaml

# Or notebook
jupyter notebook notebooks/02_training.ipynb
```

### Inference
```bash
# Single prediction
python run_inference.py --text "Check my application status"

# API server
python run_inference.py --serve --port 8000
```

### Evaluation
```bash
# Notebook (recommended for visualization)
jupyter notebook notebooks/03_evaluation.ipynb
```

---

## Dependencies Between Modules

```
data/
├── preprocessor.py ← (no dependencies)
└── dataset.py ← preprocessor.py

models/
├── intent_classifier.py ← (no src dependencies)
└── trainer.py ← intent_classifier.py

evaluation/
├── metrics.py ← (no src dependencies)
└── error_analysis.py ← metrics.py

inference/
├── predictor.py ← models/intent_classifier.py, data/preprocessor.py
└── confidence.py ← (no src dependencies)

api/
└── app.py ← inference/predictor.py, inference/confidence.py
```

**Import pattern**:
- Modules only import from their own package or packages above in this list
- No circular dependencies
- Easy to test in isolation

---

## Error Handling

### Data Module
- Empty text: returns empty string
- Missing file: raises FileNotFoundError with clear message
- Unknown intent: raises ValueError with valid intents list

### Model Module
- Shape mismatch: caught by PyTorch with informative error
- NaN loss: training stops with warning

### Inference Module
- Model not loaded: 503 Service Unavailable
- Invalid input: 422 Validation Error (from Pydantic)

### General Principle
- Fail fast with clear error messages
- Don't silently return wrong results
- Log warnings for recoverable issues
