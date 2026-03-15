# Inference Module Documentation

This module handles production inference with confidence-based fallback logic.

---

## predictor.py

### Purpose
Provides a clean API for making predictions with the trained model, including preprocessing integration and batch processing.

### Classes

#### `IntentPredictor`
Main inference class used in production.

##### `__init__(...)`
```python
def __init__(
    model: IntentClassifier = None,   # Trained model (option 1)
    model_path: str = None,           # Path to checkpoint (option 2)
    tokenizer_name: str = "xlm-roberta-base",
    max_length: int = 128,
    device: str = None,               # 'cuda', 'cpu', or auto-detect
    label_encoder: IntentLabelEncoder = None
):
```

**Initialization flow**:
```python
def __init__(...):
    # 1. Set device (auto-detect if not specified)
    if device is None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load model (from instance or checkpoint)
    if model is not None:
        self.model = model
    elif model_path is not None:
        self.model = IntentClassifier.from_pretrained(model_path)
    
    # 3. Move to device and set eval mode
    self.model.to(self.device)
    self.model.eval()  # Disable dropout
    
    # 4. Load tokenizer and preprocessor
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    self.preprocessor = TextPreprocessor()
    
    # 5. Label encoder for index ↔ intent name
    self.label_encoder = label_encoder or IntentLabelEncoder()
```

**Why two loading options?**
- `model`: For testing with newly created model
- `model_path`: For production with saved checkpoint

##### `predict(text, return_all_scores=False)`
**Purpose**: Predict intent for a single utterance.

**Returns**:
```python
{
    "text": "Check my application status",
    "intent": "check_application_status",
    "confidence": 0.92,
    "all_scores": {  # Optional, if return_all_scores=True
        "check_application_status": 0.92,
        "requirements_information": 0.03,
        ...
    }
}
```

**Flow**:
```python
def predict(self, text, return_all_scores=False):
    # Just wraps batch prediction for single item
    results = self.predict_batch([text], return_all_scores)
    return results[0]
```

##### `predict_batch(texts, return_all_scores=False, batch_size=32)`
**Purpose**: Efficiently predict multiple utterances.

**Why batch processing?**
- GPU processes batches in parallel
- Much faster than predicting one at a time
- Memory efficient (processes in chunks)

**Flow**:
```python
def predict_batch(self, texts, return_all_scores, batch_size):
    all_results = []
    
    # Process in chunks
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = self._predict_batch_internal(batch_texts, return_all_scores)
        all_results.extend(batch_results)
    
    return all_results
```

##### `_predict_batch_internal(texts, return_all_scores)`
**Purpose**: Internal batch prediction logic.

**Flow**:
```python
def _predict_batch_internal(self, texts, return_all_scores):
    # 1. Preprocess all texts
    processed_texts = self.preprocessor.batch_preprocess(texts)
    
    # 2. Tokenize
    encodings = self.tokenizer(
        processed_texts,
        max_length=self.max_length,
        padding=True,          # Pad to longest in batch
        truncation=True,       # Truncate if too long
        return_tensors="pt"    # Return PyTorch tensors
    )
    
    # 3. Move to device
    input_ids = encodings["input_ids"].to(self.device)
    attention_mask = encodings["attention_mask"].to(self.device)
    
    # 4. Predict (no gradients needed)
    with torch.no_grad():
        outputs = self.model(input_ids, attention_mask)
        probs = outputs["probabilities"].cpu().numpy()
    
    # 5. Build results
    results = []
    for i, text in enumerate(texts):
        pred_idx = int(np.argmax(probs[i]))
        confidence = float(probs[i][pred_idx])
        
        result = {
            "text": text,
            "intent": self.label_encoder.decode(pred_idx),
            "confidence": confidence,
        }
        
        if return_all_scores:
            result["all_scores"] = {
                self.label_encoder.decode(j): float(probs[i][j])
                for j in range(len(probs[i]))
            }
        
        results.append(result)
    
    return results
```

**Why `.cpu().numpy()`?**
- Model outputs are on GPU (if available)
- Need to move to CPU for Python processing
- NumPy is easier to work with than PyTorch tensors

##### `predict_top_k(text, k=3)`
**Purpose**: Get top-k intent predictions with scores.

**Returns**:
```python
[
    {"intent": "check_application_status", "confidence": 0.85},
    {"intent": "requirements_information", "confidence": 0.08},
    {"intent": "fees_information", "confidence": 0.03}
]
```

**Use case**: 
- When top prediction has medium confidence
- Offer alternatives in confirmation prompt
- "Did you mean to check status, or get requirements?"

##### `predict_with_fallback(text, high_threshold=0.85, medium_threshold=0.60)`
**Purpose**: Get prediction with recommended action.

**Returns**:
```python
{
    "text": "...",
    "intent": "...",
    "confidence": 0.72,
    "action": "confirm",  # "execute" | "confirm" | "escalate"
    "action_reason": "Medium confidence - confirm with user",
    "alternatives": [...]  # Only for "confirm" action
}
```

**Decision logic**:
```python
if confidence >= 0.85:
    action = "execute"  # Proceed automatically
elif confidence >= 0.60:
    action = "confirm"  # Ask user to confirm
    alternatives = self.predict_top_k(text, k=3)
else:
    action = "escalate"  # Route to human agent
```

---

#### `BatchPredictor`
Optimized for high-throughput scenarios like log processing.

##### `predict_dataframe(df, text_column, batch_size)`
**Purpose**: Add predictions to pandas DataFrame.

**Use case**: Evaluate on test set, process historical logs

```python
def predict_dataframe(self, df, text_column, batch_size):
    texts = df[text_column].tolist()
    results = self.predictor.predict_batch(texts, batch_size=batch_size)
    
    df = df.copy()
    df["predicted_intent"] = [r["intent"] for r in results]
    df["prediction_confidence"] = [r["confidence"] for r in results]
    
    return df
```

---

## confidence.py

### Purpose
Implements confidence-based decision logic for production Voice AI.

### Why Confidence-Based Fallback?

For government services, wrong actions have real consequences:
- Booking wrong appointment wastes citizen's time
- Wrong payment could affect application status
- Misrouted complaints delay resolution

**Three-tier strategy**:
1. **High confidence (≥85%)**: Execute automatically - fast UX
2. **Medium confidence (60-85%)**: Confirm with user - safety net
3. **Low confidence (<60%)**: Human agent - avoid errors

### Classes

#### `ConfidenceAction(Enum)`
```python
class ConfidenceAction(Enum):
    EXECUTE = "execute"    # Proceed with intent
    CONFIRM = "confirm"    # Ask user to verify
    ESCALATE = "escalate"  # Route to human agent
```

#### `ConfidenceThresholds`
```python
@dataclass
class ConfidenceThresholds:
    high: float = 0.85    # Auto-execute threshold
    medium: float = 0.60  # Confirm threshold
    
    def validate(self):
        if not (0 < self.medium < self.high < 1):
            raise ValueError("Invalid thresholds")
```

**Why these defaults?**
- 0.85: High enough to be reliable, low enough to be useful
- 0.60: Below this, too many wrong predictions
- Based on typical transformer confidence distributions

---

#### `ConfidenceHandler`

##### `__init__(thresholds, enable_logging)`
```python
def __init__(
    thresholds: ConfidenceThresholds = None,
    enable_logging: bool = True
):
    self.thresholds = thresholds or ConfidenceThresholds()
    
    # Track statistics
    self.stats = {
        "execute": 0,
        "confirm": 0,
        "escalate": 0,
    }
```

##### `get_action(confidence, intent)`
**Purpose**: Determine action based on confidence score.

**Returns**: `(ConfidenceAction, explanation_string)`

**Logic**:
```python
def get_action(self, confidence, intent):
    if confidence >= self.thresholds.high:
        return ConfidenceAction.EXECUTE, "High confidence - proceeding"
    elif confidence >= self.thresholds.medium:
        return ConfidenceAction.CONFIRM, f"Please confirm: {intent}?"
    else:
        return ConfidenceAction.ESCALATE, "Connecting to agent"
```

##### `process_prediction(prediction)`
**Purpose**: Add action info to prediction result.

**Input**:
```python
{"intent": "check_status", "confidence": 0.75}
```

**Output**:
```python
{
    "intent": "check_status",
    "confidence": 0.75,
    "action": "confirm",
    "action_enum": ConfidenceAction.CONFIRM,
    "explanation": "Medium confidence - confirm with user",
    "should_execute": False,
    "needs_confirmation": True,
    "needs_escalation": False,
}
```

##### `generate_user_prompt(prediction, alternatives)`
**Purpose**: Create user-facing message based on action.

**For EXECUTE**:
```
"I'll help you check your application status."
```

**For CONFIRM**:
```
"I think you want to check your application status. Is that correct?

Or did you mean: get requirements, check fees?"
```

**For ESCALATE**:
```
"I'm not quite sure what you need. Let me connect you with a customer service representative."
```

##### `get_stats()`
**Purpose**: Get action distribution statistics.

**Returns**:
```python
{
    "total_predictions": 1000,
    "action_counts": {"execute": 700, "confirm": 200, "escalate": 100},
    "action_percentages": {"execute": 70.0, "confirm": 20.0, "escalate": 10.0},
    "thresholds": {"high": 0.85, "medium": 0.60}
}
```

**Why track stats?**
- Monitor model health over time
- Detect drift (escalation rate increasing = model degrading)
- Optimize thresholds based on production data

---

### Functions

#### `analyze_confidence_distribution(confidences, thresholds)`
**Purpose**: Analyze confidence scores for threshold tuning.

**Returns**:
```python
{
    "count": 1000,
    "mean": 0.78,
    "std": 0.15,
    "min": 0.32,
    "max": 0.99,
    "median": 0.82,
    "percentiles": {
        "25th": 0.65,
        "50th": 0.82,
        "75th": 0.91,
        "90th": 0.95,
        "95th": 0.97
    },
    "action_distribution": {
        "execute_pct": 60.0,
        "confirm_pct": 25.0,
        "escalate_pct": 15.0
    }
}
```

**Use case**:
- Understand model confidence behavior
- Tune thresholds: if 90% are escalated, thresholds too high
- Compare across model versions

---

## Key Design Decisions

### Why three tiers instead of two?
**Option A**: Execute or Escalate
- Simple but frustrating UX (too many escalations)

**Option B**: Execute or Confirm
- Misses truly ambiguous cases that need human judgment

**Option C**: Execute / Confirm / Escalate (our choice)
- Best UX: Most requests handled automatically
- Safety net: Confirm catches medium-confidence errors
- Last resort: Escalate for truly uncertain cases

### Why 0.85 and 0.60 as defaults?
- Analyzed typical transformer confidence distributions
- 0.85: Typically corresponds to ~95% accuracy
- 0.60: Below this, error rate rises sharply
- Can be tuned based on production metrics

### Why track action statistics?
**Scenario 1**: Escalation rate increases from 5% to 20%
- Indicates model drift or new intent patterns
- Trigger for retraining

**Scenario 2**: Execute rate is 95%
- Model might be overconfident
- Check if ECE is high (miscalibrated)

### Why generate user prompts?
- Consistent user experience
- Centralized prompt management
- Easy to localize for Kinyarwanda
