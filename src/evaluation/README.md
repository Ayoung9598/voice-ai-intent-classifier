# Evaluation Module Documentation

This module provides comprehensive evaluation metrics and error analysis tools.

---

## metrics.py

### Purpose
Compute evaluation metrics for intent classification, including overall performance, per-class breakdown, and confidence calibration.

### Classes

#### `EvaluationReport`
Dataclass containing all evaluation results.

```python
@dataclass
class EvaluationReport:
    # Overall metrics
    accuracy: float                    # Correct / Total
    macro_f1: float                    # Average F1 across classes
    weighted_f1: float                 # F1 weighted by class frequency
    macro_precision: float
    macro_recall: float
    
    # Per-class metrics
    per_class_f1: Dict[str, float]     # F1 for each intent
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_support: Dict[str, int]  # Sample count per class
    
    # Language-stratified
    per_language_accuracy: Dict[str, float]  # Accuracy for en/rw/mixed
    per_language_f1: Dict[str, float]
    
    # Confidence calibration
    mean_confidence: float
    ece: float  # Expected Calibration Error
    
    # Confusion matrix
    confusion_matrix: np.ndarray
```

##### `print_summary()`
**Purpose**: Print formatted evaluation summary to console.

Displays:
- Overall metrics with visual separators
- Per-language breakdown
- Per-class F1 scores sorted by performance

##### `to_dict()` / `to_json(path)`
Export report for logging or analysis.

---

### Functions

#### `compute_metrics(y_true, y_pred, intent_names, confidences, languages)`
**Purpose**: Compute all evaluation metrics in one call.

**Parameters**:
```python
y_true: List[int]           # Ground truth indices
y_pred: List[int]           # Predicted indices
intent_names: List[str]     # For labeling results
confidences: List[float]    # Prediction confidences (0-1)
languages: List[str]        # Language per sample (for stratification)
```

**Computation flow**:
```python
def compute_metrics(...):
    report = EvaluationReport()
    
    # 1. Overall accuracy
    report.accuracy = accuracy_score(y_true, y_pred)
    
    # 2. Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    report.macro_f1 = np.mean(f1)
    
    # 3. Weighted F1 (accounts for class frequency)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    report.weighted_f1 = weighted_f1
    
    # 4. Confusion matrix
    report.confusion_matrix = confusion_matrix(y_true, y_pred)
    
    # 5. Per-language metrics
    for lang in unique_languages:
        mask = languages == lang
        lang_acc = accuracy_score(y_true[mask], y_pred[mask])
        report.per_language_accuracy[lang] = lang_acc
    
    # 6. Calibration (ECE)
    if confidences is not None:
        report.ece = compute_ece(y_true, y_pred, confidences)
    
    return report
```

**Why both macro and weighted F1?**
- **Macro F1**: Treats all classes equally - good for fairness
- **Weighted F1**: Accounts for class frequency - good for production metrics
- For government services, macro F1 ensures minority intents aren't neglected

---

#### `compute_ece(y_true, y_pred, confidences, n_bins=10)`
**Purpose**: Compute Expected Calibration Error.

**What is ECE?**
Measures how well prediction confidence matches actual accuracy.
- Perfect calibration: 80% confident predictions are correct 80% of the time
- ECE = 0 means perfectly calibrated
- ECE = 1 means completely miscalibrated

**Algorithm**:
```python
def compute_ece(y_true, y_pred, confidences, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.any():
            # Accuracy of samples in bin
            accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
            
            # Average confidence in bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Weighted absolute difference
            ece += |accuracy_in_bin - avg_confidence_in_bin| * proportion_in_bin
    
    return ece
```

**Why ECE matters for Voice AI?**
- Confidence thresholds determine whether to execute, confirm, or escalate
- Miscalibrated confidence leads to wrong routing decisions
- Low ECE means we can trust the confidence scores

---

#### `get_confused_pairs(confusion_mat, intent_names, top_k=5)`
**Purpose**: Find most commonly confused intent pairs.

**Returns**: `List[(true_intent, pred_intent, count)]`

**Algorithm**:
```python
def get_confused_pairs(confusion_mat, intent_names, top_k=5):
    confused_pairs = []
    
    for i in range(n_intents):
        for j in range(n_intents):
            if i != j and confusion_mat[i, j] > 0:
                confused_pairs.append((
                    intent_names[i],  # True
                    intent_names[j],  # Predicted
                    confusion_mat[i, j]  # Error count
                ))
    
    # Sort by count descending
    confused_pairs.sort(key=lambda x: -x[2])
    return confused_pairs[:top_k]
```

**Why useful?**
- Identifies systematic errors
- Guides data collection (add more examples of confused pairs)
- Reveals intent label ambiguity

---

#### `analyze_low_confidence_predictions(...)`
**Purpose**: Analyze predictions below confidence threshold.

**Returns**:
```python
{
    "low_confidence_count": 15,
    "low_confidence_ratio": 0.10,
    "low_confidence_accuracy": 0.45,  # Much lower than overall
    "high_confidence_accuracy": 0.95,  # Much higher
    "intent_distribution_low_conf": {...}  # Which intents are uncertain
}
```

**Why useful?**
- Validates confidence thresholds
- Identifies intents that are inherently ambiguous
- Informs escalation strategy

---

## error_analysis.py

### Purpose
Deep dive into model errors to identify patterns and improvement opportunities.

### Classes

#### `ErrorCase`
Dataclass representing a single misclassification.

```python
@dataclass
class ErrorCase:
    text: str                  # Original utterance
    true_intent: str           # Ground truth
    predicted_intent: str      # What model predicted
    confidence: float          # How confident was model
    language: str              # en/rw/mixed
    asr_confidence: float      # ASR quality
    channel: str               # voice_call/whatsapp/ivr/mobile_app
    region: str                # Kigali/Northern/etc.
```

---

#### `ErrorAnalyzer`

##### `__init__(...)`
```python
def __init__(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    intent_names: List[str],
    confidences: List[float],
    metadata: pd.DataFrame
):
    # Find all errors
    self.error_mask = y_true != y_pred
    self.error_indices = np.where(self.error_mask)[0]
    
    # Build ErrorCase objects
    self.error_cases = self._build_error_cases()
```

##### `error_rate` property
Returns: `num_errors / total_samples`

##### `get_errors_by_intent_pair()`
**Purpose**: Group errors by (true, predicted) pairs.

**Returns**: `Dict[(str, str), List[ErrorCase]]`

**Use case**: Find all cases where "check_status" was predicted as "requirements"

##### `get_errors_by_language()`
**Purpose**: Group errors by language.

**Returns**: `Dict[str, List[ErrorCase]]`

**Use case**: See if Kinyarwanda has more errors than English

##### `get_top_confused_intents(top_k=5)`
**Purpose**: Find most common error patterns with examples.

**Returns**:
```python
[
    {
        "true_intent": "check_application_status",
        "predicted_intent": "requirements_information",
        "count": 5,
        "avg_confidence": 0.72,
        "example_texts": ["Text 1", "Text 2", "Text 3"]
    },
    ...
]
```

**Why include examples?**
- Helps understand why confusion happens
- May reveal labeling issues in dataset
- Guides prompt engineering for user confirmation

##### `get_language_error_rates()`
**Purpose**: Calculate error rate per language.

**Returns**:
```python
{
    "en": {"total": 100, "errors": 5, "error_rate": 0.05},
    "rw": {"total": 80, "errors": 8, "error_rate": 0.10},
    "mixed": {"total": 50, "errors": 7, "error_rate": 0.14}
}
```

**Why important?**
- Fairness: Kinyarwanda speakers shouldn't have worse experience
- Identifies if model struggles with specific languages
- May indicate need for more training data in that language

##### `get_low_asr_confidence_analysis(threshold=0.7)`
**Purpose**: Compare errors for low vs high ASR confidence.

**Returns**:
```python
{
    "threshold": 0.7,
    "low_asr_total": 50,
    "low_asr_errors": 15,
    "low_asr_error_rate": 0.30,  # Higher!
    "high_asr_error_rate": 0.05  # Lower
}
```

**Why useful?**
- ASR errors compound with intent classification errors
- May want to escalate low ASR confidence to human
- Informs overall Voice AI pipeline design

##### `get_high_confidence_errors(threshold=0.8)`
**Purpose**: Find errors where model was confidently wrong.

**Returns**: `List[ErrorCase]` with confidence >= threshold

**Why critical?**
- These are the most dangerous errors
- High confidence means auto-execution (no confirmation)
- May indicate training data issues or inherent ambiguity

##### `print_report()`
**Purpose**: Print comprehensive error analysis.

Includes:
- Overall error statistics
- Error rate by language
- Top confused pairs with examples
- High confidence error count

---

## Key Design Decisions

### Why stratify by language?
- Fairness requirement for government services
- Kinyarwanda is low-resource, may have worse performance
- Need to ensure all citizens get equal service quality

### Why track ASR confidence in errors?
- Voice AI pipeline has multiple failure points
- Bad ASR -> bad intent -> wrong action
- Understanding correlation helps pipeline optimization

### Why include example texts in error analysis?
- Numbers alone don't explain why errors happen
- Examples reveal patterns (e.g., certain phrases always misclassified)
- Essential for iterative improvement
