# Tests Module Documentation

This module contains unit tests for the Voice AI Intent Classifier pipeline.

---

## test_pipeline.py

### Purpose
Verify that all components work correctly in isolation and together.

### Running Tests

```bash
# Run all tests
pytest tests/test_pipeline.py -v

# Run specific test class
pytest tests/test_pipeline.py::TestPreprocessor -v

# Run with coverage
pytest tests/test_pipeline.py --cov=src -v
```

---

## Test Classes

### `TestPreprocessor`
Tests for `src/data/preprocessor.py`

#### `test_preprocessor_import`
**Purpose**: Verify module can be imported without errors.

**Why test imports?**
- Catches missing dependencies
- Catches syntax errors
- First sanity check

```python
def test_preprocessor_import(self):
    from src.data.preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor()
    assert preprocessor is not None
```

#### `test_preprocess_basic`
**Purpose**: Verify basic text passes through unchanged.

```python
def test_preprocess_basic(self):
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess("Hello world")
    assert result == "Hello world"  # No changes needed
```

#### `test_preprocess_repeated_chars`
**Purpose**: Verify repeated character normalization.

```python
def test_preprocess_repeated_chars(self):
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess("Helllllo world")
    assert result == "Helo world"  # 5 l's → 1 l
```

**Why this test matters**:
- ASR often captures stuttering as repeated letters
- Without fix, tokenizer creates weird subwords

#### `test_preprocess_asr_correction`
**Purpose**: Verify known ASR errors are corrected.

```python
def test_preprocess_asr_correction(self):
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess("Check my aplikasiyo status")
    assert "application" in result
```

#### `test_preprocess_empty`
**Purpose**: Verify empty/None input handling.

```python
def test_preprocess_empty(self):
    preprocessor = TextPreprocessor()
    assert preprocessor.preprocess("") == ""
    assert preprocessor.preprocess(None) == ""
```

**Why important**: Production may receive empty strings.

---

### `TestLabelEncoder`
Tests for `src/data/dataset.py::IntentLabelEncoder`

#### `test_num_labels`
**Purpose**: Verify correct number of intents.

```python
def test_num_labels(self):
    encoder = IntentLabelEncoder()
    assert encoder.num_labels == 13
```

#### `test_encode_decode`
**Purpose**: Verify round-trip encoding.

```python
def test_encode_decode(self):
    encoder = IntentLabelEncoder()
    
    intent = "check_application_status"
    idx = encoder.encode(intent)
    decoded = encoder.decode(idx)
    
    assert decoded == intent  # Round-trip works
```

**Why important**: Mismatch between training and inference labels causes wrong predictions.

---

### `TestModel`
Tests for `src/models/intent_classifier.py`

#### `test_model_forward`
**Purpose**: Verify model produces correct output shape.

```python
def test_model_forward(self):
    model = create_model()
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 32))  # batch=2, seq=32
    attention_mask = torch.ones(2, 32)
    
    outputs = model(input_ids, attention_mask)
    
    assert outputs["logits"].shape == (2, 13)  # batch x num_classes
```

**What could go wrong**:
- Wrong hidden_size → dimension mismatch
- Wrong num_labels → wrong output size

#### `test_model_with_labels`
**Purpose**: Verify loss computation works.

```python
def test_model_with_labels(self):
    model = create_model()
    
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32)
    labels = torch.tensor([0, 5])  # Two labels
    
    outputs = model(input_ids, attention_mask, labels)
    
    assert "loss" in outputs
    assert outputs["loss"].item() > 0  # Loss should be positive
```

#### `test_model_predict`
**Purpose**: Verify prediction method returns valid output.

```python
def test_model_predict(self):
    model = create_model()
    
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32)
    
    preds, confs = model.predict(input_ids, attention_mask)
    
    assert preds.shape == (2,)  # One prediction per sample
    assert all(0 <= c <= 1 for c in confs)  # Valid probabilities
```

---

### `TestFocalLoss`
Tests for `src/models/intent_classifier.py::FocalLoss`

#### `test_focal_loss_forward`
**Purpose**: Verify focal loss computation.

```python
def test_focal_loss_forward(self):
    loss_fn = FocalLoss(gamma=2.0)
    
    logits = torch.randn(4, 13)  # 4 samples, 13 classes
    targets = torch.tensor([0, 1, 2, 3])
    
    loss = loss_fn(logits, targets)
    
    assert loss.item() > 0  # Loss should be positive
    assert not torch.isnan(loss)  # Not NaN
```

**What could go wrong**:
- Log(0) → NaN
- Numeric instability with small probabilities

---

### `TestConfidenceHandler`
Tests for `src/inference/confidence.py`

#### `test_high_confidence_action`
**Purpose**: Verify high confidence returns EXECUTE.

```python
def test_high_confidence_action(self):
    handler = ConfidenceHandler()
    action, _ = handler.get_action(0.90)
    assert action == ConfidenceAction.EXECUTE
```

#### `test_medium_confidence_action`
**Purpose**: Verify medium confidence returns CONFIRM.

```python
def test_medium_confidence_action(self):
    handler = ConfidenceHandler()
    action, _ = handler.get_action(0.70)
    assert action == ConfidenceAction.CONFIRM
```

#### `test_low_confidence_action`
**Purpose**: Verify low confidence returns ESCALATE.

```python
def test_low_confidence_action(self):
    handler = ConfidenceHandler()
    action, _ = handler.get_action(0.40)
    assert action == ConfidenceAction.ESCALATE
```

**Why test all three?**
- Critical for production behavior
- Wrong thresholds = wrong user experience

---

### `TestMetrics`
Tests for `src/evaluation/metrics.py`

#### `test_compute_metrics_basic`
**Purpose**: Verify metric computation returns valid values.

```python
def test_compute_metrics_basic(self):
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]  # One error
    intent_names = ["a", "b", "c"]
    
    report = compute_metrics(y_true, y_pred, intent_names)
    
    assert 0 <= report.accuracy <= 1
    assert 0 <= report.macro_f1 <= 1
```

#### `test_ece_computation`
**Purpose**: Verify ECE calculation.

```python
def test_ece_computation(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # Perfect
    confidences = np.array([0.9, 0.8, 0.85, 0.95])
    
    ece = compute_ece(y_true, y_pred, confidences)
    
    assert 0 <= ece <= 1  # Valid range
```

---

### `TestPredictor`
Tests for `src/inference/predictor.py`

#### `test_predict_single`
**Purpose**: Verify single prediction works.

```python
def test_predict_single(self):
    model = create_model()
    predictor = IntentPredictor(model=model)
    
    result = predictor.predict("Check my application status")
    
    assert "intent" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
```

#### `test_predict_batch`
**Purpose**: Verify batch prediction works.

```python
def test_predict_batch(self):
    model = create_model()
    predictor = IntentPredictor(model=model)
    
    texts = ["Hello", "Check status", "Payment help"]
    results = predictor.predict_batch(texts)
    
    assert len(results) == 3
    assert all("intent" in r for r in results)
```

---

## Test Design Principles

### 1. Unit Tests, Not Integration Tests
Each test focuses on one component:
- Preprocessor tests don't load model
- Model tests use dummy inputs
- Predictor tests use untrained model

### 2. Test Expected Behavior, Not Implementation
```python
# Good: Test what it does
assert result == "application"  # Expected correction

# Bad: Test how it does it
assert preprocessor.ASR_CORRECTIONS["aplikasiyo"] == "application"
```

### 3. Test Edge Cases
- Empty input
- None input
- Single item batch
- Maximum confidence
- Minimum confidence

### 4. Fast Tests
- Don't load full model (slow)
- Use small dummy inputs
- No file I/O unless testing file operations

### 5. Reproducible
- Use fixed random seeds where needed
- No dependency on external services
- Tests work offline

---

## Adding New Tests

When adding new functionality:

1. **Add test class if new module**:
```python
class TestNewModule:
    def test_new_function(self):
        ...
```

2. **Test happy path first**:
```python
def test_basic_usage(self):
    result = my_function("normal input")
    assert result == expected
```

3. **Test error cases**:
```python
def test_empty_input(self):
    result = my_function("")
    assert result == ""  # or raises exception
```

4. **Test edge cases**:
```python
def test_very_long_input(self):
    result = my_function("x" * 10000)
    assert len(result) <= MAX_LENGTH
```
