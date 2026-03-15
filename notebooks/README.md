# Notebooks Documentation

This folder contains Jupyter notebooks for exploratory data analysis, training, and evaluation.

---

## 01_eda.ipynb - Exploratory Data Analysis

### Purpose
Understand the dataset before building the model. Identifies challenges and informs design decisions.

### Sections

#### 1. Data Loading
**What it does**:
- Loads training data from `voiceai_intent_train.csv`
- Loads data dictionary from `voiceai_intent_data_dictionary.json`
- Displays sample rows and column info

**Key outputs**:
- Dataset shape (rows, columns)
- Column names and types
- Sample utterances

---

#### 2. Intent Distribution Analysis
**What it does**:
- Counts samples per intent class
- Visualizes class imbalance
- Computes imbalance ratio

**Key findings to look for**:
- Which intents have most/least samples?
- How severe is the imbalance? (max/min ratio)
- Should we use class weights or focal loss?

**Code logic**:
```python
intent_counts = df["intent"].value_counts()
plt.bar(intent_counts.index, intent_counts.values)
imbalance_ratio = intent_counts.max() / intent_counts.min()
```

---

#### 3. Language Distribution
**What it does**:
- Counts samples per language (en, rw, mixed)
- Shows cross-tabulation of language × intent
- Visualizes language balance per intent

**Key findings to look for**:
- Is English over-represented?
- Are any intents single-language only?
- Is Kinyarwanda sufficiently represented?

**Why this matters**:
- Need balanced language data for fair model
- Low-resource language (rw) may need augmentation

---

#### 4. Text Length Analysis
**What it does**:
- Computes character and word counts
- Visualizes length distribution
- Compares lengths across languages and intents

**Key findings to look for**:
- What's the max text length? (determines `max_length` config)
- Do Kinyarwanda utterances tend to be longer/shorter?
- Any outlier lengths?

**Code logic**:
```python
df["char_length"] = df["utterance_text"].str.len()
df["word_count"] = df["utterance_text"].str.split().str.len()
```

---

#### 5. ASR Confidence Analysis
**What it does**:
- Analyzes `asr_confidence` column
- Visualizes confidence distribution
- Correlates with language/channel

**Key findings to look for**:
- What % of data has low ASR confidence (<0.7)?
- Is ASR worse for Kinyarwanda?
- Which channels have lowest quality?

**Why this matters**:
- Low ASR confidence = noisier text
- May need different handling for low-confidence inputs

---

#### 6. Code-Switching Detection
**What it does**:
- Identifies utterances with mixed language markers
- Counts code-switching patterns
- Shows examples

**Detection patterns**:
- English in Kinyarwanda: `["application", "status", "password"]`
- Kinyarwanda markers: `["ndashaka", "yanjye", "cy'"]`
- Mixed = contains both

**Why this matters**:
- Code-switching is challenging for NLP
- Need model that handles mixed languages (XLM-RoBERTa)

---

#### 7. Channel and Region Analysis
**What it does**:
- Shows distribution across channels (voice_call, whatsapp, ivr, mobile_app)
- Shows distribution across regions
- Cross-tabulates with intent

**Why this matters**:
- Channel affects text quality (IVR has shorter utterances)
- Regional patterns may exist

---

#### 8. Data Quality Summary
**What it does**:
- Summarizes key findings
- Identifies potential issues
- Recommends preprocessing steps

---

## 02_training.ipynb - Model Training

### Purpose
Train the XLM-RoBERTa intent classifier with proper hyperparameters and monitoring.

### Sections

#### 1. Setup
**What it does**:
- Imports all required modules
- Loads configuration from `configs/config.yaml`
- Sets up device (GPU/CPU)

**Key configuration**:
```python
config = yaml.safe_load(open("configs/config.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

#### 2. Data Loading
**What it does**:
- Creates train/validation/test DataLoaders
- Uses TextPreprocessor for cleaning
- Shows sample batch

**Code logic**:
```python
dataloaders, label_encoder = create_dataloaders(
    train_path=config["data"]["train_path"],
    val_path=config["data"]["val_path"],
    test_path=config["data"]["test_path"],
    batch_size=config["training"]["batch_size"],
    max_length=config["model"]["max_length"]
)
```

---

#### 3. Model Initialization
**What it does**:
- Creates IntentClassifier model
- Moves to GPU if available
- Prints model architecture summary

**Code logic**:
```python
model = create_model(
    model_name=config["model"]["name"],
    num_labels=label_encoder.num_labels,
    dropout=config["model"]["dropout"],
    use_focal_loss=True,
    focal_gamma=config["focal_loss"]["gamma"]
)
model.to(device)
```

---

#### 4. Training Loop
**What it does**:
- Initializes IntentTrainer
- Runs training with early stopping
- Logs progress per epoch

**Key monitoring**:
- Training loss per epoch
- Validation loss, accuracy, F1 per epoch
- Best model checkpoint saved

**Code logic**:
```python
trainer = IntentTrainer(
    model=model,
    config=TrainingConfig(
        learning_rate=config["training"]["learning_rate"],
        num_epochs=config["training"]["num_epochs"],
        ...
    )
)
history = trainer.train(train_loader, val_loader)
```

---

#### 5. Training History Visualization
**What it does**:
- Plots training vs validation loss curves
- Plots validation F1 over epochs
- Identifies overfitting (if any)

**What to look for**:
- Loss curves should decrease then flatten
- Val loss shouldn't increase while train loss decreases (overfitting)
- F1 should increase and plateau

---

#### 6. Model Evaluation on Test Set
**What it does**:
- Loads best checkpoint
- Evaluates on held-out test set
- Reports final metrics

**Final output**:
- Test accuracy
- Test macro F1
- Per-class F1 scores

---

## 03_evaluation.ipynb - Comprehensive Evaluation

### Purpose
Deep evaluation of trained model with error analysis and confidence calibration.

### Sections

#### 1. Load Model and Data
**What it does**:
- Loads best model checkpoint
- Loads test dataset
- Initializes predictor and label encoder

---

#### 2. Generate Predictions
**What it does**:
- Runs inference on all test samples
- Extracts predicted intents and confidence scores
- Prepares data for metric computation

**Code logic**:
```python
results = predictor.predict_batch(texts, return_all_scores=False)
pred_intents = [r["intent"] for r in results]
confidences = [r["confidence"] for r in results]
```

---

#### 3. Overall Metrics
**What it does**:
- Computes accuracy, precision, recall, F1
- Computes macro and weighted averages
- Prints formatted summary

**Key metrics**:
- **Accuracy**: % of correct predictions
- **Macro F1**: Average F1 across all classes (treats all equally)
- **Weighted F1**: F1 weighted by class frequency

---

#### 4. Confusion Matrix
**What it does**:
- Creates confusion matrix heatmap
- Identifies most confused intent pairs
- Saves plot to outputs/plots/

**What to look for**:
- Diagonal should be darkest (correct predictions)
- Off-diagonal hot spots = systematic confusions
- E.g., "check_status" confused with "requirements"

---

#### 5. Per-Language Performance
**What it does**:
- Computes accuracy and F1 for each language
- Visualizes language comparison
- Checks for fairness issues

**Key question**: 
- Does Kinyarwanda have significantly worse performance?
- If yes, may need more training data or specific handling

---

#### 6. Confidence Analysis
**What it does**:
- Plots confidence score distribution
- Shows action distribution (execute/confirm/escalate)
- Validates confidence thresholds

**Key questions**:
- What % of predictions are high confidence?
- Is the model overconfident (many confident but wrong)?

---

#### 7. Error Analysis
**What it does**:
- Uses ErrorAnalyzer to categorize errors
- Shows examples of misclassifications
- Identifies patterns in errors

**Key outputs**:
- Error rate by language
- High-confidence errors (most dangerous)
- Common confusion patterns with example texts

---

#### 8. Summary
**What it does**:
- Prints comprehensive evaluation summary
- Highlights key findings
- Notes production readiness metrics

---

## Running the Notebooks

### Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Data files in `datasets/` folder
3. For training: GPU recommended (but works on CPU)

### Execution Order
1. **01_eda.ipynb**: Run first to understand data
2. **02_training.ipynb**: Train the model (saves checkpoint)
3. **03_evaluation.ipynb**: Evaluate trained model

### Tips
- Run cells in order (top to bottom)
- Check outputs before proceeding to next section
- Plots are saved to `outputs/plots/`
- Model is saved to `outputs/models/`
