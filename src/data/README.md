# Data Module Documentation

This module handles data loading, preprocessing, and dataset creation for the Voice AI Intent Classifier.

---

## preprocessor.py

### Purpose
Handles text preprocessing for noisy ASR (Automatic Speech Recognition) transcripts. Voice transcripts often contain errors, repeated characters, and code-switching (mixing languages) that need to be normalized before feeding to the model.

### Classes

#### `PreprocessingConfig`
A dataclass holding preprocessing configuration options.

```python
@dataclass
class PreprocessingConfig:
    normalize_unicode: bool = True      # Normalize Unicode characters (important for Kinyarwanda)
    fix_repeated_chars: bool = True     # Fix "helllp" -> "help"
    min_repeat_chars: int = 3           # Minimum repeats to trigger fix
    lowercase: bool = False             # Keep case for XLM-RoBERTa (it uses casing)
    preserve_code_switching: bool = True # Don't force single language
    remove_extra_whitespace: bool = True
```

**Why these defaults?**
- `lowercase=False`: XLM-RoBERTa is case-sensitive and uses casing information
- `preserve_code_switching=True`: Mixed Kinyarwanda/English is common and informative

---

#### `TextPreprocessor`

Main preprocessing class with the following methods:

##### `preprocess(text: str) -> str`
**Purpose**: Apply full preprocessing pipeline to a single text.

**Pipeline Steps**:
1. Unicode normalization (NFC) - ensures consistent character representation
2. Fix repeated characters - removes ASR stuttering artifacts
3. ASR error correction - fixes common transcription mistakes
4. Whitespace normalization - removes extra spaces

**Why this order?**
- Unicode must come first to ensure consistent character matching
- Repeated chars before spelling correction (otherwise "helllp" won't match)
- Whitespace cleanup at the end to catch any introduced spaces

```python
def preprocess(self, text: str) -> str:
    # Step 1: Unicode normalization
    if self.config.normalize_unicode:
        text = self._normalize_unicode(text)
    
    # Step 2: Fix repeated characters
    if self.config.fix_repeated_chars:
        text = self._fix_repeated_chars(text)
    
    # Step 3: ASR error correction
    text = self._correct_asr_errors(text)
    
    # Step 4: Whitespace normalization
    if self.config.remove_extra_whitespace:
        text = self._normalize_whitespace(text)
    
    return text.strip()
```

##### `_normalize_unicode(text: str) -> str`
**Purpose**: Apply NFC Unicode normalization.

**Why NFC?**
- Kinyarwanda uses diacritics (accent marks)
- Different keyboard inputs can produce visually identical but byte-different characters
- NFC combines base characters with diacritics into single code points
- Example: `é` can be stored as one character or `e` + combining accent

##### `_fix_repeated_chars(text: str) -> str`
**Purpose**: Reduce 3+ repeated characters to 1.

**Why?**
- ASR systems sometimes capture speech hesitation as repeated letters
- "Helllllo" should become "Helo" (close enough for transformer tokenization)
- We use regex: `(.)\1{2,}` matches any char repeated 3+ times

```python
pattern = r'(.)\1{2,}'  # Match 3+ of same character
return re.sub(pattern, r'\1', text)  # Replace with single instance
```

##### `_correct_asr_errors(text: str) -> str`
**Purpose**: Fix common ASR transcription mistakes.

**Correction Map** (from dataset observation):
```python
ASR_CORRECTIONS = {
    "aplikasiyo": "application",  # Kinyarwanda phonetic spelling
    "applicashon": "application", # Common ASR error
    "aplication": "application",  # Missing 'p'
    "sttatus": "status",          # Double 't' error
    "statuz": "status",           # 's' -> 'z' error
}
```

**Why conservative corrections?**
- We only correct obvious English misspellings
- Aggressive correction might break valid Kinyarwanda words
- The model should learn to handle remaining noise

##### `_has_kinyarwanda_patterns(text: str) -> bool`
**Purpose**: Detect if text contains Kinyarwanda language markers.

**Patterns checked**:
- Common words: `ndashaka`, `nashaka`, `nkeneye`, `ese`, `kuri`
- Possessive: `yanjye` (my)
- Contractions: `y'`, `cy'`, `bw'` (Kinyarwanda uses apostrophes)

**Why needed?**
- Used for text statistics and analysis
- Helps understand language distribution in errors

##### `batch_preprocess(texts: List[str]) -> List[str]`
**Purpose**: Preprocess multiple texts efficiently.

Simply applies `preprocess()` to each text in the list.

---

## dataset.py

### Purpose
Handles PyTorch Dataset creation, tokenization, and label encoding for training the intent classifier.

### Classes

#### `IntentLabelEncoder`
Maps intent names to integer indices and vice versa.

##### `INTENTS` (class variable)
```python
INTENTS = [
    "check_application_status",      # 0
    "start_new_application",         # 1
    "requirements_information",      # 2
    "fees_information",              # 3
    "appointment_booking",           # 4
    "payment_help",                  # 5
    "reset_password_login_help",     # 6
    "speak_to_agent",                # 7
    "cancel_or_reschedule_appointment", # 8
    "update_application_details",    # 9
    "document_upload_help",          # 10
    "service_eligibility",           # 11
    "complaint_or_support_ticket",   # 12
]
```

**Why fixed order?**
- Ensures consistent label indices across training/inference
- Model learns index -> intent mapping
- Changing order would break trained models

##### `encode(intent: str) -> int`
Converts intent name to index: `"speak_to_agent"` -> `7`

##### `decode(idx: int) -> str`
Converts index to intent name: `7` -> `"speak_to_agent"`

##### `num_labels` property
Returns `13` (total number of intents)

---

#### `IntentDataset(Dataset)`
PyTorch Dataset for intent classification.

##### `__init__(...)`
```python
def __init__(
    texts: List[str],           # Raw utterance texts
    labels: List[int] = None,   # Intent indices (None for inference)
    tokenizer = None,           # HuggingFace tokenizer
    max_length: int = 128,      # Max sequence length
    preprocessor = None,        # TextPreprocessor instance
    metadata: pd.DataFrame = None  # Additional features
):
```

**Initialization flow**:
1. Store texts and labels
2. Create/use preprocessor
3. Load/use tokenizer (defaults to xlm-roberta-base)
4. Preprocess all texts upfront (cached for efficiency)

**Why preprocess upfront?**
- Preprocessing is deterministic
- Avoids repeated computation during training
- Speeds up data loading in training loop

##### `__getitem__(idx: int) -> Dict[str, torch.Tensor]`
**Purpose**: Get a single tokenized sample for model input.

**Returns**:
```python
{
    "input_ids": tensor([0, 123, 456, ...]),      # Token IDs
    "attention_mask": tensor([1, 1, 1, ...]),     # 1 for real tokens, 0 for padding
    "labels": tensor(7),                          # Intent index (if training)
}
```

**Tokenization details**:
- Uses XLM-RoBERTa tokenizer (SentencePiece-based)
- Pads to `max_length` with padding tokens
- Truncates if longer than `max_length`
- Returns PyTorch tensors (squeezed to remove batch dimension)

---

### Functions

#### `load_data(data_path, tokenizer, max_length, preprocessor)`
**Purpose**: Load CSV data and create IntentDataset.

**Returns**: `(IntentDataset, IntentLabelEncoder, pd.DataFrame)`

**Flow**:
1. Read CSV file
2. Create label encoder
3. Extract texts and encode labels
4. Create IntentDataset with metadata

**Why return raw DataFrame?**
- Needed for language-stratified evaluation
- Contains metadata (ASR confidence, channel, region)

#### `create_dataloaders(...)`
**Purpose**: Create train/val/test DataLoaders in one call.

**Returns**: `(Dict[str, DataLoader], IntentLabelEncoder)`

**DataLoader settings**:
- `shuffle=True` for training (random order each epoch)
- `shuffle=False` for val/test (reproducible evaluation)
- `num_workers=0` by default (safe for Windows)

**Why shared tokenizer/preprocessor?**
- Ensures consistent tokenization across splits
- Avoids multiple model downloads

---

## Key Design Decisions

### Why max_length=128?
- Dataset analysis shows max text length ~70 characters
- 128 tokens provides comfortable margin
- Longer sequences slow training without benefit
- XLM-RoBERTa can handle up to 512, but 128 is sufficient

### Why preprocess before tokenization?
- Tokenizer handles subword splitting
- Preprocessing fixes character-level issues
- Order: Clean text -> Tokenize -> Model input

### Why keep labels optional in IntentDataset?
- Training: Labels provided for loss computation
- Inference: No labels, just predict
- Same Dataset class works for both scenarios
