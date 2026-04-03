# Voice AI Intent Classifier

A production-style **multilingual intent classification** pipeline for voice-assistant use cases: it takes **ASR text** (transcripts) and predicts one of **13 service intents**, with **confidence-based routing** (execute / confirm / escalate). Built to handle **English**, **code-switching**, and an African language track, **Yoruba** (Nigerian locale) or **Kinyarwanda** (Rwanda locale), depending on which dataset is active in config.

**Why it matters:** Voice channels (IVR, WhatsApp voice notes, call centers) need reliable intent detection across how people *actually* speak, not only clean textbook English.

## What’s in this repo

| Piece | Description |
|--------|-------------|
| **Model** | Fine-tuned **XLM-RoBERTa** (full encoder + classification head) |
| **Loss** | **Focal loss** for class imbalance and noisy labels |
| **Inference** | `IntentPredictor` + **confidence thresholds** for safe automation |
| **API** | **FastAPI** server for `/predict` and health checks |
| **Notebooks** | EDA → training → evaluation & error analysis |

## Datasets (switch in config)

The project ships with **two synthetic intent datasets** (same 13 intents, similar schema: channel, device, region, language, utterance text, ASR confidence, etc.):

| Dataset | Languages (typical) | CSV prefix | Default? |
|---------|---------------------|------------|----------|
| **Nigerian** | English, Yoruba, mixed | `voiceai_intent_nigerian_*.csv` | **Yes** — active in `configs/config.yaml` |
| **Rwanda** | English, Kinyarwanda (`rw`), mixed | `voiceai_intent_*.csv` | Optional — paths are commented in the same file |

To use Rwanda again, comment the Nigerian `data:` block and uncomment the Rwanda block in `configs/config.yaml`. Notebooks that load paths from config will follow automatically.

## Quick start

### 1. Installation

```bash
git clone https://github.com/Ayoung9598/voice-ai-intent-classifier.git
cd voice-ai

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train (produces `outputs/models/best_model.pt`)

```bash
python run_training.py

# Quick test
python run_training.py --epochs 3 --batch-size 16
```

Trained checkpoints are **not** committed to git; run training locally or upload your `best_model.pt` when using **Google Colab**.

### 3. Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | Dataset sizes, intent/language/channel distributions, sample utterances |
| `notebooks/02_training.ipynb` | Training workflow in the notebook |
| `notebooks/03_evaluation.ipynb` | Load model, metrics, confusion matrix, language-stratified results, confidence / escalation analysis |

Run notebooks from the **`notebooks/`** directory *or* project root; path logic uses `configs/config.yaml` where applicable.

### 4. Inference & API

```bash
python run_inference.py --model outputs/models/best_model.pt

python run_inference.py --demo

python run_inference.py --serve --model outputs/models/best_model.pt --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Mo fe wo ipo ti aplikasiyo mi\"}"
```

API docs: `http://localhost:8000/docs`



## Project structure

```
voice-ai/
├── configs/
│   └── config.yaml           # Data paths, training, inference thresholds
├── datasets/                  # Nigerian + Rwanda CSVs (see config)
├── docs/
│   └── requirements_analysis.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── outputs/
│   ├── models/               # best_model.pt (after training; gitignored)
│   ├── logs/
│   └── plots/
├── src/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   └── models/
├── requirements.txt
├── run_training.py
└── run_inference.py
```

## Technical approach

### XLM-RoBERTa

Multilingual encoder suited to low-resource and code-switched text; fine-tuned end-to-end for the 13-way intent head.

### Focal loss

Reduces dominance of easy/frequent classes and remains useful under mild label noise in the synthetic data.

### Confidence-based routing

| Confidence | Action |
|------------|--------|
| ≥ 0.85 | **Execute** — automate the intent flow |
| 0.60–0.85 | **Confirm** — ask the user |
| Under 0.60 | **Escalate** — human handoff |

Thresholds are configurable under `inference.confidence_thresholds` in `configs/config.yaml`.

## Intents (13)

`check_application_status`, `start_new_application`, `requirements_information`, `fees_information`, `appointment_booking`, `payment_help`, `reset_password_login_help`, `speak_to_agent`, `cancel_or_reschedule_appointment`, `update_application_details`, `document_upload_help`, `service_eligibility`, `complaint_or_support_ticket`

## Evaluation metrics

- Accuracy / F1 (macro & weighted)  
- Per-language (or stratum) accuracy  
- Expected calibration error (ECE) on confidence  
- Confusion matrix and error analysis (see `03_evaluation.ipynb`)

## Configuration snippet

```yaml
data:
  train_path: "datasets/voiceai_intent_nigerian_train.csv"
  val_path: "datasets/voiceai_intent_nigerian_val.csv"
  test_path: "datasets/voiceai_intent_nigerian_test.csv"

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 7
  early_stopping_patience: 2

inference:
  confidence_thresholds:
    high: 0.85
    medium: 0.60
```

## Hardware

- **Training:** GPU recommended (~8GB VRAM for batch 16); time depends on epochs and dataset size.  
- **Inference:** CPU viable (~100–200ms per prediction); GPU optional for lower latency.

## License

MIT

## Author

Ayomide Abiola
