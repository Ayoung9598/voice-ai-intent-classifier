# Voice AI Intent Classification System

A multilingual intent classifier for Rwanda's Voice AI assistant, handling Kinyarwanda, English, and mixed-language (code-switching) utterances from voice transcripts.

## Project Overview

This solution addresses the ML Take-Home Assignment for the Machine Learning Engineer - Voice AI position. The system classifies user intents from transcribed speech to enable natural-language voice interactions with government services.

### Key Features

- **Multilingual Support**: Handles Kinyarwanda, English, and code-switching
- **XLM-RoBERTa Model**: Fine-tuned multilingual transformer for robust classification
- **Focal Loss**: Handles class imbalance and label noise (~3%)
- **Confidence-Based Fallback**: Three-tier system (execute/confirm/escalate)
- **Production-Ready API**: FastAPI endpoint for real-time inference

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Explore the Data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train the Model

```bash
# Using the training script
python run_training.py

# Or with custom settings
python run_training.py --epochs 5 --batch-size 16
```

### 4. Run Inference

```bash
# Interactive mode
python run_inference.py --model outputs/models/best_model.pt

# Demo mode (no trained model needed)
python run_inference.py --demo

# Start API server
python run_inference.py --serve --model outputs/models/best_model.pt --port 8000
```

### 5. API Usage

Once the server is running:

```bash
# Health check
curl http://localhost:8000/health

# Predict intent
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ndashaka kureba status ya application yanjye."}'
```

API documentation available at: `http://localhost:8000/docs`

## Project Structure

```
voice-ai/
├── configs/
│   └── config.yaml           # Hyperparameters and settings
├── datasets/                  # Provided datasets
├── docs/
│   └── requirements_analysis.md  # Detailed technical decisions
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_training.ipynb     # Model training notebook
├── outputs/
│   ├── models/               # Saved model checkpoints
│   ├── logs/                 # Training logs
│   └── plots/                # Visualizations
├── src/
│   ├── api/                  # FastAPI application
│   ├── data/                 # Data loading and preprocessing
│   ├── evaluation/           # Metrics and error analysis
│   ├── inference/            # Prediction and confidence handling
│   └── models/               # Model architecture and training
├── requirements.txt
├── run_training.py           # CLI training script
└── run_inference.py          # CLI inference script
```

## Technical Approach

### Model: XLM-RoBERTa

- Pre-trained on 100+ languages including low-resource African languages
- Handles code-switching naturally
- Good balance between performance and inference latency

### Loss Function: Focal Loss

- Addresses class imbalance (13 intents with varying frequencies)
- Robust to label noise (~3% in dataset)
- Focuses learning on hard examples

### Confidence-Based Fallback

| Confidence | Action | Description |
|------------|--------|-------------|
| ≥ 85% | Execute | Proceed with intent automatically |
| 60-85% | Confirm | Ask user to confirm intent |
| < 60% | Escalate | Route to human agent |

## Intents Supported (13)

1. `check_application_status`
2. `start_new_application`
3. `requirements_information`
4. `fees_information`
5. `appointment_booking`
6. `payment_help`
7. `reset_password_login_help`
8. `speak_to_agent`
9. `cancel_or_reschedule_appointment`
10. `update_application_details`
11. `document_upload_help`
12. `service_eligibility`
13. `complaint_or_support_ticket`

## Evaluation

The model is evaluated using:

- **Macro F1**: Balanced performance across all intents
- **Per-Language Accuracy**: Ensures no language is underserved
- **Expected Calibration Error (ECE)**: Confidence reliability
- **Confusion Analysis**: Identifies commonly confused intent pairs

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  name: "xlm-roberta-base"
  max_length: 128
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 2.0e-5
  num_epochs: 10
  early_stopping_patience: 3

inference:
  confidence_thresholds:
    high: 0.85
    medium: 0.60
```

## Hardware Requirements

- **Training**: 
  - GPU recommended (CUDA-compatible)
  - ~8GB VRAM for batch_size=16
  - ~30min training on modern GPU
  
- **Inference**:
  - CPU supported (~100-200ms per prediction)
  - GPU optional (~20-50ms per prediction)

## License

This project is part of a take-home assessment and is not licensed for public use.

## Author

Machine Learning Engineer Candidate
