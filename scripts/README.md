# Scripts Documentation

This document covers the main entry-point scripts for training and inference.

---

## run_training.py

### Purpose
Command-line script to train the intent classification model.

### Usage

```bash
# Basic usage (uses default config)
python run_training.py

# With custom config
python run_training.py --config configs/config.yaml

# Override specific settings
python run_training.py --learning-rate 3e-5 --epochs 15
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | `configs/config.yaml` |
| `--learning-rate` | Learning rate override | From config |
| `--epochs` | Number of epochs override | From config |
| `--batch-size` | Batch size override | From config |
| `--output-dir` | Directory for model outputs | `outputs/models` |
| `--device` | Device to use (`cuda`/`cpu`) | Auto-detect |

### What It Does

1. **Load Configuration**
   ```python
   config = load_config(args.config)
   # Apply command-line overrides
   ```

2. **Prepare Data**
   ```python
   dataloaders, label_encoder = create_dataloaders(
       train_path=config["data"]["train_path"],
       val_path=config["data"]["val_path"],
       ...
   )
   ```

3. **Initialize Model**
   ```python
   model = create_model(
       model_name=config["model"]["name"],
       num_labels=label_encoder.num_labels,
       ...
   )
   ```

4. **Train**
   ```python
   trainer = IntentTrainer(model, training_config)
   history = trainer.train(train_loader, val_loader)
   ```

5. **Save Results**
   - Best model: `outputs/models/best_model.pt`
   - Training history: `outputs/logs/training_history.json`
   - Config used: `outputs/logs/config_used.yaml`

### Output Structure

```
outputs/
├── models/
│   ├── best_model.pt          # Best checkpoint by val F1
│   └── final_model.pt         # Last epoch checkpoint
└── logs/
    ├── training_history.json  # Loss/metrics per epoch
    └── config_used.yaml       # Copy of config used
```

### Example Output

```
Loading config from: configs/config.yaml
Device: cuda (NVIDIA GeForce RTX 3090)

Creating dataloaders...
  Train: 2400 samples
  Val: 300 samples

Initializing model: xlm-roberta-base
  Parameters: 278,043,405

Training...
Epoch 1/10: loss=1.234, val_f1=0.652
Epoch 2/10: loss=0.876, val_f1=0.723
Epoch 3/10: loss=0.654, val_f1=0.781 [BEST]
...

Training complete!
Best validation F1: 0.856
Model saved to: outputs/models/best_model.pt
```

---

## run_inference.py

### Purpose
Command-line script for inference and serving the API.

### Usage Modes

#### 1. Single Prediction
```bash
python run_inference.py --text "Check my application status"
```

Output:
```
Intent: check_application_status
Confidence: 0.92
Action: execute
Explanation: High confidence - proceeding automatically
```

#### 2. Interactive Mode
```bash
python run_inference.py --interactive
```

Starts an interactive session:
```
Voice AI Intent Classifier
Type 'quit' to exit

> Check my application status
Intent: check_application_status (0.92) → EXECUTE

> Ndashaka kureba uko application yanjye igeze
Intent: check_application_status (0.85) → EXECUTE

> quit
Goodbye!
```

#### 3. Batch Prediction
```bash
python run_inference.py --input-file texts.txt --output-file predictions.csv
```

Input file (one text per line):
```
Check my application status
What documents do I need?
Book an appointment
```

Output CSV:
```csv
text,intent,confidence,action
"Check my application status",check_application_status,0.92,execute
"What documents do I need?",requirements_information,0.88,execute
"Book an appointment",appointment_booking,0.71,confirm
```

#### 4. API Server
```bash
python run_inference.py --serve --port 8000
```

Starts FastAPI server:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model from: outputs/models/best_model.pt
INFO:     Model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model-path` | Path to model checkpoint | `outputs/models/best_model.pt` |
| `--text` | Single text to predict | - |
| `--interactive` | Start interactive mode | `False` |
| `--input-file` | File with texts (one per line) | - |
| `--output-file` | Output file for predictions | - |
| `--serve` | Start API server | `False` |
| `--host` | API server host | `0.0.0.0` |
| `--port` | API server port | `8000` |
| `--device` | Device to use | Auto-detect |

### What It Does

1. **Load Model**
   ```python
   predictor = IntentPredictor(model_path=args.model_path)
   confidence_handler = ConfidenceHandler()
   ```

2. **Based on mode**:
   
   **Single prediction**:
   ```python
   result = predictor.predict(args.text)
   processed = confidence_handler.process_prediction(result)
   print_result(processed)
   ```
   
   **Interactive**:
   ```python
   while True:
       text = input("> ")
       if text == "quit":
           break
       result = predictor.predict(text)
       print_result(result)
   ```
   
   **Batch**:
   ```python
   texts = open(args.input_file).readlines()
   results = predictor.predict_batch(texts)
   save_csv(results, args.output_file)
   ```
   
   **API server**:
   ```python
   from src.api.app import create_app
   app = create_app(model_path=args.model_path)
   uvicorn.run(app, host=args.host, port=args.port)
   ```

### Testing the API

Once server is running:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Check my application status"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Check status", "Book appointment"]}'
```

Or open http://localhost:8000/docs for Swagger UI.

---

## Common Workflows

### Training → Inference

```bash
# 1. Train model
python run_training.py

# 2. Test single prediction
python run_inference.py --text "Test utterance"

# 3. Start API for production
python run_inference.py --serve
```

### Experiment with Different Configs

```bash
# Create experiment config
cp configs/config.yaml configs/experiment.yaml
# Edit experiment.yaml...

# Train with experiment config
python run_training.py --config configs/experiment.yaml --output-dir outputs/experiment1

# Test the experiment model
python run_inference.py --model-path outputs/experiment1/best_model.pt --interactive
```

### Batch Processing Historical Data

```bash
# Prepare input file with utterances
cat > historical.txt << EOF
Check my application status
What are the requirements
I need help with payment
EOF

# Run batch prediction
python run_inference.py --input-file historical.txt --output-file predictions.csv

# Analyze predictions
python -c "import pandas as pd; print(pd.read_csv('predictions.csv')['action'].value_counts())"
```

---

## Troubleshooting

### "Model not found" Error
```
FileNotFoundError: outputs/models/best_model.pt not found
```
**Solution**: Train the model first with `python run_training.py`

### "CUDA out of memory" Error
```
RuntimeError: CUDA out of memory
```
**Solutions**:
- Reduce batch size: `--batch-size 8`
- Use CPU: `--device cpu`
- Use gradient accumulation in config

### "Module not found" Error
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root directory, or set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_training.py
```
