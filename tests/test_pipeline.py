"""
Unit Tests for Voice AI Intent Classification Pipeline

Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreprocessor:
    """Tests for text preprocessing."""
    
    def test_preprocessor_import(self):
        """Test preprocessor can be imported."""
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
    
    def test_preprocess_basic(self):
        """Test basic preprocessing."""
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        text = "Hello world"
        result = preprocessor.preprocess(text)
        assert result == "Hello world"
    
    def test_preprocess_repeated_chars(self):
        """Test repeated character normalization."""
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        text = "Helllllo world"
        result = preprocessor.preprocess(text)
        assert result == "Helo world"  # Reduced to single char
    
    def test_preprocess_asr_correction(self):
        """Test ASR error correction."""
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        text = "Check my aplikasiyo status"
        result = preprocessor.preprocess(text)
        assert "application" in result
    
    def test_preprocess_empty(self):
        """Test empty input handling."""
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess(None) == ""
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        from src.data.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        texts = ["Hello", "World", "Test"]
        results = preprocessor.batch_preprocess(texts)
        
        assert len(results) == 3
        assert results[0] == "Hello"


class TestLabelEncoder:
    """Tests for label encoding."""
    
    def test_encoder_import(self):
        """Test encoder can be imported."""
        from src.data.dataset import IntentLabelEncoder
        encoder = IntentLabelEncoder()
        assert encoder is not None
    
    def test_num_labels(self):
        """Test number of labels."""
        from src.data.dataset import IntentLabelEncoder
        encoder = IntentLabelEncoder()
        assert encoder.num_labels == 13
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        from src.data.dataset import IntentLabelEncoder
        encoder = IntentLabelEncoder()
        
        intent = "check_application_status"
        idx = encoder.encode(intent)
        decoded = encoder.decode(idx)
        
        assert decoded == intent
    
    def test_encode_batch(self):
        """Test batch encoding."""
        from src.data.dataset import IntentLabelEncoder
        encoder = IntentLabelEncoder()
        
        intents = ["check_application_status", "speak_to_agent"]
        indices = encoder.encode_batch(intents)
        
        assert len(indices) == 2
        assert all(isinstance(i, int) for i in indices)


class TestModel:
    """Tests for model architecture."""
    
    def test_model_import(self):
        """Test model can be imported."""
        from src.models.intent_classifier import IntentClassifier, create_model
        model = create_model()
        assert model is not None
    
    def test_model_forward(self):
        """Test model forward pass."""
        from src.models.intent_classifier import create_model
        model = create_model()
        
        batch_size = 2
        seq_length = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        outputs = model(input_ids, attention_mask)
        
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (batch_size, 13)
    
    def test_model_with_labels(self):
        """Test model forward pass with labels."""
        from src.models.intent_classifier import create_model
        model = create_model()
        
        batch_size = 2
        seq_length = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.tensor([0, 5])
        
        outputs = model(input_ids, attention_mask, labels)
        
        assert "loss" in outputs
        assert outputs["loss"].item() > 0
    
    def test_model_predict(self):
        """Test model prediction."""
        from src.models.intent_classifier import create_model
        model = create_model()
        
        batch_size = 2
        seq_length = 32
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        preds, confs = model.predict(input_ids, attention_mask)
        
        assert preds.shape == (batch_size,)
        assert confs.shape == (batch_size,)
        assert all(0 <= c <= 1 for c in confs)


class TestFocalLoss:
    """Tests for Focal Loss."""
    
    def test_focal_loss_import(self):
        """Test focal loss can be imported."""
        from src.models.intent_classifier import FocalLoss
        loss_fn = FocalLoss()
        assert loss_fn is not None
    
    def test_focal_loss_forward(self):
        """Test focal loss computation."""
        from src.models.intent_classifier import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 13)
        targets = torch.tensor([0, 1, 2, 3])
        
        loss = loss_fn(logits, targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestConfidenceHandler:
    """Tests for confidence handling."""
    
    def test_confidence_handler_import(self):
        """Test confidence handler can be imported."""
        from src.inference.confidence import ConfidenceHandler
        handler = ConfidenceHandler()
        assert handler is not None
    
    def test_high_confidence_action(self):
        """Test high confidence returns execute action."""
        from src.inference.confidence import ConfidenceHandler, ConfidenceAction
        handler = ConfidenceHandler()
        
        action, _ = handler.get_action(0.90)
        assert action == ConfidenceAction.EXECUTE
    
    def test_medium_confidence_action(self):
        """Test medium confidence returns confirm action."""
        from src.inference.confidence import ConfidenceHandler, ConfidenceAction
        handler = ConfidenceHandler()
        
        action, _ = handler.get_action(0.70)
        assert action == ConfidenceAction.CONFIRM
    
    def test_low_confidence_action(self):
        """Test low confidence returns escalate action."""
        from src.inference.confidence import ConfidenceHandler, ConfidenceAction
        handler = ConfidenceHandler()
        
        action, _ = handler.get_action(0.40)
        assert action == ConfidenceAction.ESCALATE
    
    def test_process_prediction(self):
        """Test prediction processing."""
        from src.inference.confidence import ConfidenceHandler
        handler = ConfidenceHandler()
        
        pred = {"intent": "speak_to_agent", "confidence": 0.75}
        result = handler.process_prediction(pred)
        
        assert "action" in result
        assert "explanation" in result
        assert result["action"] == "confirm"


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_compute_metrics_import(self):
        """Test metrics can be imported."""
        from src.evaluation.metrics import compute_metrics
        assert compute_metrics is not None
    
    def test_compute_metrics_basic(self):
        """Test basic metric computation."""
        from src.evaluation.metrics import compute_metrics
        
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 2, 2, 2]
        intent_names = ["a", "b", "c"]
        
        report = compute_metrics(y_true, y_pred, intent_names)
        
        assert 0 <= report.accuracy <= 1
        assert 0 <= report.macro_f1 <= 1
    
    def test_ece_computation(self):
        """Test Expected Calibration Error computation."""
        from src.evaluation.metrics import compute_ece
        import numpy as np
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])  # Perfect predictions
        confidences = np.array([0.9, 0.8, 0.85, 0.95])
        
        ece = compute_ece(y_true, y_pred, confidences)
        
        assert 0 <= ece <= 1


class TestPredictor:
    """Tests for inference predictor."""
    
    def test_predictor_import(self):
        """Test predictor can be imported."""
        from src.inference.predictor import IntentPredictor
        from src.models.intent_classifier import create_model
        
        model = create_model()
        predictor = IntentPredictor(model=model)
        
        assert predictor is not None
    
    def test_predict_single(self):
        """Test single prediction."""
        from src.inference.predictor import IntentPredictor
        from src.models.intent_classifier import create_model
        
        model = create_model()
        predictor = IntentPredictor(model=model)
        
        result = predictor.predict("Check my application status")
        
        assert "intent" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_predict_batch(self):
        """Test batch prediction."""
        from src.inference.predictor import IntentPredictor
        from src.models.intent_classifier import create_model
        
        model = create_model()
        predictor = IntentPredictor(model=model)
        
        texts = ["Hello", "Check status", "Payment help"]
        results = predictor.predict_batch(texts)
        
        assert len(results) == 3
        assert all("intent" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
