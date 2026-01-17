"""
Inference Pipeline for Voice AI Intent Classifier

Production-ready inference with:
- Batch prediction
- Confidence scoring
- Preprocessing integration
- Device management
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from ..data.preprocessor import TextPreprocessor
from ..data.dataset import IntentLabelEncoder
from ..models.intent_classifier import IntentClassifier, ModelConfig


class IntentPredictor:
    """
    Production-ready intent prediction pipeline.
    
    Features:
    - Single and batch prediction
    - Automatic preprocessing
    - Confidence scoring
    - Device management (CPU/GPU)
    - Top-k predictions
    """
    
    def __init__(
        self,
        model: Optional[IntentClassifier] = None,
        model_path: Optional[str] = None,
        tokenizer_name: str = "xlm-roberta-base",
        max_length: int = 128,
        device: Optional[str] = None,
        label_encoder: Optional[IntentLabelEncoder] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model: Trained IntentClassifier (optional if model_path provided)
            model_path: Path to saved model checkpoint
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            device: Device to use ('cuda', 'cpu', or None for auto)
            label_encoder: Label encoder for intent mapping
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = IntentClassifier.from_pretrained(model_path)
        else:
            raise ValueError("Either model or model_path must be provided")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Label encoder
        self.label_encoder = label_encoder or IntentLabelEncoder()
    
    def predict(
        self,
        text: str,
        return_all_scores: bool = False,
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict intent for a single utterance.
        
        Args:
            text: Input utterance text
            return_all_scores: Whether to return scores for all intents
            
        Returns:
            Dictionary with intent, confidence, and optionally all scores
        """
        results = self.predict_batch([text], return_all_scores)
        return results[0]
    
    def predict_batch(
        self,
        texts: List[str],
        return_all_scores: bool = False,
        batch_size: int = 32,
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict intents for a batch of utterances.
        
        Args:
            texts: List of input utterance texts
            return_all_scores: Whether to return scores for all intents
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._predict_batch_internal(batch_texts, return_all_scores)
            all_results.extend(batch_results)
        
        return all_results
    
    def _predict_batch_internal(
        self,
        texts: List[str],
        return_all_scores: bool,
    ) -> List[Dict]:
        """Internal batch prediction."""
        # Preprocess texts
        processed_texts = self.preprocessor.batch_preprocess(texts)
        
        # Tokenize
        encodings = self.tokenizer(
            processed_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = outputs["probabilities"].cpu().numpy()
        
        # Build results
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
    
    def predict_top_k(
        self,
        text: str,
        k: int = 3,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Get top-k intent predictions for an utterance.
        
        Useful for showing alternatives or when top prediction has low confidence.
        
        Args:
            text: Input utterance text
            k: Number of top predictions to return
            
        Returns:
            List of top-k predictions with intents and scores
        """
        result = self.predict(text, return_all_scores=True)
        
        # Sort by score
        sorted_scores = sorted(
            result["all_scores"].items(),
            key=lambda x: -x[1]
        )[:k]
        
        return [
            {"intent": intent, "confidence": score}
            for intent, score in sorted_scores
        ]
    
    def predict_with_fallback(
        self,
        text: str,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.60,
    ) -> Dict[str, Union[str, float, str]]:
        """
        Predict intent with fallback action recommendation.
        
        Args:
            text: Input utterance text
            high_threshold: Threshold for auto-execution
            medium_threshold: Threshold for user confirmation
            
        Returns:
            Prediction with action recommendation
        """
        result = self.predict(text)
        
        confidence = result["confidence"]
        
        if confidence >= high_threshold:
            result["action"] = "execute"
            result["action_reason"] = "High confidence - proceed automatically"
        elif confidence >= medium_threshold:
            result["action"] = "confirm"
            result["action_reason"] = "Medium confidence - confirm with user"
            result["alternatives"] = self.predict_top_k(text, k=3)
        else:
            result["action"] = "escalate"
            result["action_reason"] = "Low confidence - route to human agent"
        
        return result


class BatchPredictor:
    """
    Optimized batch predictor for high-throughput scenarios.
    
    Useful for:
    - Evaluating on test sets
    - Processing log files
    - Bulk classification tasks
    """
    
    def __init__(self, predictor: IntentPredictor):
        """
        Initialize batch predictor.
        
        Args:
            predictor: IntentPredictor instance
        """
        self.predictor = predictor
    
    def predict_dataframe(
        self,
        df,
        text_column: str = "utterance_text",
        batch_size: int = 64,
    ):
        """
        Add predictions to a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            batch_size: Batch size for inference
            
        Returns:
            DataFrame with added prediction columns
        """
        texts = df[text_column].tolist()
        
        results = self.predictor.predict_batch(texts, batch_size=batch_size)
        
        df = df.copy()
        df["predicted_intent"] = [r["intent"] for r in results]
        df["prediction_confidence"] = [r["confidence"] for r in results]
        
        return df


# Quick test
if __name__ == "__main__":
    from ..models.intent_classifier import create_model
    
    # Create a model (not trained, just for testing)
    model = create_model()
    
    predictor = IntentPredictor(model=model)
    
    # Test predictions
    test_texts = [
        "Ndashaka kureba status ya application yanjye.",
        "What are the requirements for passport?",
        "Help me with payment",
    ]
    
    print("Testing IntentPredictor...")
    for text in test_texts:
        result = predictor.predict_with_fallback(text)
        print(f"\nText: {text}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Action: {result['action']}")
