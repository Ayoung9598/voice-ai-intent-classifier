"""
Evaluation Metrics for Voice AI Intent Classifier

Comprehensive metrics suite including:
- Standard classification metrics (F1, accuracy, precision, recall)
- Per-language performance breakdown
- Confidence calibration metrics
- Confusion analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report for the intent classifier.
    
    Contains overall metrics, per-class breakdown, and stratified results.
    """
    # Overall metrics
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    
    # Per-class metrics
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    
    # Stratified metrics
    per_language_accuracy: Dict[str, float] = field(default_factory=dict)
    per_language_f1: Dict[str, float] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    # Confidence metrics
    mean_confidence: float = 0.0
    ece: float = 0.0  # Expected Calibration Error
    
    # Metadata
    total_samples: int = 0
    intent_names: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "overall": {
                "accuracy": self.accuracy,
                "macro_f1": self.macro_f1,
                "weighted_f1": self.weighted_f1,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall,
            },
            "per_class": {
                "f1": self.per_class_f1,
                "precision": self.per_class_precision,
                "recall": self.per_class_recall,
                "support": self.per_class_support,
            },
            "per_language": {
                "accuracy": self.per_language_accuracy,
                "f1": self.per_language_f1,
            },
            "confidence": {
                "mean": self.mean_confidence,
                "ece": self.ece,
            },
            "total_samples": self.total_samples,
        }
    
    def to_json(self, path: str):
        """Save report to JSON file."""
        report_dict = self.to_dict()
        # Convert numpy arrays to lists for JSON serialization
        if self.confusion_matrix is not None:
            report_dict["confusion_matrix"] = self.confusion_matrix.tolist()
        
        with open(path, "w") as f:
            json.dump(report_dict, f, indent=2)
    
    def print_summary(self):
        """Print a formatted summary of the evaluation."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\nOverall Metrics (n={self.total_samples})")
        print("-" * 40)
        print(f"  Accuracy:         {self.accuracy:.4f}")
        print(f"  Macro F1:         {self.macro_f1:.4f}")
        print(f"  Weighted F1:      {self.weighted_f1:.4f}")
        print(f"  Macro Precision:  {self.macro_precision:.4f}")
        print(f"  Macro Recall:     {self.macro_recall:.4f}")
        
        if self.per_language_accuracy:
            print(f"\nPer-Language Accuracy")
            print("-" * 40)
            for lang, acc in sorted(self.per_language_accuracy.items()):
                f1 = self.per_language_f1.get(lang, 0.0)
                print(f"  {lang:8s}: Acc={acc:.4f}, F1={f1:.4f}")
        
        if self.per_class_f1:
            print(f"\nPer-Intent F1 Scores")
            print("-" * 40)
            for intent, f1 in sorted(self.per_class_f1.items(), key=lambda x: -x[1]):
                support = self.per_class_support.get(intent, 0)
                print(f"  {intent:35s}: {f1:.4f} (n={support})")
        
        print("\n" + "=" * 60)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    intent_names: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    languages: Optional[List[str]] = None,
) -> EvaluationReport:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels (indices)
        y_pred: Predicted labels (indices)
        intent_names: List of intent names for labeling
        confidences: Prediction confidence scores
        languages: Language labels for stratified evaluation
        
    Returns:
        EvaluationReport with all computed metrics
    """
    report = EvaluationReport()
    report.total_samples = len(y_true)
    report.intent_names = intent_names or []
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    report.accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    report.macro_precision = float(np.mean(precision))
    report.macro_recall = float(np.mean(recall))
    report.macro_f1 = float(np.mean(f1))
    
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report.weighted_f1 = float(weighted_f1)
    
    # Per-class metrics
    if intent_names:
        for i, intent in enumerate(intent_names):
            if i < len(f1):
                report.per_class_f1[intent] = float(f1[i])
                report.per_class_precision[intent] = float(precision[i])
                report.per_class_recall[intent] = float(recall[i])
                report.per_class_support[intent] = int(support[i])
    
    # Confusion matrix
    report.confusion_matrix = confusion_matrix(y_true, y_pred)
    
    # Confidence metrics
    if confidences is not None:
        confidences = np.array(confidences)
        report.mean_confidence = float(np.mean(confidences))
        report.ece = compute_ece(y_true, y_pred, confidences)
    
    # Per-language metrics
    if languages is not None:
        languages = np.array(languages)
        unique_langs = np.unique(languages)
        
        for lang in unique_langs:
            mask = languages == lang
            if mask.sum() > 0:
                lang_acc = accuracy_score(y_true[mask], y_pred[mask])
                report.per_language_accuracy[lang] = float(lang_acc)
                
                _, _, lang_f1, _ = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average="macro", zero_division=0
                )
                report.per_language_f1[lang] = float(lang_f1)
    
    return report


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well the model's confidence aligns with its accuracy.
    A well-calibrated model should have ECE close to 0.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidences: Prediction confidence scores
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and average confidence in bin
            accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add weighted absolute difference
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return float(ece)


def get_confused_pairs(
    confusion_mat: np.ndarray,
    intent_names: List[str],
    top_k: int = 5,
) -> List[Tuple[str, str, int]]:
    """
    Find the most commonly confused intent pairs.
    
    Args:
        confusion_mat: Confusion matrix
        intent_names: List of intent names
        top_k: Number of top confused pairs to return
        
    Returns:
        List of (true_intent, pred_intent, count) tuples
    """
    confused_pairs = []
    
    n = len(intent_names)
    for i in range(n):
        for j in range(n):
            if i != j and confusion_mat[i, j] > 0:
                confused_pairs.append((
                    intent_names[i],  # True label
                    intent_names[j],  # Predicted label
                    int(confusion_mat[i, j]),  # Count
                ))
    
    # Sort by count descending
    confused_pairs.sort(key=lambda x: -x[2])
    
    return confused_pairs[:top_k]


def analyze_low_confidence_predictions(
    y_true: List[int],
    y_pred: List[int],
    confidences: List[float],
    intent_names: List[str],
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Analyze predictions with low confidence scores.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        confidences: Confidence scores
        intent_names: Intent names
        threshold: Confidence threshold for "low"
        
    Returns:
        Analysis dictionary
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)
    
    low_conf_mask = confidences < threshold
    
    analysis = {
        "low_confidence_count": int(low_conf_mask.sum()),
        "low_confidence_ratio": float(low_conf_mask.mean()),
        "low_confidence_accuracy": 0.0,
        "high_confidence_accuracy": 0.0,
        "intent_distribution_low_conf": {},
    }
    
    if low_conf_mask.sum() > 0:
        analysis["low_confidence_accuracy"] = float(
            accuracy_score(y_true[low_conf_mask], y_pred[low_conf_mask])
        )
        
        # Count intents in low confidence predictions
        for idx in y_true[low_conf_mask]:
            intent = intent_names[idx]
            analysis["intent_distribution_low_conf"][intent] = \
                analysis["intent_distribution_low_conf"].get(intent, 0) + 1
    
    high_conf_mask = ~low_conf_mask
    if high_conf_mask.sum() > 0:
        analysis["high_confidence_accuracy"] = float(
            accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
        )
    
    return analysis


def generate_classification_report(
    y_true: List[int],
    y_pred: List[int],
    intent_names: List[str],
) -> str:
    """
    Generate a formatted classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        intent_names: Intent names
        
    Returns:
        Formatted classification report string
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=intent_names,
        zero_division=0,
    )


# Quick test
if __name__ == "__main__":
    # Test with sample data
    intent_names = [
        "check_application_status",
        "start_new_application",
        "requirements_information",
    ]
    
    y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
    y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1, 0]
    confidences = [0.9, 0.8, 0.95, 0.4, 0.85, 0.7, 0.99, 0.6, 0.5, 0.88]
    languages = ["en", "rw", "en", "mixed", "rw", "en", "en", "rw", "mixed", "en"]
    
    report = compute_metrics(
        y_true, y_pred, intent_names, confidences, languages
    )
    
    report.print_summary()
    
    # Test confused pairs
    confused = get_confused_pairs(report.confusion_matrix, intent_names)
    print("\nMost confused pairs:")
    for true_int, pred_int, count in confused:
        print(f"  {true_int} -> {pred_int}: {count}")
