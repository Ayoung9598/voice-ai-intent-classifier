"""
Error Analysis for Voice AI Intent Classifier

Tools for understanding model failures and identifying improvement opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ErrorCase:
    """A single misclassification case for analysis."""
    text: str
    true_intent: str
    predicted_intent: str
    confidence: float
    language: str
    asr_confidence: Optional[float] = None
    channel: Optional[str] = None
    region: Optional[str] = None


class ErrorAnalyzer:
    """
    Analyze classification errors to identify patterns and improvement areas.
    
    Features:
    - Group errors by intent pair
    - Analyze errors by language, channel, region
    - Identify systematic failure patterns
    - Generate actionable insights
    """
    
    def __init__(
        self,
        texts: List[str],
        y_true: List[int],
        y_pred: List[int],
        intent_names: List[str],
        confidences: Optional[List[float]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize error analyzer.
        
        Args:
            texts: Original utterance texts
            y_true: Ground truth labels
            y_pred: Predicted labels
            intent_names: Intent name mapping
            confidences: Prediction confidence scores
            metadata: Additional metadata (language, channel, etc.)
        """
        self.texts = texts
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.intent_names = intent_names
        self.confidences = np.array(confidences) if confidences else np.ones(len(y_true))
        self.metadata = metadata
        
        # Find errors
        self.error_mask = self.y_true != self.y_pred
        self.error_indices = np.where(self.error_mask)[0]
        
        # Build error cases
        self.error_cases = self._build_error_cases()
    
    def _build_error_cases(self) -> List[ErrorCase]:
        """Build list of ErrorCase objects for all misclassifications."""
        cases = []
        
        for idx in self.error_indices:
            case = ErrorCase(
                text=self.texts[idx],
                true_intent=self.intent_names[self.y_true[idx]],
                predicted_intent=self.intent_names[self.y_pred[idx]],
                confidence=float(self.confidences[idx]),
                language=self._get_metadata(idx, "language", "unknown"),
                asr_confidence=self._get_metadata(idx, "asr_confidence"),
                channel=self._get_metadata(idx, "channel"),
                region=self._get_metadata(idx, "region"),
            )
            cases.append(case)
        
        return cases
    
    def _get_metadata(self, idx: int, column: str, default: Any = None) -> Any:
        """Get metadata value for an index."""
        if self.metadata is not None and column in self.metadata.columns:
            return self.metadata.iloc[idx][column]
        return default
    
    @property
    def error_rate(self) -> float:
        """Overall error rate."""
        return float(self.error_mask.mean())
    
    @property
    def num_errors(self) -> int:
        """Total number of errors."""
        return len(self.error_indices)
    
    def get_errors_by_intent_pair(self) -> Dict[Tuple[str, str], List[ErrorCase]]:
        """
        Group errors by (true_intent, predicted_intent) pairs.
        
        Returns:
            Dictionary mapping intent pairs to error cases
        """
        grouped = defaultdict(list)
        
        for case in self.error_cases:
            key = (case.true_intent, case.predicted_intent)
            grouped[key].append(case)
        
        return dict(grouped)
    
    def get_errors_by_language(self) -> Dict[str, List[ErrorCase]]:
        """Group errors by language."""
        grouped = defaultdict(list)
        
        for case in self.error_cases:
            grouped[case.language].append(case)
        
        return dict(grouped)
    
    def get_top_confused_intents(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most commonly confused intent pairs.
        
        Returns:
            List of dictionaries with confusion details
        """
        pairs = self.get_errors_by_intent_pair()
        
        sorted_pairs = sorted(
            pairs.items(),
            key=lambda x: -len(x[1])
        )[:top_k]
        
        results = []
        for (true_int, pred_int), cases in sorted_pairs:
            results.append({
                "true_intent": true_int,
                "predicted_intent": pred_int,
                "count": len(cases),
                "avg_confidence": np.mean([c.confidence for c in cases]),
                "example_texts": [c.text for c in cases[:3]],
            })
        
        return results
    
    def get_language_error_rates(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate error rates per language.
        
        Returns:
            Dictionary with error stats per language
        """
        if self.metadata is None or "language" not in self.metadata.columns:
            return {}
        
        languages = self.metadata["language"].values
        unique_langs = np.unique(languages)
        
        stats = {}
        for lang in unique_langs:
            lang_mask = languages == lang
            total = lang_mask.sum()
            errors = (self.error_mask & lang_mask).sum()
            
            stats[lang] = {
                "total": int(total),
                "errors": int(errors),
                "error_rate": float(errors / total) if total > 0 else 0.0,
            }
        
        return stats
    
    def get_low_asr_confidence_analysis(
        self,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze errors for low ASR confidence utterances.
        
        Args:
            threshold: ASR confidence threshold for "low"
            
        Returns:
            Analysis dictionary
        """
        if self.metadata is None or "asr_confidence" not in self.metadata.columns:
            return {"available": False}
        
        asr_conf = self.metadata["asr_confidence"].values
        low_asr_mask = asr_conf < threshold
        
        return {
            "available": True,
            "threshold": threshold,
            "low_asr_total": int(low_asr_mask.sum()),
            "low_asr_errors": int((self.error_mask & low_asr_mask).sum()),
            "low_asr_error_rate": float(
                (self.error_mask & low_asr_mask).sum() / low_asr_mask.sum()
            ) if low_asr_mask.sum() > 0 else 0.0,
            "high_asr_error_rate": float(
                (self.error_mask & ~low_asr_mask).sum() / (~low_asr_mask).sum()
            ) if (~low_asr_mask).sum() > 0 else 0.0,
        }
    
    def get_high_confidence_errors(
        self,
        confidence_threshold: float = 0.8
    ) -> List[ErrorCase]:
        """
        Find errors where model was highly confident but wrong.
        
        These are particularly problematic as the model is overconfident.
        """
        return [
            case for case in self.error_cases
            if case.confidence >= confidence_threshold
        ]
    
    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive error analysis summary.
        
        Returns:
            Summary dictionary with all analysis results
        """
        return {
            "overall": {
                "total_samples": len(self.y_true),
                "total_errors": self.num_errors,
                "error_rate": self.error_rate,
            },
            "top_confused_pairs": self.get_top_confused_intents(10),
            "by_language": self.get_language_error_rates(),
            "low_asr_analysis": self.get_low_asr_confidence_analysis(),
            "high_confidence_errors": len(self.get_high_confidence_errors()),
        }
    
    def print_report(self):
        """Print a formatted error analysis report."""
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\nOverall Statistics")
        print("-" * 40)
        print(f"  Total samples: {len(self.y_true)}")
        print(f"  Total errors: {self.num_errors}")
        print(f"  Error rate: {self.error_rate:.2%}")
        
        # Language breakdown
        lang_stats = self.get_language_error_rates()
        if lang_stats:
            print(f"\nError Rate by Language")
            print("-" * 40)
            for lang, stats in sorted(lang_stats.items()):
                print(f"  {lang:8s}: {stats['error_rate']:.2%} "
                      f"({stats['errors']}/{stats['total']})")
        
        # Top confused pairs
        print(f"\nMost Confused Intent Pairs")
        print("-" * 40)
        for pair in self.get_top_confused_intents(5):
            print(f"  {pair['true_intent']}")
            print(f"    -> {pair['predicted_intent']}")
            print(f"      Count: {pair['count']}, "
                  f"Avg confidence: {pair['avg_confidence']:.2f}")
            print(f"      Example: \"{pair['example_texts'][0][:50]}...\"")
        
        # High confidence errors
        hc_errors = self.get_high_confidence_errors()
        if hc_errors:
            print(f"\nHigh Confidence Errors (conf >= 0.8)")
            print("-" * 40)
            print(f"  Count: {len(hc_errors)} errors")
            if len(hc_errors) > 0:
                for case in hc_errors[:3]:
                    print(f"  * \"{case.text[:40]}...\"")
                    print(f"    True: {case.true_intent}, "
                          f"Pred: {case.predicted_intent} (conf: {case.confidence:.2f})")
        
        print("\n" + "=" * 60)


def compare_model_errors(
    analyzer1: ErrorAnalyzer,
    analyzer2: ErrorAnalyzer,
    model1_name: str = "Model A",
    model2_name: str = "Model B",
) -> Dict[str, Any]:
    """
    Compare errors between two models.
    
    Useful for A/B testing or comparing different model versions.
    """
    errors1 = set(analyzer1.error_indices)
    errors2 = set(analyzer2.error_indices)
    
    return {
        f"{model1_name}_only_errors": len(errors1 - errors2),
        f"{model2_name}_only_errors": len(errors2 - errors1),
        "shared_errors": len(errors1 & errors2),
        f"{model1_name}_error_rate": analyzer1.error_rate,
        f"{model2_name}_error_rate": analyzer2.error_rate,
    }


# Quick test
if __name__ == "__main__":
    # Test with sample data
    texts = [
        "Check my application status",
        "Ndashaka kureba status",
        "What are the fees?",
        "Help me with payment",
        "I need requirements info",
    ]
    
    intent_names = ["check_status", "fees", "payment", "requirements"]
    y_true = [0, 0, 1, 2, 3]
    y_pred = [0, 3, 1, 1, 3]  # Two errors
    confidences = [0.9, 0.6, 0.8, 0.85, 0.7]
    
    metadata = pd.DataFrame({
        "language": ["en", "rw", "en", "en", "mixed"],
        "asr_confidence": [0.9, 0.65, 0.85, 0.7, 0.8],
    })
    
    analyzer = ErrorAnalyzer(
        texts, y_true, y_pred, intent_names, confidences, metadata
    )
    
    analyzer.print_report()
