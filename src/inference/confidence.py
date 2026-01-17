"""
Confidence Handling for Voice AI Intent Classifier

Implements confidence-based fallback logic for production deployment.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ConfidenceAction(Enum):
    """Actions based on confidence level."""
    EXECUTE = "execute"       # High confidence - proceed with intent
    CONFIRM = "confirm"       # Medium confidence - ask user to confirm
    ESCALATE = "escalate"     # Low confidence - route to human agent


@dataclass
class ConfidenceThresholds:
    """Configurable confidence thresholds."""
    high: float = 0.85
    medium: float = 0.60
    
    def validate(self):
        """Validate threshold values."""
        if not (0 < self.medium < self.high < 1):
            raise ValueError(
                f"Thresholds must satisfy: 0 < medium ({self.medium}) "
                f"< high ({self.high}) < 1"
            )


class ConfidenceHandler:
    """
    Handle confidence-based decision making.
    
    For government services, it's critical to:
    1. Avoid incorrect actions on low confidence
    2. Provide good UX for medium confidence (confirm)
    3. Efficiently route to agents when needed
    """
    
    def __init__(
        self,
        thresholds: Optional[ConfidenceThresholds] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize confidence handler.
        
        Args:
            thresholds: Confidence thresholds configuration
            enable_logging: Whether to log decisions for monitoring
        """
        self.thresholds = thresholds or ConfidenceThresholds()
        self.thresholds.validate()
        self.enable_logging = enable_logging
        
        # Statistics tracking
        self.stats = {
            "execute": 0,
            "confirm": 0,
            "escalate": 0,
        }
    
    def get_action(
        self,
        confidence: float,
        intent: Optional[str] = None,
    ) -> Tuple[ConfidenceAction, str]:
        """
        Determine action based on confidence score.
        
        Args:
            confidence: Prediction confidence (0-1)
            intent: Predicted intent (for logging)
            
        Returns:
            Tuple of (action, explanation)
        """
        if confidence >= self.thresholds.high:
            action = ConfidenceAction.EXECUTE
            explanation = (
                f"High confidence ({confidence:.2%}) - "
                "Proceeding with intent automatically"
            )
        elif confidence >= self.thresholds.medium:
            action = ConfidenceAction.CONFIRM
            explanation = (
                f"Medium confidence ({confidence:.2%}) - "
                "Please confirm: Did you mean to {intent}?"
            )
        else:
            action = ConfidenceAction.ESCALATE
            explanation = (
                f"Low confidence ({confidence:.2%}) - "
                "Connecting you with a human agent for assistance"
            )
        
        # Update stats
        self.stats[action.value] += 1
        
        return action, explanation
    
    def process_prediction(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a prediction result and add fallback information.
        
        Args:
            prediction: Prediction dictionary with 'intent' and 'confidence'
            
        Returns:
            Enhanced prediction with action and explanation
        """
        intent = prediction.get("intent", "unknown")
        confidence = prediction.get("confidence", 0.0)
        
        action, explanation = self.get_action(confidence, intent)
        
        return {
            **prediction,
            "action": action.value,
            "action_enum": action,
            "explanation": explanation,
            "should_execute": action == ConfidenceAction.EXECUTE,
            "needs_confirmation": action == ConfidenceAction.CONFIRM,
            "needs_escalation": action == ConfidenceAction.ESCALATE,
        }
    
    def generate_user_prompt(
        self,
        prediction: Dict[str, Any],
        top_alternatives: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a user-facing prompt based on prediction.
        
        Args:
            prediction: Processed prediction with action
            top_alternatives: Alternative intent predictions
            
        Returns:
            User-facing prompt string
        """
        action = prediction.get("action", "escalate")
        intent = prediction.get("intent", "unknown")
        
        # Intent to human-readable mapping
        intent_descriptions = {
            "check_application_status": "check your application status",
            "start_new_application": "start a new application",
            "requirements_information": "get information about requirements",
            "fees_information": "get information about fees",
            "appointment_booking": "book an appointment",
            "payment_help": "get help with payment",
            "reset_password_login_help": "reset your password or get login help",
            "speak_to_agent": "speak to a customer service agent",
            "cancel_or_reschedule_appointment": "cancel or reschedule your appointment",
            "update_application_details": "update your application details",
            "document_upload_help": "get help uploading documents",
            "service_eligibility": "check service eligibility",
            "complaint_or_support_ticket": "file a complaint or support ticket",
        }
        
        intent_desc = intent_descriptions.get(intent, intent.replace("_", " "))
        
        if action == "execute":
            return f"I'll help you {intent_desc}."
        
        elif action == "confirm":
            prompt = f"I think you want to {intent_desc}. Is that correct?"
            
            if top_alternatives:
                alt_descs = [
                    intent_descriptions.get(a["intent"], a["intent"].replace("_", " "))
                    for a in top_alternatives[1:3]  # Skip first (already shown)
                ]
                if alt_descs:
                    prompt += f"\n\nOr did you mean: {', '.join(alt_descs)}?"
            
            return prompt
        
        else:  # escalate
            return (
                "I'm not quite sure what you need. "
                "Let me connect you with a customer service representative "
                "who can better assist you."
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get confidence handler statistics."""
        total = sum(self.stats.values())
        
        return {
            "total_predictions": total,
            "action_counts": self.stats.copy(),
            "action_percentages": {
                k: (v / total * 100) if total > 0 else 0
                for k, v in self.stats.items()
            },
            "thresholds": {
                "high": self.thresholds.high,
                "medium": self.thresholds.medium,
            },
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "execute": 0,
            "confirm": 0,
            "escalate": 0,
        }


def analyze_confidence_distribution(
    confidences: List[float],
    thresholds: Optional[ConfidenceThresholds] = None,
) -> Dict[str, Any]:
    """
    Analyze the distribution of confidence scores.
    
    Useful for:
    - Tuning thresholds
    - Monitoring model calibration
    - Identifying issues
    
    Args:
        confidences: List of confidence scores
        thresholds: Thresholds to analyze against
        
    Returns:
        Analysis dictionary
    """
    import numpy as np
    
    thresholds = thresholds or ConfidenceThresholds()
    confs = np.array(confidences)
    
    return {
        "count": len(confs),
        "mean": float(np.mean(confs)),
        "std": float(np.std(confs)),
        "min": float(np.min(confs)),
        "max": float(np.max(confs)),
        "median": float(np.median(confs)),
        "percentiles": {
            "25th": float(np.percentile(confs, 25)),
            "50th": float(np.percentile(confs, 50)),
            "75th": float(np.percentile(confs, 75)),
            "90th": float(np.percentile(confs, 90)),
            "95th": float(np.percentile(confs, 95)),
        },
        "action_distribution": {
            "execute_pct": float((confs >= thresholds.high).mean() * 100),
            "confirm_pct": float(
                ((confs >= thresholds.medium) & (confs < thresholds.high)).mean() * 100
            ),
            "escalate_pct": float((confs < thresholds.medium).mean() * 100),
        },
    }


# Quick test
if __name__ == "__main__":
    handler = ConfidenceHandler()
    
    # Test predictions
    test_cases = [
        {"intent": "check_application_status", "confidence": 0.92},
        {"intent": "start_new_application", "confidence": 0.72},
        {"intent": "speak_to_agent", "confidence": 0.45},
    ]
    
    print("Testing ConfidenceHandler...")
    print("-" * 60)
    
    for pred in test_cases:
        result = handler.process_prediction(pred)
        prompt = handler.generate_user_prompt(result)
        
        print(f"\nIntent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Action: {result['action']}")
        print(f"Prompt: {prompt}")
    
    print("\n" + "-" * 60)
    print("Stats:", handler.get_stats())
