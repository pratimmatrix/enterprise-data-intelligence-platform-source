"""
Business Rule Engine

Author: Pratim Mistry
Description:
Maps ML conversion probabilities to enterprise decision priorities and automated actions.
"""

from typing import Dict, Any


class BusinessRuleEngine:
    """
    Translates model output into operational business directives.
    """

    def evaluate(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate probability and assign priority + actionable strategy.
        Supports both decimal (0.0 - 1.0) and percentage (0 - 100) probability inputs.
        """
        # Safely extract probability from multiple possible keys
        raw_prob = (
            prediction_result.get("probability")
            if prediction_result.get("probability") is not None
            else prediction_result.get("probability_percent", 0.0)
        )

        try:
            prob = float(raw_prob)
        except (ValueError, TypeError):
            prob = 0.0

        # Auto-normalize percentage scale (0-100) to decimal (0.0-1.0)
        if prob > 1.0:
            prob = prob / 100.0

        # Business priority thresholds
        if prob >= 0.60:
            priority = "HIGH"
            action = "Prioritize customer for immediate direct-channel relationship manager follow-up."
        elif prob >= 0.30:
            priority = "MEDIUM"
            action = "Include customer in nurture email sequence and standard marketing campaign."
        else:
            priority = "LOW"
            action = "Do not prioritize customer for high-cost direct outreach; retain in baseline group."

        return {
            "priority": priority,
            "recommended_action": action
        }