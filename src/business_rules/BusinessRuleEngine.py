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
        """
        prob = float(prediction_result.get("probability", 0.0))

        if prob >= 0.70:
            priority = "HIGH"
            action = "Prioritize customer for immediate direct-channel relationship manager follow-up."
        elif prob >= 0.40:
            priority = "MEDIUM"
            action = "Include customer in nurture email sequence and standard marketing campaign."
        else:
            priority = "LOW"
            action = "Do not prioritize customer for high-cost direct outreach; retain in baseline group."

        return {
            "priority": priority,
            "recommended_action": action
        }