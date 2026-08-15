"""
Business Insight Engine

Author: Pratim Mistry
Description:
Generates natural-language customer profiling and campaign targeting recommendations.
"""

from typing import Dict, Any, List


class InsightEngine:
    """
    Extracts actionable marketing insights from prediction and customer attributes.
    """

    def generate_insights(self, customer_data: Dict[str, Any], decision_result: Dict[str, Any]) -> List[str]:
        insights = []
        pred = decision_result.get("prediction", "NO")
        prob = float(decision_result.get("probability_percent", 0.0))
        priority = decision_result.get("priority", "LOW")

        if pred == "YES":
            insights.append(f"Customer exhibits strong conversion affinity ({prob}% probability).")
        else:
            insights.append(f"Customer exhibits low propensity to subscribe ({prob}% probability).")

        # Contact history heuristics
        poutcome = customer_data.get("poutcome", "unknown")
        if poutcome == "success":
            insights.append("Prior campaign historical outcome was successful — high repeat response likelihood.")
        elif poutcome == "failure":
            insights.append("Previous campaign contact failed — consider tailored incentive packaging.")

        # Balance & engagement
        balance = float(customer_data.get("balance", 0))
        if balance > 5000:
            insights.append(f"High-balance customer (Balance: {balance:,.2f}) — prime candidate for premium term deposit products.")
        elif balance <= 0:
            insights.append("Customer maintains zero or negative account balance — financial risk constraint.")

        duration = float(customer_data.get("duration", 0))
        if duration > 300:
            insights.append(f"High historical call duration ({duration}s) signals high customer interest.")

        return insights