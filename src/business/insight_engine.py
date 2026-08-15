"""
Business Insight Engine

Author: Pratim Mistry
Description:
Generates natural-language customer profiling and campaign targeting recommendations.
"""

from typing import Dict, Any, List, Optional


class InsightEngine:
    """
    Extracts actionable marketing insights from prediction and customer attributes.
    """

    def generate_insights(
        self,
        customer_data: Optional[Dict[str, Any]] = None,
        decision_result: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> List[str]:
        """
        Safely generates business insights supporting both 1-arg and 2-arg calls,
        with full defensive parsing against None or malformed values.
        """
        c_data = customer_data if isinstance(customer_data, dict) else {}
        d_res = decision_result if isinstance(decision_result, dict) else {}

        insights: List[str] = []

        # 1. Prediction & Propensity Heuristics
        pred = str(d_res.get("prediction", "NO")).upper()
        raw_prob = d_res.get("probability_percent", d_res.get("probability", 0.0))
        try:
            prob = float(raw_prob)
            if prob <= 1.0 and prob > 0:
                prob = prob * 100.0
        except (ValueError, TypeError):
            prob = 0.0

        if pred == "YES":
            insights.append(f"Customer exhibits strong conversion affinity ({prob:.2f}% probability).")
        elif d_res:
            insights.append(f"Customer exhibits low propensity to subscribe ({prob:.2f}% probability).")

        # 2. Contact History Heuristics
        poutcome = str(c_data.get("poutcome", "unknown")).lower()
        if poutcome == "success" or c_data.get("previous_success") == 1:
            insights.append("Prior campaign historical outcome was successful — high repeat response likelihood.")
        elif poutcome == "failure":
            insights.append("Previous campaign contact failed — consider tailored incentive packaging.")

        # 3. Balance & Financial Standing
        raw_balance = c_data.get("balance", 0)
        try:
            balance = float(raw_balance)
        except (ValueError, TypeError):
            balance = 0.0

        if balance > 5000:
            insights.append(f"High-balance customer (Balance: ${balance:,.2f}) — prime candidate for premium term deposit products.")
        elif balance <= 0:
            insights.append("Customer maintains zero or negative account balance — financial risk constraint.")

        # 4. Debt Commitments
        housing = str(c_data.get("housing", "no")).lower()
        loan = str(c_data.get("loan", "no")).lower()
        if housing == "yes" and loan == "yes":
            insights.append("High debt commitment: Customer holds both housing and personal loans.")

        # 5. Call Engagement Duration
        raw_duration = c_data.get("duration", 0)
        try:
            duration = float(raw_duration)
        except (ValueError, TypeError):
            duration = 0.0

        if duration > 300:
            insights.append(f"High historical call duration ({int(duration)}s) signals active customer engagement.")
        elif 0 < duration < 100:
            insights.append("Short engagement window: Consider refining the outreach script for faster value proposition.")

        # 6. Fallback Default
        if not insights:
            insights.append("Standard banking customer profile — follow standard outreach advisory guidelines.")

        return insights