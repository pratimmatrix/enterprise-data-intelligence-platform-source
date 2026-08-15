"""
Insight Engine
Enterprise Customer Intelligence Platform

Author: Pratim Mistry
Generates operational & behavioral insights for marketing campaigns.
"""

from typing import Dict, Any, List, Optional


class InsightEngine:
    """
    Generates actionable business insights based on customer features 
    and model decision outputs.
    """

    def generate_insights(
        self,
        customer_data: Optional[Dict[str, Any]] = None,
        decision_result: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> List[str]:
        """
        Produce qualitative customer outreach directives.
        Accepts any combination of arguments without crashing.
        """
        # Handle cases where argument is passed as single dictionary or keyword
        data = customer_data or {}
        if not data and decision_result and isinstance(decision_result, dict):
            data = decision_result

        insights = []

        # 1. Contact Duration Insight
        duration = data.get("duration", 0)
        if duration > 300:
            insights.append(
                f"High customer engagement: Call duration was {duration} seconds (above benchmark)."
            )
        elif 0 < duration < 100:
            insights.append(
                "Short engagement window: Consider refining the outreach script for faster value proposition."
            )

        # 2. Financial Standing Insight
        balance = data.get("balance", 0)
        housing = str(data.get("housing", "no")).lower()
        loan = str(data.get("loan", "no")).lower()

        if balance > 3000:
            insights.append(
                f"Strong financial profile: Account balance (${balance:,.2f}) indicates high term-deposit capacity."
            )
        elif balance < 0:
            insights.append(
                "Negative account balance detected: Target with low-risk advisory rather than investment products."
            )

        if housing == "yes" and loan == "yes":
            insights.append(
                "High debt commitment: Customer holds both housing and personal loans."
            )

        # 3. Historical Campaign Affinity
        previous_success = data.get("previous_success", 0)
        poutcome = str(data.get("poutcome", "unknown")).lower()
        if previous_success == 1 or poutcome == "success":
            insights.append(
                "High-affinity prospect: Prior campaign resulted in positive conversion."
            )

        campaign = data.get("campaign", 1)
        if campaign > 3:
            insights.append(
                f"Diminishing returns warning: Customer contacted {campaign} times in the current campaign."
            )

        # 4. Fallback Default
        if not insights:
            insights.append(
                "Standard profile: Align outreach with primary banking advisory guidelines."
            )

        return insights