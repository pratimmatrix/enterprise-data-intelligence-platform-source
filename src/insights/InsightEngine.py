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
        customer_data: Dict[str, Any],
        decision_result: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Produce qualitative customer outreach directives.
        Supports both 1-argument and 2-argument calls safely.
        """
        insights = []

        # 1. Contact Duration Insight
        duration = customer_data.get("duration", 0)
        if duration > 300:
            insights.append(
                f"High customer engagement: Call duration was {duration} seconds (above benchmark)."
            )
        elif duration < 100 and duration > 0:
            insights.append(
                "Short engagement window: Consider refining the outreach script for faster value proposition."
            )

        # 2. Financial Standing Insight
        balance = customer_data.get("balance", 0)
        housing = str(customer_data.get("housing", "no")).lower()
        loan = str(customer_data.get("loan", "no")).lower()

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

        # 3. Campaign & Historical Touchpoint Insight
        previous_success = customer_data.get("previous_success", 0)
        poutcome = str(customer_data.get("poutcome", "unknown")).lower()
        if previous_success == 1 or poutcome == "success":
            insights.append(
                "High-affinity prospect: Prior campaign resulted in positive conversion."
            )

        campaign = customer_data.get("campaign", 1)
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