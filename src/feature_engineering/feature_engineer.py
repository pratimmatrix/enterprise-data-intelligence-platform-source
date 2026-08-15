"""
Feature Engineering Engine
Enterprise Customer Intelligence Platform

Author: Pratim Mistry
Description:
Constructs domain-specific engineered features, non-linear transformations,
and behavioral interaction indicators for campaign propensity modeling.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class FeatureEngineeringEngine:
    """
    Orchestrates end-to-end tabular feature transformations.
    Ensures exact feature parity between training pipelines and real-time dashboard inference.
    """

    def __init__(self):
        print("FeatureEngineeringEngine initialized.")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw tabular customer records by computing statistical logs,
        interaction flags, and behavioral buckets.
        """
        print("\n========== FEATURE ENGINEERING ==========")
        data = df.copy()

        # --------------------------------------------------
        # 1. AGE GROUP BUCKETING
        # --------------------------------------------------
        # Segmenting age into consistent enterprise decade cohorts
        if "age" in data.columns:
            data["age_group"] = pd.cut(
                data["age"],
                bins=[-np.inf, 20, 30, 40, 50, 60, np.inf],
                labels=["0-20", "21-30", "31-40", "41-50", "51-60", "61+"]
            ).astype(str)
            print("Created feature: age_group")

        # --------------------------------------------------
        # 2. LOG TRANSFORMATION OF BALANCE
        # --------------------------------------------------
        # Symmetric log-transformation handling negative & extreme balances
        if "balance" in data.columns:
            data["balance_log"] = np.sign(data["balance"]) * np.log1p(np.abs(data["balance"]))
            print("Created feature: balance_log")

        # --------------------------------------------------
        # 3. CAMPAIGN LOG TRANSFORMATION
        # --------------------------------------------------
        # Compresses extreme right-skewed campaign contact frequency
        if "campaign" in data.columns:
            data["campaign_log"] = np.log1p(np.maximum(data["campaign"], 0))
            print("Created feature: campaign_log")

        # --------------------------------------------------
        # 4. PREVIOUS CONTACT OCCURRENCE FLAG
        # --------------------------------------------------
        # Indicates historical touchpoint presence (previous > 0)
        if "previous" in data.columns:
            data["previous_contact"] = (data["previous"] > 0).astype(int)
            print("Created feature: previous_contact")

        # --------------------------------------------------
        # 5. PREVIOUS CONTACT INTERVAL FLAG
        # --------------------------------------------------
        # pdays != -1 indicates that customer was contacted in prior campaigns
        if "pdays" in data.columns:
            data["previously_contacted"] = (data["pdays"] != -1).astype(int)
            print("Created feature: previously_contacted")

        # --------------------------------------------------
        # 6. ZERO BALANCE LIQUIDITY FLAG
        # --------------------------------------------------
        if "balance" in data.columns:
            data["zero_balance"] = (data["balance"] == 0).astype(int)
            print("Created feature: zero_balance")

        # --------------------------------------------------
        # 7. DUAL LOAN BURDEN INDICATOR
        # --------------------------------------------------
        # High financial strain flag when holding both housing & personal loans
        if "housing" in data.columns and "loan" in data.columns:
            data["loan_burden"] = (
                (data["housing"].astype(str).str.lower() == "yes") &
                (data["loan"].astype(str).str.lower() == "yes")
            ).astype(int)
            print("Created feature: loan_burden")

        # --------------------------------------------------
        # 8. CAMPAIGN INTENSITY BUCKETING
        # --------------------------------------------------
        # Categorizes touchpoint frequency to prevent customer fatigue
        if "campaign" in data.columns:
            data["campaign_intensity"] = pd.cut(
                data["campaign"],
                bins=[-np.inf, 2, 5, np.inf],
                labels=["low", "medium", "high"]
            ).astype(str)
            print("Created feature: campaign_intensity")

        # --------------------------------------------------
        # 9. UNKNOWN CONTACT CHANNEL INDICATOR
        # --------------------------------------------------
        if "contact" in data.columns:
            data["contact_unknown"] = (
                data["contact"].astype(str).str.lower() == "unknown"
            ).astype(int)
            print("Created feature: contact_unknown")

        # --------------------------------------------------
        # 10. HISTORICAL CAMPAIGN SUCCESS FLAG
        # --------------------------------------------------
        if "poutcome" in data.columns:
            data["previous_success"] = (
                data["poutcome"].astype(str).str.lower() == "success"
            ).astype(int)
            print("Created feature: previous_success")

        # --------------------------------------------------
        # SUMMARY LOGGING
        # --------------------------------------------------
        new_features: List[str] = [
            "age_group", "balance_log", "campaign_log", "previous_contact",
            "previously_contacted", "zero_balance", "loan_burden",
            "campaign_intensity", "contact_unknown", "previous_success"
        ]

        active_created = [f for f in new_features if f in data.columns]

        print("\n---------- CREATED FEATURES ----------")
        for feature in active_created:
            print(f"• {feature}")

        print(f"\nOriginal columns : {df.shape[1]}")
        print(f"New columns      : {data.shape[1]}")
        print(f"Features created : {len(active_created)}")

        return data