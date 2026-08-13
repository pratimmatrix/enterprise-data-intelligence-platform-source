import numpy as np
import pandas as pd


class FeatureEngineeringEngine:

    def __init__(self):
        print("FeatureEngineeringEngine initialized.")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:

        print("\n========== FEATURE ENGINEERING ==========")

        data = df.copy()

        # --------------------------------------------------
        # 1. AGE GROUP
        # --------------------------------------------------

        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=[
                "young",
                "early_adult",
                "adult",
                "middle_age",
                "senior",
                "elderly"
            ]
        )

        print("Created feature: age_group")

        # --------------------------------------------------
        # 2. LOG TRANSFORMATION OF BALANCE
        # --------------------------------------------------
        # Bank balance is highly right-skewed.
        # log1p reduces the effect of extreme values.

        data["balance_log"] = np.sign(data["balance"]) * np.log1p(
            np.abs(data["balance"])
        )

        print("Created feature: balance_log")

        # --------------------------------------------------
        # 3. CAMPAIGN LOG
        # --------------------------------------------------
        # Campaign is also heavily right-skewed.

        data["campaign_log"] = np.log1p(data["campaign"])

        print("Created feature: campaign_log")

        # --------------------------------------------------
        # 4. PREVIOUS CONTACT FLAG
        # --------------------------------------------------
        # previous > 0 means the customer was contacted before.

        data["previous_contact"] = (
            data["previous"] > 0
        ).astype(int)

        print("Created feature: previous_contact")

        # --------------------------------------------------
        # 5. PREVIOUS CONTACT DAYS FLAG
        # --------------------------------------------------
        # pdays = -1 means the customer was never contacted
        # previously.

        data["previously_contacted"] = (
            data["pdays"] != -1
        ).astype(int)

        print("Created feature: previously_contacted")

        # --------------------------------------------------
        # 6. ZERO BALANCE FLAG
        # --------------------------------------------------

        data["zero_balance"] = (
            data["balance"] == 0
        ).astype(int)

        print("Created feature: zero_balance")

        # --------------------------------------------------
        # 7. LOAN BURDEN
        # --------------------------------------------------
        # Indicates whether the customer has both housing
        # and personal loans.

        data["loan_burden"] = (
            (data["housing"] == "yes") &
            (data["loan"] == "yes")
        ).astype(int)

        print("Created feature: loan_burden")

        # --------------------------------------------------
        # 8. CAMPAIGN INTENSITY
        # --------------------------------------------------
        # More contacts during the current campaign may
        # indicate repeated targeting.

        data["campaign_intensity"] = pd.cut(
            data["campaign"],
            bins=[0, 1, 2, 4, np.inf],
            labels=[
                "low",
                "moderate",
                "high",
                "very_high"
            ]
        )

        print("Created feature: campaign_intensity")

        # --------------------------------------------------
        # 9. CONTACT CHANNEL QUALITY
        # --------------------------------------------------
        # Unknown contact method is explicitly represented.

        data["contact_unknown"] = (
            data["contact"] == "unknown"
        ).astype(int)

        print("Created feature: contact_unknown")

        # --------------------------------------------------
        # 10. PREVIOUS OUTCOME SUCCESS FLAG
        # --------------------------------------------------

        data["previous_success"] = (
            data["poutcome"] == "success"
        ).astype(int)

        print("Created feature: previous_success")

        # --------------------------------------------------
        # FEATURE SUMMARY
        # --------------------------------------------------

        new_features = [
            "age_group",
            "balance_log",
            "campaign_log",
            "previous_contact",
            "previously_contacted",
            "zero_balance",
            "loan_burden",
            "campaign_intensity",
            "contact_unknown",
            "previous_success"
        ]

        print("\n---------- CREATED FEATURES ----------")

        for feature in new_features:
            print(f"• {feature}")

        print(
            f"\nOriginal columns : {df.shape[1]}"
        )

        print(
            f"New columns      : {data.shape[1]}"
        )

        print(
            f"Features created : {len(new_features)}"
        )

        return data