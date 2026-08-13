class InsightEngine:

    def __init__(self):
        print("InsightEngine initialized.")

    # ========================================================
    # GENERATE BUSINESS INSIGHTS
    # ========================================================

    def generate_insights(self, decision_result):

        if decision_result is None:
            raise ValueError(
                "Decision result cannot be None."
            )

        print()
        print("=" * 70)
        print("                    BUSINESS INSIGHTS")
        print("=" * 70)

        prediction = decision_result["prediction"]
        probability = decision_result["probability_percent"]
        risk = decision_result["risk_category"]
        priority = decision_result["priority"]

        insights = []

        # ----------------------------------------------------
        # INSIGHT 1: CUSTOMER RESPONSE
        # ----------------------------------------------------

        if prediction == "YES":
            insights.append(
                "Customer shows a positive likelihood of responding "
                "to the campaign."
            )
        else:
            insights.append(
                "Customer shows a lower likelihood of responding "
                "to the campaign."
            )

        # ----------------------------------------------------
        # INSIGHT 2: PROBABILITY
        # ----------------------------------------------------

        if probability >= 70:
            insights.append(
                "Prediction probability is high, indicating a "
                "strong model signal."
            )

        elif probability >= 40:
            insights.append(
                "Prediction probability is moderate, indicating "
                "some potential for customer response."
            )

        else:
            insights.append(
                "Prediction probability is low, indicating a weak "
                "likelihood of customer response."
            )

        # ----------------------------------------------------
        # INSIGHT 3: RISK
        # ----------------------------------------------------

        insights.append(
            f"Risk category is classified as {risk}."
        )

        # ----------------------------------------------------
        # INSIGHT 4: PRIORITY
        # ----------------------------------------------------

        if priority == "HIGH":
            insights.append(
                "Customer should receive high-priority campaign "
                "attention."
            )

        elif priority == "MEDIUM":
            insights.append(
                "Customer may receive campaign attention after "
                "higher-priority customers."
            )

        else:
            insights.append(
                "Customer should not be prioritized for immediate "
                "campaign outreach."
            )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print()

        for number, insight in enumerate(
            insights,
            start=1
        ):
            print(
                f"{number}. {insight}"
            )

        print()
        print("=" * 70)

        return {
            "insights": insights
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = InsightEngine()

    example_decision = {

        "prediction": "NO",

        "probability": 0.3764,

        "probability_percent": 37.64,

        "risk_category": "LOW",

        "priority": "LOW",

        "recommended_action":
            "Do not prioritize customer for immediate campaign outreach."
    }

    result = engine.generate_insights(
        example_decision
    )

    print()
    print("Insight generation completed.")