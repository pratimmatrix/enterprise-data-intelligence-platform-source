class BusinessRuleEngine:

    def __init__(self):

        print("BusinessRuleEngine initialized.")

    # ========================================================
    # CUSTOMER PRIORITY
    # ========================================================

    def determine_priority(
        self,
        probability
    ):

        if probability >= 0.70:

            return "HIGH"

        elif probability >= 0.40:

            return "MEDIUM"

        else:

            return "LOW"

    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    def determine_action(
        self,
        probability
    ):

        if probability >= 0.70:

            return (
                "Prioritize customer for "
                "immediate follow-up."
            )

        elif probability >= 0.40:

            return (
                "Include customer in "
                "standard marketing follow-up."
            )

        else:

            return (
                "Do not prioritize customer "
                "for immediate campaign outreach."
            )

    # ========================================================
    # BUSINESS DECISION
    # ========================================================

    def evaluate(
        self,
        prediction_result
    ):

        if not isinstance(
            prediction_result,
            dict
        ):

            raise TypeError(
                "Prediction result must be a dictionary."
            )

        if "probability" not in prediction_result:

            raise ValueError(
                "Prediction result must contain "
                "'probability'."
            )

        probability = float(
            prediction_result["probability"]
        )

        # ----------------------------------------------------
        # Determine priority
        # ----------------------------------------------------

        priority = self.determine_priority(
            probability
        )

        # ----------------------------------------------------
        # Determine action
        # ----------------------------------------------------

        action = self.determine_action(
            probability
        )

        # ----------------------------------------------------
        # Final business result
        # ----------------------------------------------------

        result = {

            "priority":
                priority,

            "recommended_action":
                action
        }

        return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = BusinessRuleEngine()

    # --------------------------------------------------------
    # Example prediction from ModelPredictor
    # --------------------------------------------------------

    prediction = {

        "prediction": "NO",

        "probability": 0.3764,

        "probability_percent": 37.64,

        "risk_category": "LOW"
    }

    # --------------------------------------------------------
    # Evaluate business decision
    # --------------------------------------------------------

    result = engine.evaluate(
        prediction
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "                 BUSINESS DECISION"
    )

    print("=" * 70)

    print()

    print(
        f"Priority           : "
        f"{result['priority']}"
    )

    print(
        f"Recommended Action : "
        f"{result['recommended_action']}"
    )

    print()

    print("=" * 70)