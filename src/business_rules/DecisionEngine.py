from src.modeling.ModelPredictor import ModelPredictor
from src.business_rules.BusinessRuleEngine import BusinessRuleEngine
from src.insights.InsightEngine import InsightEngine
from src.explainability.ExplainabilityEngine import ExplainabilityEngine


class DecisionEngine:

    def __init__(self):

        print("DecisionEngine initialized.")

        # ====================================================
        # ML PREDICTOR
        # ====================================================

        self.predictor = ModelPredictor()

        # ====================================================
        # BUSINESS RULE ENGINE
        # ====================================================

        self.business_rules = BusinessRuleEngine()

        # ====================================================
        # INSIGHT ENGINE
        # ====================================================

        self.insight_engine = InsightEngine()

        # ====================================================
        # EXPLAINABILITY ENGINE
        # ====================================================

        self.explainability_engine = ExplainabilityEngine()

    # ========================================================
    # RUN COMPLETE DECISION PIPELINE
    # ========================================================

    def run(
        self,
        customer_data
    ):

        print()
        print("=" * 70)
        print("                  DECISION ENGINE")
        print("=" * 70)

        # ====================================================
        # STEP 1: ML PREDICTION
        # ====================================================

        print()
        print("Step 1: Generating ML prediction...")

        prediction = self.predictor.predict(
            customer_data
        )

        print(
            f"Prediction : "
            f"{prediction['prediction']}"
        )

        print(
            f"Probability: "
            f"{prediction['probability_percent']}%"
        )

        # ====================================================
        # STEP 2: BUSINESS RULES
        # ====================================================

        print()
        print("Step 2: Applying business rules...")

        business_decision = (
            self.business_rules.evaluate(
                prediction
            )
        )

        print(
            f"Priority: "
            f"{business_decision['priority']}"
        )

        print(
            f"Action: "
            f"{business_decision['recommended_action']}"
        )

        # ====================================================
        # COMBINE ML + BUSINESS DECISION
        # ====================================================

        final_result = {

            "prediction":
                prediction["prediction"],

            "probability":
                prediction["probability"],

            "probability_percent":
                prediction["probability_percent"],

            "risk_category":
                prediction["risk_category"],

            "priority":
                business_decision["priority"],

            "recommended_action":
                business_decision[
                    "recommended_action"
                ]
        }

        # ====================================================
        # STEP 3: BUSINESS INSIGHTS
        # ====================================================

        print()
        print(
            "Step 3: Generating business insights..."
        )

        insight_result = (
            self.insight_engine.generate_insights(
                final_result
            )
        )

        final_result["insights"] = (
            insight_result["insights"]
        )

        # ====================================================
        # STEP 4: MODEL EXPLANATION
        # ====================================================

        print()
        print(
            "Step 4: Generating model explanation..."
        )

        explanation_result = (
            self.explainability_engine
            .generate_explanation(
                customer_data,
                final_result
            )
        )

        final_result["explanations"] = (
            explanation_result[
                "explanations"
            ]
        )

        final_result["top_features"] = (
            explanation_result[
                "top_features"
            ]
        )

        return final_result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Initialize Decision Engine
    # --------------------------------------------------------

    engine = DecisionEngine()

    # --------------------------------------------------------
    # Example customer
    # --------------------------------------------------------

    customer = {

        "age": 35,

        "job": "management",

        "marital": "married",

        "education": "tertiary",

        "default": "no",

        "balance": 1500,

        "housing": "yes",

        "loan": "no",

        "contact": "cellular",

        "day": 15,

        "month": "may",

        "duration": 300,

        "campaign": 2,

        "pdays": -1,

        "previous": 0,

        "poutcome": "unknown",

        "age_group": "31-40",

        "balance_log": 7.313,

        "campaign_log": 1.099,

        "previous_contact": 0,

        "previously_contacted": 0,

        "zero_balance": 0,

        "loan_burden": 0,

        "campaign_intensity": "low",

        "contact_unknown": 0,

        "previous_success": 0
    }

    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================

    result = engine.run(
        customer
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    print()
    print("=" * 70)
    print("                    FINAL DECISION")
    print("=" * 70)

    print()

    print(
        f"Prediction        : "
        f"{result['prediction']}"
    )

    print(
        f"Probability       : "
        f"{result['probability_percent']}%"
    )

    print(
        f"Risk Category     : "
        f"{result['risk_category']}"
    )

    print(
        f"Priority          : "
        f"{result['priority']}"
    )

    print(
        f"Recommended Action: "
        f"{result['recommended_action']}"
    )

    # ========================================================
    # BUSINESS INSIGHTS
    # ========================================================

    print()
    print("Business Insights:")

    for insight in result["insights"]:

        print(
            f"  • {insight}"
        )

    # ========================================================
    # MODEL EXPLANATIONS
    # ========================================================

    print()
    print("Model Explanations:")

    for explanation in result["explanations"]:

        print(
            f"  • {explanation}"
        )

    # ========================================================
    # TOP FEATURES
    # ========================================================

    print()
    print("Top Model Features:")

    for item in result["top_features"]:

        print(
            f"  • "
            f"{item['feature']}: "
            f"{item['importance']:.4f}"
        )

    print()
    print("=" * 70)