import joblib
from pathlib import Path


class ExplainabilityEngine:

    def __init__(self):

        print("ExplainabilityEngine initialized.")

        self.model_path = (
            Path.home()
            / "Documents"
            / "models"
            / "random_forest_pipeline.pkl"
        )

        self.model = None

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        if self.model is None:

            if not self.model_path.exists():

                raise FileNotFoundError(
                    f"Model not found:\n"
                    f"{self.model_path}"
                )

            self.model = joblib.load(
                self.model_path
            )

            print(
                f"Model loaded successfully:\n"
                f"{self.model_path}"
            )

        return self.model

    # ========================================================
    # EXPLAIN PREDICTION
    # ========================================================

    def explain(
        self,
        customer_data,
        prediction_result
    ):

        model = self.load_model()

        prediction = prediction_result.get(
            "prediction",
            "UNKNOWN"
        )

        probability = prediction_result.get(
            "probability_percent",
            0
        )

        risk = prediction_result.get(
            "risk_category",
            "UNKNOWN"
        )

        explanations = []

        # ----------------------------------------------------
        # MODEL RESULT
        # ----------------------------------------------------

        explanations.append(
            f"The model predicts {prediction} "
            f"with a probability of {probability:.2f}%."
        )

        explanations.append(
            f"The resulting model risk category is {risk}."
        )

        # ----------------------------------------------------
        # CUSTOMER ATTRIBUTES
        # ----------------------------------------------------

        if customer_data.get("previous", 0) == 0:

            explanations.append(
                "The customer has no previous campaign "
                "contact history."
            )

        else:

            explanations.append(
                f"The customer has been contacted "
                f"{customer_data.get('previous', 0)} "
                f"time(s) previously."
            )

        # ----------------------------------------------------
        # PREVIOUS OUTCOME
        # ----------------------------------------------------

        poutcome = customer_data.get(
            "poutcome",
            "unknown"
        )

        if poutcome == "unknown":

            explanations.append(
                "There is no known outcome from a "
                "previous campaign contact."
            )

        else:

            explanations.append(
                f"The previous campaign outcome was "
                f"{poutcome}."
            )

        # ----------------------------------------------------
        # CAMPAIGN CONTACTS
        # ----------------------------------------------------

        campaign = customer_data.get(
            "campaign",
            0
        )

        if campaign <= 2:

            explanations.append(
                f"The current campaign contact count "
                f"is {campaign}, indicating relatively "
                f"low contact intensity."
            )

        else:

            explanations.append(
                f"The current campaign contact count "
                f"is {campaign}, indicating higher "
                f"contact intensity."
            )

        # ----------------------------------------------------
        # CONTACT CHANNEL
        # ----------------------------------------------------

        contact = customer_data.get(
            "contact",
            "unknown"
        )

        explanations.append(
            f"The customer was contacted through a "
            f"{contact} communication channel."
        )

        # ----------------------------------------------------
        # CONTACT DURATION
        # ----------------------------------------------------

        duration = customer_data.get(
            "duration",
            0
        )

        if duration >= 300:

            explanations.append(
                f"The current contact duration is "
                f"{duration} seconds, indicating "
                f"relatively strong customer engagement."
            )

        else:

            explanations.append(
                f"The current contact duration is "
                f"{duration} seconds."
            )

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        age = customer_data.get(
            "age",
            None
        )

        if age is not None:

            age_group = customer_data.get(
                "age_group",
                "unknown"
            )

            explanations.append(
                f"The customer is {age} years old "
                f"and belongs to the {age_group} age group."
            )

        # ----------------------------------------------------
        # BALANCE
        # ----------------------------------------------------

        balance = customer_data.get(
            "balance",
            0
        )

        if balance > 0:

            explanations.append(
                f"The customer has a positive "
                f"account balance of {balance}."
            )

        elif balance < 0:

            explanations.append(
                f"The customer has a negative "
                f"account balance of {balance}."
            )

        else:

            explanations.append(
                "The customer has a zero account balance."
            )

        # ====================================================
        # GLOBAL FEATURE IMPORTANCE
        # ====================================================

        top_features = []

        try:

            preprocessor = (
                model.named_steps["preprocessor"]
            )

            classifier = (
                model.named_steps["classifier"]
            )

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            importances = (
                classifier.feature_importances_
            )

            feature_importance = sorted(
                zip(
                    feature_names,
                    importances
                ),
                key=lambda x: x[1],
                reverse=True
            )

            top_features = [
                {
                    "feature": name,
                    "importance": float(value)
                }

                for name, value
                in feature_importance[:10]
            ]

        except Exception as error:

            print(
                f"Feature importance unavailable: "
                f"{error}"
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        print()

        print(
            "=" * 70
        )

        for number, explanation in enumerate(
            explanations,
            start=1
        ):

            print(
                f"{number}. {explanation}"
            )

        if top_features:

            print()

            print(
                "Top Global Model Features:"
            )

            for number, feature in enumerate(
                top_features,
                start=1
            ):

                print(
                    f"{number}. "
                    f"{feature['feature']} "
                    f"(importance: "
                    f"{feature['importance']:.4f})"
                )

        print()

        print(
            "=" * 70
        )

        return {
            "explanations": explanations,
            "top_features": top_features
        }

    # ========================================================
    # ALIAS
    # ========================================================
    #
    # Keeps compatibility if another part of the project
    # calls generate_explanation().
    # ========================================================

    def generate_explanation(
        self,
        customer_data,
        prediction_result
    ):

        return self.explain(
            customer_data,
            prediction_result
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    engine = ExplainabilityEngine()

    example_customer = {

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

    example_prediction = {

        "prediction": "NO",

        "probability": 0.4539,

        "probability_percent": 45.39,

        "risk_category": "MEDIUM"
    }

    result = engine.explain(
        example_customer,
        example_prediction
    )

    print()

    print(
        "Explanation generation completed."
    )