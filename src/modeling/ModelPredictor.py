import joblib
from pathlib import Path
import pandas as pd


class ModelPredictor:

    def __init__(self):

        print("ModelPredictor initialized.")

        # ====================================================
        # MODEL LOCATION
        # ====================================================

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

        print()
        print("=" * 70)
        print("                 LOADING TRAINED MODEL")
        print("=" * 70)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model file not found:\n"
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        print(
            f"Model loaded successfully:"
        )

        print(
            self.model_path
        )

        return self.model

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        customer_data
    ):

        # ----------------------------------------------------
        # Load model if necessary
        # ----------------------------------------------------

        if self.model is None:

            self.load_model()

        # ----------------------------------------------------
        # Convert dictionary to DataFrame
        # ----------------------------------------------------

        if isinstance(
            customer_data,
            dict
        ):

            customer_data = pd.DataFrame(
                [customer_data]
            )

        elif isinstance(
            customer_data,
            pd.DataFrame
        ):

            customer_data = customer_data.copy()

        else:

            raise TypeError(
                "Input must be a dictionary "
                "or pandas DataFrame."
            )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = self.model.predict(
            customer_data
        )

        probability = (
            self.model.predict_proba(
                customer_data
            )[:, 1]
        )

        # ----------------------------------------------------
        # Convert prediction
        # ----------------------------------------------------

        prediction_value = int(
            prediction[0]
        )

        probability_value = float(
            probability[0]
        )

        # ----------------------------------------------------
        # Convert 0 / 1 to YES / NO
        # ----------------------------------------------------

        if prediction_value == 1:

            prediction_label = "YES"

        else:

            prediction_label = "NO"

        # ----------------------------------------------------
        # Risk category
        # ----------------------------------------------------

        if probability_value >= 0.70:

            risk_category = "HIGH"

        elif probability_value >= 0.40:

            risk_category = "MEDIUM"

        else:

            risk_category = "LOW"

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = {

            "prediction":
                prediction_label,

            "probability":
                round(
                    probability_value,
                    4
                ),

            "probability_percent":
                round(
                    probability_value * 100,
                    2
                ),

            "risk_category":
                risk_category
        }

        return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    predictor = ModelPredictor()

    predictor.load_model()

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

        # Engineered features
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

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predictor.predict(
        customer
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "                    PREDICTION RESULT"
    )

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

    print()

    print("=" * 70)