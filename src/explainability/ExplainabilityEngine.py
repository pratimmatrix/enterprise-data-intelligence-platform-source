"""
Explainability Engine

Author: Pratim Mistry
Generates human-readable model explanations and extracts global feature importances.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import joblib


class ExplainabilityEngine:
    """
    Enterprise Model Explainability Orchestrator.
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.model = None

    def _resolve_model_path(self) -> Path:
        candidate_paths = [
            Path.home() / "Documents" / "models" / "random_forest_pipeline.pkl",
            self.project_root / "models" / "random_forest_pipeline.pkl",
            self.project_root / "models" / "champion_model.pkl",
            Path.home() / "Documents" / "models" / "champion_model.pkl",
        ]
        for p in candidate_paths:
            if p.exists():
                return p
        return candidate_paths[0]

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):
        if self.model is None:
            model_path = self._resolve_model_path()
            if not model_path.exists():
                return None

            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                self.model = None

        return self.model

    # ========================================================
    # EXPLAIN PREDICTION
    # ========================================================

    def explain(
        self,
        customer_data: Optional[Dict[str, Any]] = None,
        prediction_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        c_data = customer_data if isinstance(customer_data, dict) else {}
        p_res = prediction_result if isinstance(prediction_result, dict) else {}

        model = self.load_model()

        prediction = p_res.get("prediction", "UNKNOWN")
        
        # Robust probability normalization
        raw_prob = p_res.get("probability_percent", p_res.get("probability", 0.0))
        try:
            probability = float(raw_prob)
            if 0 < probability <= 1.0:
                probability = probability * 100.0
        except (ValueError, TypeError):
            probability = 0.0

        risk = p_res.get("risk_category", "UNKNOWN")

        explanations: List[str] = []

        # ----------------------------------------------------
        # 1. MODEL RESULT
        # ----------------------------------------------------
        explanations.append(
            f"The model predicts {prediction} with a probability of {probability:.2f}%."
        )
        explanations.append(
            f"The resulting model risk category is {risk}."
        )

        # ----------------------------------------------------
        # 2. CUSTOMER ATTRIBUTES & CONTACT HISTORY
        # ----------------------------------------------------
        try:
            previous = int(c_data.get("previous", 0) or 0)
        except (ValueError, TypeError):
            previous = 0

        if previous == 0:
            explanations.append("The customer has no previous campaign contact history.")
        else:
            explanations.append(f"The customer has been contacted {previous} time(s) previously.")

        # ----------------------------------------------------
        # 3. PREVIOUS OUTCOME
        # ----------------------------------------------------
        poutcome = str(c_data.get("poutcome", "unknown")).lower()
        if poutcome == "unknown":
            explanations.append("There is no known outcome from a previous campaign contact.")
        else:
            explanations.append(f"The previous campaign outcome was {poutcome}.")

        # ----------------------------------------------------
        # 4. CAMPAIGN CONTACTS
        # ----------------------------------------------------
        try:
            campaign = int(c_data.get("campaign", 1) or 1)
        except (ValueError, TypeError):
            campaign = 1

        if campaign <= 2:
            explanations.append(
                f"The current campaign contact count is {campaign}, indicating relatively low contact intensity."
            )
        else:
            explanations.append(
                f"The current campaign contact count is {campaign}, indicating higher contact intensity."
            )

        # ----------------------------------------------------
        # 5. CONTACT CHANNEL
        # ----------------------------------------------------
        contact = c_data.get("contact", "unknown")
        explanations.append(f"The customer was contacted through a {contact} communication channel.")

        # ----------------------------------------------------
        # 6. CONTACT DURATION
        # ----------------------------------------------------
        try:
            duration = int(c_data.get("duration", 0) or 0)
        except (ValueError, TypeError):
            duration = 0

        if duration >= 300:
            explanations.append(
                f"The current contact duration is {duration} seconds, indicating relatively strong customer engagement."
            )
        else:
            explanations.append(f"The current contact duration is {duration} seconds.")

        # ----------------------------------------------------
        # 7. AGE & AGE GROUP
        # ----------------------------------------------------
        age = c_data.get("age")
        if age is not None:
            age_group = c_data.get("age_group", "unknown")
            explanations.append(f"The customer is {age} years old and belongs to the {age_group} age group.")

        # ----------------------------------------------------
        # 8. ACCOUNT BALANCE
        # ----------------------------------------------------
        try:
            balance = float(c_data.get("balance", 0) or 0)
            if balance > 0:
                explanations.append(f"The customer has a positive account balance of {balance:,.2f}.")
            elif balance < 0:
                explanations.append(f"The customer has a negative account balance of {balance:,.2f}.")
            else:
                explanations.append("The customer has a zero account balance.")
        except (ValueError, TypeError):
            explanations.append("Account balance information is unavailable.")

        # ====================================================
        # GLOBAL FEATURE IMPORTANCE
        # ====================================================
        top_features: List[Dict[str, Any]] = []

        if model is not None:
            try:
                preprocessor = model.named_steps.get("preprocessor")
                classifier = model.named_steps.get("classifier")

                if preprocessor and classifier and hasattr(classifier, "feature_importances_"):
                    feature_names = preprocessor.get_feature_names_out()
                    importances = classifier.feature_importances_

                    feature_importance = sorted(
                        zip(feature_names, importances),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    top_features = [
                        {
                            "feature": str(name).replace("remainder__", "").replace("cat__", "").replace("num__", ""),
                            "importance": float(value)
                        }
                        for name, value in feature_importance[:10]
                    ]
            except Exception as error:
                print(f"Feature importance unavailable: {error}")

        return {
            "explanations": explanations,
            "top_features": top_features
        }

    # ========================================================
    # ALIAS
    # ========================================================

    def generate_explanation(
        self,
        customer_data: Optional[Dict[str, Any]] = None,
        prediction_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.explain(customer_data, prediction_result)


# ============================================================
# TEST RUNNER
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
    }
    example_prediction = {
        "prediction": "NO",
        "probability_percent": 45.39,
        "risk_category": "MEDIUM"
    }

    result = engine.explain(example_customer, example_prediction)
    print("Explanations:")
    for num, exp in enumerate(result["explanations"], 1):
        print(f"{num}. {exp}")