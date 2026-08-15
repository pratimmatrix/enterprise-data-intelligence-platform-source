"""
Decision Engine
Enterprise Customer Intelligence Platform

Author: Pratim Mistry
Orchestrates ML inference, rule-based heuristics, and business actions.
"""

from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import joblib

try:
    from src.business_rules.BusinessRuleEngine import BusinessRuleEngine
except ImportError:
    from src.business.business_rules import BusinessRuleEngine

try:
    from src.insights.InsightEngine import InsightEngine
except ImportError:
    from src.business.insight_engine import InsightEngine


class DecisionEngine:
    """
    Enterprise Decision Orchestrator.
    Handles data ingestion, feature transformation, inference execution,
    and business directive synthesis.
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.business_rule_engine = BusinessRuleEngine()
        self.insight_engine = InsightEngine()
        self._load_pipeline()

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

    def _load_pipeline(self):
        model_path = self._resolve_model_path()
        if model_path.exists():
            try:
                self.pipeline = joblib.load(model_path)
            except Exception:
                self.pipeline = None
        else:
            self.pipeline = None

    def run(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute end-to-end inference and decision rules.
        """
        if self.pipeline is None:
            self._load_pipeline()

        # Convert dictionary to DataFrame for scikit-learn pipeline
        df_input = pd.DataFrame([customer_data])

        # Default fallback values
        prediction_label = "NO"
        prob_percent = 15.0
        risk_category = "MEDIUM"

        if self.pipeline is not None:
            try:
                # 1. Predict Proba
                if hasattr(self.pipeline, "predict_proba"):
                    probs = self.pipeline.predict_proba(df_input)
                    if hasattr(probs, "shape") and len(probs.shape) == 2 and probs.shape[1] > 1:
                        prob_percent = float(probs[0, 1]) * 100.0
                    elif len(probs) > 0 and isinstance(probs[0], (list, tuple)):
                        prob_percent = float(probs[0][1]) * 100.0
                    else:
                        prob_percent = float(probs[0]) * 100.0

                # 2. Predict Class
                pred_raw = self.pipeline.predict(df_input)
                raw_val = pred_raw[0] if hasattr(pred_raw, "__len__") else pred_raw
                if str(raw_val).lower() in ["1", "yes", "true"]:
                    prediction_label = "YES"
                else:
                    prediction_label = "NO"

            except Exception:
                pass

        # Adjust risk category by calculated probability
        if prob_percent >= 60.0:
            risk_category = "LOW"
            prediction_label = "YES"
        elif prob_percent >= 30.0:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"

        # 3. Evaluate Business Rules
        rule_input = {
            "prediction": prediction_label,
            "probability_percent": round(prob_percent, 2),
            "risk_category": risk_category
        }

        try:
            business_output = self.business_rule_engine.evaluate(rule_input)
        except Exception:
            business_output = {
                "priority": "HIGH" if prob_percent >= 50 else "STANDARD",
                "recommended_action": "Target with direct phone campaign for high propensity conversion."
            }

        # 4. Generate Behavioral Insights
        try:
            insights = self.insight_engine.generate_insights(customer_data, business_output)
        except Exception:
            insights = ["Standard banking customer advisory outreach recommended."]

        # 5. Extract Feature Importance for Explanations
        top_features = self._extract_top_features()

        return {
            "prediction": prediction_label,
            "probability_percent": round(prob_percent, 2),
            "risk_category": risk_category,
            "priority": business_output.get("priority", "MEDIUM"),
            "recommended_action": business_output.get("recommended_action", "Proceed with standard campaign outreach."),
            "insights": insights,
            "top_features": top_features,
            "explanations": [
                f"Customer conversion propensity scored at {prob_percent:.2f}%.",
                f"Historical touchpoint factor: {customer_data.get('previous', 0)} prior contacts recorded.",
                f"Outreach timing duration: {customer_data.get('duration', 0)}s active engagement window."
            ]
        }

    def _extract_top_features(self) -> List[Dict[str, Any]]:
        """
        Extract model feature importance safely.
        """
        if self.pipeline is None:
            return []

        try:
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            classifier = self.pipeline.named_steps.get("classifier")

            if preprocessor and classifier and hasattr(classifier, "feature_importances_"):
                feature_names = preprocessor.get_feature_names_out()
                importances = classifier.feature_importances_

                sorted_features = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                return [
                    {
                        "feature": str(f[0]).replace("remainder__", "").replace("cat__", "").replace("num__", ""),
                        "importance": float(f[1])
                    }
                    for f in sorted_features
                ]
        except Exception:
            pass

        return []