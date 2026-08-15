"""
Production Model Predictor Engine

Author: Pratim Mistry
Description:
Loads persisted champion models from local/cloud relative paths and runs real-time inference.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import joblib
import pandas as pd
import numpy as np


class ModelPredictor:
    """
    Inference engine using relative project paths for cloud and local portability.
    """

    def __init__(self, model_path: Union[str, Path] = "models/artifacts/champion_pipeline.pkl"):

        # Resolve project root dynamically across varying directory depths
        current_path = Path(__file__).resolve()
        root_dir = current_path.parent

        for _ in range(4):
            if (root_dir / "models").exists() or (root_dir / "src").exists():
                break
            root_dir = root_dir.parent

        self.project_root = root_dir

        candidate_path = Path(model_path)
        if candidate_path.is_absolute():
            self.model_path = candidate_path
        else:
            self.model_path = self.project_root / candidate_path

        self.model = None

        print(f"ModelPredictor initialized. Target model path: {self.model_path}")

    # ========================================================
    # LOAD ARTIFACT
    # ========================================================

    def load_model(self):
        """
        Load the champion model artifact into memory with fallbacks.
        """

        if not self.model_path.exists():

            fallbacks = [
                self.project_root / "models" / "champion_pipeline.pkl",
                self.project_root / "models" / "random_forest_pipeline.pkl",
                self.project_root / "models" / "gradient_boosting_pipeline.pkl",
                self.project_root / "models" / "logistic_regression_pipeline.pkl"
            ]

            found_fallback = False
            for fb in fallbacks:
                if fb.exists():
                    self.model_path = fb
                    found_fallback = True
                    break

            if not found_fallback:
                raise FileNotFoundError(
                    f"Champion model artifact not found at:\n{self.model_path}\n"
                    "Please train baseline models first."
                )

        self.model = joblib.load(self.model_path)
        return self.model

    # ========================================================
    # REAL-TIME INFERENCE
    # ========================================================

    def predict(self, customer_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Run inference on single customer dict or DataFrame.
        """

        if self.model is None:
            self.load_model()

        if isinstance(customer_data, dict):
            df_input = pd.DataFrame([customer_data])
        elif isinstance(customer_data, pd.DataFrame):
            df_input = customer_data.copy()
        else:
            raise TypeError("Input data must be a dictionary or pandas DataFrame.")

        # Safely drop target 'y' and leaky 'duration' if present in input
        cols_to_drop = []
        if "y" in df_input.columns:
            cols_to_drop.append("y")
        if "duration" in df_input.columns:
            cols_to_drop.append("duration")

        if cols_to_drop:
            df_input = df_input.drop(columns=cols_to_drop)

        # Predict class and probability
        prediction = self.model.predict(df_input)[0]

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(df_input)[:, 1][0])
        else:
            probability = 1.0 if int(prediction) == 1 else 0.0

        label = "YES" if int(prediction) == 1 else "NO"
        prob_percent = round(probability * 100, 2)

        if probability >= 0.70:
            risk_category = "HIGH"
        elif probability >= 0.40:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"

        return {
            "prediction": label,
            "probability": round(probability, 4),
            "probability_percent": prob_percent,
            "risk_category": risk_category
        }