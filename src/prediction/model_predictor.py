"""
Production Model Predictor Engine

Author: Pratim Mistry
Description:
Loads persisted champion models from local/cloud relative paths and runs real-time inference.
"""

from pathlib import Path
from typing import Dict, Any, Union
import joblib
import pandas as pd
import numpy as np


class ModelPredictor:
    """
    Inference engine using relative project paths for cloud and local portability.
    """

    def __init__(self, model_path: Union[str, Path] = "models/artifacts/champion_pipeline.pkl"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_path = self.project_root / Path(model_path)
        self.model = None

    def load_model(self):
        """
        Load the champion model artifact into memory.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Champion model artifact not found at:\n{self.model_path}\n"
                "Please train baseline models first."
            )
        self.model = joblib.load(self.model_path)
        return self.model

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

        # Drop target if accidentally included in inference payload
        if "y" in df_input.columns:
            df_input = df_input.drop(columns=["y"])

        # Predict class and probability
        prediction = self.model.predict(df_input)[0]
        probability = float(self.model.predict_proba(df_input)[:, 1][0])

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