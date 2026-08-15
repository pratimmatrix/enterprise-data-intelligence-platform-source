"""
Adaptive Model Trainer Engine

Author: Pratim Mistry
Description:
Trains candidate machine learning models dynamically using the runtime
dynamic preprocessor. Handles class imbalance and computes complete evaluation metrics.
"""

from typing import Dict, Any, Tuple, Union, Optional
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.features.dynamic_preprocessor import DynamicPreprocessorBuilder
from src.validation.target_integrity import TargetIntegrityEngine


class AdaptiveModelTrainer:
    """
    Trains and evaluates multiple model architectures adapted to runtime feature schemas.
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path] = "models/artifacts",
        test_size: float = 0.20,
        random_state: int = 42
    ):
        # Resolve project root dynamically across varying directory depths
        current_path = Path(__file__).resolve()
        root_dir = current_path.parent
        for _ in range(4):
            if (root_dir / "src").exists() or (root_dir / "models").exists():
                break
            root_dir = root_dir.parent

        candidate_dir = Path(artifacts_dir)
        if candidate_dir.is_absolute():
            self.artifacts_dir = candidate_dir
        else:
            self.artifacts_dir = root_dir / candidate_dir

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state
        self.target_engine = TargetIntegrityEngine(target_column="y")

    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Define candidate model algorithms with balanced class weights.
        """
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                random_state=self.random_state
            )
        }

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute full adaptive training across all candidate algorithms.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input dataset must be a valid non-empty pandas DataFrame.")

        # Prepare features and target
        encoded_data = self.target_engine.encode_target(df)
        if isinstance(encoded_data, tuple):
            X, y = encoded_data
        else:
            X = df.drop(columns=["y"]) if "y" in df.columns else df
            y = encoded_data

        # Train / Test Stratified Split
        stratify_target = y if len(pd.Series(y).unique()) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_target
        )

        builder = DynamicPreprocessorBuilder()
        preprocessor = builder.build_pipeline(X_train)

        models = self.get_candidate_models()
        model_results = {}
        best_model_name = None
        best_roc_auc = -1.0
        best_pipeline = None

        for name, classifier in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ])

            # Fit complete pipeline
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)

            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                roc_auc_val = round(float(roc_auc_score(y_test, y_proba)), 4)
            else:
                y_proba = None
                roc_auc_val = 0.0

            metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "roc_auc": roc_auc_val,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }

            # Save individual candidate artifact
            artifact_path = self.artifacts_dir / f"{name}_pipeline.pkl"
            joblib.dump(pipeline, artifact_path)
            metrics["artifact_path"] = str(artifact_path)

            model_results[name] = metrics

            # Track champion candidate by ROC-AUC
            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = name
                best_pipeline = pipeline

        # Save the champion model
        champion_path = self.artifacts_dir / "champion_pipeline.pkl"
        if best_pipeline is not None:
            joblib.dump(best_pipeline, champion_path)

        feature_meta = (
            builder.get_feature_metadata()
            if hasattr(builder, "get_feature_metadata")
            else {}
        )

        return {
            "best_candidate_name": best_model_name,
            "best_candidate_metrics": model_results.get(best_model_name, {}),
            "champion_path": str(champion_path),
            "all_candidates": model_results,
            "feature_metadata": feature_meta
        }

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Alias for pipeline runner.
        """
        return self.train_and_evaluate(df)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        return self.train_and_evaluate(df)