"""
Model Comparator Engine
Enterprise Customer Intelligence Platform

Author: Pratim Mistry
Description:
Trains, benchmarks, and evaluates competitive machine learning architectures
(Logistic Regression, Random Forest, Gradient Boosting) using cross-validated metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

logger = logging.getLogger(__name__)


class ModelComparator:

    TARGET_COLUMN = "y"

    def __init__(self):

        print("ModelComparator initialized.")

        self.results = []

        self.best_model = None

        self.best_model_name = None

        self.trained_models = {}

        # ====================================================
        # PROJECT MODEL DIRECTORY
        # ====================================================

        # Dynamically locate project root models directory
        current_path = Path(__file__).resolve()
        
        # Check parent depths to find project root directory
        root_dir = current_path.parent
        for _ in range(4):
            if (root_dir / "models").exists() or (root_dir / "src").exists():
                break
            root_dir = root_dir.parent

        self.model_directory = root_dir / "models"

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Model directory: "
            f"{self.model_directory}"
        )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:

        print(
            "\n========== MODEL COMPARISON DATA PREPARATION =========="
        )

        if df is None:

            raise ValueError(
                "Input dataframe is None."
            )

        if not isinstance(
            df,
            pd.DataFrame
        ):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if self.TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"Target column "
                f"'{self.TARGET_COLUMN}' "
                f"does not exist."
            )

        # ----------------------------------------------------
        # Separate features and target
        # ----------------------------------------------------

        # Safely drop target and leaky duration feature if present
        cols_to_drop = [self.TARGET_COLUMN]
        if "duration" in df.columns:
            cols_to_drop.append("duration")

        X = df.drop(columns=cols_to_drop).copy()

        # Map binary target values safely
        target_series = (
            df[self.TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        y = target_series.map(
            {
                "no": 0,
                "0": 0,
                "false": 0,
                "yes": 1,
                "1": 1,
                "true": 1
            }
        )

        if y.isna().any():

            raise ValueError(
                "Target column contains "
                "values other than "
                "'yes'/'no' or 1/0."
            )

        print(
            f"Input shape : {df.shape}"
        )

        print(
            f"Feature count: {X.shape[1]}"
        )

        print(
            "Target distribution:\n"
            f"{df[self.TARGET_COLUMN].value_counts()}"
        )

        return X, y

    # ========================================================
    # BUILD PREPROCESSOR
    # ========================================================

    def build_preprocessor(
        self,
        X: pd.DataFrame
    ) -> ColumnTransformer:

        print(
            "\n========== BUILDING PREPROCESSOR =========="
        )

        numeric_features = (
            X.select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        categorical_features = (
            X.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        print(
            f"Numeric features    : "
            f"{len(numeric_features)}"
        )

        print(
            f"Categorical features: "
            f"{len(categorical_features)}"
        )

        # ----------------------------------------------------
        # Numeric pipeline
        # ----------------------------------------------------

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # ----------------------------------------------------
        # Categorical pipeline
        # ----------------------------------------------------

        try:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            ohe = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    ohe
                )
            ]
        )

        # ----------------------------------------------------
        # Combined preprocessor
        # ----------------------------------------------------

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ],
            remainder="drop"
        )

        return preprocessor

    # ========================================================
    # BUILD MODELS
    # ========================================================

    def build_models(
        self,
        preprocessor: ColumnTransformer
    ) -> Dict[str, Pipeline]:

        print(
            "\n========== BUILDING MODELS =========="
        )

        models = {

            "Logistic Regression":
                Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor
                        ),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=42
                            )
                        )
                    ]
                ),

            "Random Forest":
                Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor
                        ),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=200,
                                max_depth=12,
                                min_samples_split=5,
                                class_weight="balanced",
                                random_state=42,
                                n_jobs=-1
                            )
                        )
                    ]
                ),

            "Gradient Boosting":
                Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor
                        ),
                        (
                            "classifier",
                            GradientBoostingClassifier(
                                n_estimators=150,
                                learning_rate=0.05,
                                max_depth=3,
                                random_state=42
                            )
                        )
                    ]
                )
        }

        for name in models:

            print(
                f"Model created: {name}"
            )

        return models

    # ========================================================
    # EVALUATE MODEL
    # ========================================================

    def evaluate_model(
        self,
        name: str,
        model: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[Pipeline, Dict[str, Any]]:

        print()

        print(
            f"========== {name.upper()} =========="
        )

        print(
            "Training model..."
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {

            "model": name,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc
        }

        self.results.append(
            result
        )

        return model, result

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save_model(
        self,
        model_name: str,
        model: Any
    ) -> Path:

        filename = (
            model_name
            .lower()
            .replace(" ", "_")
            + "_pipeline.pkl"
        )

        model_path = (
            self.model_directory
            / filename
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Saved: {model_path}"
        )

        return model_path

    # ========================================================
    # SAVE ALL TRAINED MODELS
    # ========================================================

    def save_all_models(self) -> Dict[str, Path]:

        print()

        print(
            "========== SAVING TRAINED MODELS =========="
        )

        saved_paths = {}

        for (
            name,
            model
        ) in self.trained_models.items():

            path = self.save_model(
                name,
                model
            )

            saved_paths[name] = path

        print()

        print(
            "Model files saved successfully."
        )

        print(
            f"Model directory: "
            f"{self.model_directory}"
        )

        return saved_paths

    # ========================================================
    # RUN MODEL COMPARISON
    # ========================================================

    def run(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:

        print()

        print("=" * 70)

        print(
            "                    MODEL COMPARISON"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        X, y = self.prepare_data(
            df
        )

        # ----------------------------------------------------
        # Build preprocessor
        # ----------------------------------------------------

        preprocessor = (
            self.build_preprocessor(
                X
            )
        )

        # ----------------------------------------------------
        # Build models
        # ----------------------------------------------------

        models = self.build_models(
            preprocessor
        )

        # ----------------------------------------------------
        # Train / test split
        # ----------------------------------------------------

        print(
            "\n========== TRAIN / TEST SPLIT =========="
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(
            f"Training rows: "
            f"{len(X_train)}"
        )

        print(
            f"Testing rows : "
            f"{len(X_test)}"
        )

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        self.results = []

        self.trained_models = {}

        # ----------------------------------------------------
        # Train every model
        # ----------------------------------------------------

        for (
            name,
            model
        ) in models.items():

            (
                trained_model,
                result
            ) = self.evaluate_model(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )

            self.trained_models[
                name
            ] = trained_model

        # ====================================================
        # RESULTS TABLE
        # ====================================================

        results_df = pd.DataFrame(
            self.results
        )

        results_df = (
            results_df
            .sort_values(
                by="roc_auc",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        print()

        print("=" * 70)

        print(
            "                    MODEL COMPARISON RESULTS"
        )

        print("=" * 70)

        print()

        print(
            results_df.to_string(
                index=False,
                float_format=lambda value:
                f"{value:.4f}"
            )
        )

        # ====================================================
        # BEST MODEL BY ROC-AUC
        # ====================================================

        best_row = (
            results_df.iloc[0]
        )

        self.best_model_name = (
            best_row["model"]
        )

        self.best_model = (
            self.trained_models[
                self.best_model_name
            ]
        )

        print()

        print(
            "---------- BEST MODEL BY ROC-AUC ----------"
        )

        print(
            f"Model    : "
            f"{self.best_model_name}"
        )

        print(
            f"ROC-AUC  : "
            f"{best_row['roc_auc']:.4f}"
        )

        print(
            f"F1 Score : "
            f"{best_row['f1']:.4f}"
        )

        print(
            f"Recall   : "
            f"{best_row['recall']:.4f}"
        )

        # ====================================================
        # SAVE ALL MODELS
        # ====================================================

        saved_paths = (
            self.save_all_models()
        )

        print()

        print(
            "MODEL COMPARISON COMPLETED"
        )

        # ====================================================
        # RETURN RESULTS
        # ====================================================

        return {

            "results":
                results_df,

            "best_model_name":
                self.best_model_name,

            "best_model":
                self.best_model,

            "trained_models":
                self.trained_models,

            "saved_model_paths":
                saved_paths
        }