import logging

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


logger = logging.getLogger(__name__)


class ModelTrainer:

    TARGET_COLUMN = "y"

    def __init__(self):
        print("ModelTrainer initialized.")

        self.model = None
        self.preprocessor = None
        self.feature_names = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # ============================================================
    # PREPARE DATA
    # ============================================================

    def prepare_data(self, df: pd.DataFrame):

        print("\n========== MODEL DATA PREPARATION ==========")

        if df is None:
            raise ValueError("Input dataframe is None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{self.TARGET_COLUMN}' "
                "does not exist."
            )

        print(f"Input shape: {df.shape}")

        X = df.drop(
            columns=[self.TARGET_COLUMN]
        ).copy()

        y = (
            df[self.TARGET_COLUMN]
            .astype(str)
            .str.lower()
            .map({
                "no": 0,
                "yes": 1
            })
        )

        if y.isna().any():
            raise ValueError(
                "Target column contains values "
                "other than 'yes' and 'no'."
            )

        print(
            f"Features: {X.shape[1]}"
        )

        print(
            f"Target distribution:\n"
            f"{df[self.TARGET_COLUMN].value_counts()}"
        )

        return X, y

    # ============================================================
    # BUILD PREPROCESSOR
    # ============================================================

    def build_preprocessor(self, X):

        print("\n========== PREPROCESSING ==========")

        numeric_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns.tolist()

        print(
            f"Numeric features    : "
            f"{len(numeric_features)}"
        )

        print(
            f"Categorical features: "
            f"{len(categorical_features)}"
        )

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
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
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
            ]
        )

        return self.preprocessor

    # ============================================================
    # TRAIN TEST SPLIT
    # ============================================================

    def split_data(
        self,
        X,
        y,
        test_size=0.20,
        random_state=42
    ):

        print("\n========== TRAIN / TEST SPLIT ==========")

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(
            f"Training rows: "
            f"{len(self.X_train)}"
        )

        print(
            f"Testing rows : "
            f"{len(self.X_test)}"
        )

        print(
            f"Training target rate: "
            f"{self.y_train.mean():.2%}"
        )

        print(
            f"Testing target rate : "
            f"{self.y_test.mean():.2%}"
        )

    # ============================================================
    # BUILD MODEL
    # ============================================================

    def build_model(self):

        print("\n========== MODEL BUILDING ==========")

        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor has not been created."
            )

        self.model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self.preprocessor
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
        )

        print(
            "Model: Logistic Regression"
        )

        print(
            "Class weighting: balanced"
        )

        return self.model

    # ============================================================
    # TRAIN
    # ============================================================

    def train(self):

        print("\n========== MODEL TRAINING ==========")

        if self.model is None:
            raise RuntimeError(
                "Model has not been built."
            )

        self.model.fit(
            self.X_train,
            self.y_train
        )

        print(
            "Model training completed."
        )

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate(self):

        print("\n========== MODEL EVALUATION ==========")

        if self.model is None:
            raise RuntimeError(
                "Model has not been trained."
            )

        predictions = self.model.predict(
            self.X_test
        )

        probabilities = self.model.predict_proba(
            self.X_test
        )[:, 1]

        accuracy = accuracy_score(
            self.y_test,
            predictions
        )

        precision = precision_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            self.y_test,
            probabilities
        )

        print()
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

        print("\n---------- CONFUSION MATRIX ----------")

        matrix = confusion_matrix(
            self.y_test,
            predictions
        )

        print(matrix)

        print("\n---------- CLASSIFICATION REPORT ----------")

        print(
            classification_report(
                self.y_test,
                predictions,
                target_names=[
                    "No",
                    "Yes"
                ],
                zero_division=0
            )
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": matrix
        }

    # ============================================================
    # COMPLETE TRAINING PIPELINE
    # ============================================================

    def run(self, df):

        print("\n")
        print("=" * 70)
        print(
            "                    MODEL TRAINING"
        )
        print("=" * 70)

        X, y = self.prepare_data(df)

        self.build_preprocessor(X)

        self.split_data(
            X,
            y
        )

        self.build_model()

        self.train()

        results = self.evaluate()

        print()
        print(
            "MODEL TRAINING COMPLETED"
        )

        return results