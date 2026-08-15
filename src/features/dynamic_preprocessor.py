"""
Dynamic Preprocessor Engine

Author: Pratim Mistry
Description:
Constructs dynamic, leak-free Scikit-Learn preprocessing pipelines.
Adapts to runtime schema variations (added, removed, or modified columns)
without hardcoded feature dependencies.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DynamicPreprocessorBuilder:
    """
    Constructs dynamic Scikit-Learn ColumnTransformers adapted to dataset feature types.
    """

    def __init__(
        self,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        handle_unknown_categories: str = "ignore"
    ):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.handle_unknown_categories = handle_unknown_categories
        self.fitted_preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []

    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Dynamically classify input columns into numerical and categorical types.
        """
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        self.numeric_features = sorted(numeric_cols)
        self.categorical_features = sorted(categorical_cols)

        return self.numeric_features, self.categorical_features

    def build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build an unfitted Scikit-Learn ColumnTransformer pipeline based on runtime columns.
        """
        numeric_cols, categorical_cols = self.identify_feature_types(X)
        transformers = []

        if numeric_cols:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
                    ("scaler", StandardScaler())
                ]
            )
            transformers.append(("numeric_pipeline", numeric_pipeline, numeric_cols))

        if categorical_cols:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown=self.handle_unknown_categories,
                            sparse_output=False
                        )
                    )
                ]
            )
            transformers.append(("categorical_pipeline", categorical_pipeline, categorical_cols))

        self.fitted_preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        return self.fitted_preprocessor

    def fit_transform(self, X: pd.DataFrame) -> Tuple[Any, ColumnTransformer]:
        """
        Build and fit the preprocessor pipeline on training data.
        """
        preprocessor = self.build_pipeline(X)
        transformed_data = preprocessor.fit_transform(X)
        self.fitted_preprocessor = preprocessor
        return transformed_data, preprocessor

    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Return metadata on processed feature distributions.
        """
        return {
            "numeric_count": len(self.numeric_features),
            "categorical_count": len(self.categorical_features),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features
        }