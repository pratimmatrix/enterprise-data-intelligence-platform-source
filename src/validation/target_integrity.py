"""
Target Integrity Engine

Author: Pratim Mistry
Description:
Guarantees strict protection of the supervised target variable 'y'.
Validates target existence, data types, binary class structure, 
missing values, and prevents silent target degradation during adaptive retraining.
"""

from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np


class TargetIntegrityError(Exception):
    """Raised when target variable violates integrity constraints."""
    pass


class TargetIntegrityEngine:
    """
    Enforces strict validation rules on the target column 'y'.
    """

    def __init__(self, target_column: str = "y", allowed_classes: Tuple[str, ...] = ("no", "yes")):
        self.target_column = target_column
        self.allowed_classes = set(allowed_classes)

    def validate_target(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute comprehensive validation on target variable.

        Parameters
        ----------
        df : pd.DataFrame
            The uploaded candidate dataset.

        Returns
        -------
        Dict[str, Any]
            Validation report containing status, class counts, and balance ratio.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise TargetIntegrityError("Input must be a valid pandas DataFrame.")

        # 1. Check target existence
        if self.target_column not in df.columns:
            raise TargetIntegrityError(
                f"Target column '{self.target_column}' is missing. "
                f"Supervised retraining cannot proceed without protected target '{self.target_column}'."
            )

        target_series = df[self.target_column]

        # 2. Check for empty series
        if len(target_series) == 0:
            raise TargetIntegrityError("Dataset contains 0 rows. Target evaluation aborted.")

        # 3. Check for missing values in target
        null_count = int(target_series.isnull().sum())
        if null_count > 0:
            raise TargetIntegrityError(
                f"Target column '{self.target_column}' contains {null_count} missing values. "
                "Target column must be 100% complete."
            )

        # 4. Normalize and validate class values (supports numeric 0/1 or string no/yes)
        normalized_series = target_series.astype(str).str.strip().str.lower()
        
        # Handle 0 / 1 representation gracefully
        class_mapping = {"0": "no", "1": "yes", "0.0": "no", "1.0": "yes", "false": "no", "true": "yes"}
        normalized_series = normalized_series.replace(class_mapping)

        unique_classes = set(normalized_series.unique())

        invalid_classes = unique_classes - self.allowed_classes
        if invalid_classes:
            raise TargetIntegrityError(
                f"Target column contains invalid class values: {invalid_classes}. "
                f"Allowed binary classes are: {sorted(list(self.allowed_classes))}."
            )

        if len(unique_classes) < 2:
            raise TargetIntegrityError(
                f"Target column only contains 1 class ({unique_classes}). "
                "Binary classification requires at least 2 distinct classes."
            )

        # 5. Calculate class balance statistics
        counts = normalized_series.value_counts().to_dict()
        total_samples = len(normalized_series)
        positive_count = counts.get("yes", 0)
        positive_ratio = round((positive_count / total_samples) * 100, 2)

        return {
            "status": "VALID",
            "target_column": self.target_column,
            "total_samples": total_samples,
            "class_distribution": counts,
            "positive_class_percentage": positive_ratio,
            "is_imbalanced": positive_ratio < 20.0 or positive_ratio > 80.0
        }

    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Safely separates features (X) and maps binary target (y) to (0, 1).
        """
        self.validate_target(df)

        X = df.drop(columns=[self.target_column]).copy()
        
        normalized_target = (
            df[self.target_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        
        class_mapping = {
            "no": 0, "0": 0, "0.0": 0, "false": 0,
            "yes": 1, "1": 1, "1.0": 1, "true": 1
        }
        
        y = normalized_target.map(class_mapping).astype(np.int32)

        return X, y

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        return self.validate_target(df)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validation alias.
        """
        return self.validate_target(df)