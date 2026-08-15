"""
Enterprise Data Quality Engine

Author: Pratim Mistry

Description:
Analyzes datasets for missing values, placeholder values,
duplicate rows, and basic data-quality issues.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


class DataQualityEngine:
    """
    Analyze the quality of a Pandas DataFrame.
    """

    def __init__(self):
        print("DataQualityEngine initialized.")

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset quality.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.

        Returns
        -------
        dict
            Data quality results.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "missing_values": 0,
                "duplicate_rows": 0,
                "unknown_values": 0,
                "quality_score": 0.0,
            }

        missing_values = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        unknown_values = 0

        target_placeholders = {"unknown", "?", "na", "null", "none", "n/a"}

        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        for column in text_cols:
            clean_series = df[column].astype(str).str.strip().str.lower()
            unknown_values += int(clean_series.isin(target_placeholders).sum())

        total_cells = df.shape[0] * df.shape[1]

        if total_cells == 0:
            quality_score = 0.0
        else:
            issue_count = (
                missing_values
                + (duplicate_rows * df.shape[1])
                + unknown_values
            )
            raw_score = 100.0 * (1.0 - min(1.0, issue_count / total_cells))
            quality_score = max(0.0, round(raw_score, 2))

        return {
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "unknown_values": unknown_values,
            "quality_score": quality_score,
        }

    def display(self, results: Dict[str, Any]):
        """
        Display data quality results.
        """
        print("\n========== DATA QUALITY ==========\n")

        print(
            f"Missing Values     : "
            f"{results.get('missing_values', 0)}"
        )

        print(
            f"Duplicate Rows     : "
            f"{results.get('duplicate_rows', 0)}"
        )

        print(
            f"Unknown Values     : "
            f"{results.get('unknown_values', 0)}"
        )

        print(
            f"Data Quality Score : "
            f"{results.get('quality_score', 0.0)}/100"
        )

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        results = self.analyze(df)
        self.display(results)
        return results

    def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.run(df)