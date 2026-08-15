"""
Enterprise Data Profiler

Author: Pratim Mistry

Description:
Analyzes a Pandas DataFrame and generates
a structured overview of the dataset.
"""

from typing import Dict, Any
import pandas as pd


class DataProfiler:
    """
    Analyze and profile datasets.
    """

    def __init__(self):
        print("DataProfiler initialized.")

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate dataset profiling information.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.

        Returns
        -------
        dict
            Dataset profiling information.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "rows": 0,
                "columns": 0,
                "duplicate_rows": 0,
                "missing_values": 0,
                "numeric_columns": 0,
                "categorical_columns": 0,
                "memory_usage_kb": 0.0,
            }

        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns

        profile_data = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isnull().sum().sum()),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "memory_usage_kb": round(
                float(df.memory_usage(deep=True).sum() / 1024),
                2
            ),
        }

        return profile_data

    def display(self, profile: Dict[str, Any]):
        """
        Display profiling information.
        """
        print("\n========== DATA PROFILE ==========\n")

        print(f"Rows               : {profile.get('rows', 0)}")
        print(f"Columns            : {profile.get('columns', 0)}")
        print(f"Duplicate Rows     : {profile.get('duplicate_rows', 0)}")
        print(f"Missing Values     : {profile.get('missing_values', 0)}")
        print(f"Numeric Columns    : {profile.get('numeric_columns', 0)}")
        print(f"Categorical Columns: {profile.get('categorical_columns', 0)}")
        print(f"Memory Usage       : {profile.get('memory_usage_kb', 0.0)} KB")

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Direct pipeline runner that profiles and displays summary.
        """
        profile_data = self.profile(df)
        self.display(profile_data)
        return profile_data

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Alias for pipeline execution.
        """
        return self.run(df)