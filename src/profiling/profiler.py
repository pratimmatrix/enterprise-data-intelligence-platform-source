"""
Enterprise Data Profiler

Author: Pratim Mistry

Description:
Analyzes a Pandas DataFrame and generates
a structured overview of the dataset.
"""

import pandas as pd


class DataProfiler:
    """
    Analyze and profile datasets.
    """

    def profile(self, df: pd.DataFrame) -> dict:
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

        profile = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isnull().sum().sum()),
            "numeric_columns": len(
                df.select_dtypes(include="number").columns
            ),
            "categorical_columns": len(
                df.select_dtypes(include="object").columns
            ),
            "memory_usage_kb": round(
                df.memory_usage(deep=True).sum() / 1024,
                2
            ),
        }

        return profile

    def display(self, profile: dict):
        """
        Display profiling information.
        """

        print("\n========== DATA PROFILE ==========\n")

        print(f"Rows              : {profile['rows']}")
        print(f"Columns           : {profile['columns']}")
        print(f"Duplicate Rows    : {profile['duplicate_rows']}")
        print(f"Missing Values    : {profile['missing_values']}")
        print(f"Numeric Columns   : {profile['numeric_columns']}")
        print(f"Categorical Columns: {profile['categorical_columns']}")
        print(f"Memory Usage      : {profile['memory_usage_kb']} KB")
        