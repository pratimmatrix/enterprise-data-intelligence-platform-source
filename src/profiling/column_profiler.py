"""
Enterprise Column Profiler

Author: Pratim Mistry

Description:
Performs detailed column-level analysis for
numeric and categorical columns.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


class ColumnProfiler:
    """
    Analyze individual columns in a dataset.
    """

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate detailed information for every column.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.

        Returns
        -------
        dict
            Column-level profiling information.
        """

        if df is None or not isinstance(df, pd.DataFrame):
            return {}

        results = {}

        for column in df.columns:

            series = df[column]

            column_info = {
                "data_type": str(series.dtype),
                "unique_values": int(series.nunique(dropna=True)),
                "missing_values": int(series.isnull().sum()),
            }

            if pd.api.types.is_numeric_dtype(series):

                clean_series = series.dropna()

                if len(clean_series) > 0:
                    min_val = float(clean_series.min())
                    max_val = float(clean_series.max())
                    mean_val = round(float(clean_series.mean()), 2)
                    median_val = float(clean_series.median())
                else:
                    min_val, max_val, mean_val, median_val = None, None, None, None

                column_info.update({
                    "type": "numeric",
                    "minimum": min_val,
                    "maximum": max_val,
                    "mean": mean_val,
                    "median": median_val,
                })

            else:

                value_counts = series.value_counts()

                top_value = (
                    value_counts.index[0]
                    if not value_counts.empty
                    else None
                )

                column_info.update({
                    "type": "categorical",
                    "top_value": str(top_value)
                    if top_value is not None
                    else None,
                    "top_value_count": (
                        int(value_counts.iloc[0])
                        if not value_counts.empty
                        else 0
                    ),
                })

            results[column] = column_info

        return results

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias for profile.
        """
        results = self.profile(df)
        self.display(results)
        return results

    def display(self, results: Dict[str, Any]):
        """
        Display column-level profiling information.
        """

        print("\n========== COLUMN INTELLIGENCE ==========\n")

        for column, info in results.items():

            print(f"Column: {column}")
            print(f"  Type           : {info.get('type')}")
            print(f"  Data Type      : {info.get('data_type')}")
            print(f"  Unique Values  : {info.get('unique_values')}")
            print(f"  Missing Values : {info.get('missing_values')}")

            if info.get("type") == "numeric":

                print(f"  Minimum        : {info.get('minimum')}")
                print(f"  Maximum        : {info.get('maximum')}")
                print(f"  Mean           : {info.get('mean')}")
                print(f"  Median         : {info.get('median')}")

            else:

                print(f"  Top Value      : {info.get('top_value')}")
                print(
                    f"  Top Value Count: "
                    f"{info.get('top_value_count')}"
                )

            print()