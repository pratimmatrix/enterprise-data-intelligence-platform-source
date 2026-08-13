"""
Enterprise Column Profiler

Author: Pratim Mistry

Description:
Performs detailed column-level analysis for
numeric and categorical columns.
"""

import pandas as pd


class ColumnProfiler:
    """
    Analyze individual columns in a dataset.
    """

    def profile(self, df: pd.DataFrame) -> dict:
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

        results = {}

        for column in df.columns:

            series = df[column]

            column_info = {
                "data_type": str(series.dtype),
                "unique_values": int(series.nunique()),
                "missing_values": int(series.isnull().sum()),
            }

            if pd.api.types.is_numeric_dtype(series):

                column_info.update({
                    "type": "numeric",
                    "minimum": float(series.min()),
                    "maximum": float(series.max()),
                    "mean": round(float(series.mean()), 2),
                    "median": float(series.median()),
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

    def display(self, results: dict):
        """
        Display column-level profiling information.
        """

        print("\n========== COLUMN INTELLIGENCE ==========\n")

        for column, info in results.items():

            print(f"Column: {column}")
            print(f"  Type           : {info['type']}")
            print(f"  Data Type      : {info['data_type']}")
            print(f"  Unique Values  : {info['unique_values']}")
            print(f"  Missing Values : {info['missing_values']}")

            if info["type"] == "numeric":

                print(f"  Minimum        : {info['minimum']}")
                print(f"  Maximum        : {info['maximum']}")
                print(f"  Mean           : {info['mean']}")
                print(f"  Median         : {info['median']}")

            else:

                print(f"  Top Value      : {info['top_value']}")
                print(
                    f"  Top Value Count: "
                    f"{info['top_value_count']}"
                )

            print()