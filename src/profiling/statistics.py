"""
Enterprise Statistical Intelligence Engine

Author: Pratim Mistry

Description:
Analyzes numerical columns to identify statistical
characteristics, skewness, and potential outliers.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


class StatisticalAnalyzer:
    """
    Perform statistical analysis on numerical columns.
    """

    def __init__(self):
        print("StatisticalAnalyzer initialized.")

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.

        Returns
        -------
        dict
            Statistical analysis results.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {}

        results = {}

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        for column in numeric_columns:

            series = df[column].dropna()

            if series.empty:
                continue

            mean = float(series.mean())
            median = float(series.median())
            std = float(series.std()) if len(series) > 1 else 0.0

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))

            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outlier_count = int(
                ((series < lower_bound) |
                 (series > upper_bound)).sum()
            )

            skewness = float(series.skew()) if len(series) > 2 else 0.0

            if np.isnan(skewness):
                distribution = "Undefined / Constant"
            elif skewness > 1:
                distribution = "Highly Right-Skewed"
            elif skewness > 0.5:
                distribution = "Right-Skewed"
            elif skewness < -1:
                distribution = "Highly Left-Skewed"
            elif skewness < -0.5:
                distribution = "Left-Skewed"
            else:
                distribution = "Approximately Symmetric"

            results[column] = {
                "mean": round(mean, 2),
                "median": round(median, 2),
                "standard_deviation": round(std, 2) if not np.isnan(std) else 0.0,
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": outlier_count,
                "skewness": round(skewness, 3) if not np.isnan(skewness) else 0.0,
                "distribution": distribution,
            }

        return results

    def display(self, results: Dict[str, Any]):
        """
        Display statistical intelligence.
        """

        print("\n========== STATISTICAL INTELLIGENCE ==========\n")

        for column, info in results.items():

            print(f"Column: {column}")

            print(
                f"  Mean              : "
                f"{info.get('mean')}"
            )

            print(
                f"  Median            : "
                f"{info.get('median')}"
            )

            print(
                f"  Standard Deviation: "
                f"{info.get('standard_deviation')}"
            )

            print(
                f"  Q1                : "
                f"{info.get('q1')}"
            )

            print(
                f"  Q3                : "
                f"{info.get('q3')}"
            )

            print(
                f"  IQR               : "
                f"{info.get('iqr')}"
            )

            print(
                f"  Outliers          : "
                f"{info.get('outlier_count')}"
            )

            print(
                f"  Skewness          : "
                f"{info.get('skewness')}"
            )

            print(
                f"  Distribution      : "
                f"{info.get('distribution')}"
            )

            print()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        results = self.analyze(df)
        self.display(results)
        return results