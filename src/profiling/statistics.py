"""
Enterprise Statistical Intelligence Engine

Author: Pratim Mistry

Description:
Analyzes numerical columns to identify statistical
characteristics, skewness, and potential outliers.
"""

import pandas as pd


class StatisticalAnalyzer:
    """
    Perform statistical analysis on numerical columns.
    """

    def analyze(self, df: pd.DataFrame) -> dict:
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

        results = {}

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:

            series = df[column].dropna()

            if series.empty:
                continue

            mean = float(series.mean())
            median = float(series.median())
            std = float(series.std())

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))

            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outlier_count = int(
                ((series < lower_bound) |
                 (series > upper_bound)).sum()
            )

            skewness = float(series.skew())

            if skewness > 1:
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
                "standard_deviation": round(std, 2),
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "outlier_count": outlier_count,
                "skewness": round(skewness, 3),
                "distribution": distribution,
            }

        return results

    def display(self, results: dict):
        """
        Display statistical intelligence.
        """

        print("\n========== STATISTICAL INTELLIGENCE ==========\n")

        for column, info in results.items():

            print(f"Column: {column}")

            print(
                f"  Mean              : "
                f"{info['mean']}"
            )

            print(
                f"  Median            : "
                f"{info['median']}"
            )

            print(
                f"  Standard Deviation: "
                f"{info['standard_deviation']}"
            )

            print(
                f"  Q1                : "
                f"{info['q1']}"
            )

            print(
                f"  Q3                : "
                f"{info['q3']}"
            )

            print(
                f"  IQR               : "
                f"{info['iqr']}"
            )

            print(
                f"  Outliers          : "
                f"{info['outlier_count']}"
            )

            print(
                f"  Skewness          : "
                f"{info['skewness']}"
            )

            print(
                f"  Distribution      : "
                f"{info['distribution']}"
            )

            print()