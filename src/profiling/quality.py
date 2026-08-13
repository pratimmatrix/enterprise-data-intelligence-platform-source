"""
Enterprise Data Quality Engine

Author: Pratim Mistry

Description:
Analyzes datasets for missing values, placeholder values,
duplicate rows, and basic data-quality issues.
"""

import pandas as pd


class DataQualityEngine:
    """
    Analyze the quality of a Pandas DataFrame.
    """

    def analyze(self, df: pd.DataFrame) -> dict:
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

        missing_values = int(df.isnull().sum().sum())

        duplicate_rows = int(df.duplicated().sum())

        unknown_values = 0

        for column in df.select_dtypes(include="object").columns:
            unknown_values += int(
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("unknown")
                .sum()
            )

        total_cells = df.shape[0] * df.shape[1]

        if total_cells == 0:
            quality_score = 0
        else:
            issue_count = (
                missing_values
                + duplicate_rows
                + unknown_values
            )

            quality_score = max(
                0,
                round(
                    100 * (1 - issue_count / total_cells),
                    2
                )
            )

        return {
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "unknown_values": unknown_values,
            "quality_score": quality_score,
        }

    def display(self, results: dict):
        """
        Display data quality results.
        """

        print("\n========== DATA QUALITY ==========\n")

        print(
            f"Missing Values     : "
            f"{results['missing_values']}"
        )

        print(
            f"Duplicate Rows     : "
            f"{results['duplicate_rows']}"
        )

        print(
            f"Unknown Values     : "
            f"{results['unknown_values']}"
        )

        print(
            f"Data Quality Score : "
            f"{results['quality_score']}/100"
        )