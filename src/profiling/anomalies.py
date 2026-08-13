"""
Anomaly Intelligence Engine

Author: Pratim Mistry

Description:
Detects potential numeric anomalies using
the IQR and Z-score methods.
"""

import pandas as pd
import numpy as np


class AnomalyEngine:
    """
    Enterprise anomaly detection engine.
    """

    def __init__(self):
        print("AnomalyEngine initialized.")

    def analyze(self, df: pd.DataFrame):
        """
        Analyze numeric columns for potential anomalies.
        """

        print("\n========== ANOMALY INTELLIGENCE ==========")

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) == 0:
            print("No numeric columns found.")
            return

        for column in numeric_columns:

            series = df[column].dropna()

            if len(series) == 0:
                continue

            # -----------------------------
            # IQR METHOD
            # -----------------------------

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            iqr_anomalies = (
                (series < lower_bound) |
                (series > upper_bound)
            ).sum()

            # -----------------------------
            # Z-SCORE METHOD
            # -----------------------------

            mean = series.mean()
            std = series.std()

            if std == 0:
                zscore_anomalies = 0
            else:
                z_scores = (
                    (series - mean) / std
                )

                zscore_anomalies = (
                    z_scores.abs() > 3
                ).sum()

            # -----------------------------
            # ANOMALY PERCENTAGE
            # -----------------------------

            anomaly_percentage = (
                iqr_anomalies / len(series)
            ) * 100

            # -----------------------------
            # REPORT
            # -----------------------------

            print(f"\nColumn: {column}")

            print(
                f"  IQR Lower Bound     : "
                f"{lower_bound:.2f}"
            )

            print(
                f"  IQR Upper Bound     : "
                f"{upper_bound:.2f}"
            )

            print(
                f"  IQR Anomalies       : "
                f"{iqr_anomalies}"
            )

            print(
                f"  Z-Score Anomalies   : "
                f"{zscore_anomalies}"
            )

            print(
                f"  Anomaly Percentage  : "
                f"{anomaly_percentage:.2f}%"
            )