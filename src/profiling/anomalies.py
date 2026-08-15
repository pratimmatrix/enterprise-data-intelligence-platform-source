"""
Anomaly Intelligence Engine

Author: Pratim Mistry
Description:
Detects potential numeric anomalies using IQR (Interquartile Range) 
and Z-Score statistical methods.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class AnomalyEngine:
    """
    Enterprise anomaly detection engine for statistical data quality audits.
    """

    def __init__(self):
        print("AnomalyEngine initialized.")

    # ============================================================
    # ANALYZE ANOMALIES
    # ============================================================

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze numeric columns for potential anomalies and statistical outliers.
        """

        print("\n========== ANOMALY INTELLIGENCE ==========")

        if df is None or not isinstance(df, pd.DataFrame):
            print("Invalid input: DataFrame is required.")
            return {}

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        if len(numeric_columns) == 0:
            print("No numeric columns found.")
            return {}

        anomaly_summary = {}

        for column in numeric_columns:

            series = df[column].dropna()

            if len(series) == 0:
                continue

            # -----------------------------
            # IQR METHOD
            # -----------------------------

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))

            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            iqr_anomalies = int((
                (series < lower_bound) |
                (series > upper_bound)
            ).sum())

            # -----------------------------
            # Z-SCORE METHOD
            # -----------------------------

            mean = float(series.mean())
            std = float(series.std())

            if std == 0 or np.isnan(std):
                zscore_anomalies = 0
            else:
                z_scores = (series - mean) / std
                zscore_anomalies = int((z_scores.abs() > 3).sum())

            # -----------------------------
            # ANOMALY PERCENTAGE
            # -----------------------------

            total_count = len(series)
            anomaly_percentage = (iqr_anomalies / total_count) * 100 if total_count > 0 else 0.0

            # -----------------------------
            # REPORTING
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

            anomaly_summary[column] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "iqr_anomalies": iqr_anomalies,
                "zscore_anomalies": zscore_anomalies,
                "anomaly_percentage": round(anomaly_percentage, 2)
            }

        return anomaly_summary

    # ============================================================
    # PIPELINE ALIASES
    # ============================================================

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.analyze(df)

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.analyze(df)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.analyze(df)