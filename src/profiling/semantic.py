"""
Enterprise Semantic Intelligence Engine

Author: Pratim Mistry

Description:
Detects common semantic values such as sentinel values
and placeholder values that may have special meaning
in a dataset.
"""

from typing import Dict, Any
import pandas as pd


class SemanticAnalyzer:
    """
    Analyze columns for meaningful sentinel and placeholder values.
    """

    def __init__(self):
        print("SemanticAnalyzer initialized.")

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect common semantic values in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to analyze.

        Returns
        -------
        dict
            Semantic analysis results.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {}

        results = {}

        # Common placeholder values
        placeholder_values = {
            "unknown",
            "missing",
            "not available",
            "n/a",
            "na",
            "none",
            "null",
            "?",
        }

        for column in df.columns:

            series = df[column]

            column_results = {
                "placeholder_values": {},
                "sentinel_values": {},
            }

            # ==========================================
            # TEXT PLACEHOLDERS
            # ==========================================

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):

                normalized = (
                    series
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                for value in placeholder_values:

                    count = int(
                        normalized.eq(value).sum()
                    )

                    if count > 0:
                        column_results[
                            "placeholder_values"
                        ][value] = count

            # ==========================================
            # NUMERIC SENTINEL VALUES
            # ==========================================

            if pd.api.types.is_numeric_dtype(series):

                clean_series = series.dropna()
                value_counts = clean_series.value_counts()
                total_len = len(clean_series)

                for value, count in value_counts.items():

                    # Negative sentinel values (e.g. -1 in pdays)
                    if value < 0:

                        column_results[
                            "sentinel_values"
                        ][str(value)] = int(count)

                    # High density of 0 (can be a default / flag)
                    elif value == 0 and total_len > 0:

                        zero_ratio = count / total_len

                        if zero_ratio > 0.05:

                            column_results[
                                "sentinel_values"
                            ][str(value)] = int(count)

                    # 999 / 9999 common sentinel codes
                    elif value in {999, 9999}:
                        column_results[
                            "sentinel_values"
                        ][str(value)] = int(count)

            results[column] = column_results

        return results

    def display(self, results: Dict[str, Any]):
        """
        Display semantic intelligence results.
        """

        print(
            "\n========== SEMANTIC INTELLIGENCE ==========\n"
        )

        for column, info in results.items():

            placeholders = info.get("placeholder_values", {})
            sentinels = info.get("sentinel_values", {})

            if not placeholders and not sentinels:
                continue

            print(f"Column: {column}")

            if placeholders:

                print("  Placeholder Values:")

                for value, count in placeholders.items():

                    print(
                        f"    {value} -> {count} records"
                    )

            if sentinels:

                print("  Sentinel Values:")

                for value, count in sentinels.items():

                    print(
                        f"    {value} -> {count} records"
                    )

            print()

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        results = self.analyze(df)
        self.display(results)
        return results