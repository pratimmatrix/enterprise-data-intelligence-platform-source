"""
Enterprise Semantic Intelligence Engine

Author: Pratim Mistry

Description:
Detects common semantic values such as sentinel values
and placeholder values that may have special meaning
in a dataset.
"""

import pandas as pd


class SemanticAnalyzer:
    """
    Analyze columns for meaningful sentinel and placeholder values.
    """

    def analyze(self, df: pd.DataFrame) -> dict:
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

            if pd.api.types.is_object_dtype(series):

                normalized = (
                    series
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

                value_counts = series.value_counts()

                for value, count in value_counts.items():

                    # Negative sentinel values
                    if value < 0:

                        column_results[
                            "sentinel_values"
                        ][str(value)] = int(count)

                    # Zero can be meaningful in some columns
                    elif value == 0:

                        zero_ratio = count / len(series)

                        if zero_ratio > 0.05:

                            column_results[
                                "sentinel_values"
                            ][str(value)] = int(count)

            results[column] = column_results

        return results

    def display(self, results: dict):
        """
        Display semantic intelligence results.
        """

        print(
            "\n========== SEMANTIC INTELLIGENCE ==========\n"
        )

        for column, info in results.items():

            placeholders = info["placeholder_values"]
            sentinels = info["sentinel_values"]

            if not placeholders and not sentinels:
                continue

            print(f"Column: {column}")

            if placeholders:

                print("  Placeholder Values:")

                for value, count in placeholders.items():

                    print(
                        f"    {value} → {count} records"
                    )

            if sentinels:

                print("  Sentinel Values:")

                for value, count in sentinels.items():

                    print(
                        f"    {value} → {count} records"
                    )

            print()