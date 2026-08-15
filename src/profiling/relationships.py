"""
Relationship Intelligence Engine

Author: Pratim Mistry

Description:
Analyzes relationships between dataset columns,
including numeric correlations and categorical
relationships with the target variable.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class RelationshipAnalyzer:
    """
    Enterprise relationship analysis engine.
    """

    def __init__(self):
        print("RelationshipAnalyzer initialized.")

    # ============================================================
    # MAIN ANALYZER
    # ============================================================

    def analyze(self, df: pd.DataFrame, target: str = "y") -> Dict[str, Any]:
        """
        Analyze relationships between dataset columns.
        """

        print("\n========== RELATIONSHIP INTELLIGENCE ==========")

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            print("Invalid input: Non-empty DataFrame is required.")
            return {"numeric_correlations": [], "categorical_relationships": {}}

        numeric_rels = self._numeric_correlations(df)
        cat_rels = self._categorical_target_relationships(df, target)

        return {
            "numeric_correlations": numeric_rels,
            "categorical_relationships": cat_rels
        }

    # ============================================================
    # NUMERIC CORRELATIONS
    # ============================================================

    def _numeric_correlations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze correlations between numeric columns.
        """

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            print("\nNot enough numeric columns for correlation analysis.")
            return []

        correlation_matrix = numeric_df.corr()

        print("\n---------- NUMERIC CORRELATIONS ----------")

        relationships = []
        columns = correlation_matrix.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):

                col1 = columns[i]
                col2 = columns[j]

                correlation = correlation_matrix.loc[col1, col2]

                if not np.isnan(correlation) and abs(correlation) >= 0.3:
                    relationships.append((col1, col2, float(correlation)))

        relationships.sort(
            key=lambda x: abs(x[2]),
            reverse=True
        )

        if not relationships:
            print("No strong numeric relationships detected (threshold >= 0.3).")
            return []

        formatted_rels = []

        for col1, col2, correlation in relationships:

            if correlation >= 0.7:
                strength = "Very Strong Positive"
            elif correlation >= 0.5:
                strength = "Strong Positive"
            elif correlation >= 0.3:
                strength = "Moderate Positive"
            elif correlation <= -0.7:
                strength = "Very Strong Negative"
            elif correlation <= -0.5:
                strength = "Strong Negative"
            else:
                strength = "Moderate Negative"

            print(f"{col1} <-> {col2}")
            print(f"  Correlation : {correlation:.3f}")
            print(f"  Relationship: {strength}\n")

            formatted_rels.append({
                "feature_1": col1,
                "feature_2": col2,
                "correlation": round(correlation, 4),
                "strength": strength
            })

        return formatted_rels

    # ============================================================
    # CATEGORICAL TARGET RELATIONSHIPS
    # ============================================================

    def _categorical_target_relationships(
        self,
        df: pd.DataFrame,
        target: str
    ) -> Dict[str, Any]:
        """
        Analyze categorical columns against target.
        """

        if target not in df.columns:
            print(f"\nTarget column '{target}' not found in dataset.")
            return {}

        print("\n---------- CATEGORICAL TARGET RELATIONSHIPS ----------")

        categorical_columns = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        relationships_summary = {}

        for column in categorical_columns:

            if column == target:
                continue

            print(f"\nColumn: {column}")

            # Cross-tabulate with target
            grouped = (
                df.groupby(column, dropna=False)[target]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )

            # Identify positive class dynamically (yes, 1, True, or highest index)
            pos_label = None
            for candidate in ["yes", "YES", 1, "1", True, "True"]:
                if candidate in grouped.columns:
                    pos_label = candidate
                    break

            if pos_label is None and len(grouped.columns) == 2:
                pos_label = grouped.columns[1]

            if pos_label is not None:
                grouped_sorted = grouped.sort_values(
                    by=pos_label,
                    ascending=False
                )

                col_breakdown = {}
                for category, row in grouped_sorted.head(5).iterrows():
                    rate = float(row[pos_label]) * 100
                    col_breakdown[str(category)] = round(rate, 2)
                    print(
                        f"  {category} -> "
                        f"Target={pos_label}: {rate:.2f}%"
                    )

                relationships_summary[column] = {
                    "positive_class": str(pos_label),
                    "top_categories_rate": col_breakdown
                }
            else:
                print(f"  Multi-class or undetermined positive class for target: {list(grouped.columns)}")

        return relationships_summary

    # ============================================================
    # PIPELINE ALIASES
    # ============================================================

    def run(self, df: pd.DataFrame, target: str = "y") -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        return self.analyze(df, target=target)