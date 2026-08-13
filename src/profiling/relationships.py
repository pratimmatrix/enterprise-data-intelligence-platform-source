"""
Relationship Intelligence Engine

Author: Pratim Mistry

Description:
Analyzes relationships between dataset columns,
including numeric correlations and categorical
relationships with the target variable.
"""

import pandas as pd
import numpy as np


class RelationshipAnalyzer:
    """
    Enterprise relationship analysis engine.
    """

    def __init__(self):
        print("RelationshipAnalyzer initialized.")

    def analyze(self, df: pd.DataFrame, target: str = "y"):
        """
        Analyze relationships between dataset columns.
        """

        print("\n========== RELATIONSHIP INTELLIGENCE ==========")

        self._numeric_correlations(df)
        self._categorical_target_relationships(df, target)

    def _numeric_correlations(self, df: pd.DataFrame):
        """
        Analyze correlations between numeric columns.
        """

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            print("\nNot enough numeric columns for correlation analysis.")
            return

        correlation_matrix = numeric_df.corr()

        print("\n---------- NUMERIC CORRELATIONS ----------")

        relationships = []

        columns = correlation_matrix.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):

                col1 = columns[i]
                col2 = columns[j]

                correlation = correlation_matrix.loc[col1, col2]

                if abs(correlation) >= 0.3:
                    relationships.append(
                        (col1, col2, correlation)
                    )

        relationships.sort(
            key=lambda x: abs(x[2]),
            reverse=True
        )

        if not relationships:
            print("No strong numeric relationships detected.")
            return

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

            print(
                f"{col1} <-> {col2}"
            )
            print(
                f"  Correlation : {correlation:.3f}"
            )
            print(
                f"  Relationship: {strength}"
            )

    def _categorical_target_relationships(
        self,
        df: pd.DataFrame,
        target: str
    ):
        """
        Analyze categorical columns against target.
        """

        if target not in df.columns:
            print(
                f"\nTarget column '{target}' not found."
            )
            return

        print("\n---------- CATEGORICAL TARGET RELATIONSHIPS ----------")

        categorical_columns = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns

        for column in categorical_columns:

            if column == target:
                continue

            print(f"\nColumn: {column}")

            grouped = (
                df.groupby(column, dropna=False)[target]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )

            if "yes" in grouped.columns:

                grouped = grouped.sort_values(
                    by="yes",
                    ascending=False
                )

                for category, row in grouped.head(5).iterrows():

                    yes_rate = row["yes"] * 100

                    print(
                        f"  {category} -> "
                        f"Target=yes: {yes_rate:.2f}%"
                    )

            else:
                print(
                    "  Target does not contain 'yes' category."
                )