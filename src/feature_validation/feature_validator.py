import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class FeatureValidator:

    EXPECTED_FEATURES = [
        "age_group",
        "balance_log",
        "campaign_log",
        "previous_contact",
        "previously_contacted",
        "zero_balance",
        "loan_burden",
        "campaign_intensity",
        "contact_unknown",
        "previous_success",
    ]

    BINARY_FEATURES = [
        "previous_contact",
        "previously_contacted",
        "zero_balance",
        "loan_burden",
        "contact_unknown",
        "previous_success",
    ]

    CATEGORICAL_FEATURES = [
        "age_group",
        "campaign_intensity",
    ]

    NUMERIC_FEATURES = [
        "balance_log",
        "campaign_log",
    ]

    def __init__(self):
        print("FeatureValidator initialized.")

    def validate(self, df: pd.DataFrame) -> bool:

        print("\n========== FEATURE VALIDATION ==========")

        if df is None:
            raise ValueError("Input dataframe is None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        # --------------------------------------------------
        # FEATURE EXISTENCE
        # --------------------------------------------------

        print("\n---------- FEATURE EXISTENCE ----------")

        missing_features = [
            feature
            for feature in self.EXPECTED_FEATURES
            if feature not in df.columns
        ]

        if missing_features:

            print("Missing engineered features:")

            for feature in missing_features:
                print(f"  X {feature}")

            raise ValueError(
                f"Missing engineered features: "
                f"{missing_features}"
            )

        print(
            f"All {len(self.EXPECTED_FEATURES)} "
            "engineered features exist."
        )

        # --------------------------------------------------
        # ROW INTEGRITY
        # --------------------------------------------------

        print("\n---------- ROW INTEGRITY ----------")

        if df.shape[0] == 0:
            raise ValueError(
                "Dataset contains zero rows."
            )

        print(
            f"Valid row count: {df.shape[0]}"
        )

        # --------------------------------------------------
        # BINARY FEATURES
        # --------------------------------------------------

        print("\n---------- BINARY FEATURES ----------")

        for feature in self.BINARY_FEATURES:

            values = set(
                df[feature]
                .dropna()
                .unique()
            )

            invalid_values = values - {0, 1}

            if invalid_values:

                raise ValueError(
                    f"{feature} contains invalid "
                    f"binary values: {invalid_values}"
                )

            print(
                f"{feature:<25} "
                f"values={sorted(values)}"
            )

        # --------------------------------------------------
        # NUMERIC FEATURES
        # --------------------------------------------------

        print("\n---------- NUMERIC FEATURES ----------")

        for feature in self.NUMERIC_FEATURES:

            if not pd.api.types.is_numeric_dtype(
                df[feature]
            ):

                raise TypeError(
                    f"{feature} must be numeric."
                )

            nan_count = df[feature].isna().sum()

            inf_count = np.isinf(
                df[feature]
            ).sum()

            if nan_count > 0:

                raise ValueError(
                    f"{feature} contains "
                    f"{nan_count} NaN values."
                )

            if inf_count > 0:

                raise ValueError(
                    f"{feature} contains "
                    f"{inf_count} infinite values."
                )

            print(
                f"{feature:<25} "
                f"NaN=0 | Inf=0"
            )

        # --------------------------------------------------
        # CATEGORICAL FEATURES
        # --------------------------------------------------

        print(
            "\n---------- CATEGORICAL FEATURES ----------"
        )

        for feature in self.CATEGORICAL_FEATURES:

            missing_count = (
                df[feature].isna().sum()
            )

            unique_count = (
                df[feature].nunique(
                    dropna=True
                )
            )

            print(
                f"{feature:<25} "
                f"unique={unique_count} | "
                f"missing={missing_count}"
            )

        # --------------------------------------------------
        # DUPLICATE CHECK
        # --------------------------------------------------

        print("\n---------- DUPLICATE CHECK ----------")

        duplicate_count = df.duplicated().sum()

        print(
            f"Duplicate rows: {duplicate_count}"
        )

        # --------------------------------------------------
        # VALIDATION SUMMARY
        # --------------------------------------------------

        print(
            "\n---------- VALIDATION SUMMARY ----------"
        )

        print(
            f"Expected features : "
            f"{len(self.EXPECTED_FEATURES)}"
        )

        print(
            f"Binary features   : "
            f"{len(self.BINARY_FEATURES)}"
        )

        print(
            f"Numeric features  : "
            f"{len(self.NUMERIC_FEATURES)}"
        )

        print(
            f"Categorical       : "
            f"{len(self.CATEGORICAL_FEATURES)}"
        )

        print("\nFEATURE VALIDATION PASSED")

        return True