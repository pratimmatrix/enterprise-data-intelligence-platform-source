import pandas as pd


class DataValidator:
    """
    Validate datasets before analysis.
    """

    def validate(self, df: pd.DataFrame):

        print("\n========== DATA VALIDATION ==========\n")

        print(f"Rows              : {df.shape[0]}")
        print(f"Columns           : {df.shape[1]}")

        print(f"\nDuplicate Rows    : {df.duplicated().sum()}")

        print("\nMissing Values")

        print(df.isnull().sum())

        print("\nData Types")

        print(df.dtypes)

        print("\nMemory Usage")

        print(
            f"{df.memory_usage(deep=True).sum()/1024:.2f} KB"
        )