import pandas as pd


class MetadataExtractor:
    """
    Extract metadata from a dataset.
    """

    def extract(self, df: pd.DataFrame):

        print("\n========== DATASET METADATA ==========\n")

        print(f"Rows             : {df.shape[0]}")
        print(f"Columns          : {df.shape[1]}")

        print(f"\nMemory Usage     : {df.memory_usage(deep=True).sum()/1024:.2f} KB")

        print(f"Numeric Columns  : {len(df.select_dtypes(include='number').columns)}")
        print(f"Text Columns     : {len(df.select_dtypes(include='object').columns)}")

        print("\nColumn List")

        for column in df.columns:
            print(f"• {column}")