"""
Dataset Metadata Extractor

Author: Pratim Mistry
Description:
Extracts high-level architectural metadata, dimensions,
memory consumption, and column composition from datasets.
"""

from typing import Dict, Any
import pandas as pd


class MetadataExtractor:
    """
    Extract metadata and structural dimensions from a dataset.
    """

    def __init__(self):
        pass

    def extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract dataset structural metadata and print summary.
        """

        print("\n========== DATASET METADATA ==========\n")

        if df is None or not isinstance(df, pd.DataFrame):
            print("Invalid input: DataFrame is required.")
            return {}

        rows = int(df.shape[0])
        cols = int(df.shape[1])
        memory_kb = float(df.memory_usage(deep=True).sum() / 1024)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        print(f"Rows             : {rows}")
        print(f"Columns          : {cols}")
        print(f"\nMemory Usage     : {memory_kb:.2f} KB")
        print(f"Numeric Columns  : {len(numeric_cols)}")
        print(f"Text Columns     : {len(text_cols)}")

        print("\nColumn List")
        for column in df.columns:
            print(f"• {column}")

        return {
            "rows": rows,
            "columns": cols,
            "memory_kb": round(memory_kb, 2),
            "numeric_columns_count": len(numeric_cols),
            "text_columns_count": len(text_cols),
            "column_names": df.columns.tolist()
        }

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        return self.extract(df)