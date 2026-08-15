"""
Data Validator Engine
Enterprise Customer Intelligence Platform

Author: Pratim Mistry
Description:
Performs defensive structural validation, null-value audits,
duplicate detection, and memory profiling on ingested tabular datasets.
"""

from typing import Dict, Any, Optional
import pandas as pd


class DataValidator:
    """
    Validate datasets before analysis and pipeline execution.
    """

    def __init__(self):

        print("DataValidator initialized.")


    # ========================================================
    # DATASET VALIDATION AUDIT
    # ========================================================

    def validate(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes structural audit on candidate DataFrames.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to evaluate.

        Returns
        -------
        Dict[str, Any]
            Audit summary report including counts, duplicates, and memory usage.
        """

        print("\n========== DATA VALIDATION ==========\n")

        # ----------------------------------------------------
        # 1. INPUT TYPE VALIDATION
        # ----------------------------------------------------

        if df is None:
            raise ValueError("Input dataset is None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a valid pandas DataFrame.")

        # ----------------------------------------------------
        # 2. STRUCTURAL METRICS
        # ----------------------------------------------------

        row_count = df.shape[0]
        col_count = df.shape[1]
        duplicate_rows = int(df.duplicated().sum())
        total_missing = int(df.isnull().sum().sum())
        memory_kb = float(df.memory_usage(deep=True).sum() / 1024.0)

        print(f"Rows              : {row_count}")
        print(f"Columns           : {col_count}")
        print(f"\nDuplicate Rows    : {duplicate_rows}")

        # ----------------------------------------------------
        # 3. MISSING VALUE PROFILE
        # ----------------------------------------------------

        print("\nMissing Values:")
        print(df.isnull().sum())

        # ----------------------------------------------------
        # 4. DATA TYPES
        # ----------------------------------------------------

        print("\nData Types:")
        print(df.dtypes)

        # ----------------------------------------------------
        # 5. MEMORY PROFILE
        # ----------------------------------------------------

        print("\nMemory Usage:")
        print(f"{memory_kb:.2f} KB")

        print("\n=====================================\n")

        # Return structured audit dictionary for programmatic usage
        return {
            "rows": row_count,
            "columns": col_count,
            "duplicate_rows": duplicate_rows,
            "total_missing": total_missing,
            "memory_kb": round(memory_kb, 2),
            "is_valid": row_count > 0 and col_count > 0
        }