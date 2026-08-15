"""
Dataset Gatekeeper Engine

Author: Pratim Mistry
Description:
The primary security and defensive filter for the platform.
Evaluates candidate CSVs against baseline fingerprints, data quality thresholds,
and target contracts before retraining or schema mutation is allowed.
"""

from typing import Dict, Any, List, Optional
import pandas as pd


# ============================================================
# TARGET INTEGRITY CONTRACT (SELF-CONTAINED TO PREVENT IMPORT ERRORS)
# ============================================================

class TargetIntegrityError(Exception):
    """Raised when target column fails defensive contract."""
    pass


class TargetIntegrityEngine:
    """
    Validates existence, distribution, and validity of target column.
    """

    def __init__(self, target_column: str = "y"):
        self.target_column = target_column

    def validate_target(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.target_column not in df.columns:
            raise TargetIntegrityError(
                f"Mandatory target column '{self.target_column}' is missing from candidate dataset."
            )

        unique_targets = df[self.target_column].dropna().unique()
        
        # Check non-empty target
        if len(unique_targets) == 0:
            raise TargetIntegrityError(
                f"Target column '{self.target_column}' contains only null values."
            )

        # Check binary classification validity
        if len(unique_targets) < 2:
            raise TargetIntegrityError(
                f"Target column '{self.target_column}' must have at least 2 distinct classes. Found: {unique_targets}"
            )

        return {
            "valid": True,
            "target_column": self.target_column,
            "classes": [str(c) for c in unique_targets]
        }


# ============================================================
# GATEKEEPER DECISION ENUM
# ============================================================

class GatekeeperDecision:

    ACCEPT = "ACCEPT"
    WARNING = "WARNING"
    REJECT = "REJECT"


# ============================================================
# DATASET GATEKEEPER
# ============================================================

class DatasetGatekeeper:
    """
    Defensive gatekeeper validating candidate dataset eligibility.
    """

    def __init__(
        self,
        baseline_columns: Optional[List[str]] = None,
        min_rows: int = 100,
        max_missing_ratio: float = 0.50,
        min_schema_overlap_ratio: float = 0.50
    ):

        self.target_engine = TargetIntegrityEngine(target_column="y")
        self.min_rows = min_rows
        self.max_missing_ratio = max_missing_ratio
        self.min_schema_overlap_ratio = min_schema_overlap_ratio
        
        # Default baseline features if none provided
        self.baseline_columns = baseline_columns or [
            "age", "job", "marital", "education", "default", "balance", 
            "housing", "loan", "contact", "day", "month", "duration", 
            "campaign", "pdays", "previous", "poutcome", "y"
        ]


    # ========================================================
    # CANDIDATE EVALUATION PIPELINE
    # ========================================================

    def evaluate_candidate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full gatekeeper validation checks.

        Returns
        -------
        Dict[str, Any]
            Detailed gatekeeper audit report with ACCEPT / WARNING / REJECT status.
        """

        reasons: List[str] = []
        status = GatekeeperDecision.ACCEPT

        # ----------------------------------------------------
        # 1. STRUCTURAL CHECK
        # ----------------------------------------------------

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": ["Uploaded file could not be parsed into a valid DataFrame."]
            }

        if len(df) < self.min_rows:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [
                    f"Insufficient rows ({len(df)}). Minimum required: {self.min_rows}."
                ]
            }

        # ----------------------------------------------------
        # 2. PROTECTED TARGET INTEGRITY CHECK
        # ----------------------------------------------------

        try:
            target_report = self.target_engine.validate_target(df)
        except (TargetIntegrityError, Exception) as err:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [f"Target Integrity Violation: {str(err)}"]
            }

        # ----------------------------------------------------
        # 3. SCHEMA OVERLAP & DOMAIN RELEVANCE CHECK
        # ----------------------------------------------------

        candidate_cols = set(df.columns)
        baseline_set = set(self.baseline_columns)
        
        overlap = candidate_cols.intersection(baseline_set)
        overlap_ratio = len(overlap) / max(len(baseline_set), 1)

        if overlap_ratio < self.min_schema_overlap_ratio:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [
                    f"Dataset appears unrelated to banking campaign domain. "
                    f"Feature overlap with baseline is only {overlap_ratio:.1%} "
                    f"(Minimum required: {self.min_schema_overlap_ratio:.1%})."
                ]
            }

        # ----------------------------------------------------
        # 4. DATA QUALITY & CORRUPT COLUMN CHECKS
        # ----------------------------------------------------

        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            reasons.append(f"Columns with 100% missing values detected: {empty_columns}.")
            status = GatekeeperDecision.WARNING

        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            overall_missing = df.isnull().sum().sum() / total_cells
            if overall_missing > self.max_missing_ratio:
                return {
                    "decision": GatekeeperDecision.REJECT,
                    "reasons": [
                        f"Overall missingness ({overall_missing:.1%}) exceeds threshold ({self.max_missing_ratio:.1%})."
                    ]
                }

        # ----------------------------------------------------
        # 5. ASSEMBLE VERDICT REPORT
        # ----------------------------------------------------

        return {
            "decision": status,
            "reasons": reasons if reasons else ["Dataset passed all defensive checks."],
            "target_report": target_report,
            "schema_overlap_ratio": round(overlap_ratio, 4),
            "candidate_shape": df.shape
        }