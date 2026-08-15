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
from src.validation.target_integrity import TargetIntegrityEngine, TargetIntegrityError


class GatekeeperDecision:
    ACCEPT = "ACCEPT"
    WARNING = "WARNING"
    REJECT = "REJECT"


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

        # 1. Structural Check
        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": ["Uploaded file could not be parsed into a DataFrame."]
            }

        if len(df) < self.min_rows:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [f"Insufficient rows ({len(df)}). Minimum required: {self.min_rows}."]
            }

        # 2. Protected Target Integrity Check
        try:
            target_report = self.target_engine.validate_target(df)
        except TargetIntegrityError as err:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [f"Target Integrity Violation: {str(err)}"]
            }

        # 3. Schema Overlap & Project Relevance Check
        candidate_cols = set(df.columns)
        baseline_set = set(self.baseline_columns)
        
        overlap = candidate_cols.intersection(baseline_set)
        overlap_ratio = len(overlap) / len(baseline_set)

        if overlap_ratio < self.min_schema_overlap_ratio:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [
                    f"Dataset appears unrelated to banking campaign domain. "
                    f"Feature overlap with baseline is only {overlap_ratio:.1%} "
                    f"(Minimum required: {self.min_schema_overlap_ratio:.1%})."
                ]
            }

        # 4. Data Quality Checks (100% Empty / Corrupt Columns)
        empty_columns = [col for col in df.columns if df[col].isnull().all()]
        if empty_columns:
            reasons.append(f"Columns with 100% missing values detected: {empty_columns}.")
            status = GatekeeperDecision.WARNING

        overall_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if overall_missing > self.max_missing_ratio:
            return {
                "decision": GatekeeperDecision.REJECT,
                "reasons": [f"Overall missingness ({overall_missing:.1%}) exceeds threshold ({self.max_missing_ratio:.1%})."]
            }

        return {
            "decision": status,
            "reasons": reasons if reasons else ["Dataset passed all defensive checks."],
            "target_report": target_report,
            "schema_overlap_ratio": round(overlap_ratio, 4),
            "candidate_shape": df.shape
        }