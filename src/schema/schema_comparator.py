"""
Schema Comparator Engine

Author: Pratim Mistry
Description:
Computes schema differences (ADD, DELETE, MODIFY) between an active baseline
and incoming candidate datasets during adaptive retraining cycles.
"""

from typing import Dict, Any, List, Set, Optional
import pandas as pd
from src.schema.schema_registry import SchemaRegistry


class SchemaComparator:
    """
    Detects dynamic feature mutations between baseline and candidate datasets.
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None):
        self.registry = registry or SchemaRegistry()

    def compare_against_active(
        self,
        candidate_df: pd.DataFrame,
        target_column: str = "y"
    ) -> Dict[str, Any]:
        """
        Compare candidate DataFrame against active registered schema.

        Returns
        -------
        Dict[str, Any]
            Detailed schema diff containing added, removed, preserved, and type-modified columns.
        """
        if candidate_df is None or not isinstance(candidate_df, pd.DataFrame) or candidate_df.empty:
            return {
                "is_initial_schema": False,
                "has_mutations": True,
                "added_features": [],
                "deleted_features": [],
                "preserved_features": [],
                "type_mutations": {},
                "unseen_categories": {},
                "candidate_fingerprint": {}
            }

        active_fingerprint = self.registry.get_active_schema()
        candidate_fingerprint = self.registry.extract_schema_fingerprint(
            candidate_df, target_column=target_column
        )

        if not active_fingerprint:
            all_candidate = sorted(
                candidate_fingerprint.get("numeric_features", []) +
                candidate_fingerprint.get("categorical_features", [])
            )
            return {
                "is_initial_schema": True,
                "has_mutations": False,
                "added_features": all_candidate,
                "deleted_features": [],
                "preserved_features": [],
                "type_mutations": {},
                "unseen_categories": {},
                "candidate_fingerprint": candidate_fingerprint
            }

        baseline_features: Set[str] = set(
            active_fingerprint.get("numeric_features", []) +
            active_fingerprint.get("categorical_features", [])
        )
        candidate_features: Set[str] = set(
            candidate_fingerprint.get("numeric_features", []) +
            candidate_fingerprint.get("categorical_features", [])
        )

        added: List[str] = sorted(list(candidate_features - baseline_features))
        deleted: List[str] = sorted(list(baseline_features - candidate_features))
        preserved: List[str] = sorted(list(baseline_features.intersection(candidate_features)))

        # Detect data type migrations
        type_mutations: Dict[str, Dict[str, str]] = {}
        active_dtypes = active_fingerprint.get("column_dtypes", {})
        candidate_dtypes = candidate_fingerprint.get("column_dtypes", {})

        for feat in preserved:
            old_type = active_dtypes.get(feat)
            new_type = candidate_dtypes.get(feat)
            if old_type and new_type and old_type != new_type:
                type_mutations[feat] = {
                    "previous_dtype": str(old_type),
                    "new_dtype": str(new_type)
                }

        # Check for new unseen levels in categorical features
        unseen_categories: Dict[str, List[str]] = {}
        active_cat_levels = active_fingerprint.get("categorical_levels", {})
        candidate_cat_levels = candidate_fingerprint.get("categorical_levels", {})

        for feat in preserved:
            if feat in active_cat_levels and feat in candidate_cat_levels:
                old_levels = set(active_cat_levels[feat])
                new_levels = set(candidate_cat_levels[feat])
                novel_categories = sorted(list(new_levels - old_levels))
                if novel_categories:
                    unseen_categories[feat] = novel_categories

        has_mutations = bool(added or deleted or type_mutations or unseen_categories)

        return {
            "is_initial_schema": False,
            "has_mutations": has_mutations,
            "added_features": added,
            "deleted_features": deleted,
            "preserved_features": preserved,
            "type_mutations": type_mutations,
            "unseen_categories": unseen_categories,
            "candidate_fingerprint": candidate_fingerprint
        }

    def compare(self, candidate_df: pd.DataFrame, target_column: str = "y") -> Dict[str, Any]:
        """
        Alias for pipeline runner.
        """
        return self.compare_against_active(candidate_df, target_column=target_column)

    def run(self, candidate_df: pd.DataFrame, target_column: str = "y") -> Dict[str, Any]:
        """
        Pipeline runner execution alias.
        """
        return self.compare_against_active(candidate_df, target_column=target_column)