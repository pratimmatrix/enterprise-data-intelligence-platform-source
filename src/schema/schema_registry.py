"""
Schema Registry Engine

Author: Pratim Mistry
Description:
Maintains versioned metadata fingerprints of dataset schemas across pipeline retraining.
Enables schema provenance, tracks evolutionary changes, and provides baseline signatures.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import pandas as pd


class SchemaRegistry:
    """
    Persists and retrieves dataset schema definitions across version cycles.
    """

    def __init__(self, schemas_directory: Union[str, Path] = "data/schemas"):
        # Dynamically discover project root if relative path is passed
        current_path = Path(__file__).resolve()
        root_dir = current_path.parent
        for _ in range(4):
            if (root_dir / "src").exists() or (root_dir / "data").exists():
                break
            root_dir = root_dir.parent

        candidate_dir = Path(schemas_directory)
        if candidate_dir.is_absolute():
            self.schemas_dir = candidate_dir
        else:
            self.schemas_dir = root_dir / candidate_dir

        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.schemas_dir / "schema_versions.json"

    def extract_schema_fingerprint(self, df: pd.DataFrame, target_column: str = "y") -> Dict[str, Any]:
        """
        Extract detailed architectural signature of a dataframe.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "total_columns": 0,
                "feature_count": 0,
                "numeric_features": [],
                "categorical_features": [],
                "categorical_levels": {},
                "column_dtypes": {},
                "target_column": target_column
            }

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        categorical_levels = {}
        for col in categorical_cols:
            categorical_levels[col] = sorted(df[col].dropna().astype(str).unique().tolist())

        return {
            "total_columns": int(df.shape[1]),
            "feature_count": int(df.shape[1] - (1 if target_column in df.columns else 0)),
            "numeric_features": sorted(numeric_cols),
            "categorical_features": sorted(categorical_cols),
            "categorical_levels": categorical_levels,
            "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "target_column": target_column
        }

    def register_schema(
        self,
        df: pd.DataFrame,
        version_tag: Optional[str] = None,
        target_column: str = "y",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Register a new schema version in the persistent JSON store.
        """
        registry = self.load_registry()
        fingerprint = self.extract_schema_fingerprint(df, target_column=target_column)

        # Auto-increment version if not specified
        if not version_tag:
            version_number = len(registry.get("versions", [])) + 1
            version_tag = f"v{version_number}.0"

        schema_entry = {
            "version": version_tag,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "fingerprint": fingerprint
        }

        registry["active_version"] = version_tag
        registry["active_fingerprint"] = fingerprint

        # Append or replace version record
        registry["versions"] = [
            v for v in registry.get("versions", []) if v.get("version") != version_tag
        ]
        registry["versions"].append(schema_entry)

        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4)

        return schema_entry

    def load_registry(self) -> Dict[str, Any]:
        """
        Load the persisted schema registry from disk.
        """
        if not self.registry_file.exists():
            return {
                "active_version": None,
                "active_fingerprint": None,
                "versions": []
            }

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "active_version": None,
                "active_fingerprint": None,
                "versions": []
            }

    def get_active_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active schema fingerprint.
        """
        reg = self.load_registry()
        return reg.get("active_fingerprint")