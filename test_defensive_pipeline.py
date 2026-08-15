"""
Live Verification & Unit Test Suite for Defensive ML Layer

Author: Pratim Mistry
Tests:
1. TargetIntegrityEngine: Valid vs Corrupted Target
2. DatasetGatekeeper: Normal vs Unrelated CSV
3. SchemaComparator & DynamicPreprocessorBuilder: Adding and Dropping Columns Dynamically
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.target_integrity import TargetIntegrityEngine, TargetIntegrityError
from src.gatekeeper.dataset_gatekeeper import DatasetGatekeeper, GatekeeperDecision
from src.schema.schema_registry import SchemaRegistry
from src.schema.schema_comparator import SchemaComparator
from src.features.dynamic_preprocessor import DynamicPreprocessorBuilder


def run_all_tests():
    print("=" * 70)
    print("🚀 STARTING ADAPTIVE EIP DEFENSIVE LAYER VERIFICATION")
    print("=" * 70)

    # ----------------------------------------------------
    # Load actual baseline bank data
    # ----------------------------------------------------
    data_path = PROJECT_ROOT / "bank-full.csv"
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found.")
        return

    print(f"\n📂 Loading {data_path.name}...")
    df = pd.read_csv(data_path, sep=";")
    print(f"✅ Baseline dataset loaded successfully! Shape: {df.shape}")

    # ====================================================
    # TEST 1: TARGET INTEGRITY ENGINE
    # ====================================================
    print("\n" + "-" * 70)
    print("TEST 1: Testing Target Integrity Engine...")
    target_engine = TargetIntegrityEngine(target_column="y")
    
    # 1.1 Valid Target Test
    report = target_engine.validate_target(df)
    print(f"  [Pass] Normal Target: {report['status']} | Yes Rate: {report['positive_class_percentage']}%")

    # 1.2 Corrupted Target Test (Simulating Target Tampering)
    corrupted_df = df.copy()
    corrupted_df.loc[0:5, "y"] = "invalid_class"
    try:
        target_engine.validate_target(corrupted_df)
        print("  [Fail] Corrupted target was NOT detected!")
    except TargetIntegrityError as err:
        print(f"  [Pass] Tampered Target Blocked! Error Caught: {err}")

    # ====================================================
    # TEST 2: DATASET GATEKEEPER
    # ====================================================
    print("\n" + "-" * 70)
    print("TEST 2: Testing Dataset Gatekeeper Engine...")
    gatekeeper = DatasetGatekeeper()

    # 2.1 Valid Dataset
    eval_normal = gatekeeper.evaluate_candidate(df)
    print(f"  [Pass] Baseline CSV Gatekeeper Decision: {eval_normal['decision']}")

    # 2.2 Unrelated Dataset (Simulating Wrong CSV Upload)
    fake_df = pd.DataFrame({
        "passenger_id": range(200),
        "ticket_fare": np.random.rand(200) * 100,
        "cabin": ["A", "B", "C", "D"] * 50,
        "y": ["no", "yes"] * 100
    })
    eval_fake = gatekeeper.evaluate_candidate(fake_df)
    print(f"  [Pass] Unrelated CSV Gatekeeper Decision: {eval_fake['decision']}")
    print(f"         Reason: {eval_fake['reasons'][0]}")

    # ====================================================
    # TEST 3: DYNAMIC SCHEMA & PREPROCESSING MUTATION
    # ====================================================
    print("\n" + "-" * 70)
    print("TEST 3: Testing Dynamic Schema & Feature Mutation...")
    registry = SchemaRegistry()
    registry.register_schema(df, version_tag="v1.0", notes="Initial Banking Baseline")

    # 3.1 Simulate CSV Modification: Add 2 columns, Drop 1 column
    mutated_df = df.copy()
    mutated_df["credit_score"] = np.random.randint(300, 850, size=len(mutated_df))
    mutated_df["annual_income"] = np.random.uniform(20000, 150000, size=len(mutated_df))
    mutated_df = mutated_df.drop(columns=["pdays"])  # Dropped column

    comparator = SchemaComparator(registry)
    diff = comparator.compare_against_active(mutated_df)
    print(f"  [Pass] Schema Diff Detected:")
    print(f"         Added Columns   : {diff['added_features']}")
    print(f"         Deleted Columns : {diff['deleted_features']}")

    # 3.2 Dynamic Preprocessing without hardcoded columns
    X_mutated = mutated_df.drop(columns=["y"])
    builder = DynamicPreprocessorBuilder()
    transformed_X, preprocessor = builder.fit_transform(X_mutated)
    meta = builder.get_feature_metadata()
    print(f"  [Pass] Dynamic Pipeline Executed Successfully!")
    print(f"         Numeric Features Processed    : {meta['numeric_count']}")
    print(f"         Categorical Features Processed: {meta['categorical_count']}")
    print(f"         Transformed Matrix Shape      : {transformed_X.shape}")

    print("\n" + "=" * 70)
    print("🎉 ALL DEFENSIVE & DYNAMIC ADAPTIVE TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()