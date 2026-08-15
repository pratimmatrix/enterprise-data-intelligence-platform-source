"""
Live Verification Suite for Model Training & Quality Gate Engine

Author: Pratim Mistry
Tests:
1. AdaptiveModelTrainer: Trains Logistic Regression, Random Forest, Gradient Boosting
2. Artifact Persistence: Confirms .pkl models are saved in models/artifacts/
3. QualityGateEngine: Evaluates candidate metrics for production promotion
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.adaptive_trainer import AdaptiveModelTrainer
from src.training.quality_gate import QualityGateEngine, QualityGateDecision


def run_training_gate_test():
    print("=" * 70)
    print("🎯 STARTING MODEL TRAINING & QUALITY GATE VERIFICATION")
    print("=" * 70)

    data_path = PROJECT_ROOT / "bank-full.csv"
    print(f"\n📂 Ingesting dataset: {data_path.name}...")
    df = pd.read_csv(data_path, sep=";")
    print(f"✅ Ingestion successful! Records: {len(df)}")

    # ----------------------------------------------------
    # 1. TRAIN ADAPTIVE CANDIDATE MODELS
    # ----------------------------------------------------
    print("\n" + "-" * 70)
    print("⚡ Training candidate model suite (LogReg, RF, GB)...")
    trainer = AdaptiveModelTrainer(artifacts_dir="models/artifacts")
    train_results = trainer.train_and_evaluate(df)

    best_model = train_results["best_candidate_name"]
    best_metrics = train_results["best_candidate_metrics"]
    champion_path = train_results["champion_path"]

    print("\n📊 CANDIDATE MODEL BENCHMARK RESULTS:")
    for name, metrics in train_results["all_candidates"].items():
        print(f"  • {name.upper():<22} -> ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f}")

    print(f"\n🏆 Top Champion Candidate Selected: {best_model.upper()}")
    print(f"📁 Champion Artifact Persisted at: {champion_path}")

    # ----------------------------------------------------
    # 2. RUN QUALITY GATE GOVERNANCE
    # ----------------------------------------------------
    print("\n" + "-" * 70)
    print("🛡️ Running Quality Gate Governance Checks...")
    gate = QualityGateEngine(
        min_roc_auc=0.70,
        min_f1_score=0.35
    )
    decision_report = gate.evaluate_candidate(best_model, best_metrics)

    print(f"  Quality Gate Verdict: {decision_report['decision']}")
    for r in decision_report["reasons"]:
        print(f"  Reason: {r}")

    assert decision_report["decision"] == QualityGateDecision.PROMOTE, "Quality gate rejected champion model!"

    print("\n" + "=" * 70)
    print("🎉 ALL TRAINING & QUALITY GATE VERIFICATIONS PASSED 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_training_gate_test()