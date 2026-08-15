"""
Quality Gate Engine

Author: Pratim Mistry
Description:
Evaluates candidate retrained model metrics against active production baseline.
Decides whether to PROMOTE candidate model to production or ROLLBACK to baseline.
"""

from typing import Dict, Any, Optional, Union
import json
from pathlib import Path


class QualityGateDecision:
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"


class QualityGateEngine:
    """
    Automated governance gate enforcing model deployment performance constraints.
    """

    def __init__(
        self,
        registry_path: Union[str, Path] = "models/registry/model_registry.json",
        min_roc_auc: float = 0.70,
        min_f1_score: float = 0.35,
        max_performance_drop: float = 0.05
    ):
        # Resolve project root dynamically across varying directory depths
        current_path = Path(__file__).resolve()
        root_dir = current_path.parent
        for _ in range(4):
            if (root_dir / "src").exists() or (root_dir / "models").exists():
                break
            root_dir = root_dir.parent

        candidate_file = Path(registry_path)
        if candidate_file.is_absolute():
            self.registry_path = candidate_file
        else:
            self.registry_path = root_dir / candidate_file

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_roc_auc = min_roc_auc
        self.min_f1_score = min_f1_score
        self.max_performance_drop = max_performance_drop

    def load_active_baseline_metrics(self) -> Optional[Dict[str, float]]:
        """
        Load metrics of the currently deployed champion model.
        """
        if not self.registry_path.exists():
            return None

        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)

            active_model = registry.get("selected_model")
            for m in registry.get("models", []):
                if m.get("model_name") == active_model:
                    return m.get("metrics")
        except Exception:
            return None

        return None

    def evaluate_candidate(
        self,
        candidate_name: str,
        candidate_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Compare candidate metrics against active baseline and absolute minimum thresholds.

        Returns
        -------
        Dict[str, Any]
            Quality gate evaluation with decision (PROMOTE or REJECT) and detailed reasons.
        """
        cand_roc = float(candidate_metrics.get("roc_auc", 0.0))
        cand_f1 = float(candidate_metrics.get("f1", 0.0))
        reasons = []

        # 1. Absolute Threshold Checks
        if cand_roc < self.min_roc_auc:
            reasons.append(
                f"Candidate ROC-AUC ({cand_roc:.4f}) is below minimum acceptable threshold ({self.min_roc_auc:.4f})."
            )

        if cand_f1 < self.min_f1_score:
            reasons.append(
                f"Candidate F1-Score ({cand_f1:.4f}) is below minimum acceptable threshold ({self.min_f1_score:.4f})."
            )

        # 2. Baseline Comparison Checks (Regression Prevention)
        baseline = self.load_active_baseline_metrics()
        if baseline:
            base_roc = float(baseline.get("roc_auc", 0.0))
            if cand_roc < (base_roc - self.max_performance_drop):
                reasons.append(
                    f"Performance degradation: Candidate ROC-AUC ({cand_roc:.4f}) dropped more than "
                    f"{self.max_performance_drop*100:.1f}% compared to active baseline ({base_roc:.4f})."
                )

        decision = QualityGateDecision.REJECT if reasons else QualityGateDecision.PROMOTE

        return {
            "decision": decision,
            "candidate_name": candidate_name,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline,
            "reasons": reasons if reasons else ["Candidate model passed all quality constraints. Approved for production promotion."]
        }

    def evaluate(self, candidate_name: str, candidate_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Alias for pipeline runner.
        """
        return self.evaluate_candidate(candidate_name, candidate_metrics)

    def run(self, candidate_name: str, candidate_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Pipeline runner alias.
        """
        return self.evaluate_candidate(candidate_name, candidate_metrics)