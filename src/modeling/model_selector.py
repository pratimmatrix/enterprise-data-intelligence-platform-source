import logging

import pandas as pd


logger = logging.getLogger(__name__)


class ModelSelector:

    def __init__(
        self,
        strategy="f1",
        minimum_recall=None
    ):
        """
        Select the final ML model based on business strategy.

        Strategies:
            - "f1"       : Select highest F1 score
            - "recall"   : Select highest recall
            - "precision" : Select highest precision
            - "roc_auc"  : Select highest ROC-AUC

        minimum_recall:
            Optional minimum recall requirement.
            Models below this threshold are excluded.
        """

        self.strategy = strategy
        self.minimum_recall = minimum_recall

        self.selected_model = None
        self.selection_results = None

        print("ModelSelector initialized.")

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    def _validate_results(self, results):

        if results is None:
            raise ValueError(
                "Model comparison results are None."
            )

        if not isinstance(results, pd.DataFrame):
            raise TypeError(
                "Model comparison results must be "
                "a pandas DataFrame."
            )

        required_columns = {
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        }

        missing_columns = (
            required_columns
            - set(results.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required model metrics: "
                f"{sorted(missing_columns)}"
            )

        if results.empty:
            raise ValueError(
                "Model comparison results are empty."
            )

    # ============================================================
    # SELECT MODEL
    # ============================================================

    def select(self, results):

        print()
        print("=" * 70)
        print(
            "                    MODEL SELECTION"
        )
        print("=" * 70)

        self._validate_results(results)

        candidates = results.copy()

        # --------------------------------------------------------
        # APPLY MINIMUM RECALL CONSTRAINT
        # --------------------------------------------------------

        if self.minimum_recall is not None:

            print()
            print(
                f"Minimum recall requirement: "
                f"{self.minimum_recall:.2%}"
            )

            candidates = candidates[
                candidates["recall"]
                >= self.minimum_recall
            ]

            if candidates.empty:

                raise ValueError(
                    "No model satisfies the "
                    "minimum recall requirement."
                )

        # --------------------------------------------------------
        # SELECT BASED ON STRATEGY
        # --------------------------------------------------------

        strategy_columns = {
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
            "roc_auc": "roc_auc"
        }

        if self.strategy not in strategy_columns:

            raise ValueError(
                f"Unsupported selection strategy: "
                f"{self.strategy}. "
                f"Choose from: "
                f"{list(strategy_columns.keys())}"
            )

        metric = strategy_columns[
            self.strategy
        ]

        selected_row = (
            candidates
            .sort_values(
                by=metric,
                ascending=False
            )
            .iloc[0]
        )

        self.selected_model = (
            selected_row["model"]
        )

        self.selection_results = (
            selected_row.to_dict()
        )

        # --------------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------------

        print()
        print(
            f"Selection strategy: "
            f"{self.strategy.upper()}"
        )

        print()
        print(
            "Candidate models:"
        )

        display_columns = [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]

        print(
            candidates[
                display_columns
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "---------- SELECTED MODEL ----------"
        )

        print(
            f"Model     : "
            f"{self.selected_model}"
        )

        print(
            f"Accuracy  : "
            f"{selected_row['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{selected_row['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{selected_row['recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{selected_row['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{selected_row['roc_auc']:.4f}"
        )

        print()
        print(
            "Model selection completed."
        )

        return self.selected_model

    # ============================================================
    # GET SELECTION DETAILS
    # ============================================================

    def get_selection_details(self):

        if self.selection_results is None:

            raise RuntimeError(
                "No model has been selected yet."
            )

        return self.selection_results