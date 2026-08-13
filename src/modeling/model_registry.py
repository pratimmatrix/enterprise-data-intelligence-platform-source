import json
from pathlib import Path
from datetime import datetime


class ModelRegistry:

    def __init__(self, model_directory="models"):

        self.model_directory = Path(model_directory)

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.registry_path = (
            self.model_directory / "model_registry.json"
        )

        print("ModelRegistry initialized.")

    # ========================================================
    # REGISTER MODEL
    # ========================================================

    def register_model(
        self,
        model_name,
        model_path,
        metrics,
        selection_strategy,
        selected=False
    ):

        # ----------------------------------------------------
        # Load existing registry
        # ----------------------------------------------------

        registry = self.load_registry()

        # ----------------------------------------------------
        # Create model record
        # ----------------------------------------------------

        model_record = {
            "model_name": model_name,
            "model_path": str(model_path),
            "selection_strategy": selection_strategy,
            "selected": selected,
            "metrics": {
                "accuracy": float(
                    metrics["accuracy"]
                ),
                "precision": float(
                    metrics["precision"]
                ),
                "recall": float(
                    metrics["recall"]
                ),
                "f1": float(
                    metrics["f1"]
                ),
                "roc_auc": float(
                    metrics["roc_auc"]
                )
            },
            "registered_at": (
                datetime.now().isoformat()
            )
        }

        # ----------------------------------------------------
        # If selected, remove previous selection
        # ----------------------------------------------------

        if selected:

            for record in registry["models"]:

                record["selected"] = False

            registry["selected_model"] = model_name

            registry["selected_model_path"] = str(
                model_path
            )

        # ----------------------------------------------------
        # Replace existing model record
        # ----------------------------------------------------

        registry["models"] = [
            record
            for record in registry["models"]
            if record["model_name"] != model_name
        ]

        registry["models"].append(
            model_record
        )

        # ----------------------------------------------------
        # Save registry
        # ----------------------------------------------------

        self.save_registry(
            registry
        )

        print(
            f"Registered model: {model_name}"
        )

    # ========================================================
    # LOAD REGISTRY
    # ========================================================

    def load_registry(self):

        if not self.registry_path.exists():

            return {
                "selected_model": None,
                "selected_model_path": None,
                "models": []
            }

        with open(
            self.registry_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # ========================================================
    # SAVE REGISTRY
    # ========================================================

    def save_registry(self, registry):

        with open(
            self.registry_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                registry,
                file,
                indent=4
            )

        print(
            f"Registry saved: "
            f"{self.registry_path}"
        )

    # ========================================================
    # GET SELECTED MODEL
    # ========================================================

    def get_selected_model(self):

        registry = self.load_registry()

        selected_model = (
            registry.get(
                "selected_model"
            )
        )

        if selected_model is None:

            raise RuntimeError(
                "No model has been selected."
            )

        return selected_model

    # ========================================================
    # GET SELECTED MODEL PATH
    # ========================================================

    def get_selected_model_path(self):

        registry = self.load_registry()

        model_path = (
            registry.get(
                "selected_model_path"
            )
        )

        if model_path is None:

            raise RuntimeError(
                "No selected model path exists."
            )

        return model_path

    # ========================================================
    # DISPLAY REGISTRY
    # ========================================================

    def display_registry(self):

        registry = self.load_registry()

        print()
        print("=" * 70)

        print(
            "                    MODEL REGISTRY"
        )

        print("=" * 70)

        print()

        print(
            f"Selected model : "
            f"{registry.get('selected_model')}"
        )

        print(
            f"Selected path  : "
            f"{registry.get('selected_model_path')}"
        )

        print()

        for model in registry.get(
            "models",
            []
        ):

            print(
                f"Model     : "
                f"{model['model_name']}"
            )

            print(
                f"Selected  : "
                f"{model['selected']}"
            )

            print(
                f"Accuracy  : "
                f"{model['metrics']['accuracy']:.4f}"
            )

            print(
                f"Precision : "
                f"{model['metrics']['precision']:.4f}"
            )

            print(
                f"Recall    : "
                f"{model['metrics']['recall']:.4f}"
            )

            print(
                f"F1        : "
                f"{model['metrics']['f1']:.4f}"
            )

            print(
                f"ROC-AUC   : "
                f"{model['metrics']['roc_auc']:.4f}"
            )

            print(
                "-" * 70
            )