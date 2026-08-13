# ============================================================
# ENTERPRISE DATA INTELLIGENCE PLATFORM
# MAIN APPLICATION PIPELINE
# ============================================================

import logging
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# IMPORTS
# ============================================================

from src.ingestion.loader import DataLoader
from src.ingestion.validator import DataValidator

from src.profiling.profiler import DataProfiler
from src.profiling.anomalies import AnomalyEngine

from src.feature_engineering.feature_engineer import (
    FeatureEngineeringEngine
)

from src.feature_validation.feature_validator import (
    FeatureValidator
)

from src.modeling.model_trainer import ModelTrainer
from src.modeling.model_comparator import ModelComparator
from src.modeling.model_selector import ModelSelector


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = PROJECT_ROOT / "bank-full.csv"

TARGET_COLUMN = "y"

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    """Print a consistent section heading."""

    print()
    print("=" * 70)
    print(f"                    {title}")
    print("=" * 70)


def find_method(obj, possible_names):
    """
    Find the first available callable method from a list.
    """

    for name in possible_names:

        method = getattr(obj, name, None)

        if callable(method):
            return method

    return None


# ============================================================
# DATA INGESTION
# ============================================================

def data_ingestion():

    print_section("DATA INGESTION")

    logger.info("Starting data ingestion...")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_FILE}\n\n"
            f"Make sure bank-full.csv is located in "
            f"the project root."
        )

    # --------------------------------------------------------
    # Initialize loader
    # --------------------------------------------------------

    loader = DataLoader()

    # --------------------------------------------------------
    # Find loader method
    # --------------------------------------------------------

    method = find_method(
        loader,
        [
            "load_csv",
            "load_data",
            "load",
            "read_csv",
            "ingest"
        ]
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if method is None:

        logger.warning(
            "No compatible DataLoader method found. "
            "Using pandas directly."
        )

        df = pd.read_csv(
            DATA_FILE,
            sep=";"
        )

    else:

        try:

            df = method(
                str(DATA_FILE)
            )

        except TypeError:

            try:

                df = method(
                    DATA_FILE
                )

            except TypeError:

                df = method()

    # --------------------------------------------------------
    # Handle tuple return
    # --------------------------------------------------------

    if isinstance(df, tuple):

        dataframe = None

        for item in df:

            if isinstance(
                item,
                pd.DataFrame
            ):

                dataframe = item
                break

        df = dataframe

    # --------------------------------------------------------
    # Handle dictionary return
    # --------------------------------------------------------

    elif isinstance(df, dict):

        dataframe = None

        for value in df.values():

            if isinstance(
                value,
                pd.DataFrame
            ):

                dataframe = value
                break

        df = dataframe

    # --------------------------------------------------------
    # Validate DataFrame
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "DataLoader did not return "
            "a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Bank Marketing dataset normally uses ;
    # --------------------------------------------------------

    if len(df.columns) == 1:

        logger.warning(
            "Dataset appears to contain one column. "
            "Retrying with ';' separator."
        )

        df = pd.read_csv(
            DATA_FILE,
            sep=";"
        )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in dataset."
        )

    print()

    print(
        "Data successfully loaded."
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def data_validation(df):

    print_section("DATA VALIDATION")

    logger.info(
        "Starting data validation..."
    )

    validator = DataValidator()

    method = find_method(
        validator,
        [
            "validate",
            "validate_data",
            "run",
            "check"
        ]
    )

    if method is not None:

        try:

            method(df)

        except TypeError:

            try:

                method()

            except Exception as exc:

                logger.warning(
                    "Data validation warning: %s",
                    exc
                )

        except Exception as exc:

            logger.warning(
                "Data validation warning: %s",
                exc
            )

    # --------------------------------------------------------
    # Universal validation summary
    # --------------------------------------------------------

    print()

    print(
        f"Rows              : {len(df)}"
    )

    print(
        f"Columns           : {len(df.columns)}"
    )

    print()

    print(
        f"Duplicate Rows    : {df.duplicated().sum()}"
    )

    print()

    print(
        "Missing Values"
    )

    print(
        df.isnull().sum()
    )

    print()

    print(
        "Data Types"
    )

    print(
        df.dtypes
    )

    print()

    print(
        "Memory Usage"
    )

    memory_kb = (
        df.memory_usage(
            deep=True
        ).sum()
        / 1024
    )

    print(
        f"{memory_kb:.2f} KB"
    )

    return df


# ============================================================
# DATA PROFILING
# ============================================================

def profiling(df):

    print_section("DATASET PROFILING")

    logger.info(
        "Starting dataset profiling..."
    )

    profiler = DataProfiler()

    method = find_method(
        profiler,
        [
            "profile",
            "profile_data",
            "run",
            "analyze",
            "generate_profile"
        ]
    )

    if method is not None:

        try:

            method(df)

        except TypeError:

            try:

                method()

            except Exception as exc:

                logger.warning(
                    "Profiling warning: %s",
                    exc
                )

        except Exception as exc:

            logger.warning(
                "Profiling warning: %s",
                exc
            )

    else:

        logger.warning(
            "No compatible DataProfiler method found."
        )

    # --------------------------------------------------------
    # Basic profiling
    # --------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            exclude="number"
        ).columns.tolist()
    )

    print()

    print(
        f"Rows             : {len(df)}"
    )

    print(
        f"Columns          : {len(df.columns)}"
    )

    print(
        f"Numeric Columns  : {len(numeric_columns)}"
    )

    print(
        f"Text Columns     : {len(categorical_columns)}"
    )

    return df


# ============================================================
# ANOMALY INTELLIGENCE
# ============================================================

def anomaly_analysis(df):

    print_section("ANOMALY INTELLIGENCE")

    logger.info(
        "Starting anomaly detection..."
    )

    engine = AnomalyEngine()

    method = find_method(
        engine,
        [
            "detect",
            "detect_anomalies",
            "analyze",
            "run",
            "find_anomalies"
        ]
    )

    if method is not None:

        try:

            method(df)

        except TypeError:

            try:

                method()

            except Exception as exc:

                logger.warning(
                    "Anomaly detection warning: %s",
                    exc
                )

        except Exception as exc:

            logger.warning(
                "Anomaly detection warning: %s",
                exc
            )

    else:

        logger.warning(
            "No compatible anomaly method found."
        )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    print_section("FEATURE ENGINEERING")

    logger.info(
        "Starting feature engineering..."
    )

    engine = FeatureEngineeringEngine()

    method = find_method(
        engine,
        [
            "create_features",
            "engineer_features",
            "transform",
            "feature_engineering",
            "run",
            "create"
        ]
    )

    if method is None:

        raise AttributeError(
            "FeatureEngineeringEngine does not expose "
            "a supported feature-engineering method."
        )

    # --------------------------------------------------------
    # Execute feature engineering
    # --------------------------------------------------------

    try:

        result = method(df)

    except TypeError:

        result = method()

    # --------------------------------------------------------
    # DataFrame result
    # --------------------------------------------------------

    if isinstance(
        result,
        pd.DataFrame
    ):

        df = result

    # --------------------------------------------------------
    # Tuple result
    # --------------------------------------------------------

    elif isinstance(
        result,
        tuple
    ):

        dataframe_found = False

        for item in result:

            if isinstance(
                item,
                pd.DataFrame
            ):

                df = item
                dataframe_found = True
                break

        if not dataframe_found:

            raise TypeError(
                "Feature engineering returned a tuple "
                "without a DataFrame."
            )

    # --------------------------------------------------------
    # None means in-place modification
    # --------------------------------------------------------

    elif result is None:

        pass

    # --------------------------------------------------------
    # Unsupported
    # --------------------------------------------------------

    else:

        raise TypeError(
            "Feature engineering returned "
            f"unsupported type: "
            f"{type(result).__name__}"
        )

    print()

    print(
        "Feature engineering completed."
    )

    print(
        "Dataset shape after feature engineering: "
        f"{df.shape}"
    )

    print()

    print(
        "Current columns:"
    )

    for column in df.columns:

        print(
            f"• {column}"
        )

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def feature_validation(df):

    print_section("FEATURE VALIDATION")

    logger.info(
        "Starting feature validation..."
    )

    validator = FeatureValidator()

    method = find_method(
        validator,
        [
            "validate",
            "validate_features",
            "run",
            "check"
        ]
    )

    if method is None:

        logger.warning(
            "No compatible FeatureValidator method found."
        )

        return df

    try:

        method(df)

    except TypeError:

        try:

            method()

        except Exception as exc:

            logger.warning(
                "Feature validation warning: %s",
                exc
            )

    except Exception as exc:

        logger.warning(
            "Feature validation warning: %s",
            exc
        )

    return df


# ============================================================
# BASELINE MODEL
# ============================================================

def baseline_model(df):

    print_section("BASELINE MODEL TRAINING")

    logger.info(
        "Starting baseline model training..."
    )

    trainer = ModelTrainer()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # ModelTrainer.run(df) is the complete training pipeline.
    #
    # It performs:
    #
    # prepare_data()
    # build_preprocessor()
    # split_data()
    # build_model()
    # train()
    # evaluate()
    #
    # Therefore DO NOT call trainer.train(df).
    # --------------------------------------------------------

    try:

        result = trainer.run(
            df
        )

    except Exception as exc:

        logger.exception(
            "Baseline model training failed."
        )

        raise RuntimeError(
            f"Baseline model training failed: {exc}"
        ) from exc

    print()

    print(
        "Baseline model training completed."
    )

    return result


# ============================================================
# MODEL COMPARISON
# ============================================================

def model_comparison(df):

    print_section("MODEL COMPARISON")

    logger.info(
        "Starting model comparison..."
    )

    comparator = ModelComparator()

    # --------------------------------------------------------
    # Run comparator
    # --------------------------------------------------------

    comparison_output = comparator.run(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # ModelComparator from your project returns:
    #
    # {
    #     "results": DataFrame,
    #     "best_model_name": str,
    #     "best_model": pipeline
    # }
    # --------------------------------------------------------

    if isinstance(
        comparison_output,
        dict
    ):

        results_df = (
            comparison_output.get(
                "results"
            )
        )

        best_model_name = (
            comparison_output.get(
                "best_model_name"
            )
        )

        best_model = (
            comparison_output.get(
                "best_model"
            )
        )

    elif isinstance(
        comparison_output,
        pd.DataFrame
    ):

        results_df = comparison_output

        best_model_name = None

        best_model = None

    else:

        raise TypeError(
            "ModelComparator.run() must return "
            "a pandas DataFrame or a dictionary "
            "containing 'results'."
        )

    # --------------------------------------------------------
    # Validate DataFrame
    # --------------------------------------------------------

    if not isinstance(
        results_df,
        pd.DataFrame
    ):

        raise TypeError(
            "Model comparison results must be "
            "a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

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
        - set(results_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Model comparison results are missing "
            f"columns: {sorted(missing_columns)}"
        )

    print()

    print(
        "Model comparison completed."
    )

    print()

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value:
            f"{value:.4f}"
        )
    )

    return {
        "results": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model
    }


# ============================================================
# MODEL SELECTION
# ============================================================

def model_selection(
    comparison_output
):

    print_section("MODEL SELECTION")

    logger.info(
        "Starting model selection..."
    )

    # --------------------------------------------------------
    # Extract DataFrame
    # --------------------------------------------------------

    if isinstance(
        comparison_output,
        dict
    ):

        comparison_results = (
            comparison_output.get(
                "results"
            )
        )

    else:

        comparison_results = (
            comparison_output
        )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # ModelSelector.select() requires a DataFrame.
    # --------------------------------------------------------

    if not isinstance(
        comparison_results,
        pd.DataFrame
    ):

        raise TypeError(
            "Model comparison must provide "
            "a pandas DataFrame under the "
            "'results' key."
        )

    # --------------------------------------------------------
    # Select based on F1
    # --------------------------------------------------------

    selector = ModelSelector(
        strategy="f1"
    )

    selected_model_name = (
        selector.select(
            comparison_results
        )
    )

    selection_details = (
        selector.get_selection_details()
    )

    print()

    print(
        "FINAL MODEL SELECTION"
    )

    print(
        f"Selected model: "
        f"{selected_model_name}"
    )

    print(
        "Selection metric: F1"
    )

    return {
        "selected_model_name":
            selected_model_name,

        "selection_details":
            selection_details
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(
    df,
    comparison_output,
    selection_output
):

    print_section(
        "ENTERPRISE PIPELINE SUMMARY"
    )

    results_df = (
        comparison_output[
            "results"
        ]
    )

    selected_model_name = (
        selection_output[
            "selected_model_name"
        ]
    )

    # --------------------------------------------------------
    # Locate selected model
    # --------------------------------------------------------

    selected_rows = results_df[
        results_df["model"]
        == selected_model_name
    ]

    if selected_rows.empty:

        raise ValueError(
            f"Selected model '{selected_model_name}' "
            "was not found in comparison results."
        )

    selected_row = (
        selected_rows.iloc[0]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        f"Dataset rows       : "
        f"{len(df)}"
    )

    print(
        f"Dataset columns    : "
        f"{len(df.columns)}"
    )

    print()

    print(
        "---------- SELECTED MODEL ----------"
    )

    print(
        f"Model      : "
        f"{selected_model_name}"
    )

    print(
        f"Accuracy   : "
        f"{selected_row['accuracy']:.4f}"
    )

    print(
        f"Precision  : "
        f"{selected_row['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{selected_row['recall']:.4f}"
    )

    print(
        f"F1 Score   : "
        f"{selected_row['f1']:.4f}"
    )

    print(
        f"ROC-AUC    : "
        f"{selected_row['roc_auc']:.4f}"
    )

    print()

    print(
        "Enterprise Data Intelligence "
        "Pipeline completed successfully."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    try:

        print()

        print(
            "=" * 70
        )

        print(
            "       ENTERPRISE DATA INTELLIGENCE PLATFORM"
        )

        print(
            "=" * 70
        )

        print()

        # ----------------------------------------------------
        # 1. DATA INGESTION
        # ----------------------------------------------------

        df = data_ingestion()

        # ----------------------------------------------------
        # 2. DATA VALIDATION
        # ----------------------------------------------------

        df = data_validation(
            df
        )

        # ----------------------------------------------------
        # 3. DATA PROFILING
        # ----------------------------------------------------

        df = profiling(
            df
        )

        # ----------------------------------------------------
        # 4. ANOMALY INTELLIGENCE
        # ----------------------------------------------------

        df = anomaly_analysis(
            df
        )

        # ----------------------------------------------------
        # 5. FEATURE ENGINEERING
        # ----------------------------------------------------

        df = feature_engineering(
            df
        )

        # ----------------------------------------------------
        # 6. FEATURE VALIDATION
        # ----------------------------------------------------

        df = feature_validation(
            df
        )

        # ----------------------------------------------------
        # 7. BASELINE MODEL
        # ----------------------------------------------------

        baseline_result = baseline_model(
            df
        )

        # Prevent unused-variable issues while keeping the
        # baseline result available for future reporting.
        _ = baseline_result

        # ----------------------------------------------------
        # 8. MODEL COMPARISON
        # ----------------------------------------------------

        comparison_output = model_comparison(
            df
        )

        # ----------------------------------------------------
        # 9. MODEL SELECTION
        # ----------------------------------------------------

        selection_output = model_selection(
            comparison_output
        )

        # ----------------------------------------------------
        # 10. FINAL SUMMARY
        # ----------------------------------------------------

        final_summary(
            df,
            comparison_output,
            selection_output
        )

        return 0

    except KeyboardInterrupt:

        print()

        logger.warning(
            "Pipeline interrupted by user."
        )

        return 1

    except Exception as exc:

        logger.exception(
            "Pipeline execution failed."
        )

        print()

        print(
            "ERROR:",
            str(exc)
        )

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )