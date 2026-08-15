"""
Enterprise Data Loader

Author: Pratim Mistry

Description:
A professional data loader that automatically detects
the file type and loads datasets into a Pandas DataFrame.
"""

from pathlib import Path
from typing import Union
import logging
import pandas as pd

# Safe logger setup with fallback
try:
    from src.utils.logger import setup_logger
    logger = setup_logger()
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DataLoader")


class DataLoader:
    """
    Enterprise Data Loader

    Supports:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Parquet (.parquet)
    """

    def __init__(self):

        logger.info("DataLoader initialized.")


    # ========================================================
    # MAIN INGESTION DISPATCHER
    # ========================================================

    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Automatically detect file type and load dataset.

        Parameters
        ----------
        file_path : str or Path
            Path of the dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(path)

        elif suffix in [".xlsx", ".xls"]:
            return self._load_excel(path)

        elif suffix == ".parquet":
            return self._load_parquet(path)

        else:
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )


    # ========================================================
    # INTERNAL FORMAT LOADERS
    # ========================================================

    def _load_csv(self, path: Path) -> pd.DataFrame:
        """
        Internal method for loading CSV files with auto-delimiter sniffing.
        """

        logger.info(f"Loading CSV file: {path.name}")

        try:
            # First attempt with auto delimiter detection
            return pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
        except Exception:
            # Fallback to standard comma separated
            return pd.read_csv(path, encoding_errors="replace")


    def _load_excel(self, path: Path) -> pd.DataFrame:
        """
        Internal method for loading Excel files.
        """

        logger.info(f"Loading Excel file: {path.name}")

        return pd.read_excel(path)


    def _load_parquet(self, path: Path) -> pd.DataFrame:
        """
        Internal method for loading Parquet files.
        """

        logger.info(f"Loading Parquet file: {path.name}")

        return pd.read_parquet(path)