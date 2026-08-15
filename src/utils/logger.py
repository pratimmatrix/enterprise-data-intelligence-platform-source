"""
Application Logger Configuration

Author: Pratim Mistry
Description:
Configures centralized logging formatting and stream handling 
for the enterprise platform with duplicate handler protection.
"""

import logging
from typing import Optional, Union


def setup_logger(
    name: str = "EnterpriseDataPlatform",
    level: Union[int, str] = logging.INFO
) -> logging.Logger:
    """
    Configure and return a centralized application logger.

    Parameters
    ----------
    name : str, optional
        Logger name identifier, by default "EnterpriseDataPlatform"
    level : Union[int, str], optional
        Logging severity level, by default logging.INFO

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    # Set log level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Formatter configuration
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Convenience helper to retrieve an existing configured logger.
    """
    if name:
        return logging.getLogger(f"EnterpriseDataPlatform.{name}")
    return logging.getLogger("EnterpriseDataPlatform")