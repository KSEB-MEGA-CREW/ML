import logging
import sys
from app.core.config import settings


def setup_logger():
    """Setup application logging"""

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    root_logger.addHandler(console_handler)

    # Prevent duplicate logs
    root_logger.propagate = False

    # Configure specific loggers
    loggers = ["app", "uvicorn.access", "uvicorn.error"]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        logger.addHandler(console_handler)
        logger.propagate = False
