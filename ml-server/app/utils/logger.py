import logging
import sys
from ..config import settings


def setup_logger():
    """Enhanced logging setup with structured format"""
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler for debug mode
    handlers = [console_handler]
    if settings.DEBUG:
        try:
            file_handler = logging.FileHandler("/app/logs/app.log", mode="a")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except (OSError, PermissionError):
            # Fallback if file logging fails
            pass

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()), handlers=handlers
    )

    # Adjust specific logger levels
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
