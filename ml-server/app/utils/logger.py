import logging
import sys
from ..config import settings


def setup_logger():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            (
                logging.FileHandler("/app/logs/app.log", mode="a")
                if settings.DEBUG
                else logging.NullHandler()
            ),
        ],
    )

    # 특정 로거 레벨 조정
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
