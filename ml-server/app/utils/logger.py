import logging
import sys
from typing import Optional


def setup_logger(
    name: str, level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """로거 설정"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 핸들러가 이미 있으면 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 포맷터 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택사항)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
