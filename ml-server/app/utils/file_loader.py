import json
import os
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FileLoader:
    """local file loading utility"""

    @staticmethod
    def load_json(file_path: str, encoding: str = "utf-8") -> Optional[Dict]:
        """load json file"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"can not find JSON file: {file_path}")
                return None

            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)

            logger.info(f"load JSON file: {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"parsing JSON error: {file_path} - {e}")
            return None
        except Exception as e:
            logger.error(f"JSON file load filed: {file_path} - {e}")
            return None

    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """check file exists"""
        exists = os.path.exists(file_path)
        if not exists:
            logger.warning(f"file does not exist: {file_path}")
        return exists

    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Union[str, int, float]]:
        """파일 정보 반환"""
        if not os.path.exists(file_path):
            return {"exists": False}

        stat = os.stat(file_path)
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "is_readable": os.access(file_path, os.R_OK),
        }
