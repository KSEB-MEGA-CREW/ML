import logging
from typing import Dict, List, Optional
from pathlib import Path
from ..utils.file_loader import FileLoader

logger = logging.getLogger(__name__)


class LabelLoader:
    """simple string label loading"""

    def __init__(self, json_path: str = "ml-server/models/label_map.json"):
        self.json_path = json_path
        self.labels: Dict[int, str] = {}
        self.label_list: List[str] = []
        self.loaded = False

    def load_labels(self) -> bool:
        """load label mapping"""

        # try JSON file
        if self._load_from_json_array():
            logger.info("JSON array file label loaded")
            self.loaded = True
            return True

        logger.error("label loading failed")
        return False

    def _load_from_json_array(self) -> bool:
        """JSON array file label load"""
        # alternate paths
        json_paths = [self.json_path, "../../models/label_map.json"]

        for path in json_paths:
            if FileLoader.validate_file_exists(path):
                data = FileLoader.load_json(path)
                if data and self._parse_json_array(data):
                    logger.info(f"JSON label file: {path}")
                    return True
        return False

    def _parse_json_array(self, data) -> bool:
        """parsing JSON array data"""
        try:
            # check if it is array
            if not isinstance(data, list):
                logger.error("JSON file is not array")
                return False

            # check if it is empty
            if len(data) == 0:
                logger.error("JSON file is empty")
                return False

            self.label_list = [str(label).strip() for label in data]

            self.labels = {}
            for i, label in enumerate(self.label_list):
                if label:  # not empty label
                    self.labels[i] = label

            logger.info(f"JSON file {len(self.labels)} loaded")
            logger.debug(
                f"label: {self.label_list[:5]}{'...' if len(self.label_list) > 5 else ''}"
            )

            return len(self.labels) > 0

        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return False

    def get_all_labels(self) -> Dict[int, str]:
        """return labels"""
        return self.labels.copy()

    def get_label_list(self) -> List[str]:
        """return label_list"""
        return self.label_list.copy()

    def reload_labels(self) -> bool:
        """reload labels"""
        logger.info("reload labels starting")
        self.labels.clear()
        self.label_list.clear()
        self.loaded = False
        return self.load_labels()
