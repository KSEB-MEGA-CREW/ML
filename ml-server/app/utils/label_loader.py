# label data loader -> cache or take from S3 bucket
import json
import os
from typing import List, Dict, Any
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class LabelLoader:
    def __init__(self):
        self._labels_cache = None

    async def load_class_labels(self) -> List[str]:
        """Load class labels with caching"""
        if self._labels_cache is not None:
            return self._labels_cache

        try:
            labels_data = await self._load_from_local()

            # Extract class list from JSON data
            if isinstance(labels_data, list):
                self._labels_cache = labels_data
            elif isinstance(labels_data, dict) and "labels" in labels_data:
                self._labels_cache = labels_data["labels"]
            else:
                raise ValueError("Invalid label file format")

            logger.info(f"Class labels loaded: {len(self._labels_cache)} classes")
            logger.debug(f"Loaded labels: {self._labels_cache}")

            return self._labels_cache

        except Exception as e:
            logger.error(f"Failed to load labels: {str(e)}")
            return []

    async def _load_from_local(self) -> Dict[str, Any]:
        """Load labels from local file"""
        try:
            if not os.path.exists(settings.CLASS_LABELS_FILE):
                raise FileNotFoundError(
                    f"Label file not found: {settings.CLASS_LABELS_FILE}"
                )

            with open(settings.CLASS_LABELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load labels from local: {str(e)}")
            raise

    async def validate_model_compatibility(self, model_num_classes: int) -> bool:
        """Validate compatibility between model and labels"""
        labels = await self.load_class_labels()
        if len(labels) != model_num_classes:
            logger.warning(
                f"Model expects {model_num_classes} classes but labels have {len(labels)}"
            )
            return False
        return True

    def clear_cache(self):
        """Clear label cache"""
        self._labels_cache = None
        logger.info("Label cache cleared")
