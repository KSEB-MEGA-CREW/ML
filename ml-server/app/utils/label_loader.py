# utils/label_loader.py
import json
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class LabelLoader:
    _instance: Optional["LabelLoader"] = None
    _labels_cache: Optional[List[str]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def load_class_labels(self) -> Optional[List[str]]:
        """Load class labels (with caching)"""
        if self._labels_cache is not None:
            return self._labels_cache

        try:
            # Get paths from settings
            from app.config import settings

            for path in settings.LABEL_PATHS:
                if os.path.exists(path):
                    logger.info(f"Loading label file: {path}")
                    with open(path, "r", encoding="utf-8") as f:
                        labels_data = json.load(f)

                    if isinstance(labels_data, list):
                        self._labels_cache = labels_data
                        logger.info(
                            f"Labels loaded successfully: {len(self._labels_cache)} classes"
                        )
                        return self._labels_cache

            # Use default labels
            logger.warning("Label file not found, using default labels")
            self._labels_cache = [
                "좋다1",
                "지시1#",
                "돕다1",
                "무엇1",
                "지시2",
                "때2",
                "오늘1",
                "일하다1",
                "재미1",
                "필요1",
                "회사1",
                "요리1",
                "괜찮다1",
                "잘하다2",
            ]
            return self._labels_cache

        except Exception as e:
            logger.error(f"Label loading failed: {e}")
            return None

    async def validate_model_compatibility(self, expected_num_classes: int) -> bool:
        """Validate model-label compatibility"""
        try:
            labels = await self.load_class_labels()
            if labels is None:
                logger.error("Labels not loaded, cannot validate compatibility")
                return False

            actual_num_classes = len(labels)

            if actual_num_classes != expected_num_classes:
                logger.warning(
                    f"Model-label mismatch: model expects {expected_num_classes} classes, "
                    f"but {actual_num_classes} labels loaded"
                )
                return False

            logger.info(
                f"Model-label compatibility verified: {actual_num_classes} classes"
            )
            return True

        except Exception as e:
            logger.error(f"Model compatibility validation failed: {e}")
            return False

    def clear_cache(self):
        """Clear cache"""
        self._labels_cache = None
        logger.info("Label cache cleared")
