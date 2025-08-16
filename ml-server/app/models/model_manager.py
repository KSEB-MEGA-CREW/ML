import os
import json
import logging
from typing import Optional, Dict, List
import tensorflow as tf
import keras
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.labels = None
        self.is_loaded = False
        self._initialized = True

        logger.info("ModelManager initialized")

    async def load_model(self) -> bool:
        """load tensorflow model and lables from local files"""
        try:
            from app.core.config import settings

            # load labels first
            if not os.path.exists(settings.LABELS_PATH):
                logger.error(f"label file not found: {settings.LABELS_PATH}")
                return False

            with open(settings.LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

            # load model
            if not os.path.exists(settings.MODEL_PATH):
                logger.error(f"model file not found: {settings.MODEL_PATH}")
                return False

            logger.info(f"loading model from: {settings.MODEL_PATH}")
            self.model = keras.models.load_model(settings.MODEL_PATH)

            self.is_loaded = True

            logger.info(
                f"model and label loaded successfully. input shape: {self.model.input_shape}"
            )

            return True

        except Exception as e:
            logger.error(f"model loading failed: {e}")
            self.is_loaded = False
            return False

    def predict(self, keypoints_sequence: List[List[float]]) -> Dict:
        """predict sign language from keypoints sequence"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("model not loaded")

        try:
            # convert to numpy array and reshape for model input
            # expected input shape: (1, 10, 194)
            features = np.array(keypoints_sequence).reshape(1, 10, 194)

            # predict
            predictions = self.model.predict(features, verbose=0)

            # get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # get label
            label = self.labels.get(str(predicted_class), "Unknown")

            return {
                "label": label,
                "confidence": confidence,
                "class_id": int(predicted_class),
            }

        except Exception as e:
            logger.error(f"prediction error: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.is_loaded and self.model is not None

    def unload_model(self):
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.labels is not None:
            del self.labels
            self.labels = None

        self.is_loaded = False
        logger.info("Model unloaded")

    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "input_shape": str(self.model.input_shape) if self.model else None,
            "output_shape": str(self.model.output_shape) if self.model else None,
            "num_classes": len(self.labels) if self.labels else 0,
        }
