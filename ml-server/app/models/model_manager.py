import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.model: Optional[tf.keras.Model] = None
            self.labels: Optional[Dict] = None
            self.is_loaded = False
            self.initialized = True

    async def load_model(self) -> bool:
        """Load TensorFlow model and labels"""
        try:
            # 실제 파일 경로로 수정
            model_path = os.path.join("models", "gesture_model.h5")
            labels_path = os.path.join("models", "label_map.json")

            # 파일 존재 확인
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found: {labels_path}")

            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Loading labels from: {labels_path}")

            # 모델 로딩
            self.model = tf.keras.models.load_model(model_path)

            # 라벨 로딩 - 배열을 인덱스:라벨 딕셔너리로 변환
            with open(labels_path, "r", encoding="utf-8") as f:
                label_list = json.load(f)

            # 배열을 딕셔너리로 변환 (인덱스 -> 라벨)
            self.labels = {str(i): label for i, label in enumerate(label_list)}

            self.is_loaded = True

            logger.info(
                f"✅ Model loaded successfully. "
                f"Input shape: {self.model.input_shape}, "
                f"Output shape: {self.model.output_shape}, "
                f"Labels: {len(self.labels)} classes"
            )

            # 로딩된 라벨들 출력
            logger.info(f"Available labels: {list(self.labels.values())}")

            return True

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(
                f"Files in models/: {os.listdir('models') if os.path.exists('models') else 'models directory not found'}"
            )
            self.is_loaded = False
            return False

    def predict(self, keypoints_sequence: List[List[float]]) -> Dict:
        """모델 예측: 확률 벡터 출력 후 argmax로 클래스 선택"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # (1, 10, 194) 형태로 변환
            features = np.array(keypoints_sequence).reshape(1, 10, 194)

            # 모델 예측 - 확률 벡터 반환
            probability_vector = self.model.predict(features, verbose=0)

            # argmax로 최대 확률의 인덱스 찾기
            predicted_class_index = np.argmax(probability_vector[0])
            confidence = float(probability_vector[0][predicted_class_index])

            # 라벨 매핑
            label = self.labels.get(str(predicted_class_index), "Unknown")

            return {
                "label": label,
                "confidence": confidence,
                "class_id": int(predicted_class_index),
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def is_ready(self) -> bool:
        return self.is_loaded and self.model is not None

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None

        if self.labels is not None:
            del self.labels
            self.labels = None

        self.is_loaded = False
        logger.info("Model unloaded")

    def get_model_info(self) -> Dict:
        if not self.is_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "input_shape": str(self.model.input_shape) if self.model else None,
            "output_shape": str(self.model.output_shape) if self.model else None,
            "num_classes": len(self.labels) if self.labels else 0,
            "labels": list(self.labels.values()) if self.labels else [],
        }
