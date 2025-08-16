import tensorflow as tf
import os
import time
import logging
from typing import Optional, List, Dict, Any
from .label_loader import LabelLoader

logger = logging.getLogger(__name__)


class ModelManager:
    """local h5 file and JSON array label model manager"""

    def __init__(
        self,
        model_path: str = "ml-server/models/gesture_model.h5",
        labels_json_path: str = "ml-server/models/label_map.json",
    ):
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.model_loaded = False
        self.load_time = None

        # initialize label loader
        self.label_loader = LabelLoader(labels_json_path)

        self._configure_tensorflow()

    def _configure_tensorflow(self):
        """TensorFlow optimization settings"""
        try:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU: {len(gpus)}")
            else:
                logger.info("CPU mode")

            tf.get_logger().setLevel("ERROR")

        except Exception as e:
            logger.warning(f"TensorFlow settings error: {e}")

    async def load_model(self) -> bool:
        """load model and label"""
        if self.model_loaded:
            logger.info("model loaded")
            return True

        start_time = time.time()

        try:
            # load label first
            if not self.label_loader.load_labels():
                logger.error("label load failed")
                return False

            logger.info(f"label loaded: {self.label_loader.get_label_list}")

            # check model file and load
            if not self._find_and_load_model():
                return False

            # model warm up
            dummy_input = tf.random.normal((1, 10, 194))
            _ = self.model(dummy_input, training=False)

            self.model_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"model and label loaded! ({self.load_time} s)")
            return True

        except Exception as e:
            logger.error(f"model loading fail: {e}")
            return False

    def _find_and_load_model(self) -> bool:
        """find and load model file"""
        model_paths = [
            self.model_path,
            "./models/sign_language_model.h5",
            "models/sign_language_model.h5",
        ]

        for path in model_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"📦 모델 로딩 시작: {path}")

                    # 파일 크기 확인
                    file_size_mb = os.path.getsize(path) / (1024 * 1024)
                    logger.info(f"모델 파일 크기: {file_size_mb:.2f} MB")

                    # 모델 로드
                    self.model = tf.keras.models.load_model(path, compile=False)
                    self.model_path = path

                    logger.info(f"model input size: {self.model.input_shape}")
                    logger.info(f"model output size: {self.model.output_shape}")
                    logger.info(f"parameter: {self.model.count_params():,}개")

                    return True

                except Exception as e:
                    logger.error(f"model load failed ({path}): {e}")
                    continue

        logger.error("can not find model file")
        return False

    async def predict(self, input_data: tf.Tensor) -> Dict[str, Any]:
        """model predict"""
        if not self.is_loaded():
            raise RuntimeError("model Unloaded")

        try:
            start_time = time.time()

            # predict
            predictions = self.model(input_data, training=False)

            # predict result
            predicted_class = int(tf.argmax(predictions[0]))
            confidence = float(tf.reduce_max(predictions[0]))

            # get label
            label = self.label_loader.get_label(predicted_class)

            inference_time = time.time() - start_time

            result = {
                "label": label,
                "confidence": confidence,
                "predicted_class": predicted_class,
                "inference_time": inference_time,
                "timestamp": time.time(),
            }

            return result

        except Exception as e:
            logger.error(f"predict failed: {e}")
            raise

    def is_loaded(self) -> bool:
        """check loading"""
        return (
            self.model_loaded
            and self.model is not None
            and self.label_loader.is_loaded()
        )

    def get_model_info(self) -> Dict[str, Any]:
        """return model and label info"""
        if not self.is_loaded():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_path": self.model_path,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
            "load_time": self.load_time,
            "file_size_mb": round(os.path.getsize(self.model_path) / (1024 * 1024), 2),
            "labels": {
                "total_classes": self.label_loader.get_total_classes(),
                "metadata": self.label_loader.get_metadata(),
                "all_labels": self.label_loader.get_label_list(),
            },
        }

    def get_all_labels(self) -> List[str]:
        """return all labels"""
        return self.label_loader.get_label_list()

    def unload_model(self):
        """model unloaded"""
        if self.model:
            del self.model
            self.model = None
            self.model_loaded = False
            logger.info("model unloading completed")

    async def reload_model(self) -> bool:
        """reloading"""
        logger.info("start reloading")
        self.unload_model()
        return await self.load_model()
