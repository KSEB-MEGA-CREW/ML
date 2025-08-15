# models/model_loader.py
import tensorflow as tf
import asyncio
import os
from typing import Optional
from app.utils.s3_client import S3Client
from app.models.model_cache import ModelCache
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.s3_client = S3Client()
        self.cache = ModelCache()
        self.model_loaded = False
        self._configure_tensorflow()
        self._initialized = True

    def _configure_tensorflow(self):
        """Configure TensorFlow optimization"""
        try:
            # Enable GPU memory growth
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available: {len(gpus)} devices")
            else:
                logger.info("Running in CPU mode")

            # Suppress unnecessary logs
            tf.get_logger().setLevel("ERROR")

        except Exception as e:
            logger.warning(f"TensorFlow configuration warning: {e}")

    async def load_model(self) -> bool:
        """Load model (local files first, S3 as backup)"""
        if self.model_loaded:
            logger.info("Model already loaded")
            return True

        try:
            # 1. Check local files first
            local_model_paths = [
                "/app/models/gesture_model.h5",
                "./models/gesture_model.h5",
                "gesture_model.h5",
            ]

            for local_path in local_model_paths:
                if os.path.exists(local_path):
                    logger.info(f"Using local model file: {local_path}")
                    self.model = await self._load_h5_model(local_path)
                    self.model_loaded = True
                    return True

            # 2. Download from S3
            logger.info("No local model found, attempting S3 download")
            model_path = await self._get_model_path()
            if not model_path:
                logger.error("Model loading failed")
                return False

            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_h5_model, model_path
            )

            self.model_loaded = True
            logger.info("S3 model loading completed")
            return True

        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            return False

    async def _load_h5_model(self, model_path: str):
        """Load TensorFlow model"""
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Model loaded successfully: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Keras model loading failed: {e}")
            raise

    async def _get_model_path(self) -> Optional[str]:
        """Download model from S3"""
        s3_key = settings.MODEL_S3_KEY

        # Check cache
        if self.cache.is_cached(s3_key):
            cache_path = self.cache.get_cache_path(s3_key)
            logger.info(f"Using cached model: {cache_path}")
            return cache_path

        # Check S3 file existence
        if not await self.s3_client.check_file_exists(s3_key):
            logger.error(f"Model file not found in S3: {s3_key}")
            return None

        # Download
        cache_path = self.cache.get_cache_path(s3_key)
        success = await self.s3_client.download_file(s3_key, cache_path)

        return cache_path if success else None

    def predict(self, input_data):
        """Model prediction"""
        if not self.is_ready():
            raise RuntimeError("Model is not loaded")

        try:
            predictions = self.model(input_data, training=False)
            return predictions.numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def is_ready(self) -> bool:
        return self.model_loaded and self.model is not None

    def get_model_info(self) -> dict:
        if not self.is_ready():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
        }

    def unload_model(self):
        """Unload model"""
        if self.model:
            del self.model
            self.model = None
            self.model_loaded = False
            logger.info("Model unloaded successfully")
