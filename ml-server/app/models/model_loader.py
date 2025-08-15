import tensorflow as tf
import asyncio
import threading
from app.utils.s3_client import S3Client
from app.models.model_cache import ModelCache
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            self.model = None
            self.s3_client = S3Client()
            self.cache = ModelCache()
            self.model_loaded = False
            self._configure_tensorflow()
            ModelManager._initialized = True

    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal performance"""
        try:
            # GPU memory growth
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available: {len(gpus)} devices")
            else:
                logger.info("Running in CPU mode")

            # Suppress unnecessary logs
            tf.get_logger().setLevel("ERROR")

            # Set thread configuration for better performance
            tf.config.threading.set_inter_op_parallelism_threads(settings.MAX_WORKERS)
            tf.config.threading.set_intra_op_parallelism_threads(settings.MAX_WORKERS)

        except Exception as e:
            logger.warning(f"TensorFlow configuration warning: {e}")

    async def load_model(self) -> bool:
        """Load model from S3 with caching"""
        try:
            model_path = await self._get_model_path()
            if not model_path:
                logger.error("Failed to get model path")
                return False

            # Load TensorFlow model asynchronously
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_h5_model, model_path
            )

            self.model_loaded = True
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False

    def _load_h5_model(self, model_path: str):
        """Load TensorFlow Keras model"""
        try:
            model = tf.keras.models.load_model(
                model_path, compile=False  # Load without compilation for inference only
            )

            logger.info(f"Model loaded from: {model_path}")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")

            return model

        except Exception as e:
            logger.error(f"Keras model loading failed: {e}")
            raise

    async def _get_model_path(self) -> str:
        """Get model file path (check cache then download from S3)"""
        s3_key = settings.MODEL_S3_KEY

        # Check cache first
        if self.cache.is_cached(s3_key):
            cache_path = self.cache.get_cache_path(s3_key)
            logger.info(f"Using cached model: {cache_path}")
            return cache_path

        # Download from S3
        logger.info(f"Downloading model from S3: {s3_key}")

        # Check if file exists in S3
        if not await self.s3_client.check_file_exists(s3_key):
            logger.error(f"Model file not found in S3: {s3_key}")
            return None

        # Download
        cache_path = self.cache.get_cache_path(s3_key)
        success = await self.s3_client.download_file(s3_key, cache_path)

        return cache_path if success else None

    async def predict(self, input_data):
        """Async model prediction"""
        if not self.is_ready():
            raise RuntimeError("Model not loaded")

        try:
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None, self._sync_predict, input_data
            )
            return predictions

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def _sync_predict(self, input_data):
        """Synchronous prediction wrapper"""
        predictions = self.model(input_data, training=False)
        return predictions.numpy()

    def is_ready(self) -> bool:
        """Check if model is ready for use"""
        return self.model_loaded and self.model is not None

    def is_loaded(self) -> bool:
        """Compatibility alias"""
        return self.is_ready()

    def get_model_info(self) -> dict:
        """Return model information"""
        if not self.is_ready():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
        }

    def unload_model(self):
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            tf.keras.backend.clear_session()
            logger.info("Model unloaded")
