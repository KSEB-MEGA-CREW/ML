import tensorflow as tf
import asyncio
from app.utils.s3_client import S3Client
from app.models.model_cache import ModelCache
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.model = None
        self.s3_client = S3Client()
        self.cache = ModelCache()
        self.model_loaded = False

    async def load_model(self) -> bool:
        """S3에서 h5 모델을 로드(캐시 우선 사용)"""
        try:
            model_path = await self._get_model_path()
            if not model_path:
                logger.error("모델 로드 실패")
                return False

            # Tensorflow/Keras 모델 로드
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_h5_model, model_path
            )

            self.model_loaded = True
            logger.info("frameToGloss 모델 로드 완료")
            return True

        except Exception as e:
            logger.info(f"모델 로드 중 오류: {e}")
            return False

    def _load_h5_model(self, model_path: str):
        """TensorFlow 2.16.1 기준 Keras 모델 로드"""
        try:
            # TensorFlow 2.16.1에서 권장되는 방식
            model = tf.keras.models.load_model(
                model_path, compile=False  # 컴파일 없이 로드 (추론만 사용)
            )

            logger.info(f"모델 로드 완료: {model_path}")
            logger.info(f"모델 구조: {model.summary()}")

            return model

        except Exception as e:
            logger.error(f"Keras 모델 로드 실패: {e}")
            raise

    async def _get_model_path(self) -> str:
        """모델 파일 경로 획득 (캐시 확인 후 S3 다운로드)"""
        s3_key = settings.MODEL_S3_KEY

        # 1. 캐시 확인
        if self.cache.is_cached(s3_key):
            cache_path = self.cache.get_cache_path(s3_key)
            logger.info(f"캐시된 모델 사용: {cache_path}")
            return cache_path

        # 2. S3에서 다운로드
        logger.ingfo(f"S3에서 모델 다운로드 시작: {s3_key}")

        # 3. S3 파일 존재 확인
        if not await self.s3_client.check_file_exists(s3_key):
            logger.error(f"S3에 모델 파일이 없음: {s3_key}")
            return None

        # 4. download
        cache_path = self.cache.get_cache_path(s3_key)
        success = await self.s3_client.download_file(s3_key, cache_path)

        return cache_path if success else None

    def predict(self, input_data):
        """모델 예측 (TensorFlow 2.16.1 방식)"""
        if not self.is_ready():
            raise RuntimeError("모델이 로드되지 않았습니다")

        try:
            # TensorFlow 2.16.1에서 권장되는 예측 방식
            predictions = self.model(input_data, training=False)
            return predictions.numpy()  # numpy 배열로 변환

        except Exception as e:
            logger.error(f"예측 중 오류: {e}")
            raise

    def is_ready(self) -> bool:
        """모델이 로드되어 사용 가능한지 확인"""
        return self.model_loaded and self.model is not None

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        if not self.is_ready():
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
        }
