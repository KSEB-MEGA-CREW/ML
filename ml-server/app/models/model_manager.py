# app/models/model_manager.py
import asyncio
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import tensorflow as tf
import json

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """수어 인식 모델 관리자 (WebSocket용)"""

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.label_map: Optional[Dict[int, str]] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self._model_loaded = False
        self._loading_lock = asyncio.Lock()

    async def load_model(self) -> bool:
        """모델 비동기 로딩"""
        async with self._loading_lock:
            if self._model_loaded:
                return True

            try:
                logger.info("수어 인식 모델 로딩 시작...")

                # ThreadPoolExecutor 초기화
                self.executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="model_"
                )

                # 모델 로딩을 별도 스레드에서 실행
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    self.executor, self._load_model_sync
                )

                if success:
                    self._model_loaded = True
                    logger.info("✅ 모델 로딩 완료")
                else:
                    logger.error("❌ 모델 로딩 실패")

                return success

            except Exception as e:
                logger.error(f"모델 로딩 중 오류: {e}")
                return False

    def _load_model_sync(self) -> bool:
        """동기식 모델 로딩 (별도 스레드에서 실행)"""
        try:
            # TensorFlow 모델 로딩
            self.model = tf.keras.models.load_model(settings.model_path)
            logger.info(f"모델 로딩 완료: {settings.model_path}")

            # 라벨 맵 로딩
            with open(settings.labels_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            logger.info(f"라벨 맵 로딩 완료: {len(self.label_map)}개 클래스")

            # 모델 워밍업 (첫 번째 예측이 느린 것을 방지)
            dummy_input = np.random.random((1, 10, 194)).astype(np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            logger.info("모델 워밍업 완료")

            return True

        except Exception as e:
            logger.error(f"동기식 모델 로딩 오류: {e}")
            return False

    async def predict_async(
        self, keypoints_batch: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        비동기 모델 예측

        Args:
            keypoints_batch: (1, 10, 194) 형태의 키포인트 배치

        Returns:
            Dict with 'gloss' and 'confidence' keys
        """
        if not self._model_loaded or self.model is None:
            logger.warning("모델이 로딩되지 않음")
            return None

        try:
            # 입력 검증
            if keypoints_batch.shape != (1, 10, 194):
                logger.error(
                    f"잘못된 입력 형태: {keypoints_batch.shape}, 예상: (1, 10, 194)"
                )
                return None

            # 모델 예측을 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._predict_sync, keypoints_batch
            )

            return result

        except Exception as e:
            logger.error(f"비동기 예측 중 오류: {e}")
            return None

    def _predict_sync(self, keypoints_batch: np.ndarray) -> Optional[Dict[str, Any]]:
        """동기식 모델 예측 (별도 스레드에서 실행)"""
        try:
            # 모델 추론
            predictions = self.model.predict(keypoints_batch, verbose=0)

            # 최고 신뢰도 클래스 선택
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # 라벨 매핑
            gloss = self.label_map.get(
                str(predicted_class_idx), f"unknown_{predicted_class_idx}"
            )

            return {
                "gloss": gloss,
                "confidence": confidence,
                "class_index": int(predicted_class_idx),
            }

        except Exception as e:
            logger.error(f"동기식 예측 오류: {e}")
            return None

    def is_model_ready(self) -> bool:
        """모델 준비 상태 확인"""
        return self._model_loaded and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_path": settings.model_path,
            "num_classes": len(self.label_map) if self.label_map else 0,
            "input_shape": self.model.input_shape if self.model else None,
            "output_shape": self.model.output_shape if self.model else None,
        }

    async def cleanup(self):
        """리소스 정리"""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("모델 executor 정리 완료")


# 전역 모델 매니저 인스턴스
model_manager = ModelManager()
