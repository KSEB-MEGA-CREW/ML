# model_manager.py
import tensorflow as tf
import os
import logging
import json
import asyncio
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# TensorFlow 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.experimental.enable_op_determinism()


class ModelManager:
    def __init__(self):
        self.model = None
        self.labels = None
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model")
        self._load_lock = asyncio.Lock()

    async def load_model(self) -> bool:
        """모델 로딩 - 중복 로딩 방지"""
        if self.is_loaded:
            return True

        async with self._load_lock:
            if self.is_loaded:  # 다시 확인
                return True

            try:
                logger.info("🚀 모델 로딩 시작...")

                # 라벨 파일 경로
                label_paths = [
                    Path("models/label_map.json"),
                    Path("../models/label_map.json"),
                    Path("/app/models/label_map.json"),
                ]

                # 모델 파일 경로
                model_paths = [
                    Path("models/gesture_model.h5"),
                    Path("../models/gesture_model.h5"),
                    Path("/app/models/gesture_model.h5"),
                ]

                # 라벨 로딩
                labels_path = next((p for p in label_paths if p.exists()), None)
                if not labels_path:
                    logger.error("❌ 라벨 파일을 찾을 수 없음")
                    return False

                with open(labels_path, "r", encoding="utf-8") as f:
                    label_list = json.load(f)
                self.labels = {str(i): label for i, label in enumerate(label_list)}
                logger.info(f"📋 라벨 로딩 완료: {len(self.labels)}개")

                # 모델 로딩
                model_path = next((p for p in model_paths if p.exists()), None)
                if not model_path:
                    logger.error("❌ 모델 파일을 찾을 수 없음")
                    return False

                logger.info(f"📦 모델 로딩: {model_path}")
                self.model = tf.keras.models.load_model(model_path, compile=False)

                # 워밍업
                dummy_input = np.random.random((1, 10, 194)).astype(np.float32)
                _ = self.model.predict(dummy_input, verbose=0)

                self.is_loaded = True
                logger.info("✅ 모델 로딩 완료!")
                return True

            except Exception as e:
                logger.error(f"❌ 모델 로딩 실패: {e}")
                return False

    def is_model_ready(self) -> bool:
        """모델 준비 상태 확인"""
        return self.is_loaded and self.model is not None

    def _predict_sync(self, keypoints_sequence: List[List[float]]) -> Dict:
        """동기 예측 함수 - 스레드에서 실행됨"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # 입력 데이터 변환 및 검증
            features = np.array(keypoints_sequence, dtype=np.float32)

            # 형태 검증
            if features.shape != (10, 194):
                raise ValueError(
                    f"Invalid input shape: {features.shape}, expected (10, 194)"
                )

            # 모델 입력 형태로 변환: (1, 10, 194)
            features = features.reshape(1, 10, 194)

            # 예측 수행
            probability_vector = self.model.predict(features, verbose=0)
            predicted_class_index = np.argmax(probability_vector[0])
            confidence = float(probability_vector[0][predicted_class_index])
            label = self.labels.get(str(predicted_class_index), "Unknown")

            logger.debug(f"예측 완료: {label} (신뢰도: {confidence:.3f})")

            return {
                "label": label,
                "confidence": confidence,
                "class_id": int(predicted_class_index),
            }

        except Exception as e:
            logger.error(f"예측 수행 중 오류: {e}")
            raise

    async def predict_async(
        self, keypoints_sequence: List[List[float]]
    ) -> Optional[Dict]:
        """비동기 예측 - 강화된 예외 처리"""
        if not self.is_loaded:
            logger.error("모델이 로딩되지 않음")
            return None

        try:
            # 입력 데이터 사전 검증
            if not keypoints_sequence:
                logger.error("빈 키포인트 시퀀스")
                return None

            if len(keypoints_sequence) != 10:
                logger.error(f"잘못된 시퀀스 길이: {len(keypoints_sequence)}, 예상: 10")
                return None

            # 스레드 풀에서 예측 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._predict_sync, keypoints_sequence
            )

            return result

        except Exception as e:
            logger.error(f"비동기 예측 실패: {e}")
            logger.exception("예측 실패 상세 정보:")
            return None


# 전역 인스턴스
model_manager = ModelManager()
