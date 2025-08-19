# app/models/predictor.py
import logging
import numpy as np
from typing import List, Dict
from app.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SignLanguagePredictor:

    def __init__(self):
        self.model_manager = ModelManager()

    async def predict_sequence(self, keypoints_sequence: List[List[float]]) -> Dict:
        """10프레임 키포인트 시퀀스로부터 수화 예측 (비동기)"""
        try:
            if not self.model_manager.is_ready():
                raise RuntimeError("Model not ready")

            # 입력 검증
            if len(keypoints_sequence) != 10:
                raise ValueError(f"Expected 10 frames, got {len(keypoints_sequence)}")

            for i, frame in enumerate(keypoints_sequence):
                if len(frame) != 194:
                    raise ValueError(
                        f"Frame {i}: expected 194 keypoints, got {len(frame)}"
                    )

            # ✅ 비동기 모델 예측 수행
            result = await self.model_manager.predict_async(keypoints_sequence)

            # 명확한 라벨 출력 추가
            logger.info(
                f"🎯 예측 결과: {result['label']} (신뢰도: {result['confidence']:.3f})"
            )

            return {
                "success": True,
                "prediction": result,
                "timestamp": int(
                    np.datetime64("now").astype("datetime64[ms]").astype(int)
                ),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e), "prediction": None}

    def predict_sequence_sync(self, keypoints_sequence: List[List[float]]) -> Dict:
        """동기 예측 (하위 호환용)"""
        try:
            if not self.model_manager.is_ready():
                raise RuntimeError("Model not ready")

            # 입력 검증
            if len(keypoints_sequence) != 10:
                raise ValueError(f"Expected 10 frames, got {len(keypoints_sequence)}")

            for i, frame in enumerate(keypoints_sequence):
                if len(frame) != 194:
                    raise ValueError(
                        f"Frame {i}: expected 194 keypoints, got {len(frame)}"
                    )

            # 동기 모델 예측 수행
            result = self.model_manager.predict(keypoints_sequence)

            logger.info(
                f"🎯 예측 결과: {result['label']} (신뢰도: {result['confidence']:.3f})"
            )

            return {
                "success": True,
                "prediction": result,
                "timestamp": int(
                    np.datetime64("now").astype("datetime64[ms]").astype(int)
                ),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"success": False, "error": str(e), "prediction": None}
