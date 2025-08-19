# app/models/model_manager.py
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional
import concurrent.futures
import os
import tensorflow as tf
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.executor = None
        self.model_path = None
        self.class_labels = []
        self._initialization_lock = asyncio.Lock()

    async def initialize(self, model_path: str = None):
        """비동기 초기화 - 한 번만 실행되도록 보장"""
        async with self._initialization_lock:
            if not self.is_loaded:
                success = await self.load_model(model_path)
                return success
            return True

    async def load_model(self, model_path: str = None):
        """비동기 모델 로드"""
        try:
            # 모델 경로 확인
            self.model_path = self._find_model_path(model_path)
            if not self.model_path:
                raise FileNotFoundError("모델 파일을 찾을 수 없습니다")

            logger.info(f"🔄 모델 로드 시작: {self.model_path}")

            # 별도 스레드에서 모델 로드 (I/O 블로킹 방지)
            loop = asyncio.get_event_loop()
            model_data = await loop.run_in_executor(
                None, self._load_model_from_file, self.model_path
            )

            self.model = model_data["model"]
            self.class_labels = model_data["labels"]

            # 예측용 스레드 풀 생성
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="model_prediction"
            )

            # 모델 워밍업
            await self._warmup_model()

            self.is_loaded = True
            logger.info(f"✅ 모델 로드 완료: {len(self.class_labels)}개 클래스")

            return True

        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            self.is_loaded = False
            return False

    def _find_model_path(self, model_path: str = None) -> Optional[str]:
        """모델 파일 경로 찾기"""
        if model_path and os.path.exists(model_path):
            return model_path

        # 기본 경로들 검색
        possible_paths = [
            "./models/sign_language_model.h5",
            "./models/sign_language_model.keras",
            "./models/sign_language_model",
            "../models/sign_language_model.h5",
            "../../models/sign_language_model.h5",
            os.getenv("MODEL_PATH", "./models/sign_language_model.h5"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _load_model_from_file(self, model_path: str) -> Dict[str, Any]:
        """파일에서 모델 로드 (동기 - 스레드에서 실행)"""
        try:
            # TensorFlow 모델 로드
            if model_path.endswith((".h5", ".keras")):
                model = tf.keras.models.load_model(model_path, compile=False)
            else:
                model = tf.saved_model.load(model_path)

            # 클래스 레이블 로드
            labels = self._load_class_labels(model_path)

            logger.info(
                f"🏗️ 모델 정보: input_shape={getattr(model, 'input_shape', 'Unknown')}"
            )

            return {"model": model, "labels": labels}

        except Exception as e:
            logger.error(f"❌ 파일 로드 실패: {e}")
            raise

    def _load_class_labels(self, model_path: str) -> list:
        """클래스 레이블 파일 로드"""
        try:
            base_path = Path(model_path).parent
            label_files = [
                base_path / "class_labels.txt",
                base_path / "labels.txt",
                base_path / "classes.txt",
            ]

            for label_file in label_files:
                if label_file.exists():
                    with open(label_file, "r", encoding="utf-8") as f:
                        labels = [
                            line.strip() for line in f.readlines() if line.strip()
                        ]
                    logger.info(f"📋 레이블 파일 로드: {len(labels)}개")
                    return labels

            # 기본 수어 레이블
            logger.warning("⚠️ 레이블 파일 없음 - 기본 레이블 사용")
            return [
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
                "기타",
            ]

        except Exception as e:
            logger.error(f"❌ 레이블 로드 실패: {e}")
            return [f"class_{i}" for i in range(15)]

    async def _warmup_model(self):
        """모델 워밍업 - 첫 예측 속도 향상"""
        try:
            logger.info("🔥 모델 워밍업...")
            dummy_input = np.zeros((1, 10, 194), dtype=np.float32)
            result = await self.predict(dummy_input)

            if "error" not in result:
                logger.info("✅ 워밍업 완료")
            else:
                logger.warning(f"⚠️ 워밍업 경고: {result['error']}")

        except Exception as e:
            logger.warning(f"⚠️ 워밍업 실패: {e}")

    async def predict(self, keypoints_batch: np.ndarray) -> Dict[str, Any]:
        """비동기 예측 - 메인 인터페이스"""
        # 모델 초기화 확인
        if not self.is_loaded:
            init_success = await self.initialize()
            if not init_success:
                return {
                    "label": "초기화_실패",
                    "confidence": 0.0,
                    "error": "Model initialization failed",
                }

        try:
            # 입력 검증
            validated_input = self._validate_input(keypoints_batch)
            if "error" in validated_input:
                return validated_input

            # 스레드 풀에서 예측 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._predict_in_thread, validated_input["data"]
            )

            return result

        except Exception as e:
            logger.error(f"❌ 예측 실패: {e}")
            return {"label": "예측_실패", "confidence": 0.0, "error": str(e)}

    def _validate_input(self, keypoints_batch) -> Dict[str, Any]:
        """입력 데이터 검증"""
        try:
            # NumPy 배열로 변환
            if not isinstance(keypoints_batch, np.ndarray):
                keypoints_batch = np.array(keypoints_batch, dtype=np.float32)

            # 형태 검증
            if keypoints_batch.shape != (1, 10, 194):
                return {
                    "error": f"Invalid shape: {keypoints_batch.shape}, expected: (1, 10, 194)"
                }

            # 값 범위 검증 (선택적)
            if np.any(np.isnan(keypoints_batch)) or np.any(np.isinf(keypoints_batch)):
                return {"error": "Input contains NaN or Inf values"}

            return {"data": keypoints_batch}

        except Exception as e:
            return {"error": f"Input validation failed: {str(e)}"}

    def _predict_in_thread(self, keypoints_batch: np.ndarray) -> Dict[str, Any]:
        """스레드에서 실행되는 실제 예측"""
        try:
            start_time = time.time()

            # 모델 예측 실행
            if hasattr(self.model, "predict"):
                # Keras 모델
                predictions = self.model.predict(keypoints_batch, verbose=0)
            else:
                # SavedModel
                predictions = self.model(keypoints_batch).numpy()

            prediction_time = time.time() - start_time

            # 결과 처리
            prediction = predictions[0] if len(predictions.shape) > 1 else predictions
            max_idx = np.argmax(prediction)
            confidence = float(prediction[max_idx])

            # 레이블 매핑
            label = (
                self.class_labels[max_idx]
                if max_idx < len(self.class_labels)
                else f"unknown_{max_idx}"
            )

            logger.debug(
                f"🎯 예측: {label} ({confidence:.3f}) - {prediction_time:.3f}s"
            )

            return {
                "label": label,
                "confidence": confidence,
                "prediction_time": prediction_time,
                "class_index": int(max_idx),
            }

        except Exception as e:
            logger.error(f"❌ 스레드 예측 실패: {e}")
            return {"label": "스레드_에러", "confidence": 0.0, "error": str(e)}

    async def predict_multiple(self, keypoints_list: list) -> list:
        """여러 입력에 대한 동시 예측"""
        if not keypoints_list:
            return []

        # 모든 예측을 병렬로 실행
        tasks = [self.predict(keypoints) for keypoints in keypoints_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"배치 예측 {i} 실패: {result}")
                processed_results.append(
                    {"label": "배치_에러", "confidence": 0.0, "error": str(result)}
                )
            else:
                processed_results.append(result)

        return processed_results

    def is_ready(self) -> bool:
        """모델 준비 상태"""
        return self.is_loaded and self.model is not None

    def get_info(self) -> Dict[str, Any]:
        """모델 정보"""
        return {
            "loaded": self.is_loaded,
            "model_path": self.model_path,
            "num_classes": len(self.class_labels),
            "class_labels": self.class_labels[:10],  # 처음 10개만
            "model_type": type(self.model).__name__ if self.model else None,
        }

    async def reload(self, model_path: str = None):
        """모델 재로드"""
        logger.info("🔄 모델 재로드...")
        await self.cleanup()
        return await self.initialize(model_path)

    async def cleanup(self):
        """리소스 정리"""
        try:
            # 스레드 풀 종료
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None

            # 모델 메모리 해제
            if self.model:
                del self.model
                self.model = None

            # TensorFlow 세션 정리
            try:
                import gc

                gc.collect()
                if tf.config.list_physical_devices("GPU"):
                    tf.keras.backend.clear_session()
            except Exception:
                pass

            self.is_loaded = False
            self.class_labels = []
            self.model_path = None

            logger.info("🧹 정리 완료")

        except Exception as e:
            logger.error(f"❌ 정리 실패: {e}")


# 전역 싱글톤 인스턴스
model_manager = ModelManager()


# 애플리케이션 시작 시 자동 초기화 (선택적)
async def initialize_model_on_startup():
    """애플리케이션 시작 시 모델 초기화"""
    try:
        success = await model_manager.initialize()
        if success:
            logger.info("🚀 모델 매니저 시작 완료")
        else:
            logger.warning("⚠️ 모델 로드 실패 - 런타임에 재시도")
    except Exception as e:
        logger.error(f"❌ 시작 시 모델 초기화 실패: {e}")


# FastAPI 이벤트에서 사용할 초기화 함수
async def startup_event():
    """FastAPI startup 이벤트"""
    await initialize_model_on_startup()


async def shutdown_event():
    """FastAPI shutdown 이벤트"""
    await model_manager.cleanup()
