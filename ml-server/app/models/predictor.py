# 예측 수행 코드
import torch
import numpy as np
from typing import Dict, Any
from ..preprocessing.image_processor import ImageProcessor
from ..preprocessing.key_point_extractor import KeypointExtractor
from ..preprocessing.feature_processor import FeatureProcessor
from ..utils.label_loader import LabelLoader
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class SignLanguagePredictor:
    def __init__(self, model):
        self.model = model
        self.image_processor = ImageProcessor()
        self.keypoint_extractor = KeypointExtractor()
        self.feature_processor = FeatureProcessor()
        self.label_loader = LabelLoader()
        self.class_names = None  # 지연 로딩
        self._model_validated = False

    async def _ensure_labels_loaded(self):
        """라벨이 로드되지 않았다면 로드하고 모델과 호환성 검증"""
        if self.class_names is None:
            self.class_names = await self.label_loader.load_class_labels()

            # 모델과 라벨 수 호환성 검증 (한 번만)
            if not self._model_validated:
                is_compatible = await self.label_loader.validate_model_compatibility(
                    settings.MODEL_NUM_CLASSES
                )
                if not is_compatible:
                    logger.warning("모델과 라벨 수가 일치하지 않습니다.")
                self._model_validated = True

    async def predict(
        self, base64_image: str, session_id: str, frame_index: int
    ) -> Dict[str, Any]:
        """수어 예측 수행"""
        try:
            # 라벨 로드 및 체크
            await self._ensure_labels_loaded()

            # 1. Base64 -> OpenCV 이미지 변환
            image = self.image_processor.decoded_base64(base64_image)
            if image is None:
                return self._create_no_detection_response(session_id, frame_index)

            # 2. keypoint 추출
            keypoints = self.keypoint_extractor.extract_keypoints(image)

            # 3. feature vector 생성
            feature_vector = self.feature_processor.create_feature_vector(keypoints)

            # 4. feature vector 검증
            if not self.feature_processor.validate_features(feature_vector):
                return self._create_no_detection_response(session_id, frame_index)

            # 5. model 예측
            prediction_result = await self._run_inference(feature_vector)

            # 6. 결과 후처리
            final_result = self._postprocess_prediction(
                prediction_result, keypoints, session_id, frame_index
            )

            return final_result

        except Exception as e:
            logger.error(f"예측 오류: session={session_id}, error={str(e)}")
            return self._create_error_response(session_id, frame_index, str(e))

    async def _run_inference(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """모델 추론 실행"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(device)

        self.model.eval()  # 함수 추가 구현 필요
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        probs_numpy = probabilities.cpu().numpy()[0]

        # 현재 라벨 수에 맞춰 top 3 계산
        num_classes = len(self.class_names)
        top_k = min(3, num_classes)  # 최소 3개 또는 전체 클래스 수

        top_indices = np.argsort(probs_numpy)[-top_k:][::-1]

        # output data shape!!!
        return {
            "prbabilities": probs_numpy,
            "top_classes": top_indices,
            "top_scores": probs_numpy[top_indices],
        }

    def _postprocess_prediction(
        self,
        prediction_result: Dict,
        keypoints: Dict,
        session_id: str,
        frame_index: int,
    ) -> Dict[str, Any]:
        """예측 결과 후처리"""
        top_class_idx = prediction_result["top_classes"][0]
        confidence = float(prediction_result["top_scores"][0])

        # 인덱스 범위 검증
        if top_class_idx >= len(self.class_names):
            logger.error(
                f"클래스 인덱스 범위 초과: {top_class_idx} >= {len(self.class_names)}"
            )
            predicted_text = "인식 오류"
            confidence_level = "ERROR"
        elif confidence < settings.CONFIDENCE_THRESHOLD:
            predicted_text = "인식 중..."
            confidence_level = "LOW"
        elif confidence < 0.8:
            predicted_text = self.class_names[top_class_idx]
            confidence_level = "MEDIUM"
        else:
            predicted_text = self.class_names[top_class_idx]
            confidence_level = "HIGH"

        # Top predictions 생성 (인덱스 범위 검증 포함)
        top_predictions = []
        for idx, score in zip(
            prediction_result["top_classes"], prediction_result["top_scores"]
        ):
            if idx < len(self.class_names):
                top_predictions.append(
                    {"class": self.class_names[idx], "confidence": float(score)}
                )

        return {
            "session_id": session_id,
            "frame_index": frame_index,
            "predicted_class": predicted_text,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "top_predictions": top_predictions,
            "has_hand_detection": keypoints["hand_landmarks"]["left_hand"] is not None
            or keypoints["hand_landmarks"]["right_hand"] is not None,
            "has_pose_detection": keypoints["pose_landmarks"] is not None,
        }

    def _create_no_detection_response(
        self, session_id: str, frame_index: int
    ) -> Dict[str, Any]:
        """검출 실패시 응답"""
        return {
            "session_id": session_id,
            "frame_index": frame_index,
            "predicted_class": "손동작을 인식할 수 없습니다",
            "confidence": 0.0,
            "confidence_level": "NONE",
            "top_predictions": [],
            "has_hand_detection": False,
            "has_pose_detection": False,
        }

    def _create_error_response(
        self, session_id: str, frame_index: int, error_msg: str
    ) -> Dict[str, Any]:
        """에러 응답"""
        return {
            "session_id": session_id,
            "frame_index": frame_index,
            "predicted_class": "처리 중 오류 발생",
            "confidence": 0.0,
            "confidence_level": "ERROR",
            "top_predictions": [],
            "has_hand_detection": False,
            "has_pose_detection": False,
            "error": error_msg,
        }
