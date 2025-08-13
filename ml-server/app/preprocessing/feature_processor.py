# 194차원 feature vector 생성
import numpy as np
from typing import Dict, List, Optional
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self):
        self.feature_size = settings.MODEL_INPUT_SIZE

    def create_feature_vector(self, keypoints: Dict[str, any]) -> np.ndarray:
        """keypoints를 194차원 feature vector로 변환(모델 입력 데이터 크기)"""
        features = []

        # 손 랜드마크 특징 (21 * 3 * 2 = 126차원)
        hand_landmarks = keypoints["hand_landmarks"]

        # 왼손 (21 * 3 = 63차원)
        if hand_landmarks["left_hand"]:
            left_hand_flat = np.array(hand_landmarks["left_hand"]).flatten()
            features.extend(left_hand_flat)
        else:
            features.extend([0.0] * 63)

        # 오른손 (21 * 3 = 63차원)
        if hand_landmarks["right_hand"]:
            right_hand_flat = np.array(hand_landmarks["right_hand"]).flatten()
            features.extend(right_hand_flat)
        else:
            features.extend([0.0] * 63)

        # 포즈 랜드마크 특징 (17 * 4 = 68차원)
        pose_landmarks = keypoints["pose_landmarks"]
        if pose_landmarks:
            pose_flat = np.array(pose_landmarks).flatten()
            features.extend(pose_flat)
        else:
            features.extend([0.0] * 68)

        # 총 194차원 확인
        feature_vector = np.array(features, dtype=np.float32)
        if len(feature_vector) != self.feature_size:
            logger.warning(
                f"특징벡터 크기 불일치: {len(feature_vector)} != {self.feature_size}"
            )
            # 크기 맞추기
            if len(feature_vector) < self.feature_size:
                feature_vector = np.pad(
                    feature_vector, (0, self.feature_size - len(feature_vector))
                )
            else:
                feature_vector = feature_vector[: self.feature_size]

        return feature_vector

    def validate_features(self, feature_vector: np.ndarray) -> bool:
        """feature vector 유효성 검증"""
        if np.sum(np.abs(feature_vector)) < 0.01:
            return False

        # NaN이나 inf 값 확인
        if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
            return False

        return True
