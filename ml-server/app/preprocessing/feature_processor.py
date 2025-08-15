import numpy as np
from typing import Dict, Any
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    def __init__(self):
        self.feature_size = settings.MODEL_INPUT_SIZE

    def create_feature_vector(self, keypoints: Dict[str, Any]) -> np.ndarray:
        """Convert keypoints to 194-dimensional feature vector"""
        try:
            features = []

            # Hand landmark features (21 * 3 * 2 = 126 dimensions)
            hand_landmarks = keypoints.get("hand_landmarks", {})

            # Left hand (21 * 3 = 63 dimensions)
            if hand_landmarks.get("left_hand"):
                left_hand_flat = np.array(hand_landmarks["left_hand"]).flatten()
                if len(left_hand_flat) == 63:
                    features.extend(left_hand_flat)
                else:
                    features.extend([0.0] * 63)
            else:
                features.extend([0.0] * 63)

            # Right hand (21 * 3 = 63 dimensions)
            if hand_landmarks.get("right_hand"):
                right_hand_flat = np.array(hand_landmarks["right_hand"]).flatten()
                if len(right_hand_flat) == 63:
                    features.extend(right_hand_flat)
                else:
                    features.extend([0.0] * 63)
            else:
                features.extend([0.0] * 63)

            # Pose landmark features (17 * 4 = 68 dimensions)
            pose_landmarks = keypoints.get("pose_landmarks")
            if pose_landmarks:
                pose_flat = np.array(pose_landmarks).flatten()
                if len(pose_flat) == 68:
                    features.extend(pose_flat)
                else:
                    features.extend([0.0] * 68)
            else:
                features.extend([0.0] * 68)

            # Ensure exactly 194 dimensions
            feature_vector = np.array(features, dtype=np.float32)
            if len(feature_vector) != self.feature_size:
                logger.warning(
                    f"Feature vector size mismatch: {len(feature_vector)} != {self.feature_size}"
                )
                # Adjust size
                if len(feature_vector) < self.feature_size:
                    feature_vector = np.pad(
                        feature_vector, (0, self.feature_size - len(feature_vector))
                    )
                else:
                    feature_vector = feature_vector[: self.feature_size]

            return feature_vector

        except Exception as e:
            logger.error(f"Feature vector creation failed: {e}")
            return np.zeros(self.feature_size, dtype=np.float32)

    def validate_features(self, feature_vector: np.ndarray) -> bool:
        """Validate feature vector"""
        try:
            # Check for zero vector
            if np.sum(np.abs(feature_vector)) < 0.01:
                return False

            # Check for NaN or inf values
            if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                return False

            # Check vector size
            if len(feature_vector) != self.feature_size:
                return False

            return True

        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
