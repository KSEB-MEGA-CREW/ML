import logging
from typing import List, Union

logger = logging.getLogger(__name__)


class KeypointProcessor:

    @staticmethod
    def validate_keypoints(keypoints: List[Union[int, float]]) -> bool:
        """Validate keypoints format and values"""
        try:
            # Check length
            if len(keypoints) != 194:
                logger.warning(
                    f"Invalid keypoints length: {len(keypoints)}, expected 194"
                )
                return False

            # Check data types
            for i, point in enumerate(keypoints):
                if not isinstance(point, (int, float)):
                    logger.warning(f"Invalid keypoint type at index {i}: {type(point)}")
                    return False

                # Check for NaN or infinite values
                if not (
                    -10 <= point <= 10
                ):  # Reasonable range for normalized coordinates
                    logger.warning(f"Keypoint value out of range at index {i}: {point}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Keypoint validation error: {e}")
            return False

    @staticmethod
    def normalize_keypoints(keypoints: List[Union[int, float]]) -> List[float]:
        """Normalize keypoints to standard range"""
        try:
            # Convert to float and ensure range
            normalized = []
            for point in keypoints:
                normalized_point = float(point)
                # Clamp to reasonable range
                normalized_point = max(-5.0, min(5.0, normalized_point))
                normalized.append(normalized_point)

            return normalized

        except Exception as e:
            logger.error(f"Keypoint normalization error: {e}")
            return keypoints
