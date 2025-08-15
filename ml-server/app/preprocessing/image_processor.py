import cv2
import numpy as np
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    def decode_base64(self, base64_data: str) -> Optional[np.ndarray]:
        """Convert Base64 string to OpenCV image"""
        try:
            # Validate base64 format
            if not base64_data:
                raise ValueError("Empty base64 data")

            # Remove data URL prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]

            # Base64 decoding
            img_data = base64.b64decode(base64_data)

            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)

            # Decode to OpenCV image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.warning("Image decoding failed")
                return None

            return image

        except Exception as e:
            logger.error(f"Base64 image conversion failed: {str(e)}")
            return None

    def validate_image(self, image: np.ndarray) -> bool:
        """Validate image properties"""
        if image is None:
            return False

        # Check dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("Invalid image format: expected 3-channel color image")
            return False

        # Check size
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            logger.warning(f"Image too small: {width}x{height}")
            return False

        return True
