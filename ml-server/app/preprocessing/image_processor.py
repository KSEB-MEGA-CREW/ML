# convert Base64 -> OpenCV image
import cv2
import numpy as np
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    def decoded_base64(self, base64_data: str) -> Optional[np.ndarray]:
        """Base64 문자열을 OpenCV 이미지로 변환"""
        try:
            # Base64 decoding
            img_data = base64.b64decode(base64_data)
            # convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            # OpenCV image로 decoding
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.warning("이미지 디코딩 실패")
                return None

            return image

        except Exception as e:
            logger.error(f"Base64 image 변환 실패: {str(e)}")
            return None
