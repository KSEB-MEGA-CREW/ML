import os
from typing import List


class Settings:
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # AWS 설정
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")

    # S3 설정
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "sign-language-models")
    MODEL_S3_KEY: str = os.getenv("MODEL_S3_KEY", "models/sign_language_model.pth")

    # 모델 설정 input data shape
    MODEL_INPUT_SIZE: int = 194
    MODEL_NUM_CLASSES: int = 14
    CONFIDENCE_THRESHOLD: float = 0.6
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")

    # TensorFlow 설정
    TF_ENABLE_GPU_MEMORY_GROWTH: bool = (
        os.getenv("TF_ENABLE_GPU_MEMORY_GROWTH", "True").lower() == "true"
    )

    # label file 설정
    CLASS_LABELS_FILE: str = os.getenv(
        "CLASS_LABELS_FILE", "/app/labels/label_map.json"
    )

    # 전처리 설정
    TARGET_IMAGE_SIZE: tuple = (640, 480)
    MAX_HANDS: int = 2
    HAND_CONFIDENCE: float = 0.7
    POSE_CONFIDENCE: float = 0.5

    # 세션 관리
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_BUFFER_SIZE: int = 100

    # CORS 설정
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://your-frontend-domain.com",
    ]

    # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
