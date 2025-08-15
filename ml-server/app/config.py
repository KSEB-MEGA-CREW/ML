import os
from typing import List
from dotenv import load_dotenv

# load .env
load_dotenv()


class Settings:
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # AWS settings
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")

    # S3 settings
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "mega-crew-ml-models-dev")
    MODEL_S3_KEY: str = os.getenv("MODEL_S3_KEY", "frame to gloss/v1/gesture_model.h5")

    # Model settings
    MODEL_INPUT_SIZE: int = 194
    MODEL_NUM_CLASSES: int = 14
    CONFIDENCE_THRESHOLD: float = 0.6
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")

    # TensorFlow settings
    TF_ENABLE_GPU_MEMORY_GROWTH: bool = (
        os.getenv("TF_ENABLE_GPU_MEMORY_GROWTH", "True").lower() == "true"
    )

    # Label settings
    CLASS_LABELS_FILE: str = os.getenv(
        "CLASS_LABELS_FILE", "/app/labels/label_map.json"
    )

    # Preprocessing settings
    TARGET_IMAGE_SIZE: tuple = (640, 480)
    MAX_HANDS: int = 2
    HAND_CONFIDENCE: float = 0.7
    POSE_CONFIDENCE: float = 0.5

    # Session management
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_BUFFER_SIZE: int = 100

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://your-frontend-domain.com",
    ]

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Async processing settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    PROCESS_POOL_SIZE: int = int(os.getenv("PROCESS_POOL_SIZE", "2"))


settings = Settings()
