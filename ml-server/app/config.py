# config.py
import os
from typing import List, Optional


class Settings:
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # S3
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "mega-crew-ml-models-dev")
    MODEL_S3_KEY: str = os.getenv("MODEL_S3_KEY", "frame to gloss/v1/gesture_model.h5")

    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")

    @property
    def has_aws_credentials(self) -> bool:
        return bool(self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY)

    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")

    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    MAX_BUFFER_SIZE: int = int(os.getenv("MAX_BUFFER_SIZE", "100"))

    # label_paths(priority order)
    LABEL_PATHS = [
        "/app/labels/label_map.json",
        "./labels/label_map.json",
        "labels/label_map.json",
    ]

    # TensorFlow
    TF_ENABLE_GPU_MEMORY_GROWTH: bool = (
        os.getenv("TF_ENABLE_GPU_MEMORY_GROWTH", "True").lower() == "true"
    )

    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080",
    ).split(",")

    # logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
