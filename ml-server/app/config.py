# config.py
import os
from typing import List, Optional


class Settings:
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

    # label_paths(priority order)
    LABEL_PATHS = [
        "/app/labels/label_map.json",
        "./labels/label_map.json",
        "labels/label_map.json",
    ]


settings = Settings()
