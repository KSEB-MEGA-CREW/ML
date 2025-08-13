import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-northeast-2")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "mega-crew-ml-models-dev")
    MODEL_S3_KEY: str = os.getenv(
        "MODEL_S3_KEY", "frame to gloss/V0/frame_to_gloss_v0.h5"
    )
    MODEL_CACHE_DIR: str = "./cache/models"


settings = Settings()
