from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://mega-crew-react-deploy.s3-website.ap-northeast-2.amazonaws.com",
    ]

    # Model settings
    MODEL_PATH: str = "./models/sign_language_model.h5"
    LABELS_PATH: str = "./models/labels.json"

    # WebSocket settings
    MAX_CONNECTIONS: int = 100
    FRAME_BUFFER_SIZE: int = 10

    # Backend verification
    BACKEND_URL: str = "http://localhost:8080"
    BACKEND_TOKEN_VERIFY_ENDPOINT: str = "/api/auth/verify-token"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
