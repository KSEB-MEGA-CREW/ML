# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from typing import Optional, List
import os


class Settings(BaseSettings):
    """환경 설정 클래스"""

    model_config = SettingsConfigDict(
        env_file=(
            ".env.dev"
            if os.getenv("ENVIRONMENT", "development") == "development"
            else ".env.prod"
        ),
        protected_namespaces=("settings_",),
    )

    # Claude API 설정
    claude_api_key: str
    claude_model: str = "claude-3-haiku-20240307"
    claude_max_tokens: int = 200
    claude_enabled: bool = True

    # 서버 설정
    ai_server_host: str = "0.0.0.0"
    ai_server_port: int = 8000

    # 백엔드 연동 설정
    backend_host: str = "localhost"
    backend_port: str = "8080"
    backend_protocol: str = "http"
    backend_token_verify_endpoint: str = "/api/auth/verify-token"

    # WebSocket 설정
    websocket_timeout: int = 300
    max_connections: int = 100

    # 모델 설정
    model_path: str = "./models/gesture_model.h5"
    labels_path: str = "./models/label_map.json"
    frame_buffer_size: int = 10

    # Gloss 수집 설정
    gloss_confidence_threshold: float = 0.8
    gloss_max_count: int = 20
    gloss_deduplication: bool = True

    # 로깅 설정
    log_level: str = "INFO"
    debug: bool = False

    # CORS 설정
    allowed_origins: List[str] = ["*"]

    # 환경 구분
    environment: str = "development"

    # F2T (Frame-to-Text) 서버 설정
    enable_frame_processing: bool = True
    frame_batch_size: int = 10
    mediapipe_model_complexity: int = 1  # 0(lite), 1(full), 2(heavy)
    frame_processing_threads: int = 2
    frame_processing_timeout: int = 30  # seconds

    # Computed field로 backend_url 생성 (Pydantic v2 방식)
    @computed_field
    @property
    def backend_url(self) -> str:
        return f"{self.backend_protocol}://{self.backend_host}:{self.backend_port}"


# 전역 설정 인스턴스
settings = Settings()
