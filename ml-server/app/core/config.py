# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    websocket_timeout: int = 300  # 5분
    max_connections: int = 100

    # 모델 설정
    model_path: str = "./models/gesture_model.h5"
    labels_path: str = "./models/label_map.json"
    frame_buffer_size: int = 10

    # 로깅 설정
    log_level: str = "INFO"
    debug: bool = False

    # CORS 설정
    allowed_origins: List[str] = ["*"]  # 개발환경용, 운영시 특정 도메인으로 제한

    # 환경 구분
    environment: str = "development"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 백엔드 URL 동적 생성
        if not hasattr(self, "backend_url") or not self.backend_url:
            self.backend_url = (
                f"{self.backend_protocol}://{self.backend_host}:{self.backend_port}"
            )


# 전역 설정 인스턴스
settings = Settings()
