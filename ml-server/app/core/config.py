# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Claude API 설정
    claude_api_key: str
    claude_model: str = "claude-3-haiku-20240307"
    claude_max_tokens: int = 200

    # 서버 설정
    ai_server_host: str = "0.0.0.0"
    ai_server_port: int = 8000

    # 백엔드 연동 설정 (환경별 동적 설정)
    backend_host: Optional[str] = None
    backend_port: Optional[str] = None
    backend_protocol: str = "http"
    backend_url: Optional[str] = None

    # Redis 설정 (환경별 동적 설정)
    redis_host: Optional[str] = None
    redis_port: Optional[str] = None
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_url: Optional[str] = None

    # 환경 구분
    environment: str = "development"  # development, production

    # WebSocket 설정
    websocket_timeout: int = 300
    max_connections: int = 100

    # 로깅 설정
    log_level: str = "INFO"
    log_file_path: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_dynamic_urls()

    def _build_dynamic_urls(self):
        """환경 변수를 기반으로 동적 URL 생성"""
        # 백엔드 URL 동적 생성
        if not self.backend_url:
            if self.backend_host and self.backend_port:
                self.backend_url = (
                    f"{self.backend_protocol}://{self.backend_host}:{self.backend_port}"
                )
            elif self.environment == "production":
                # 프로덕션 환경: 로드밸런서나 도메인 사용
                self.backend_url = os.getenv(
                    "BACKEND_URL", "http://backend-service:8080"
                )
            elif self.environment == "development":
                # 개발 환경: localhost
                self.backend_url = "http://localhost:8080"
            else:
                # 기본값 : localhost
                self.backend_url = "http://localhost:8080"

        # Redis URL 동적 생성
        if not self.redis_url:
            if self.redis_host and self.redis_port:
                auth_part = f":{self.redis_password}@" if self.redis_password else ""
                self.redis_url = f"redis://{auth_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"
            elif self.environment == "production":
                # 프로덕션 환경: ElastiCache 등
                self.redis_url = os.getenv("REDIS_URL", "redis://redis-cluster:6379/0")
            elif self.environment == "development":
                # 개발 환경: localhost
                self.redis_url = "redis://localhost:6379/0"
            else:
                # Docker Compose 환경
                self.redis_url = "redis://redis:6379/0"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    class Config:
        env_file = (
            ".env.dev"
            if os.getenv("ENVIRONMENT", "development") == "development"
            else ".env.prod"
        )


settings = Settings()
