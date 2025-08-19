# app/core/dependencies.py
from fastapi import Depends, HTTPException
import aioredis
import httpx
import asyncio
import logging
from typing import Optional, Annotated
from .config import settings
from app.services.auth_service import token_verifier
from app.services.claude_service import claude_service
from app.models.model_manager import ModelManager
from app.models.predictor import SignLanguagePredictor

logger = logging.getLogger(__name__)


class ConnectionManager:
    """연결 관리 클래스"""

    def __init__(self):
        self.redis_pool: Optional[aioredis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self._redis_lock = asyncio.Lock()
        self._http_lock = asyncio.Lock()

    async def get_redis(self) -> aioredis.Redis:
        """Redis 연결 풀 관리 (싱글톤 패턴)"""
        if self.redis_pool is None:
            async with self._redis_lock:
                if self.redis_pool is None:
                    try:
                        # Redis URL이 설정되지 않은 경우 기본값 사용
                        redis_url = getattr(
                            settings, "redis_url", "redis://localhost:6379/0"
                        )

                        self.redis_pool = await aioredis.from_url(
                            redis_url,
                            encoding="utf-8",
                            decode_responses=True,
                            max_connections=20,
                            retry_on_timeout=True,
                            socket_connect_timeout=5,
                            socket_timeout=5,
                        )

                        # 연결 테스트
                        await self.redis_pool.ping()
                        logger.info(f"Redis 연결 성공: {redis_url}")

                    except Exception as e:
                        logger.warning(f"Redis 연결 실패 (선택사항): {e}")
                        # Redis 없이도 동작하도록 None 유지
                        self.redis_pool = None

        return self.redis_pool

    async def get_http_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 관리 (백엔드 통신용)"""
        if self.http_client is None:
            async with self._http_lock:
                if self.http_client is None:
                    timeout = httpx.Timeout(10.0, connect=5.0)
                    limits = httpx.Limits(
                        max_keepalive_connections=20, max_connections=100
                    )

                    self.http_client = httpx.AsyncClient(
                        timeout=timeout, limits=limits, follow_redirects=True
                    )

                    logger.info("HTTP 클라이언트 생성 완료")

        return self.http_client

    async def close_connections(self):
        """모든 연결 정리"""
        if self.redis_pool:
            await self.redis_pool.close()
            logger.info("Redis 연결 정리 완료")

        if self.http_client:
            await self.http_client.aclose()
            logger.info("HTTP 클라이언트 정리 완료")


# 전역 연결 관리자
connection_manager = ConnectionManager()


# 의존성 주입 함수들
async def get_redis() -> Optional[aioredis.Redis]:
    """Redis 의존성 주입"""
    return await connection_manager.get_redis()


async def get_http_client() -> httpx.AsyncClient:
    """HTTP 클라이언트 의존성 주입"""
    return await connection_manager.get_http_client()


async def verify_token_with_backend(token: str) -> str:
    """백엔드 서버에서 JWT 토큰 검증"""
    try:
        user_id = await token_verifier.verify_token(token)
        if user_id:
            return user_id
        else:
            raise HTTPException(status_code=401, detail="Invalid token")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"토큰 검증 중 오류: {e}")
        raise HTTPException(
            status_code=503, detail="Token verification service unavailable"
        )


async def check_backend_health() -> bool:
    """백엔드 서비스 상태 확인"""
    try:
        http_client = await get_http_client()
        health_url = f"{settings.backend_url}/health"

        response = await http_client.get(health_url, timeout=3.0)
        return response.status_code == 200

    except Exception as e:
        logger.error(f"백엔드 헬스체크 실패: {e}")
        return False


async def check_redis_health() -> bool:
    """Redis 서비스 상태 확인"""
    try:
        redis = await get_redis()
        if redis:
            await redis.ping()
            return True
        return False

    except Exception as e:
        logger.error(f"Redis 헬스체크 실패: {e}")
        return False


def get_model_manager() -> ModelManager:
    """모델 매니저 의존성 주입"""
    return ModelManager()


def get_predictor() -> SignLanguagePredictor:
    """예측기 의존성 주입"""
    return SignLanguagePredictor()


def get_claude_service():
    """Claude 서비스 의존성 주입"""
    return claude_service


def get_token_verifier():
    """토큰 검증기 의존성 주입"""
    return token_verifier


# 타입 어노테이션 별칭
ModelManagerDep = Annotated[ModelManager, Depends(get_model_manager)]
PredictorDep = Annotated[SignLanguagePredictor, Depends(get_predictor)]
ClaudeServiceDep = Annotated[object, Depends(get_claude_service)]
TokenVerifierDep = Annotated[object, Depends(get_token_verifier)]
