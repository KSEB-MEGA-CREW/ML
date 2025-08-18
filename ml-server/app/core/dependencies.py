# manage dependencies
# app/core/dependencies.py
from fastapi import Depends, HTTPException
import aioredis
import httpx
import asyncio
import logging
from typing import Optional
from .config import settings

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
                        self.redis_pool = await aioredis.from_url(
                            settings.redis_url,
                            encoding="utf-8",
                            decode_responses=True,
                            max_connections=20,
                            retry_on_timeout=True,
                            socket_connect_timeout=5,
                            socket_timeout=5,
                        )

                        # 연결 테스트
                        await self.redis_pool.ping()
                        logger.info(f"Redis 연결 성공: {settings.redis_url}")

                    except Exception as e:
                        logger.error(f"Redis 연결 실패: {e}")
                        raise HTTPException(
                            status_code=503, detail="Redis service unavailable"
                        )

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
async def get_redis() -> aioredis.Redis:
    """Redis 의존성 주입"""
    return await connection_manager.get_redis()


async def get_http_client() -> httpx.AsyncClient:
    """HTTP 클라이언트 의존성 주입"""
    return await connection_manager.get_http_client()


async def verify_token_with_backend(token: str) -> str:
    """백엔드 서버에서 JWT 토큰 검증 (동적 URL 사용)"""
    http_client = await get_http_client()

    verify_url = f"{settings.backend_url}/api/verify-token"

    try:
        response = await http_client.post(
            verify_url, json={"token": token}, timeout=5.0
        )

        if response.status_code == 200:
            data = response.json()
            user_id = data.get("user_id")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token response")
            return str(user_id)
        else:
            logger.warning(f"토큰 검증 실패: status={response.status_code}")
            raise HTTPException(status_code=401, detail="Invalid token")

    except httpx.RequestError as e:
        logger.error(f"백엔드 서비스 연결 실패: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")
    except httpx.TimeoutException:
        logger.error("백엔드 서비스 타임아웃")
        raise HTTPException(status_code=504, detail="Backend service timeout")


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
        await redis.ping()
        return True

    except Exception as e:
        logger.error(f"Redis 헬스체크 실패: {e}")
        return False
