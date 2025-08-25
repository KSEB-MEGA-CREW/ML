# app/core/dependencies.py
import httpx
import logging
from typing import Optional
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


class HTTPClientManager:
    """HTTP 클라이언트 관리자"""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 인스턴스 반환"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(5.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self):
        """HTTP 클라이언트 정리"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# 전역 HTTP 클라이언트 매니저
http_client_manager = HTTPClientManager()


async def get_http_client() -> httpx.AsyncClient:
    """HTTP 클라이언트 의존성"""
    return await http_client_manager.get_client()


async def verify_token_with_backend(token: str) -> str:
    """
    백엔드에 JWT 토큰 검증 요청

    Args:
        token: JWT 토큰 문자열

    Returns:
        str: 검증된 사용자 ID

    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
    """
    try:
        http_client = await get_http_client()

        # 백엔드 토큰 검증 API 호출
        response = await http_client.post(
            f"{settings.backend_url}{settings.backend_token_verify_endpoint}",
            json={"token": token},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            # debug log 추가
            logger.info(f"백엔드 토큰 검증 응답 데이터: {data}")

            user_id = None

            # 케이스 1: 직접 userId 필드
            if "userId" in data:
                user_id = data["userId"]
            # 케이스 2: data 객체 내부의 userId
            elif "data" in data and isinstance(data["data"], dict):
                user_id = data["data"].get("userId")
            # 케이스 3: success와 함께 중첩된 구조
            elif data.get("success", False) and "data" in data:
                token_data = data["data"]
                user_id = (
                    token_data.get("userId") if isinstance(token_data, dict) else None
                )
            # 케이스 4: 다른 필드명들 시도
            elif "user_id" in data:
                user_id = data["user_id"]
            elif "id" in data:
                user_id = data["id"]

            if user_id:
                logger.info(f"토큰 검증 성공: 사용자 ID {user_id}")
                return str(user_id)
            else:
                logger.warning(f"토큰 검증 응답에 userId가 없음. 응답 구조: {data}")
                return HTTPException(status_code=401, detail="Invalid token response")
            
        elif response.status_code == 401:
            logger.warning("유효하지 않은 토큰")
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        else:
            logger.error(f"토큰 검증 API 오류: {response.status_code}")
            raise HTTPException(status_code=503, detail="Authentication service error")

    except httpx.TimeoutException:
        logger.error("토큰 검증 요청 타임아웃")
        raise HTTPException(status_code=503, detail="Authentication service timeout")

    except httpx.RequestError as e:
        logger.error(f"토큰 검증 요청 실패: {e}")
        raise HTTPException(
            status_code=503, detail="Authentication service unavailable"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"토큰 검증 중 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def cleanup_dependencies():
    """의존성 정리 함수 (서버 종료 시 호출)"""
    await http_client_manager.close()
