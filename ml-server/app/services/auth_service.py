# app/services/auth_service.py
import httpx
import logging
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self):
        self.backend_url = settings.backend_url
        self.timeout = 10.0

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """백엔드 서버에 토큰 검증 요청"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 🔄 수정: 백엔드 토큰 검증 엔드포인트 호출
                response = await client.post(
                    f"{self.backend_url}/api/auth/verify",
                    json={"token": token},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"토큰 검증 성공: {data}")

                    # 🔄 수정: 응답 데이터 구조 확인 및 정규화
                    if isinstance(data, dict):
                        # 성공적인 응답인 경우
                        if data.get("valid", False) or data.get("success", False):
                            return {
                                "user_id": data.get("userId")
                                or data.get("user_id")
                                or data.get("sub"),
                                "email": data.get("email") or data.get("sub"),
                                "valid": True,
                            }
                        else:
                            logger.warning(f"토큰이 유효하지 않음: {data}")
                            return None
                    else:
                        logger.error(f"예상치 못한 응답 형식: {type(data)} - {data}")
                        return None
                else:
                    logger.error(
                        f"토큰 검증 실패: {response.status_code} - {response.text}"
                    )
                    return None

        except httpx.TimeoutException:
            logger.error("백엔드 서버 연결 타임아웃")
            return None
        except httpx.RequestError as e:
            logger.error(f"백엔드 서버 연결 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"토큰 검증 중 예외 발생: {e}")
            return None

    def extract_user_id_from_token(self, token: str) -> Optional[str]:
        """JWT 토큰에서 직접 user_id 추출 (백엔드 연결 실패 시 대안)"""
        try:
            import jwt

            # JWT 디코딩 (서명 검증 없이 - 개발용만)
            decoded = jwt.decode(token, options={"verify_signature": False})
            user_id = (
                decoded.get("userId") or decoded.get("user_id") or decoded.get("sub")
            )

            logger.warning(f"백엔드 검증 실패로 로컬 디코딩 사용: user_id={user_id}")
            return str(user_id) if user_id else None

        except Exception as e:
            logger.error(f"JWT 토큰 디코딩 실패: {e}")
            return None


# 싱글톤 인스턴스
auth_service = AuthService()
