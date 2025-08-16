import httpx
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class TokenVerifier:
    # verify token backend endpoint
    # "http://localhost:8080/api/auth/verify-token"
    @staticmethod
    async def verify_token(token: str) -> Optional[str]:
        """verify jwt with backend server and return user_id"""
        try:
            url = f"{settings.BACKEND_URL}{settings.BACKEND_TOKEN_VERIFY_ENDPOINT}"

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    url,
                    json={"token": token},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get("valid", False):
                        user_id = data.get("user_id")
                        logger.info(f"Token verified for user: {user_id}")
                        return user_id
                    else:
                        logger.warning("Invalid token received")
                        return None
                else:
                    logger.error(f"Token verification failed: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
