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
            logger.info(f"Verifying token with backend: {url}")

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    url,
                    json={"token": token},
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Backend response: {data}")

                    # backend response structure
                    if data.get("success", False):
                        token_data = data.get("data", {})
                        if token_data.get("valid", False):
                            user_id = token_data.get("userId")
                            logger.info(f"✅ Token verified for user: {user_id}")
                            return str(user_id)
                        else:
                            logger.warning(
                                f"❌ Token invalid: {token_data.get('message')}"
                            )
                            return None
                    else:
                        logger.warning(f"❌ Backend error: {data.get('message')}")
                        return None
                else:
                    logger.error(f"❌ HTTP error: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
