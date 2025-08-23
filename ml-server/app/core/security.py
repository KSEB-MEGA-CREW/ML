# app/core/security.py 수정
import httpx
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class TokenVerifier:
    @staticmethod
    async def verify_token(token: str) -> Optional[str]:
        """verify jwt with backend server and return user_id"""
        try:
            url = f"{settings.backend_url}{settings.backend_token_verify_endpoint}"
            logger.info(f"🔍 Verifying token with backend: {url}")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    url,
                    json={"token": token},
                    headers={"Content-Type": "application/json"},
                )

                logger.info(f"📡 Backend response status: {response.status_code}")

                if response.status_code == 200:
                    response_text = response.text
                    logger.info(f"📄 Raw response text: {response_text}")

                    data = response.json()
                    logger.info(f"📋 Parsed JSON data: {data}")
                    logger.info(f"📋 Data type: {type(data)}")

                    # 응답 구조 확인
                    success = data.get("success", False)
                    logger.info(f"🔍 success field: {success} (type: {type(success)})")

                    if success:
                        token_data = data.get("data", {})
                        logger.info(
                            f"🔍 token_data: {token_data} (type: {type(token_data)})"
                        )

                        if isinstance(token_data, dict):
                            valid = token_data.get("valid", False)
                            logger.info(
                                f"🔍 valid field: {valid} (type: {type(valid)})"
                            )

                            if valid:
                                user_id = token_data.get("userId")
                                logger.info(
                                    f"🔍 userId field: {user_id} (type: {type(user_id)})"
                                )
                                logger.info(
                                    f"🔍 Available keys in token_data: {list(token_data.keys())}"
                                )

                                if user_id is not None:  # 0도 유효한 user_id일 수 있음
                                    logger.info(
                                        f"✅ Token verified for user: {user_id}"
                                    )
                                    return str(user_id)
                                else:
                                    logger.error(
                                        f"❌ userId is None! token_data: {token_data}"
                                    )

                                    # 대안 1: 다른 필드명 시도
                                    alt_user_id = (
                                        token_data.get("user_id")
                                        or token_data.get("id")
                                        or token_data.get(
                                            "email"
                                        )  # email을 user_id로 사용
                                    )

                                    if alt_user_id:
                                        logger.warning(
                                            f"⚠️ 대안 user_id 사용: {alt_user_id}"
                                        )
                                        return str(alt_user_id)

                                    return None
                            else:
                                logger.warning(
                                    f"❌ Token invalid: {token_data.get('message', 'Unknown reason')}"
                                )
                                return None
                        else:
                            logger.error(
                                f"❌ token_data is not dict: {type(token_data)}"
                            )
                            return None
                    else:
                        logger.error(
                            f"❌ Backend request failed: {data.get('message', 'Unknown error')}"
                        )
                        return None
                else:
                    logger.error(f"❌ Backend HTTP error: {response.status_code}")
                    logger.error(f"❌ Response text: {response.text}")
                    return None

        except Exception as e:
            logger.error(f"❌ Token verification error: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
