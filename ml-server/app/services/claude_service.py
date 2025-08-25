# app/services/claude_service.py
import asyncio
import logging
from typing import List, Optional, Dict, Any
import anthropic
from anthropic import AsyncAnthropic

from app.core.config import settings

logger = logging.getLogger(__name__)


class ClaudeService:
    """Claude API 서비스 (WebSocket 최적화)"""

    def __init__(self):
        self.client: Optional[AsyncAnthropic] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> AsyncAnthropic:
        """Claude 클라이언트 인스턴스 반환"""
        if self.client is None:
            async with self._client_lock:
                if self.client is None:
                    self.client = AsyncAnthropic(api_key=settings.claude_api_key)
        return self.client

    async def generate_sentence(
        self, gloss_sequence: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        수어 gloss 시퀀스를 한국어 문장으로 변환

        Args:
            gloss_sequence: 수어 단어(gloss) 리스트

        Returns:
            Dict with 'sentence' and 'processing_time' keys
        """
        if not settings.claude_enabled:
            logger.info("Claude API가 비활성화됨")
            return None

        if not gloss_sequence:
            logger.warning("빈 gloss 시퀀스")
            return None

        try:
            start_time = asyncio.get_event_loop().time()

            # 프롬프트 생성
            prompt = self._create_translation_prompt(gloss_sequence)

            # Claude API 호출
            client = await self._get_client()
            response = await client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 일관성 있는 번역을 위해 낮은 temperature
            )

            # 응답 처리
            if response.content and len(response.content) > 0:
                sentence = response.content[0].text.strip()
                processing_time = asyncio.get_event_loop().time() - start_time

                logger.info(
                    f"Claude 번역 완료 ({processing_time:.2f}s): {' '.join(gloss_sequence)} → {sentence}"
                )

                return {
                    "sentence": sentence,
                    "processing_time": processing_time,
                    "gloss_count": len(gloss_sequence),
                }
            else:
                logger.warning("Claude API 응답이 비어있음")
                return None

        except anthropic.APIError as e:
            logger.error(f"Claude API 오류: {e}")
            return None

        except Exception as e:
            logger.error(f"Claude 서비스 오류: {e}")
            return None

    def _create_translation_prompt(self, gloss_sequence: List[str]) -> str:
        """번역용 프롬프트 생성"""
        gloss_text = " ".join(gloss_sequence)

        prompt = f"""다음은 한국 수어의 gloss(수어 단어) 시퀀스입니다. 이를 자연스러운 한국어 문장으로 번역해 주세요.

수어 gloss: {gloss_text}

번역 규칙:
1. 수어의 특성을 고려하여 자연스러운 한국어 문장으로 변환
2. 문법적으로 올바른 문장 구성
3. 간결하고 명확한 표현 사용
4. 문장 부호 포함하여 완성된 문장으로 응답
5. 동일한 gloss가 반복해서 등장할 경우, gloss 하나로 처리
6. 오직 변환된 한국어 문장만 출력

한국어 번역:"""

        return prompt

    async def test_connection(self) -> bool:
        """Claude API 연결 테스트"""
        try:
            test_result = await self.generate_sentence(["안녕"])
            return test_result is not None
        except Exception as e:
            logger.error(f"Claude 연결 테스트 실패: {e}")
            return False

    async def close(self):
        """Claude 클라이언트 정리"""
        if self.client:
            await self.client.close()
            logger.info("Claude 클라이언트 정리 완료")


# 전역 Claude 서비스 인스턴스
claude_service = ClaudeService()
