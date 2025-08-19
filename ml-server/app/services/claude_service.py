# app/services/claude_service.py
import asyncio
import logging
from typing import List, Optional
from anthropic import Anthropic, AsyncAnthropic
from app.core.config import settings

logger = logging.getLogger(__name__)


class ClaudeService:
    def __init__(self):
        """Claude API client 초기화"""
        try:
            # 동기 클라이언트와 비동기 클라이언트 모두 생성
            self.sync_client = Anthropic(api_key=settings.claude_api_key)
            self.async_client = AsyncAnthropic(api_key=settings.claude_api_key)
            self.model = settings.claude_model
            self.max_tokens = settings.claude_max_tokens

            logger.info(f"Claude 서비스 초기화 완료: model={self.model}")

        except Exception as e:
            logger.error(f"Claude 서비스 초기화 실패: {e}")
            raise

    async def translate_glosses_to_korean(self, glosses: List[str]) -> str:
        """gloss 배열 -> 한국어 문장 변환"""
        try:
            if not glosses:
                return ""

            gloss_text = " ".join(glosses)
            logger.info(f"Claude 번역 요청: {gloss_text}")

            prompt = f"""
한국어 수어의 글로스를 입력받아 글로스의 의미를 이해하고 문법적으로 올바른 한국어 문장으로 변환하려 한다.

입력될 수 있는 글로스는 다음과 같다:
[ "좋다1", "지시1#", "돕다1", "무엇1", "지시2", "때2", "오늘1", "일하다1", "재미1", "필요1", "회사1", "요리1", "괜찮다1", "잘하다2"]

한국어 문장으로 변환할 때, 구어체로 자연스럽게 변환하시오.
수어의 문법 순서는 한국어와 다를 수 있으므로, 글로스의 의미를 이해하고 조합을 중심으로 한국어 문법에 맞게 변환하시오.

입력된 한국 수어의 글로스(gloss) 시퀀스: {gloss_text}

변환된 한국어 문장:"""

            # 비동기 API 호출
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            # 응답에서 번역된 텍스트 추출
            if response.content and len(response.content) > 0:
                translated_text = response.content[0].text.strip()
                logger.info(f"Claude 번역 완료: {gloss_text} -> {translated_text}")
                return translated_text
            else:
                logger.warning("Claude API 응답에 내용이 없음")
                return " ".join(glosses)

        except Exception as e:
            logger.error(f"Claude API 호출 실패: {e}")
            logger.exception("Claude API 상세 오류:")
            # 기본 응답 반환 (서비스 중단 방지)
            return " ".join(glosses)

    def translate_glosses_to_korean_sync(self, glosses: List[str]) -> str:
        """동기 버전 (테스트용)"""
        try:
            if not glosses:
                return ""

            gloss_text = " ".join(glosses)
            logger.info(f"Claude 동기 번역 요청: {gloss_text}")

            prompt = f"""
한국어 수어의 글로스를 입력받아 글로스의 의미를 이해하고 문법적으로 올바른 한국어 문장으로 변환하려 한다.

입력될 수 있는 글로스는 다음과 같다:
[ "좋다1", "지시1#", "돕다1", "무엇1", "지시2", "때2", "오늘1", "일하다1", "재미1", "필요1", "회사1", "요리1", "괜찮다1", "잘하다2"]

한국어 문장으로 변환할 때, 구어체로 자연스럽게 변환하시오.
수어의 문법 순서는 한국어와 다를 수 있으므로, 글로스의 의미를 이해하고 조합을 중심으로 한국어 문법에 맞게 변환하시오.

입력된 한국 수어의 글로스(gloss) 시퀀스: {gloss_text}

변환된 한국어 문장:"""

            # 동기 API 호출
            response = self.sync_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            if response.content and len(response.content) > 0:
                translated_text = response.content[0].text.strip()
                logger.info(f"Claude 동기 번역 완료: {gloss_text} -> {translated_text}")
                return translated_text
            else:
                logger.warning("Claude API 응답에 내용이 없음")
                return " ".join(glosses)

        except Exception as e:
            logger.error(f"Claude API 동기 호출 실패: {e}")
            return " ".join(glosses)

    async def test_connection(self) -> bool:
        """Claude API 연결 테스트"""
        try:
            test_response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=50,
                messages=[
                    {"role": "user", "content": "안녕하세요. 연결 테스트입니다."}
                ],
            )

            if test_response.content:
                logger.info("✅ Claude API 연결 테스트 성공")
                return True
            else:
                logger.error("❌ Claude API 연결 테스트 실패: 응답 없음")
                return False

        except Exception as e:
            logger.error(f"❌ Claude API 연결 테스트 실패: {e}")
            return False


# 싱글톤 인스턴스
claude_service = ClaudeService()
