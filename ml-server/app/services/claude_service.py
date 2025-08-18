# cluade api
from anthropic import Anthropic
import asyncio
import logging
from typing import List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class ClaudeService:
    def __init__(self):
        """Claude API client 초기화"""
        self.client = Anthropic(api_key=settings.claude_api_key)
        self.model = settings.claude_model
        self.max_tokens = settings.claude_max_tokens

    async def translate_glosses_to_korean(self, glosses: List[str]) -> str:
        """gloss 배열 -> text 변환 prompt"""
        try:
            gloss_text = " ".join(glosses)

            prompt = f"""
            한국어 수어의 글로스를 입력받아 글로스의 의미를 이해하고 문법적으로 올바른 한국어 문장으로 변환하려 한다.
            입력될 수 있는 글로스는 다음과 같다.
            [ "좋다1", "지시1#", "돕다1", "무엇1", "지시2", "때2", "오늘1", "일하다1", "재미1", "필요1", "회사1", "요리1","괜찮다1", "잘하다2"]
            한국어 문장으로 변환할 때, 구어체로 자연스럽게 변환하시오.
            수어의 문법 순서는 한국어와 다를 수 있으므로, 글로스의 의미를 이해하고
            조합을 중심으로 한국어 문법에 맞게 변환하시오.
            구어체로 자연스러운 문장으로 변환하시오.
            
            입력된 한국 수어의 글로스(gloss) 시퀀스:{gloss_text}
            변환된 한국어 문장:"""

            # 비동기 API 호출
            response = await asyncio.create_task(self._call_claude_api(prompt))

            # 응답에서 번역된 텍스트 추출
            translated_text = response.content[0].text.strip()

            logger.info(f"Claude 번역 완료: {gloss_text} -> {translated_text}")
            return translated_text

        except Exception as e:
            logger.error(f"Claude API 호출 실패: {e}")
            # 기본 응답 반환
            return " ".join(glosses)

    async def _call_claude_api(self, prompt: str):
        """Claude API 비동기 호출"""
        loop = asyncio.get_event_loop()

        # Claude API는 동기이므로 스레드 풀에서 실행
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ),
        )

        return response


# 싱글톤 인스턴스
claude_service = ClaudeService()
