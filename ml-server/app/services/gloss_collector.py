# app/services/gloss_collector.py
from collections import deque
from typing import List, Optional
import time
import logging
from .claude_service import claude_service
from .local_translator import local_translator

logger = logging.getLogger(__name__)


class GlossCollector:
    def __init__(
        self,
        confidence_threshold: float = 0.8,  # confidence_threshold 0.8로 하향
        max_glosses: int = 100,
        timeout_seconds: int = 5,
        use_local_fallback: bool = True,  # 로컬 백업 사용 여부
    ):
        """
        Gloss 수집기 초기화

        Args:
            confidence_threshold: 신뢰도 임계값 (0.8+)
            max_glosses: 최대 수집할 gloss 개수
            timeout_seconds: 타임아웃 시간 (초)
            use_local_fallback: Claude API 실패 시 로컬 번역 사용 여부
        """
        self.confidence_threshold = confidence_threshold
        self.max_glosses = max_glosses
        self.timeout_seconds = timeout_seconds
        self.use_local_fallback = use_local_fallback

        self.glosses = deque(maxlen=max_glosses)
        self.last_gloss_time = time.time()

        # API 상태 추적
        self.claude_api_available = True
        self.consecutive_claude_failures = 0
        self.max_claude_failures = 3  # 3회 연속 실패 시 로컬로 전환

    def has_glosses(self) -> bool:
        """수집된 gloss가 있는지 확인"""
        return len(self.glosses) > 0

    def get_current_glosses(self) -> List[str]:
        """현재 수집된 gloss 목록 반환 (clear 전에 호출)"""
        return list(self.glosses)

    def get_gloss_count(self) -> int:
        """현재 수집된 gloss 개수"""
        return len(self.glosses)

    def get_status(self) -> dict:
        """현재 상태 정보 반환"""
        return {
            "gloss_count": len(self.glosses),
            "current_glosses": list(self.glosses),
            "last_gloss_time": self.last_gloss_time,
            "claude_api_available": self.claude_api_available,
            "consecutive_failures": self.consecutive_claude_failures,
            "confidence_threshold": self.confidence_threshold,
            "max_glosses": self.max_glosses,
            "timeout_seconds": self.timeout_seconds,
        }

    async def force_generate_sentence(self) -> str:
        """강제로 현재 수집된 gloss로 문장 생성"""
        if not self.glosses:
            return ""

        logger.info(f"강제 문장 생성: {list(self.glosses)}")
        return await self._generate_sentence_with_fallback()

    async def clear_glosses(self):
        """수집된 gloss 강제 초기화"""
        self.glosses.clear()
        self.last_gloss_time = time.time()
        logger.info("Gloss 버퍼 수동 초기화")

    def reset_claude_api_status(self):
        """Claude API 상태 리셋 (수동 재시도용)"""
        self.claude_api_available = True
        self.consecutive_claude_failures = 0
        logger.info("Claude API 상태 리셋 완료")

    async def add_prediction(self, label: str, confidence: float) -> Optional[str]:
        """
        예측 결과를 추가하고 문장 생성 조건을 확인

        Args:
            label: 예측된 gloss
            confidence: 신뢰도 점수

        Returns:
            생성된 한국어 문장 (조건 만족 시) 또는 None
        """
        current_time = time.time()

        # 신뢰도가 임계값 이상인 경우만 추가
        if confidence >= self.confidence_threshold:
            self.glosses.append(label)
            self.last_gloss_time = current_time

            logger.debug(f"고신뢰도 gloss 추가: {label} (신뢰도: {confidence:.3f})")

        # 문장 생성 조건 확인
        if self._should_generate_sentence(current_time):
            try:
                return await self._generate_sentence_with_fallback()
            except Exception as e:
                logger.error(f"문장 생성 중 오류: {e}")
                # 최종 백업: 단순 조합
                result = " ".join(list(self.glosses))
                self.glosses.clear()
                return result

        return None

    def _should_generate_sentence(self, current_time: float) -> bool:
        """문장 생성 조건 확인"""
        # 조건 1: 최대 gloss 개수에 도달
        if len(self.glosses) >= self.max_glosses:
            return True

        # 조건 2: 타임아웃 도달 (gloss가 1개 이상 있는 경우)
        if (
            len(self.glosses) > 0
            and (current_time - self.last_gloss_time) >= self.timeout_seconds
        ):
            return True

        return False

    async def _generate_sentence_with_fallback(self) -> str:
        """Claude API와 로컬 번역을 활용한 문장 생성"""
        if not self.glosses:
            return ""

        glosses_list = list(self.glosses)
        logger.info(f"문장 생성 시작: {glosses_list}")

        # 1차: Claude API 시도 (사용 가능한 경우)
        if self.claude_api_available:
            try:
                sentence = await claude_service.translate_glosses_to_korean(
                    glosses_list
                )

                # 성공 시 실패 카운터 리셋
                self.consecutive_claude_failures = 0

                # 초기화
                self.glosses.clear()
                self.last_gloss_time = time.time()

                return sentence

            except Exception as e:
                logger.warning(f"Claude API 호출 실패: {e}")
                self.consecutive_claude_failures += 1

                # 연속 실패 시 Claude API 일시 비활성화
                if self.consecutive_claude_failures >= self.max_claude_failures:
                    self.claude_api_available = False
                    logger.warning(
                        f"Claude API 연속 {self.max_claude_failures}회 실패 - 로컬 번역으로 전환"
                    )

        # 2차: 로컬 번역 시도
        if self.use_local_fallback:
            try:
                logger.info("로컬 번역 서비스 사용")
                sentence = await local_translator.translate_glosses_to_korean(
                    glosses_list
                )

                # 초기화
                self.glosses.clear()
                self.last_gloss_time = time.time()

                return sentence

            except Exception as e:
                logger.error(f"로컬 번역 실패: {e}")

        # 3차: 최종 백업 (단순 조합)
        logger.warning("모든 번역 서비스 실패 - 단순 조합 사용")
        sentence = " ".join(glosses_list)
        self.glosses.clear()
        self.last_gloss_time = time.time()

        return sentence

    def reset_claude_api_status(self):
        """Claude API 상태 리셋 (수동 재시도용)"""
        self.claude_api_available = True
        self.consecutive_claude_failures = 0
        logger.info("Claude API 상태 리셋 완료")


# 사용 예시
if __name__ == "__main__":
    import asyncio

    async def test_local_translation():
        collector = GlossCollector(use_local_fallback=True)

        # 테스트 데이터
        test_predictions = [
            ("재미1", 0.97),
            ("좋다1", 0.98),
            ("일하다1", 0.96),
            ("회사1", 0.99),
            ("오늘1", 0.97),
        ]

        for label, confidence in test_predictions:
            sentence = await collector.add_prediction(label, confidence)
            if sentence:
                print(f"생성된 문장: {sentence}")

        # 강제 문장 생성
        if len(collector.glosses) > 0:
            final_sentence = await collector._generate_sentence_with_fallback()
            print(f"최종 문장: {final_sentence}")

    asyncio.run(test_local_translation())

gloss_collector = GlossCollector()
