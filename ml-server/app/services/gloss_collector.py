# confidence 0.96+ gloss collector
# app/services/gloss_collector.py
from collections import deque
from typing import List, Optional
import time
import logging
from .claude_service import claude_service

logger = logging.getLogger(__name__)


class GlossCollector:
    def __init__(
        self,
        confidence_threshold: float = 0.96,
        max_glosses: int = 100,  # 최대 수집할 gloss 개수 -> 추후 조정 필요
        timeout_seconds: int = 5,
    ):
        """
        Gloss 수집기 초기화

        Args:
            confidence_threshold: 신뢰도 임계값 (0.96+)
            max_glosses: 최대 수집할 gloss 개수
            timeout_seconds: 타임아웃 시간 (초)
        """
        self.confidence_threshold = confidence_threshold
        self.max_glosses = max_glosses
        self.timeout_seconds = timeout_seconds

        self.glosses = deque(maxlen=max_glosses)
        self.last_gloss_time = time.time()

    def add_prediction(self, label: str, confidence: float) -> Optional[str]:
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
            return self._generate_sentence()

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

    async def _generate_sentence(self) -> str:
        """수집된 gloss를 문장으로 변환"""
        if not self.glosses:
            return ""

        glosses_list = list(self.glosses)
        logger.info(f"문장 생성 시작: {glosses_list}")

        try:
            # Claude API를 통해 번역
            sentence = await claude_service.translate_glosses_to_korean(glosses_list)

            # 초기화
            self.glosses.clear()
            self.last_gloss_time = time.time()

            return sentence

        except Exception as e:
            logger.error(f"문장 생성 실패: {e}")
            # 에러 시 기본 문장 반환
            sentence = " ".join(glosses_list)
            self.glosses.clear()
            return sentence
