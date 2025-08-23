# app/services/gloss_collector.py
import time
from collections import deque
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GlossCollector:
    """고신뢰도 gloss 수집 및 번역 관리 (사용자 제어 방식)"""

    def __init__(self, max_glosses: int = 20):  # 더 많은 gloss 수집 가능
        self.max_glosses = max_glosses

        # gloss 저장소: (gloss, confidence, timestamp)
        self.glosses: deque = deque(maxlen=max_glosses)
        self.translation_active = False  # 번역 활성 상태
        self.translation_start_time: Optional[float] = None

        # 통계
        self.total_glosses_added = 0
        self.translations_generated = 0

    def start_translation(self):
        """번역 시작 (사용자가 시작 버튼 클릭)"""
        self.translation_active = True
        self.translation_start_time = time.time()
        self.glosses.clear()  # 새 번역 시작 시 기존 gloss 초기화

        logger.info("번역 시작: gloss 수집 활성화")

    def stop_translation(self) -> bool:
        """
        번역 종료 (사용자가 종료 버튼 클릭)

        Returns:
            bool: 번역할 gloss가 있는지 여부
        """
        self.translation_active = False
        has_glosses = len(self.glosses) > 0

        if has_glosses:
            logger.info(f"번역 종료: {len(self.glosses)}개 gloss로 번역 수행")
        else:
            logger.info("번역 종료: 수집된 gloss 없음")

        return has_glosses

    def add_gloss(self, gloss: str, confidence: float) -> bool:
        """
        고신뢰도 gloss 추가 (번역이 활성화된 경우에만)

        Args:
            gloss: 수어 단어
            confidence: 신뢰도 (0.0 ~ 1.0)

        Returns:
            bool: 추가 성공 여부
        """
        # 번역이 비활성화된 경우 추가하지 않음
        if not self.translation_active:
            return False

        # 신뢰도 임계값 확인 0.85+
        if confidence < 0.85:
            return False

        # 중복 제거 (마지막 gloss와 같으면 무시)
        if self.glosses and self.glosses[-1][0] == gloss:
            return False

        # gloss 추가
        current_time = time.time()
        self.glosses.append((gloss, confidence, current_time))
        self.total_glosses_added += 1

        logger.debug(
            f"Gloss 추가: {gloss} (신뢰도: {confidence:.3f}) [{len(self.glosses)}/{self.max_glosses}]"
        )

        return True

    def should_generate_translation(self) -> str:
        """
        번역 생성 조건 확인

        Returns:
            str: 번역 트리거 유형 ("user_end", "none")
        """
        if not self.glosses:
            return "none"

        # 사용자가 번역을 종료한 경우만 번역 수행
        if not self.translation_active and len(self.glosses) > 0:
            return "user_end"

        return "none"

    def get_collected_glosses(self) -> List[str]:
        """수집된 gloss 리스트 반환"""
        return [gloss for gloss, _, _ in self.glosses]

    def get_gloss_count(self) -> int:
        """현재 수집된 gloss 개수"""
        return len(self.glosses)

    def get_average_confidence(self) -> float:
        """평균 신뢰도 반환"""
        if not self.glosses:
            return 0.0

        total_confidence = sum(confidence for _, confidence, _ in self.glosses)
        return total_confidence / len(self.glosses)

    def get_confidence_details(self) -> List[Tuple[str, float]]:
        """각 gloss의 신뢰도 상세 정보 반환"""
        return [(gloss, confidence) for gloss, confidence, _ in self.glosses]

    def reset_after_translation(self):
        """번역 완료 후 초기화"""
        gloss_count = len(self.glosses)
        self.glosses.clear()
        self.translation_active = False
        self.translation_start_time = None

        if gloss_count > 0:
            self.translations_generated += 1
            logger.info(f"번역 완료 후 초기화: {gloss_count}개 gloss 처리됨")

    def is_translation_active(self) -> bool:
        """번역 활성 상태 확인"""
        return self.translation_active

    def has_glosses(self) -> bool:
        """수집된 gloss가 있는지 확인"""
        return len(self.glosses) > 0

    def get_translation_duration(self) -> Optional[float]:
        """현재 번역 세션 지속 시간 (초)"""
        if self.translation_start_time is None:
            return None
        return time.time() - self.translation_start_time

    def get_stats(self) -> dict:
        """통계 정보 반환"""
        return {
            "current_glosses": len(self.glosses),
            "translation_active": self.translation_active,
            "total_glosses_added": self.total_glosses_added,
            "translations_generated": self.translations_generated,
            "average_confidence": self.get_average_confidence(),
            "translation_duration": self.get_translation_duration(),
        }


gloss_collector = GlossCollector()
