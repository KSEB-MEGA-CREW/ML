# app/services/local_translator.py
import logging
from typing import List, Dict
import random

logger = logging.getLogger(__name__)


class LocalTranslator:
    """Claude API 대안 - 로컬 번역 서비스"""

    def __init__(self):
        # 글로스별 한국어 매핑
        self.gloss_mapping = {
            "좋다1": "좋다",
            "재미1": "재미있다",
            "일하다1": "일하다",
            "회사1": "회사",
            "오늘1": "오늘",
            "지시1#": "이것",
            "지시2": "저것",
            "돕다1": "돕다",
            "무엇1": "무엇",
            "때2": "때",
            "필요1": "필요하다",
            "요리1": "요리",
            "괜찮다1": "괜찮다",
            "잘하다2": "잘하다",
        }

        # 문장 패턴 템플릿
        self.sentence_patterns = [
            "{subject} {predicate}",
            "{time} {subject} {predicate}",
            "{subject} {object} {predicate}",
            "{time} {subject} {object} {predicate}",
        ]

        # 품사별 분류
        self.word_categories = {
            "time": ["오늘"],
            "subject": ["회사", "이것", "저것"],
            "object": ["요리", "무엇"],
            "predicate": [
                "좋다",
                "재미있다",
                "일하다",
                "돕다",
                "필요하다",
                "괜찮다",
                "잘하다",
            ],
        }

    async def translate_glosses_to_korean(self, glosses: List[str]) -> str:
        """글로스를 한국어 문장으로 변환"""
        try:
            if not glosses:
                return ""

            # 중복 제거하면서 순서 유지
            unique_glosses = list(dict.fromkeys(glosses))

            logger.info(f"로컬 번역 시작: {unique_glosses}")

            # 글로스를 한국어로 변환
            korean_words = []
            for gloss in unique_glosses:
                korean_word = self.gloss_mapping.get(
                    gloss, gloss.replace("1", "").replace("2", "").replace("#", "")
                )
                korean_words.append(korean_word)

            # 문장 생성 시도
            sentence = self._generate_sentence(korean_words, unique_glosses)

            logger.info(f"로컬 번역 완료: {unique_glosses} -> {sentence}")
            return sentence

        except Exception as e:
            logger.error(f"로컬 번역 실패: {e}")
            # 최소한의 결과라도 반환
            return " ".join(glosses)

    def _generate_sentence(
        self, korean_words: List[str], original_glosses: List[str]
    ) -> str:
        """문장 생성 로직"""

        # 1. 단순 연결 (기본값)
        if len(korean_words) == 1:
            return korean_words[0]

        # 2. 패턴 기반 문장 생성
        try:
            # 품사별 단어 분류
            categorized = {
                "time": [w for w in korean_words if w in self.word_categories["time"]],
                "subject": [
                    w for w in korean_words if w in self.word_categories["subject"]
                ],
                "object": [
                    w for w in korean_words if w in self.word_categories["object"]
                ],
                "predicate": [
                    w for w in korean_words if w in self.word_categories["predicate"]
                ],
            }

            # 서술어가 있는 경우 문장 구성
            if categorized["predicate"]:
                predicate = categorized["predicate"][-1]  # 마지막 서술어 사용

                # 시간 + 주어 + 서술어
                if categorized["time"] and categorized["subject"]:
                    return f"{categorized['time'][0]} {categorized['subject'][0]}에서 {predicate}"

                # 주어 + 목적어 + 서술어
                elif categorized["subject"] and categorized["object"]:
                    return f"{categorized['subject'][0]}에서 {categorized['object'][0]}를 {predicate}"

                # 주어 + 서술어
                elif categorized["subject"]:
                    return f"{categorized['subject'][0]}가 {predicate}"

                # 서술어만
                else:
                    return predicate

            # 3. 특별한 조합 처리
            return self._handle_special_combinations(korean_words)

        except Exception as e:
            logger.warning(f"패턴 기반 생성 실패: {e}")
            return self._handle_special_combinations(korean_words)

    def _handle_special_combinations(self, words: List[str]) -> str:
        """특별한 단어 조합 처리"""

        # 자주 나오는 조합들
        if "재미있다" in words and "좋다" in words:
            return "재미있고 좋다"

        if "오늘" in words and "일하다" in words:
            return "오늘 일한다"

        if "회사" in words and "일하다" in words:
            return "회사에서 일한다"

        if "좋다" in words:
            if len(words) > 3:
                return "정말 좋다"
            else:
                return "좋다"

        # 기본: 마지막 단어를 중심으로
        if len(words) >= 2:
            return f"{words[0]} {words[-1]}"

        return " ".join(words)

    def test_connection(self) -> bool:
        """연결 테스트 (항상 성공)"""
        return True


# 싱글톤 인스턴스
local_translator = LocalTranslator()
