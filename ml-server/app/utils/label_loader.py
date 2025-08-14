# label data loader -> cache 방식, S3에서 가져오는 방식 구현
import json
import os
from typing import List, Dict, Any, Union
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class LabelLoader:
    def __init__(self):
        self._labels_cache = None

    async def load_class_labels(self) -> List[str]:
        """클래스 라벨을 로드 (캐시 사용)"""
        if self._labels_cache is not None:
            return self._labels_cache

        try:
            # 우선 로컬에서 라벨 로드하는 경우만 구현
            # 추후 필요 시 S3 버킷에서 로드하는 경우 구현하기
            labels_data = await self._load_from_local()

            # JSON 데이터에서 클래스 리스트 추출
            # 단순 배열 형태
            if isinstance(labels_data, list):
                self._labels_cache = labels_data

            logger.info(f"클래스 라벨 로드 완료: {len(self._labels_cache)} 개 클래스")
            logger.debug(f"로드된 라벨: {self._labels_cache}")

            return self._labels_cache

        except Exception as e:
            logger.error(f"라벨 로드 실패: {str(e)}")
            return None

    async def _load_from_local(self) -> Dict[str, Any]:
        """로컬에서 라벨 로드"""
        try:
            if not os.path.exists(settings.CLASS_LABELS_FILE):
                return FileNotFoundError(
                    f"라벨 파일이 존재하지 않음: {settings.CLASS_LABELS_FILE}"
                )

            with open(settings.CLASS_LABELS_FILE, "r", encoding="utf-8") as f:
                return json.loads(f)
        except Exception as e:
            logger.error(f"로컬에서 라벨 로드 실패: {str(e)}")
            raise

    def clear_cache(self):
        """캐시 초기화"""
        self._labels_cache = None
        logger.info("라벨 캐시 초기화")
