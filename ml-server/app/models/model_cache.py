# 다운로드 받은 모델 캐시 관리
import os
import hashlib
from pathlib import Path
from ..config import settings
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    def __init__(self):
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, s3_key: str) -> str:
        """S3 키를 해싱하여 캐시 파일 경로 생성"""
        key_hash = hashlib.md5(s3_key.encode()).hexdigest()
        return str(self.cache_dir / f"{key_hash}.h5")  # 모델 확장자 h5

    def is_cached(self, s3_key) -> bool:
        """모델 캐시 여부 확인"""
        cache_path = self.get_cache_path(s3_key)
        return os.path.exists(cache_path)

    def clear_cache(self):
        """캐시 디렉토리 정리 unlink"""
        for file in self.cache_dir.glob("*.h5"):
            file.unlink()
        logger.info("모델 캐시 정리 완료")
