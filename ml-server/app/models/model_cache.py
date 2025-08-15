# model cache
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
        """Generate cache file path by hashing S3 key"""
        key_hash = hashlib.md5(s3_key.encode()).hexdigest()
        return str(self.cache_dir / f"{key_hash}.h5")

    def is_cached(self, s3_key: str) -> bool:
        """Check if model is cached"""
        cache_path = self.get_cache_path(s3_key)
        exists = os.path.exists(cache_path)
        if exists:
            # Verify file is not corrupted
            try:
                size = os.path.getsize(cache_path)
                if size == 0:
                    logger.warning(f"Cached model file is empty: {cache_path}")
                    os.remove(cache_path)
                    return False
            except OSError:
                return False
        return exists

    def clear_cache(self):
        """Clear cache directory"""
        try:
            for file in self.cache_dir.glob("*.h5"):
                file.unlink()
            logger.info("Model cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
