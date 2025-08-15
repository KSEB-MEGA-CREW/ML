import boto3
import asyncio
from pathlib import Path
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self):
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download model from S3 asynchronously"""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False

        try:
            # Create directory
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Async download using thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.download_file,
                settings.S3_BUCKET_NAME,
                s3_key,
                local_path,
            )

            logger.info(f"Model download completed: {s3_key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    async def check_file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3"""
        if not self.s3_client:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.s3_client.head_object, settings.S3_BUCKET_NAME, s3_key
            )
            return True
        except Exception as e:
            logger.debug(f"File not found in S3: {s3_key}, error: {e}")
            return False
