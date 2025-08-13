import boto3
import asyncio
from pathlib import Path
from app.utils.config import settings
import logging

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )

    async def download_file(self, s3_key: str, local_path: str) -> bool:
        """S3에서 모델을 비동기로 다운로드"""
        try:
            # 디렉토리 생성
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # 비동기 다운로드
            # 동기식일 경우 모델 다운로드 중 다른 작업 불가
            # main event loop를 가져와 별도 쓰레드 생성 후
            # 별드 스레드에서 다운로드 진행 -> 동시에 다른 작업 가능
            # blocking 작업을 thread pool에서 실행
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.download_file,
                settings.S3_BUCKET,
                s3_key,
                local_path,
            )

            logger.info(f"모델 다운로드 완료: {s3_key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"S3 다운로드 실패: {e}")
            return False

    async def check_file_exists(self, s3_key: str) -> bool:
        """S3에 파일이 존재하는지 확인"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.s3_client.head_object, settings.S3_BUCKET, s3_key
            )
            return True
        except:
            return False
