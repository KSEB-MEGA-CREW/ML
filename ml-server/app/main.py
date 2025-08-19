# main.py
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
import time

from app.websockets.handlers import websocket_handler
from app.core.config import settings
from app.models.model_manager import model_manager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시 초기화
    logger.info("🚀 AI 서버 시작 중...")

    # 모델 로딩 (동기적으로 완료 대기)
    model_loaded = await model_manager.load_model()
    if not model_loaded:
        logger.error("❌ 모델 로딩 실패 - 서버 시작 중단")
        raise RuntimeError("Model loading failed")

    logger.info(f"🌐 서버 실행: {settings.ai_server_host}:{settings.ai_server_port}")

    yield  # 서버 실행 중

    # 종료 시 정리
    logger.info("🛑 AI 서버 종료 중...")

    # 모델 정리
    if hasattr(model_manager, "executor"):
        model_manager.executor.shutdown(wait=True)

    logger.info("✅ AI 서버 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="수어 인식 AI 서버",
    description="Claude API를 사용한 실시간 수어 인식 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경용, 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "수어 인식 AI 서버", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_ready": model_manager.is_model_ready(),
        "timestamp": time.time(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """
    WebSocket 엔드포인트

    Args:
        websocket: WebSocket 연결
        token: JWT 인증 토큰
    """
    await websocket_handler.handle_connection(websocket, token)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.ai_server_host,
        port=settings.ai_server_port,
        reload=True,
        log_level="info",
    )
