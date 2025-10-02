# main.py
from fastapi import FastAPI, WebSocket, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from contextlib import asynccontextmanager
import logging
import uvicorn
import time

from app.websockets.handlers import websocket_handler
from app.core.config import settings
from app.models.model_manager import model_manager

# 로깅 설정 (환경변수 기반)
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    logger.info("🚀 AI 서버 시작 중...")

    # 모델 로딩
    model_loaded = await model_manager.load_model()
    if not model_loaded:
        logger.error("❌ 모델 로딩 실패")
        raise RuntimeError("Model loading failed")

    logger.info(
        f"🌐 WebSocket 서버 실행: {settings.ai_server_host}:{settings.ai_server_port}"
    )
    yield

    # 종료 시 정리
    logger.info("🛑 AI 서버 종료 중...")
    await websocket_handler.cleanup_all_sessions()
    logger.info("✅ AI 서버 종료 완료")


# FastAPI 앱 생성
app = FastAPI(
    title="수어 인식 AI 서버 (WebSocket)",
    description="Claude API + WebSocket 기반 실시간 수어 인식",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "수어 인식 AI 서버",
        "version": "2.0.0",
        "status": "running",
        "websocket_url": "wss://korean-signlanguage-sudam.com/ws",
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_ready": model_manager.is_model_ready(),
        "active_sessions": len(websocket_handler.session_manager.active_sessions),
        "timestamp": time.time(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    """
    WebSocket 엔드포인트

    Args:
        websocket: WebSocket 연결 객체
        token: JWT 인증 토큰 (쿼리 파라미터)
    """
    try:
        await websocket_handler.handle_connection(websocket, token)
    except WebSocketDisconnect:
        logger.info("WebSocket 연결이 정상적으로 종료되었습니다")
    except Exception as e:
        logger.error(f"WebSocket 연결 중 오류 발생: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.ai_server_host,
        port=settings.ai_server_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
