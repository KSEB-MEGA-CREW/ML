# main.py
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import time

from app.websockets.handlers import websocket_handler
from app.core.config import settings
from app.models.model_manager import ModelManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="수어 인식 AI 서버",
    description="Claude API를 사용한 실시간 수어 인식 시스템",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경용, 운영에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 매니저 인스턴스
model_manager = ModelManager()


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "수어 인식 AI 서버", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_ready": model_manager.is_ready(),  # ✅ 실제 모델 상태 확인
        "model_info": model_manager.get_model_info(),
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


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    logger.info(
        f"수어 인식 AI 서버 시작: {settings.ai_server_host}:{settings.ai_server_port}"
    )

    # ✅ 모델 로딩
    logger.info("모델 로딩 시작...")
    success = await model_manager.load_model()

    if success:
        logger.info("✅ 모델 로딩 완료")
        model_info = model_manager.get_model_info()
        logger.info(f"모델 정보: {model_info}")
    else:
        logger.error("❌ 모델 로딩 실패")
        raise RuntimeError("모델 로딩에 실패했습니다. 서버를 시작할 수 없습니다.")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("수어 인식 AI 서버 종료")
    # ✅ 모델 메모리 해제
    model_manager.unload_model()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.ai_server_host,
        port=settings.ai_server_port,
        reload=True,  # 개발 환경용
        log_level="info",
    )
