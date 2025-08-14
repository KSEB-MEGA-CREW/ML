import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.model_loader import ModelManager
from app.api.endpoints import router
from app.utils.logger import setup_logger
from app.config import settings

# logging setting
setup_logger()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # starting...
    logger.info("🚀 AI 서버 시작")
    logger.info(f"환경: {settings.DEBUG and 'Development' or 'Production'}")
    logger.info(f"TensorFlow GPU 사용: {settings.TF_ENABLE_GPU_MEMORY_GROWTH}")

    try:
        # 모델 매니저 싱글톤 인스턴스 생성
        model_manager = ModelManager()

        # 모델 로드
        logger.info("📦 모델 로드 시작...")
        success = await model_manager.load_model()

        if success:
            logger.info("✅ 모델 로드 완료")
            model_info = model_manager.get_model_info()
            logger.info(f"모델 정보: {model_info}")
        else:
            logger.error("❌ 모델 로드 실패")
            # 개발 환경에서는 계속 진행, 프로덕션에서는 종료
            if not settings.DEBUG:
                raise RuntimeError("모델 로드 필수")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        if not settings.DEBUG:
            raise

    yield

    # 종료 시
    logger.info("🛑 AI 서버 종료")
    try:
        model_manager = ModelManager()
        if model_manager.is_ready():
            model_manager.unload_model()
            logger.info("모델 언로드 완료")
    except Exception as e:
        logger.error(f"종료 중 오류: {e}")

# FastAPI 앱 생성
app = FastAPI(
    title="Sign Language AI Server",
    description="수화 인식 AI 서버 - TensorFlow 기반",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 요청 로깅 미들웨어 (개발 환경에서만)
if settings.DEBUG:

    @app.middleware("http")
    async def log_requests(request, call_next):
        import time

        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        return response


# API 라우터 등록
app.include_router(router, prefix="/api/v1", tags=["AI Analysis"])


# 루트 엔드포인트
@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    model_manager = ModelManager()

    return {
        "message": "Sign Language AI Server is running",
        "version": "1.0.0",
        "status": "healthy",
        "model_status": "loaded" if model_manager.is_ready() else "not_loaded",
        "framework": "TensorFlow",
        "endpoints": {
            "health": "/api/v1/health",
            "analyze": "/api/v1/analyze-frame",
            "labels": "/api/v1/labels",
            "model_info": "/api/v1/model-info",
            "docs": "/docs",
        },
    }


# 개발 서버 실행용 (로컬 테스트용)
if __name__ == "__main__":
    import uvicorn

    # 환경변수에서 설정 로드
    host = settings.HOST
    port = settings.PORT
    reload = settings.DEBUG

    logger.info(f"개발 서버 시작: http://{host}:{port}")
    logger.info(f"API 문서: http://{host}:{port}/docs")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower(),
    )
