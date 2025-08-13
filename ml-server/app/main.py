import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.model_loader import ModelManager
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Mega-Crew SLT AI Server",
    description="실시간 수화 번역 AI 서버",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
# model loader/cache
model_manager = ModelManager()

# CORS 설정 (FE,BE 연결용)
# BE ->(frame data) -> AI
# AI ->(gloss data) -> FE
# AI ->(interpreted text) -> FE
# front : http://mega-crew-react-deploy.s3-website.ap-northeast-2.amazonaws.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://mega-crew-react-deploy.s3-website.ap-northeast-2.amazonaws.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 모델 정의
# class HealthResponse(BaseModel):
#     status: str
#     message: str
#     version: str


# class FrameRequest(BaseModel):
#     frame_data: str  # Base64 인코딩된 이미지
#     timestamp: int
#     session_id: str
#     frame_index: int


# class FrameResponse(BaseModel):
#     success: bool
#     result: dict = None  # text -> aws bedrock에 연결


# 기본 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "Sign Language AI Server", "status": "running", "docs": "/docs"}


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """헬스체크 엔드포인트"""
#     return HealthResponse(
#         status="healthy", message="AI 서버가 정상적으로 동작 중입니다.", version="1.0.0"
#     )


# 임시 프레임 분석 엔드포인트 (모델 로드 전)
# @app.post("/analyze-frame", response_model=FrameResponse)
# async def analyze_frame(request: FrameRequest):
#     """프레임 분석 엔드포인트 (임시 구현)"""
#     try:
#         logger.info(
#             f"프레임 분석 요청 - Session: {request.session_id}, Frame: {request.frame_index}"
#         )

#         # 임시 응답 (실제 모델 구현 전)
#         return FrameResponse(
#             success=True,
#             message="프레임 분석 완료 (임시)",
#             result={
#                 "predicted_text": "안녕하세요",
#                 "confidence": 0.85,
#                 "processing_time": 0.1,
#             },
#         )

#     except Exception as e:
#         logger.error(f"프레임 분석 오류: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail=f"프레임 분석 중 오류가 발생했습니다: {str(e)}"
#         )


# 서버 정보 엔드포인트
@app.get("/info")
async def server_info():
    """서버 정보 조회"""
    return {
        "server": "Sign Language AI Server",
        "version": "1.0.0",
        "python_version": "3.9+",
        "framework": "FastAPI",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze-frame",
            "docs": "/docs",
        },
    }


# 애플리케이션 시작/종료 이벤트
@app.on_event("startup")
async def startup_event():
    logger.info("AI 서버 시작 중...")
    success = await model_manager.load_model()
    if not success:
        logger.error("모델 로드 실패 - 서버 시작 중단")
        raise Exception("모델 로드 실패")
    logger.info("AI 서버 시작 완료")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_manager.is_ready()}


@app.get("/model/status")
async def model_status():
    return {
        "loaded": model_manager.is_ready(),
        "cache_info": (
            "cached"
            if model_manager.cache.is_cached(
                model_manager.s3_client.settings.MODEL_S3_KEY
            )
            else "not_cached"
        ),
    }


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    logger.info("🛑 AI 서버가 종료되었습니다.")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"

    uvicorn.run("main:app", host=host, port=port, reload=debug, log_level="info")
