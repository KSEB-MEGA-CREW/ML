# API endpoints
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import time
from typing import Dict, Any

from ..models.model_loader import ModelManager
from ..models.predictor import SignLanguagePredictor
from ..core.session_manager import SessionManager
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# session manager
session_manager = SessionManager()


# 의존성 주입
def get_predictor() -> SignLanguagePredictor:
    model_manager = ModelManager()
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    return SignLanguagePredictor(model_manager.model)


# API 경로 지정
@router.post("/analyze-frame")
async def analyze_frame(
    request: dict, predictor: SignLanguagePredictor = Depends(get_predictor)
) -> Dict[str, Any]:
    """프레임 분석 API"""
    start_time = time.time()

    try:
        # 세션 관리
        session_id = request.get("session_id")
        frame_index = request.get("frame_index")

        # 수어 예측 수행
        prediction_result = await predictor.predict(
            base64_image=request.get("frame_data"),
            session_id=session_id,
            frame_index=frame_index,
        )

        # session에 결과 저장
        session_manager.add_frame_result(session_id, frame_index, prediction_result)

        processing_time = time.time() - start_time

        response = {
            "success": True,
            "data": {
                **prediction_result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "timestamp": request.get("timestamp"),
                "request_id": request.get("request_id"),
            },
            "message": "분석 완료",
        }

        logger.info(f"분석 완료: session={session_id}, time={processing_time:.3f}s")
        return response

    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="내부 서버 오류")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """헬스체크 API"""
    model_manager = ModelManager()

    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "timestamp": time.time(),
    }
