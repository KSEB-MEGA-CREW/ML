from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import logging
import time
import base64
from typing import Dict, Any, Optional

from ..models.model_loader import ModelManager
from ..models.predictor import SignLanguagePredictor
from ..core.session_manager import SessionManager
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Session manager instance
session_manager = SessionManager()


# Request/Response models
class FrameAnalysisRequest(BaseModel):
    frame_data: str
    session_id: str
    frame_index: int
    timestamp: Optional[int] = None
    request_id: Optional[str] = None
    user_id: Optional[int] = None

    @validator("frame_data")
    def validate_base64(cls, v):
        try:
            # Remove data URL prefix if present
            if "," in v:
                v = v.split(",")[1]
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 frame_data")

    @validator("frame_index")
    def validate_frame_index(cls, v):
        if v < 0:
            raise ValueError("frame_index must be non-negative")
        return v


class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: str
    processing_time_ms: Optional[float] = None


# Dependency injection
async def get_predictor() -> SignLanguagePredictor:
    """Get predictor instance with error handling"""
    try:
        model_manager = ModelManager()
        if not model_manager.is_loaded():
            raise HTTPException(
                status_code=503, detail="Model not loaded. Please try again later."
            )
        return SignLanguagePredictor(model_manager.model)
    except Exception as e:
        logger.error(f"Failed to get predictor: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")


# API endpoints
@router.post("/analyze-frame", response_model=AnalysisResponse)
async def analyze_frame(
    request: FrameAnalysisRequest,
    predictor: SignLanguagePredictor = Depends(get_predictor),
) -> Dict[str, Any]:
    """Frame analysis API with comprehensive error handling"""
    start_time = time.time()

    try:
        # Validate request
        if not request.frame_data:
            raise HTTPException(status_code=400, detail="Missing frame_data")

        if not request.session_id:
            raise HTTPException(status_code=400, detail="Missing session_id")

        # Perform sign language prediction
        prediction_result = await predictor.predict(
            base64_image=request.frame_data,
            session_id=request.session_id,
            frame_index=request.frame_index,
        )

        # Store result in session
        await session_manager.add_frame_result(
            request.session_id, request.frame_index, prediction_result
        )

        processing_time = time.time() - start_time

        response = {
            "success": True,
            "data": {
                **prediction_result,
                "processing_time_ms": round(processing_time * 1000, 2),
                "timestamp": request.timestamp or int(time.time() * 1000),
                "request_id": request.request_id,
            },
            "message": "Analysis completed successfully",
        }

        logger.info(
            f"Analysis completed: session={request.session_id}, "
            f"frame={request.frame_index}, time={processing_time:.3f}s"
        )
        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in frame analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check API"""
    try:
        model_manager = ModelManager()
        session_stats = await session_manager.get_session_stats()

        return {
            "status": "healthy",
            "model_loaded": model_manager.is_loaded(),
            "model_info": model_manager.get_model_info(),
            "session_stats": session_stats,
            "timestamp": time.time(),
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}


@router.get("/labels")
async def get_labels() -> Dict[str, Any]:
    """Get available class labels"""
    try:
        from ..utils.label_loader import LabelLoader

        label_loader = LabelLoader()
        labels = await label_loader.load_class_labels()

        return {"success": True, "labels": labels, "count": len(labels)}
    except Exception as e:
        logger.error(f"Failed to get labels: {e}")
        raise HTTPException(status_code=500, detail="Failed to load labels")


@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Get model information"""
    try:
        model_manager = ModelManager()
        return {"success": True, "model_info": model_manager.get_model_info()}
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@router.get("/session/{session_id}/results")
async def get_session_results(session_id: str, count: int = 10) -> Dict[str, Any]:
    """Get recent results for a session"""
    try:
        if count <= 0 or count > 100:
            raise HTTPException(
                status_code=400, detail="Count must be between 1 and 100"
            )

        results = await session_manager.get_recent_results(session_id, count)

        return {
            "success": True,
            "session_id": session_id,
            "results": results,
            "count": len(results),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session results: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session results")
