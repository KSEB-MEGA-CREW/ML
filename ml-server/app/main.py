import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.models.model_manager import ModelManager
from app.websockets.handlers import router as websocket_router
from app.utils.logger import setup_logger
from app.core.config import settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger = logging.getLogger(__name__)

    # Startup
    logger.info("🚀 AI Server starting")

    try:
        model_manager = ModelManager()
        await model_manager.load_model()

        if model_manager.is_ready():
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ Model loading failed")
            raise RuntimeError("Model loading is required")

    except Exception as e:
        logger.error(f"❌ Server startup error: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 AI Server shutting down")
    try:
        model_manager = ModelManager()
        if model_manager.is_ready():
            model_manager.unload_model()
            logger.info("Model unloaded successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


app = FastAPI(
    title="Sign Language AI Server",
    description="Real-time sign language recognition via WebSocket",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket 라우터 포함
app.include_router(websocket_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger = logging.getLogger(__name__)
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


@app.get("/health")
async def health_check():
    model_manager = ModelManager()
    return {
        "status": "healthy",
        "model_ready": model_manager.is_ready(),
        "model_info": model_manager.get_model_info(),
    }
