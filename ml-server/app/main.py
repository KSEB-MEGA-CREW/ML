import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.models.model_loader import ModelManager
from app.api.endpoints import router
from app.utils.logger import setup_logger
from app.config import settings

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("🚀 AI Server starting")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    logger.info(f"TensorFlow GPU enabled: {settings.TF_ENABLE_GPU_MEMORY_GROWTH}")

    try:
        # Initialize model manager singleton
        model_manager = ModelManager()

        # Load model
        logger.info("📦 Starting model load...")
        success = await model_manager.load_model()

        if success:
            logger.info("✅ Model loaded successfully")
            model_info = model_manager.get_model_info()
            logger.info(f"Model info: {model_info}")
        else:
            logger.error("❌ Model loading failed")
            if not settings.DEBUG:
                raise RuntimeError("Model loading is required")

    except Exception as e:
        logger.error(f"❌ Server startup error: {e}")
        if not settings.DEBUG:
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


# Create FastAPI app
app = FastAPI(
    title="Sign Language AI Server",
    description="Sign Language Recognition AI Server - TensorFlow based",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request logging middleware (development only)
if settings.DEBUG:

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"📥 {request.method} {request.url.path}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"📤 {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )

        return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred",
        },
    )


# Register API router
app.include_router(router, prefix="/api/v1", tags=["AI Analysis"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    try:
        model_manager = ModelManager()

        return {
            "message": "Sign Language AI Server is running",
            "version": "1.0.0",
            "status": "healthy",
            "model_status": "loaded" if model_manager.is_ready() else "not_loaded",
            "framework": "TensorFlow",
            "async_enabled": True,
            "endpoints": {
                "health": "/api/v1/health",
                "analyze": "/api/v1/analyze-frame",
                "labels": "/api/v1/labels",
                "model_info": "/api/v1/model-info",
                "docs": "/docs",
            },
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {
            "message": "Sign Language AI Server",
            "status": "error",
            "error": str(e),
        }


# Development server
if __name__ == "__main__":
    import uvicorn

    host = settings.HOST
    port = settings.PORT
    reload = settings.DEBUG

    # debug
    logger.info(
        f"AWS_ACCESS_KEY_ID: {settings.AWS_ACCESS_KEY_ID[:10] if settings.AWS_ACCESS_KEY_ID else 'None'}..."
    )
    logger.info(
        f"AWS_SECRET_ACCESS_KEY: {'Set' if settings.AWS_SECRET_ACCESS_KEY else 'None'}"
    )
    logger.info(f"S3_BUCKET_NAME: {settings.S3_BUCKET_NAME}")
    logger.info(f"MODEL_S3_KEY: {settings.MODEL_S3_KEY}")

    logger.info(f"Development server starting: http://{host}:{port}")
    logger.info(f"API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1,  # Single worker for development
    )
