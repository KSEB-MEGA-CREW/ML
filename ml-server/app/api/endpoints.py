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
    if not model_manager.is_loaded