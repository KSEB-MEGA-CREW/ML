# app/websockets/message_types.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Union
from enum import Enum
import numpy as np


class MessageType(str, Enum):
    """WebSocket 메시지 타입"""

    REGISTER_SESSION = "register_session"
    START_TRANSLATION = "start_translation"
    STOP_TRANSLATION = "stop_translation"
    KEYPOINTS = "keypoints"
    PREDICTION_RESULT = "prediction_result"
    SENTENCE_GENERATED = "sentence_generated"
    ERROR = "error"
    STATUS = "status"


class RegisterSessionMessage(BaseModel):
    """세션 등록 메시지"""

    type: MessageType = MessageType.REGISTER_SESSION
    session_id: str
    timestamp: float


class KeypointsMessage(BaseModel):
    """키포인트 데이터 메시지"""

    type: MessageType = MessageType.KEYPOINTS
    session_id: str
    frame_index: int
    keypoints: List[List[float]] = Field(
        ..., description="10프레임 배치: (10, 194)"
    )  # 🔄 수정
    timestamp: float

    @validator("keypoints")
    def validate_keypoints(cls, v):
        """키포인트 데이터 검증"""
        if not isinstance(v, list):
            raise ValueError("keypoints must be a list")

        # 배치 데이터인 경우: (10, 194)
        if len(v) == 10:
            for i, frame in enumerate(v):
                if not isinstance(frame, list) or len(frame) != 194:
                    raise ValueError(f"Frame {i} must be a list of 194 floats")
                # 각 요소를 float로 변환
                v[i] = [float(x) for x in frame]
        else:
            raise ValueError("keypoints must contain exactly 10 frames")

        return v


class StartTranslationMessage(BaseModel):
    """번역 시작 메시지"""  # 🔄 추가

    type: MessageType = MessageType.START_TRANSLATION
    session_id: str
    timestamp: float


class StopTranslationMessage(BaseModel):
    """번역 종료 메시지"""  # 🔄 추가

    type: MessageType = MessageType.STOP_TRANSLATION
    session_id: str
    timestamp: float


class PredictionResultMessage(BaseModel):
    """예측 결과 메시지"""

    type: MessageType = MessageType.PREDICTION_RESULT
    session_id: str
    label: str
    confidence: float
    frame_index: int
    timestamp: float


class SentenceGeneratedMessage(BaseModel):
    """문장 생성 메시지"""

    type: MessageType = MessageType.SENTENCE_GENERATED
    session_id: str
    sentence: str
    glosses: List[str]
    timestamp: float


class ErrorMessage(BaseModel):
    """에러 메시지"""

    type: MessageType = MessageType.ERROR
    session_id: str
    error_code: str
    error_message: str
    timestamp: float


class StatusMessage(BaseModel):
    """상태 메시지"""

    type: MessageType = MessageType.STATUS
    session_id: str
    status: str
    details: Optional[dict] = None
    timestamp: float
