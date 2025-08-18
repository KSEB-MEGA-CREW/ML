# declare Websocket message type
from pydantic import BaseModel
from typing import List, Optional, Any
from enum import Enum


class MessageType(str, Enum):
    """WebSocket 메시지 타입"""

    KEYPOINTS = "keypoints"
    PREDICTION_RESULT = "prediction_result"
    SENTENCE_GENERATED = "sentence_generated"
    ERROR = "error"
    STATUS = "status"


class KeypointsMessage(BaseModel):
    """키포인트 데이터 메시지"""

    type: MessageType = MessageType.KEYPOINTS
    session_id: str
    user_id: str
    frame_index: int
    keypoints: List[float]  # 194개 특징점
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
