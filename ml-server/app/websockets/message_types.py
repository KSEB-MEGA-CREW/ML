# app/websockets/message_types.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time


class MessageType(str, Enum):
    """WebSocket 메시지 타입"""

    # client => server
    KEYPOINTS = "keypoints"
    TRANSLATION_START = "translation_start"
    TRANSLATION_END = "translation_end"
    PING = "ping"

    # server => client
    PREDICTION_RESULT = "prediction_result"
    TRANSLATION_RESULT = "translation_result"
    TRANSLATION_STATUS = "translation_status"
    ERROR = "error"
    PONG = "pong"


class BaseMessage(BaseModel):
    """base message structure"""

    type: MessageType
    timestamp: float = Field(default_factory=time.time)
    session_id: Optional[str] = None
    data: Optional[Dict[str,Any]] = None


class TranslationStartMessage(BaseMessage):
    """translation start message"""

    type: MessageType = MessageType.TRANSLATION_START
    user_id: str = Field(..., description="user ID")


class TranslationEndMessage(BaseMessage):
    """translation end message"""

    type: MessageType = MessageType.TRANSLATION_END
    user_id: str = Field(..., description="user ID")
    # force_translation: bool = Field(default=True, description= "강제 번역 수행 여부")


class TranslationStatusMessage(BaseMessage):
    """translation status message"""

    type: MessageType = MessageType.TRANSLATION_STATUS
    status: str = Field(..., description="번역 상태")
    session_id: str = Field(..., description="세션 ID")
    message: Optional[str] = None
    gloss_count: Optional[int] = None  # 현재 수집된 gloss 수


class TranslationResultMessage(BaseMessage):
    """translation result message"""

    type: MessageType = MessageType.TRANSLATION_RESULT
    sentence: str = Field(..., description="번역된 한국어 문장")
    gloss_sequence: List[str] = Field(..., description="사용된 gloss 시퀀스")
    confidence_avg: float = Field(..., description="평균 신뢰도")


class ErrorMessage(BaseMessage):
    """에러 메시지"""

    type: MessageType = MessageType.ERROR
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = None


class PingMessage(BaseMessage):
    """핑 메시지"""

    type: MessageType = MessageType.PING


class PongMessage(BaseMessage):
    """퐁 메시지"""

    type: MessageType = MessageType.PONG


class KeypointDataMessage(BaseMessage):
    """키포인트 데이터 메시지"""

    type: MessageType = MessageType.KEYPOINTS
    keypoints: List[float] = Field(..., description="194차원 키포인트 배열")
    frame_index: int = Field(..., description="프레임 순서")
    user_id: str = Field(..., description="사용자 ID")


class MessageFactory:
    """메시지 생성 헬퍼 클래스"""

    @staticmethod
    def create_ping_message(session_id: Optional[str] = None) -> PingMessage:
        """핑 메시지 생성 (오류 해결)"""
        return PingMessage(session_id=session_id)

    @staticmethod
    def create_pong_message(session_id: Optional[str] = None) -> PongMessage:
        """퐁 메시지 생성"""
        return PongMessage(session_id=session_id)

    @staticmethod
    def create_translation_result(
        session_id: str,
        sentence: str,
        gloss_sequence: List[str],
        confidence_avg: float,
    ) -> TranslationResultMessage:
        return TranslationResultMessage(
            session_id=session_id,
            sentence=sentence,
            gloss_sequence=gloss_sequence,
            confidence_avg=confidence_avg,
        )

    @staticmethod
    def create_translation_status(
        session_id: str,
        status: str,
        message: Optional[str] = None,
        gloss_count: Optional[int] = None,
    ) -> TranslationStatusMessage:
        return TranslationStatusMessage(
            session_id=session_id,
            status=status,
            message=message,
            gloss_count=gloss_count,
        )

    @staticmethod
    def create_error_message(
        session_id: str,
        error_code: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> ErrorMessage:
        """에러 메시지 생성"""
        return ErrorMessage(
            session_id=session_id,
            error_code=error_code,
            error_message=error_message,
            details=details,
        )

    @staticmethod
    def create_prediction_result(
        session_id: str, gloss: str, confidence: float, frame_count: int
    ) -> BaseMessage:
        """예측 결과 메시지 생성"""
        return BaseMessage(
            type=MessageType.PREDICTION_RESULT,
            session_id=session_id,
            data={"gloss": gloss, "confidence": confidence, "frame_count": frame_count},
        )
