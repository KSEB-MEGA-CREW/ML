# app/websockets/handlers.py
import json
import logging
import time
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any, List

from .message_types import (
    MessageType,
    KeypointsMessage,
    StartTranslationMessage,
    StopTranslationMessage,
    PredictionResultMessage,
    SentenceGeneratedMessage,
    ErrorMessage,
)
from .session_manager import session_manager
from ..services.auth_service import auth_service
from ..models.model_manager import model_manager
from ..services.gloss_collector import GlossCollector

logger = logging.getLogger(__name__)


class WebSocketHandler:
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}

    async def handle_connection(self, websocket: WebSocket, token: str):
        """WebSocket 연결 처리"""
        user_id = None
        try:
            logger.info(f"WebSocket 연결 시도: token={token[:20]}...")

            # 토큰 검증 및 에러 처리 강화
            user_info = await auth_service.verify_token(token)

            if not user_info or not isinstance(user_info, dict):
                logger.warning(f"토큰 검증 실패: user_info={user_info}")

                # 대안으로 JWT 직접 디코딩 시도
                user_id = auth_service.extract_user_id_from_token(token)
                if user_id:
                    user_info = {"user_id": user_id, "valid": True}
                    logger.info(f"JWT 직접 디코딩으로 user_id 추출: {user_id}")
                else:
                    await websocket.close(code=1008, reason="Invalid token")
                    return

            await websocket.accept()

            # user_id 추출 방식 개선
            if isinstance(user_info, dict):
                user_id = (
                    user_info.get("user_id")
                    or user_info.get("userId")
                    or user_info.get("sub")
                )

            if not user_id:
                logger.error(f"user_id를 찾을 수 없음: user_info={user_info}")
                await websocket.close(code=1008, reason="User ID not found")
                return

            user_id = str(user_id)  # 문자열로 변환
            logger.info(f"WebSocket 연결 승인: user_id={user_id}")

            # 메시지 루프
            while True:
                try:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)

                    await self.handle_message(websocket, message_data, user_id)

                except WebSocketDisconnect:
                    logger.info(f"WebSocket 연결 종료: user_id={user_id}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 에러: {e}")
                    await self.send_error(
                        websocket, "", "INVALID_JSON", "Invalid JSON format"
                    )
                except Exception as e:
                    logger.error(f"메시지 처리 에러: {e}")
                    await self.send_error(websocket, "", "MESSAGE_ERROR", str(e))

        except WebSocketDisconnect:
            logger.info(f"WebSocket 연결 해제: user_id={user_id}")
        except Exception as e:
            logger.error(f"WebSocket 연결 에러: {e}")
            logger.exception("상세 에러 정보:")
        finally:
            # 정리 작업
            await self.cleanup_user_sessions(user_id)

    async def cleanup_user_sessions(self, user_id: str):
        """사용자 세션 정리"""
        if not user_id:
            return

        sessions_to_remove = []
        for session_id, session_info in self.active_sessions.items():
            if session_info.get("user_id") == user_id:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            logger.info(f"세션 정리: session_id={session_id}, user_id={user_id}")

    async def handle_message(
        self, websocket: WebSocket, data: Dict[Any, Any], user_id: str
    ):
        """메시지 타입별 처리"""
        message_type = data.get("type")
        session_id = data.get("session_id")

        if not message_type:
            await self.send_error(
                websocket, "", "MISSING_TYPE", "Message type is required"
            )
            return

        if not session_id:
            await self.send_error(
                websocket, "", "MISSING_SESSION_ID", "Session ID is required"
            )
            return

        try:
            # 메시지 타입별 처리
            if message_type == MessageType.START_TRANSLATION:
                await self.handle_start_translation(websocket, data, user_id)
            elif message_type == MessageType.STOP_TRANSLATION:
                await self.handle_stop_translation(websocket, data, user_id)
            elif message_type == MessageType.KEYPOINTS:
                await self.handle_keypoints(websocket, data, user_id)
            else:
                logger.warning(f"알 수 없는 메시지 타입: {message_type}")
                await self.send_error(
                    websocket,
                    session_id,
                    "UNKNOWN_MESSAGE_TYPE",
                    f"Unknown type: {message_type}",
                )

        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            logger.exception("메시지 처리 에러 상세:")
            await self.send_error(websocket, session_id, "PROCESSING_ERROR", str(e))

    async def handle_start_translation(
        self, websocket: WebSocket, data: Dict, user_id: str
    ):
        """번역 시작 처리"""
        try:
            message = StartTranslationMessage(**data)
            session_id = message.session_id

            # 기존 세션이 있으면 정리
            if session_id in self.active_sessions:
                logger.info(f"기존 세션 정리: session_id={session_id}")
                del self.active_sessions[session_id]

            # 새 세션 초기화
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "websocket": websocket,
                "gloss_collector": GlossCollector(),
                "start_time": time.time(),
                "frame_count": 0,
            }

            logger.info(f"번역 세션 시작: session_id={session_id}, user_id={user_id}")

            # 상태 메시지 전송
            await self.send_status(websocket, session_id, "translation_started")

        except Exception as e:
            logger.error(f"번역 시작 처리 실패: {e}")
            await self.send_error(
                websocket, data.get("session_id", ""), "START_TRANSLATION_ERROR", str(e)
            )

    async def handle_stop_translation(
        self, websocket: WebSocket, data: Dict, user_id: str
    ):
        """번역 종료 처리"""
        try:
            message = StopTranslationMessage(**data)
            session_id = message.session_id

            # 세션 정리
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]

                # 마지막 문장 생성 시도
                gloss_collector = session_info.get("gloss_collector")
                if gloss_collector and len(gloss_collector.glosses) > 0:
                    try:
                        final_sentence = await gloss_collector._generate_sentence()
                        if final_sentence:
                            await self.send_sentence(
                                websocket,
                                session_id,
                                final_sentence,
                                list(gloss_collector.glosses),
                            )
                    except Exception as e:
                        logger.error(f"마지막 문장 생성 실패: {e}")

                del self.active_sessions[session_id]
                logger.info(f"번역 세션 종료: session_id={session_id}")

            # 상태 메시지 전송
            await self.send_status(websocket, session_id, "translation_stopped")

        except Exception as e:
            logger.error(f"번역 종료 처리 실패: {e}")
            await self.send_error(
                websocket, data.get("session_id", ""), "STOP_TRANSLATION_ERROR", str(e)
            )

    async def handle_keypoints(self, websocket: WebSocket, data: Dict, user_id: str):
        """키포인트 처리"""
        try:
            # Pydantic 모델로 검증
            message = KeypointsMessage(**data)
            session_id = message.session_id

            # 세션 확인
            if session_id not in self.active_sessions:
                logger.warning(f"활성 세션 없음: session_id={session_id}")
                await self.send_error(
                    websocket,
                    session_id,
                    "SESSION_NOT_FOUND",
                    "Translation session not found",
                )
                return

            session_info = self.active_sessions[session_id]
            session_info["frame_count"] += 1

            # 키포인트 데이터 변환 (10, 194) → (1, 10, 194)
            keypoints_array = np.array(message.keypoints, dtype=np.float32)

            # 데이터 형태 검증
            if keypoints_array.shape != (10, 194):
                logger.error(
                    f"잘못된 키포인트 형태: {keypoints_array.shape}, 예상: (10, 194)"
                )
                await self.send_error(
                    websocket,
                    session_id,
                    "INVALID_KEYPOINTS_SHAPE",
                    f"Expected shape (10, 194), got {keypoints_array.shape}",
                )
                return

            keypoints_batch = keypoints_array.reshape(1, 10, 194)

            logger.debug(
                f"키포인트 배치 처리: shape={keypoints_batch.shape}, session_id={session_id}"
            )

            # 모델 예측
            try:
                prediction = await model_manager.predict(keypoints_batch)
                label = prediction.get("label", "")
                confidence = float(prediction.get("confidence", 0.0))

                logger.info(f"예측 결과: label={label}, confidence={confidence:.3f}")

                # 개별 예측 결과 전송
                await self.send_prediction(
                    websocket, session_id, label, confidence, message.frame_index
                )

                # GlossCollector로 문장 생성 확인
                gloss_collector = session_info.get("gloss_collector")
                if gloss_collector:
                    sentence = await gloss_collector.add_prediction(label, confidence)
                    if sentence:
                        logger.info(f"문장 생성됨: {sentence}")
                        await self.send_sentence(
                            websocket,
                            session_id,
                            sentence,
                            list(gloss_collector.glosses),
                        )

            except Exception as model_error:
                logger.error(f"모델 예측 실패: {model_error}")
                await self.send_error(
                    websocket, session_id, "MODEL_PREDICTION_ERROR", str(model_error)
                )

        except ValueError as ve:
            logger.error(f"키포인트 데이터 검증 실패: {ve}")
            await self.send_error(
                websocket, data.get("session_id", ""), "VALIDATION_ERROR", str(ve)
            )
        except Exception as e:
            logger.error(f"키포인트 처리 실패: {e}")
            logger.exception("키포인트 처리 에러 상세:")
            await self.send_error(
                websocket, data.get("session_id", ""), "KEYPOINTS_ERROR", str(e)
            )

    async def send_prediction(
        self,
        websocket: WebSocket,
        session_id: str,
        label: str,
        confidence: float,
        frame_index: int,
    ):
        """예측 결과 전송"""
        try:
            message = PredictionResultMessage(
                session_id=session_id,
                label=label,
                confidence=confidence,
                frame_index=frame_index,
                timestamp=time.time(),
            )
            await websocket.send_text(message.model_dump_json())
            logger.debug(f"예측 결과 전송: {label} ({confidence:.3f})")
        except Exception as e:
            logger.error(f"예측 결과 전송 실패: {e}")

    async def send_sentence(
        self, websocket: WebSocket, session_id: str, sentence: str, glosses: List[str]
    ):
        """문장 생성 결과 전송"""
        try:
            message = SentenceGeneratedMessage(
                session_id=session_id,
                sentence=sentence,
                glosses=glosses,
                timestamp=time.time(),
            )
            await websocket.send_text(message.model_dump_json())
            logger.info(f"문장 전송: {sentence}")
        except Exception as e:
            logger.error(f"문장 전송 실패: {e}")

    async def send_status(self, websocket: WebSocket, session_id: str, status: str):
        """상태 메시지 전송"""
        try:
            message = {
                "type": MessageType.STATUS,
                "session_id": session_id,
                "status": status,
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(message))
            logger.debug(f"상태 메시지 전송: {status}")
        except Exception as e:
            logger.error(f"상태 메시지 전송 실패: {e}")

    async def send_error(
        self, websocket: WebSocket, session_id: str, error_code: str, error_message: str
    ):
        """에러 메시지 전송"""
        try:
            message = ErrorMessage(
                session_id=session_id,
                error_code=error_code,
                error_message=error_message,
                timestamp=time.time(),
            )
            await websocket.send_text(message.model_dump_json())
            logger.warning(f"에러 메시지 전송: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"에러 메시지 전송 실패: {e}")


# 싱글톤 인스턴스
websocket_handler = WebSocketHandler()
